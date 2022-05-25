# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn
from pytorch3d import _C
from pytorch3d.ops.graph_conv import gather_scatter, gather_scatter_python, GraphConv
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils import ico_sphere

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestGraphConv(TestCaseMixin, unittest.TestCase):
    def test_undirected(self):
        dtype = torch.float32
        device = get_random_cuda_device()
        verts = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype, device=device
        )
        edges = torch.tensor([[0, 1], [0, 2]], device=device)
        w0 = torch.tensor([[1, 1, 1]], dtype=dtype, device=device)
        w1 = torch.tensor([[-1, -1, -1]], dtype=dtype, device=device)

        expected_y = torch.tensor(
            [
                [1 + 2 + 3 - 4 - 5 - 6 - 7 - 8 - 9],
                [4 + 5 + 6 - 1 - 2 - 3],
                [7 + 8 + 9 - 1 - 2 - 3],
            ],
            dtype=dtype,
            device=device,
        )

        conv = GraphConv(3, 1, directed=False).to(device)
        conv.w0.weight.data.copy_(w0)
        conv.w0.bias.data.zero_()
        conv.w1.weight.data.copy_(w1)
        conv.w1.bias.data.zero_()

        y = conv(verts, edges)
        self.assertClose(y, expected_y)

    def test_no_edges(self):
        dtype = torch.float32
        verts = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        edges = torch.zeros(0, 2, dtype=torch.int64)
        w0 = torch.tensor([[1, -1, -2]], dtype=dtype)
        expected_y = torch.tensor(
            [[1 - 2 - 2 * 3], [4 - 5 - 2 * 6], [7 - 8 - 2 * 9]], dtype=dtype
        )
        conv = GraphConv(3, 1).to(dtype)
        conv.w0.weight.data.copy_(w0)
        conv.w0.bias.data.zero_()

        y = conv(verts, edges)
        self.assertClose(y, expected_y)

    def test_no_verts_and_edges(self):
        dtype = torch.float32
        verts = torch.tensor([], dtype=dtype, requires_grad=True)
        edges = torch.tensor([], dtype=dtype)
        w0 = torch.tensor([[1, -1, -2]], dtype=dtype)

        conv = GraphConv(3, 1).to(dtype)
        conv.w0.weight.data.copy_(w0)
        conv.w0.bias.data.zero_()
        y = conv(verts, edges)
        self.assertClose(y, torch.zeros((0, 1)))
        self.assertTrue(y.requires_grad)

        conv2 = GraphConv(3, 2).to(dtype)
        conv2.w0.weight.data.copy_(w0.repeat(2, 1))
        conv2.w0.bias.data.zero_()
        y = conv2(verts, edges)
        self.assertClose(y, torch.zeros((0, 2)))
        self.assertTrue(y.requires_grad)

    def test_directed(self):
        dtype = torch.float32
        verts = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=dtype)
        edges = torch.tensor([[0, 1], [0, 2]])
        w0 = torch.tensor([[1, 1, 1]], dtype=dtype)
        w1 = torch.tensor([[-1, -1, -1]], dtype=dtype)

        expected_y = torch.tensor(
            [[1 + 2 + 3 - 4 - 5 - 6 - 7 - 8 - 9], [4 + 5 + 6], [7 + 8 + 9]], dtype=dtype
        )

        conv = GraphConv(3, 1, directed=True).to(dtype)
        conv.w0.weight.data.copy_(w0)
        conv.w0.bias.data.zero_()
        conv.w1.weight.data.copy_(w1)
        conv.w1.bias.data.zero_()

        y = conv(verts, edges)
        self.assertClose(y, expected_y)

    def test_backward(self):
        device = get_random_cuda_device()
        mesh = ico_sphere()
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        verts_cpu = verts.clone()
        edges_cpu = edges.clone()
        verts_cuda = verts.clone().to(device)
        edges_cuda = edges.clone().to(device)
        verts.requires_grad = True
        verts_cpu.requires_grad = True
        verts_cuda.requires_grad = True

        neighbor_sums_cuda = gather_scatter(verts_cuda, edges_cuda, False)
        neighbor_sums_cpu = gather_scatter(verts_cpu, edges_cpu, False)
        neighbor_sums = gather_scatter_python(verts, edges, False)
        randoms = torch.rand_like(neighbor_sums)
        (neighbor_sums_cuda * randoms.to(device)).sum().backward()
        (neighbor_sums_cpu * randoms).sum().backward()
        (neighbor_sums * randoms).sum().backward()

        self.assertClose(verts.grad, verts_cuda.grad.cpu())
        self.assertClose(verts.grad, verts_cpu.grad)

    def test_repr(self):
        conv = GraphConv(32, 64, directed=True)
        self.assertEqual(repr(conv), "GraphConv(32 -> 64, directed=True)")

    def test_cpu_cuda_tensor_error(self):
        device = get_random_cuda_device()
        verts = torch.tensor(
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32, device=device
        )
        edges = torch.tensor([[0, 1], [0, 2]])
        conv = GraphConv(3, 1, directed=True).to(torch.float32)
        with self.assertRaises(Exception) as err:
            conv(verts, edges)
        self.assertTrue("tensors must be on the same device." in str(err.exception))

    def test_gather_scatter(self):
        """
        Check gather_scatter cuda and python versions give the same results.
        Check that gather_scatter cuda version throws an error if cpu tensors
        are given as input.
        """
        device = get_random_cuda_device()
        mesh = ico_sphere()
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        w0 = nn.Linear(3, 1)
        input = w0(verts)

        # undirected
        output_python = gather_scatter_python(input, edges, False)
        output_cuda = _C.gather_scatter(
            input.to(device=device), edges.to(device=device), False, False
        )
        self.assertClose(output_cuda.cpu(), output_python)

        output_cpu = _C.gather_scatter(input.cpu(), edges.cpu(), False, False)
        self.assertClose(output_cpu, output_python)

        # directed
        output_python = gather_scatter_python(input, edges, True)
        output_cuda = _C.gather_scatter(
            input.to(device=device), edges.to(device=device), True, False
        )
        self.assertClose(output_cuda.cpu(), output_python)
        output_cpu = _C.gather_scatter(input.cpu(), edges.cpu(), True, False)
        self.assertClose(output_cpu, output_python)

    @staticmethod
    def graph_conv_forward_backward(
        gconv_dim,
        num_meshes,
        num_verts,
        num_faces,
        directed: bool,
        backend: str = "cuda",
    ):
        device = torch.device("cuda") if backend == "cuda" else "cpu"
        verts_list = torch.tensor(num_verts * [[0.11, 0.22, 0.33]], device=device).view(
            -1, 3
        )
        faces_list = torch.tensor(num_faces * [[1, 2, 3]], device=device).view(-1, 3)
        meshes = Meshes(num_meshes * [verts_list], num_meshes * [faces_list])
        gconv = GraphConv(gconv_dim, gconv_dim, directed=directed)
        gconv.to(device)
        edges = meshes.edges_packed()
        total_verts = meshes.verts_packed().shape[0]

        # Features.
        x = torch.randn(total_verts, gconv_dim, device=device, requires_grad=True)
        torch.cuda.synchronize()

        def run_graph_conv():
            y1 = gconv(x, edges)
            y1.sum().backward()
            torch.cuda.synchronize()

        return run_graph_conv
