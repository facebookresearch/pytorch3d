# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
import torch.nn as nn
from common_testing import TestCaseMixin, get_random_cuda_device
from pytorch3d import _C
from pytorch3d.ops.graph_conv import GraphConv, gather_scatter, gather_scatter_python
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils import ico_sphere


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
        verts_cuda = verts.clone().to(device)
        edges_cuda = edges.clone().to(device)
        verts.requires_grad = True
        verts_cuda.requires_grad = True

        neighbor_sums_cuda = gather_scatter(verts_cuda, edges_cuda, False)
        neighbor_sums = gather_scatter_python(verts, edges, False)
        neighbor_sums_cuda.sum().backward()
        neighbor_sums.sum().backward()

        self.assertClose(verts.grad.cpu(), verts_cuda.grad.cpu())

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

        # output
        output_cpu = gather_scatter_python(input, edges, False)
        output_cuda = _C.gather_scatter(
            input.to(device=device), edges.to(device=device), False, False
        )
        self.assertClose(output_cuda.cpu(), output_cpu)
        with self.assertRaises(Exception) as err:
            _C.gather_scatter(input.cpu(), edges.cpu(), False, False)
        self.assertTrue("Not implemented on the CPU" in str(err.exception))

        # directed
        output_cpu = gather_scatter_python(input, edges, True)
        output_cuda = _C.gather_scatter(
            input.to(device=device), edges.to(device=device), True, False
        )
        self.assertClose(output_cuda.cpu(), output_cpu)

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
