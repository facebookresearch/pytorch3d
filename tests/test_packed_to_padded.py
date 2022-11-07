# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.ops import packed_to_padded, padded_to_packed
from pytorch3d.structures.meshes import Meshes

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestPackedToPadded(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    @staticmethod
    def init_meshes(
        num_meshes: int = 10,
        num_verts: int = 1000,
        num_faces: int = 3000,
        device: str = "cpu",
    ):
        device = torch.device(device)
        verts_list = []
        faces_list = []
        for _ in range(num_meshes):
            verts = torch.rand((num_verts, 3), dtype=torch.float32, device=device)
            faces = torch.randint(
                num_verts, size=(num_faces, 3), dtype=torch.int64, device=device
            )
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts_list, faces_list)

        return meshes

    @staticmethod
    def packed_to_padded_python(inputs, first_idxs, max_size, device):
        """
        PyTorch implementation of packed_to_padded function.
        """
        num_meshes = first_idxs.size(0)
        if inputs.dim() == 1:
            inputs_padded = torch.zeros((num_meshes, max_size), device=device)
        else:
            inputs_padded = torch.zeros(
                (num_meshes, max_size, *inputs.shape[1:]), device=device
            )
        for m in range(num_meshes):
            s = first_idxs[m]
            if m == num_meshes - 1:
                f = inputs.shape[0]
            else:
                f = first_idxs[m + 1]
            inputs_padded[m, : f - s] = inputs[s:f]

        return inputs_padded

    @staticmethod
    def padded_to_packed_python(inputs, first_idxs, num_inputs, device):
        """
        PyTorch implementation of padded_to_packed function.
        """
        num_meshes = inputs.size(0)
        if inputs.dim() == 2:
            inputs_packed = torch.zeros((num_inputs,), device=device)
        else:
            inputs_packed = torch.zeros((num_inputs, *inputs.shape[2:]), device=device)
        for m in range(num_meshes):
            s = first_idxs[m]
            if m == num_meshes - 1:
                f = num_inputs
            else:
                f = first_idxs[m + 1]
            inputs_packed[s:f] = inputs[m, : f - s]

        return inputs_packed

    def _test_packed_to_padded_helper(self, dims, device):
        """
        Check the results from packed_to_padded and PyTorch implementations
        are the same.
        """
        meshes = self.init_meshes(16, 100, 300, device=device)
        faces = meshes.faces_packed()
        mesh_to_faces_packed_first_idx = meshes.mesh_to_faces_packed_first_idx()
        max_faces = meshes.num_faces_per_mesh().max().item()

        if len(dims) == 0:
            values = torch.rand((faces.shape[0],), device=device, requires_grad=True)
        else:
            values = torch.rand(
                (faces.shape[0], *dims), device=device, requires_grad=True
            )
        values_torch = values.detach().clone()
        values_torch.requires_grad = True
        values_padded = packed_to_padded(
            values, mesh_to_faces_packed_first_idx, max_faces
        )
        values_padded_torch = TestPackedToPadded.packed_to_padded_python(
            values_torch, mesh_to_faces_packed_first_idx, max_faces, device
        )
        # check forward
        self.assertClose(values_padded, values_padded_torch)

        # check backward
        if len(dims) == 0:
            grad_inputs = torch.rand((len(meshes), max_faces), device=device)
        else:
            grad_inputs = torch.rand((len(meshes), max_faces, *dims), device=device)
        values_padded.backward(grad_inputs)
        grad_outputs = values.grad
        values_padded_torch.backward(grad_inputs)
        grad_outputs_torch1 = values_torch.grad
        grad_outputs_torch2 = TestPackedToPadded.padded_to_packed_python(
            grad_inputs, mesh_to_faces_packed_first_idx, values.size(0), device=device
        )
        self.assertClose(grad_outputs, grad_outputs_torch1)
        self.assertClose(grad_outputs, grad_outputs_torch2)

    def test_packed_to_padded_flat_cpu(self):
        self._test_packed_to_padded_helper([], "cpu")

    def test_packed_to_padded_D1_cpu(self):
        self._test_packed_to_padded_helper([1], "cpu")

    def test_packed_to_padded_D16_cpu(self):
        self._test_packed_to_padded_helper([16], "cpu")

    def test_packed_to_padded_D16_9_cpu(self):
        self._test_packed_to_padded_helper([16, 9], "cpu")

    def test_packed_to_padded_D16_3_2_cpu(self):
        self._test_packed_to_padded_helper([16, 3, 2], "cpu")

    def test_packed_to_padded_flat_cuda(self):
        device = get_random_cuda_device()
        self._test_packed_to_padded_helper([], device)

    def test_packed_to_padded_D1_cuda(self):
        device = get_random_cuda_device()
        self._test_packed_to_padded_helper([1], device)

    def test_packed_to_padded_D16_cuda(self):
        device = get_random_cuda_device()
        self._test_packed_to_padded_helper([16], device)

    def test_packed_to_padded_D16_9_cuda(self):
        device = get_random_cuda_device()
        self._test_packed_to_padded_helper([16, 9], device)

    def test_packed_to_padded_D16_3_2_cuda(self):
        device = get_random_cuda_device()
        self._test_packed_to_padded_helper([16, 3, 2], device)

    def _test_padded_to_packed_helper(self, dims, device):
        """
        Check the results from packed_to_padded and PyTorch implementations
        are the same.
        """
        meshes = self.init_meshes(16, 100, 300, device=device)
        mesh_to_faces_packed_first_idx = meshes.mesh_to_faces_packed_first_idx()
        num_faces_per_mesh = meshes.num_faces_per_mesh()
        max_faces = num_faces_per_mesh.max().item()
        if len(dims) == 0:
            values = torch.rand((len(meshes), max_faces), device=device)
        else:
            values = torch.rand((len(meshes), max_faces, *dims), device=device)
        for i, num in enumerate(num_faces_per_mesh):
            values[i, num:] = 0
        values.requires_grad = True
        values_torch = values.detach().clone()
        values_torch.requires_grad = True
        values_packed = padded_to_packed(
            values, mesh_to_faces_packed_first_idx, num_faces_per_mesh.sum().item()
        )
        values_packed_torch = TestPackedToPadded.padded_to_packed_python(
            values_torch,
            mesh_to_faces_packed_first_idx,
            num_faces_per_mesh.sum().item(),
            device,
        )
        # check forward
        self.assertClose(values_packed, values_packed_torch)

        if len(dims) > 0:
            values_packed_dim2 = padded_to_packed(
                values.transpose(1, 2),
                mesh_to_faces_packed_first_idx,
                num_faces_per_mesh.sum().item(),
                max_size_dim=2,
            )
            # check forward
            self.assertClose(values_packed_dim2, values_packed_torch)

        # check backward
        if len(dims) == 0:
            grad_inputs = torch.rand((num_faces_per_mesh.sum().item()), device=device)
        else:
            grad_inputs = torch.rand(
                (num_faces_per_mesh.sum().item(), *dims), device=device
            )
        values_packed.backward(grad_inputs)
        grad_outputs = values.grad
        values_packed_torch.backward(grad_inputs)
        grad_outputs_torch1 = values_torch.grad
        grad_outputs_torch2 = TestPackedToPadded.packed_to_padded_python(
            grad_inputs, mesh_to_faces_packed_first_idx, values.size(1), device=device
        )
        self.assertClose(grad_outputs, grad_outputs_torch1)
        self.assertClose(grad_outputs, grad_outputs_torch2)

    def test_padded_to_packed_flat_cpu(self):
        self._test_padded_to_packed_helper([], "cpu")

    def test_padded_to_packed_D1_cpu(self):
        self._test_padded_to_packed_helper([1], "cpu")

    def test_padded_to_packed_D16_cpu(self):
        self._test_padded_to_packed_helper([16], "cpu")

    def test_padded_to_packed_D16_9_cpu(self):
        self._test_padded_to_packed_helper([16, 9], "cpu")

    def test_padded_to_packed_D16_3_2_cpu(self):
        self._test_padded_to_packed_helper([16, 3, 2], "cpu")

    def test_padded_to_packed_flat_cuda(self):
        device = get_random_cuda_device()
        self._test_padded_to_packed_helper([], device)

    def test_padded_to_packed_D1_cuda(self):
        device = get_random_cuda_device()
        self._test_padded_to_packed_helper([1], device)

    def test_padded_to_packed_D16_cuda(self):
        device = get_random_cuda_device()
        self._test_padded_to_packed_helper([16], device)

    def test_padded_to_packed_D16_9_cuda(self):
        device = get_random_cuda_device()
        self._test_padded_to_packed_helper([16, 9], device)

    def test_padded_to_packed_D16_3_2_cuda(self):
        device = get_random_cuda_device()
        self._test_padded_to_packed_helper([16, 3, 2], device)

    @staticmethod
    def packed_to_padded_with_init(
        num_meshes: int, num_verts: int, num_faces: int, num_d: int, device: str = "cpu"
    ):
        meshes = TestPackedToPadded.init_meshes(
            num_meshes, num_verts, num_faces, device
        )
        faces = meshes.faces_packed()
        mesh_to_faces_packed_first_idx = meshes.mesh_to_faces_packed_first_idx()
        max_faces = meshes.num_faces_per_mesh().max().item()
        if num_d == 0:
            values = torch.rand((faces.shape[0],), device=meshes.device)
        else:
            values = torch.rand((faces.shape[0], num_d), device=meshes.device)
        torch.cuda.synchronize()

        def out():
            packed_to_padded(values, mesh_to_faces_packed_first_idx, max_faces)
            torch.cuda.synchronize()

        return out

    @staticmethod
    def packed_to_padded_with_init_torch(
        num_meshes: int, num_verts: int, num_faces: int, num_d: int, device: str = "cpu"
    ):
        meshes = TestPackedToPadded.init_meshes(
            num_meshes, num_verts, num_faces, device
        )
        faces = meshes.faces_packed()
        mesh_to_faces_packed_first_idx = meshes.mesh_to_faces_packed_first_idx()
        max_faces = meshes.num_faces_per_mesh().max().item()
        if num_d == 0:
            values = torch.rand((faces.shape[0],), device=meshes.device)
        else:
            values = torch.rand((faces.shape[0], num_d), device=meshes.device)
        torch.cuda.synchronize()

        def out():
            TestPackedToPadded.packed_to_padded_python(
                values, mesh_to_faces_packed_first_idx, max_faces, device
            )
            torch.cuda.synchronize()

        return out
