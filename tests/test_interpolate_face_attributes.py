# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.ops.interp_face_attrs import (
    interpolate_face_attributes,
    interpolate_face_attributes_python,
)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestInterpolateFaceAttributes(TestCaseMixin, unittest.TestCase):
    def _test_interp_face_attrs(self, interp_fun, device):
        pix_to_face = [0, 2, -1, 0, 1, -1]
        barycentric_coords = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.8, 0.0, 0.2],
            [0.25, 0.5, 0.25],
        ]
        face_attrs = [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[13, 14], [15, 16], [17, 18]],
        ]
        pix_attrs = [
            [1, 2],
            [15, 16],
            [0, 0],
            [2, 3],
            [0.8 * 7 + 0.2 * 11, 0.8 * 8 + 0.2 * 12],
            [0, 0],
        ]
        N, H, W, K, D = 1, 2, 1, 3, 2
        pix_to_face = torch.tensor(pix_to_face, dtype=torch.int64, device=device)
        pix_to_face = pix_to_face.view(N, H, W, K)
        barycentric_coords = torch.tensor(
            barycentric_coords, dtype=torch.float32, device=device
        )
        barycentric_coords = barycentric_coords.view(N, H, W, K, 3)
        face_attrs = torch.tensor(face_attrs, dtype=torch.float32, device=device)
        pix_attrs = torch.tensor(pix_attrs, dtype=torch.float32, device=device)
        pix_attrs = pix_attrs.view(N, H, W, K, D)

        args = (pix_to_face, barycentric_coords, face_attrs)
        pix_attrs_actual = interp_fun(*args)
        self.assertClose(pix_attrs_actual, pix_attrs)

    def test_python(self):
        device = torch.device("cuda:0")
        self._test_interp_face_attrs(interpolate_face_attributes_python, device)

    def test_cuda(self):
        device = torch.device("cuda:0")
        self._test_interp_face_attrs(interpolate_face_attributes, device)

    def test_python_vs_cuda(self):
        N, H, W, K = 2, 32, 32, 5
        F = 1000
        D = 3
        device = get_random_cuda_device()
        torch.manual_seed(598)
        pix_to_face = torch.randint(-F, F, (N, H, W, K), device=device)
        barycentric_coords = torch.randn(
            N, H, W, K, 3, device=device, requires_grad=True
        )
        face_attrs = torch.randn(F, 3, D, device=device, requires_grad=True)
        grad_pix_attrs = torch.randn(N, H, W, K, D, device=device)
        args = (pix_to_face, barycentric_coords, face_attrs)

        # Run the python version
        pix_attrs_py = interpolate_face_attributes_python(*args)
        pix_attrs_py.backward(gradient=grad_pix_attrs)
        grad_bary_py = barycentric_coords.grad.clone()
        grad_face_attrs_py = face_attrs.grad.clone()

        # Clear gradients
        barycentric_coords.grad.zero_()
        face_attrs.grad.zero_()

        # Run the CUDA version
        pix_attrs_cu = interpolate_face_attributes(*args)
        pix_attrs_cu.backward(gradient=grad_pix_attrs)
        grad_bary_cu = barycentric_coords.grad.clone()
        grad_face_attrs_cu = face_attrs.grad.clone()

        # Check they are the same
        self.assertClose(pix_attrs_py, pix_attrs_cu, rtol=2e-3)
        self.assertClose(grad_bary_py, grad_bary_cu, rtol=1e-4)
        self.assertClose(grad_face_attrs_py, grad_face_attrs_cu, rtol=1e-3)

    def test_interpolate_attributes(self):
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        vert_tex = torch.tensor(
            [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32
        )
        tex = TexturesVertex(verts_features=vert_tex[None, :])
        mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        pix_to_face = torch.tensor([0, 1], dtype=torch.int64).view(1, 1, 1, 2)
        barycentric_coords = torch.tensor(
            [[0.5, 0.3, 0.2], [0.3, 0.6, 0.1]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        expected_vals = torch.tensor(
            [[0.5, 1.0, 0.3], [0.3, 1.0, 0.9]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=barycentric_coords,
            zbuf=torch.ones_like(pix_to_face),
            dists=torch.ones_like(pix_to_face),
        )

        verts_features_packed = mesh.textures.verts_features_packed()
        faces_verts_features = verts_features_packed[mesh.faces_packed()]

        texels = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_features
        )
        self.assertTrue(torch.allclose(texels, expected_vals[None, :]))

    def test_interpolate_attributes_grad(self):
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        vert_tex = torch.tensor(
            [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tex = TexturesVertex(verts_features=vert_tex[None, :])
        mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        pix_to_face = torch.tensor([0, 1], dtype=torch.int64).view(1, 1, 1, 2)
        barycentric_coords = torch.tensor(
            [[0.5, 0.3, 0.2], [0.3, 0.6, 0.1]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=barycentric_coords,
            zbuf=torch.ones_like(pix_to_face),
            dists=torch.ones_like(pix_to_face),
        )
        grad_vert_tex = torch.tensor(
            [[0.3, 0.3, 0.3], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]],
            dtype=torch.float32,
        )
        verts_features_packed = mesh.textures.verts_features_packed()
        faces_verts_features = verts_features_packed[mesh.faces_packed()]

        texels = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_features
        )
        texels.sum().backward()
        self.assertTrue(hasattr(vert_tex, "grad"))
        self.assertTrue(torch.allclose(vert_tex.grad, grad_vert_tex[None, :]))

    def test_interpolate_face_attributes_fail(self):
        # 1. A face can only have 3 verts
        #   i.e. face_attributes must have shape (F, 3, D)
        face_attributes = torch.ones(1, 4, 3)
        pix_to_face = torch.ones((1, 1, 1, 1))
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=pix_to_face[..., None].expand(-1, -1, -1, -1, 3),
            zbuf=pix_to_face,
            dists=pix_to_face,
        )
        with self.assertRaises(ValueError):
            interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, face_attributes
            )

        # 2. pix_to_face must have shape (N, H, W, K)
        pix_to_face = torch.ones((1, 1, 1, 1, 3))
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=pix_to_face,
            zbuf=pix_to_face,
            dists=pix_to_face,
        )
        with self.assertRaises(ValueError):
            interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, face_attributes
            )
