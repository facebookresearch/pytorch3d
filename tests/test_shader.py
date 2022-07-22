# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shader import (
    HardDepthShader,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftDepthShader,
    SoftPhongShader,
    SplatterPhongShader,
)
from pytorch3d.structures.meshes import Meshes

from .common_testing import TestCaseMixin


class TestShader(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        self.shader_classes = [
            HardDepthShader,
            HardFlatShader,
            HardGouraudShader,
            HardPhongShader,
            SoftDepthShader,
            SoftPhongShader,
            SplatterPhongShader,
        ]

    def test_to(self):
        cpu_device = torch.device("cpu")
        cuda_device = torch.device("cuda:0")

        R, T = look_at_view_transform()

        for shader_class in self.shader_classes:
            for cameras_class in (None, PerspectiveCameras):
                if cameras_class is None:
                    cameras = None
                else:
                    cameras = PerspectiveCameras(device=cpu_device, R=R, T=T)

                cpu_shader = shader_class(device=cpu_device, cameras=cameras)
                if cameras is None:
                    self.assertIsNone(cpu_shader.cameras)
                else:
                    self.assertEqual(cpu_device, cpu_shader.cameras.device)
                self.assertEqual(cpu_device, cpu_shader.materials.device)
                self.assertEqual(cpu_device, cpu_shader.lights.device)

                cuda_shader = cpu_shader.to(cuda_device)
                self.assertIs(cpu_shader, cuda_shader)
                if cameras is None:
                    self.assertIsNone(cuda_shader.cameras)
                    with self.assertRaisesRegex(ValueError, "Cameras must be"):
                        cuda_shader._get_cameras()
                else:
                    self.assertEqual(cuda_device, cuda_shader.cameras.device)
                    self.assertIsInstance(cuda_shader._get_cameras(), cameras_class)
                self.assertEqual(cuda_device, cuda_shader.materials.device)
                self.assertEqual(cuda_device, cuda_shader.lights.device)

    def test_cameras_check(self):
        verts = torch.tensor(
            [[-1, -1, 0], [1, -1, 1], [1, 1, 0], [-1, 1, 1]], dtype=torch.float32
        )
        faces = torch.tensor([[0, 1, 2], [2, 3, 0]], dtype=torch.int64)
        meshes = Meshes(verts=[verts], faces=[faces])

        pix_to_face = torch.tensor([0, 1], dtype=torch.int64).view(1, 1, 1, 2)
        barycentric_coords = torch.tensor(
            [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=barycentric_coords,
            zbuf=torch.ones_like(pix_to_face),
            dists=torch.ones_like(pix_to_face),
        )

        for shader_class in self.shader_classes:
            shader = shader_class()

            with self.assertRaises(ValueError):
                shader(fragments, meshes)

    def test_depth_shader(self):
        shader_classes = [
            HardDepthShader,
            SoftDepthShader,
        ]

        verts = torch.tensor(
            [[-1, -1, 0], [1, -1, 1], [1, 1, 0], [-1, 1, 1]], dtype=torch.float32
        )
        faces = torch.tensor([[0, 1, 2], [2, 3, 0]], dtype=torch.int64)
        meshes = Meshes(verts=[verts], faces=[faces])

        pix_to_face = torch.tensor([0, 1], dtype=torch.int64).view(1, 1, 1, 2)
        barycentric_coords = torch.tensor(
            [[0.1, 0.2, 0.7], [0.3, 0.5, 0.2]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        for faces_per_pixel in [1, 2]:
            fragments = Fragments(
                pix_to_face=pix_to_face[:, :, :, :faces_per_pixel],
                bary_coords=barycentric_coords[:, :, :, :faces_per_pixel],
                zbuf=torch.ones_like(pix_to_face),
                dists=torch.ones_like(pix_to_face),
            )
            R, T = look_at_view_transform()
            cameras = PerspectiveCameras(R=R, T=T)

            for shader_class in shader_classes:
                shader = shader_class()

                out = shader(fragments, meshes, cameras=cameras)
                self.assertEqual(out.shape, (1, 1, 1, 1))
