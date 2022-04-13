# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from common_testing import TestCaseMixin
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shader import (
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftPhongShader,
)
from pytorch3d.structures.meshes import Meshes


class TestShader(TestCaseMixin, unittest.TestCase):
    def test_to(self):
        cpu_device = torch.device("cpu")
        cuda_device = torch.device("cuda:0")

        R, T = look_at_view_transform()

        shader_classes = [
            HardFlatShader,
            HardGouraudShader,
            HardPhongShader,
            SoftPhongShader,
        ]

        for shader_class in shader_classes:
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
                else:
                    self.assertEqual(cuda_device, cuda_shader.cameras.device)
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
        shader_classes = [
            HardFlatShader,
            HardGouraudShader,
            HardPhongShader,
            SoftPhongShader,
        ]

        for shader_class in shader_classes:
            shader = shader_class()

            with self.assertRaises(ValueError):
                shader(fragments, meshes)
