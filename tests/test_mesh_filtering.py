# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from pytorch3d.ops import taubin_smoothing
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestTaubinSmoothing(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_taubin(self):
        N = 3
        device = get_random_cuda_device()

        mesh = ico_sphere(4, device).extend(N)
        ico_verts = mesh.verts_padded()
        ico_faces = mesh.faces_padded()

        rand_noise = torch.rand_like(ico_verts) * 0.2 - 0.1
        z_mask = (ico_verts[:, :, -1] > 0).view(N, -1, 1)
        rand_noise = rand_noise * z_mask
        verts = ico_verts + rand_noise
        mesh = Meshes(verts=verts, faces=ico_faces)

        smooth_mesh = taubin_smoothing(mesh, num_iter=50)
        smooth_verts = smooth_mesh.verts_padded()

        smooth_dist = (smooth_verts - ico_verts).norm(dim=-1).mean()
        dist = (verts - ico_verts).norm(dim=-1).mean()
        self.assertTrue(smooth_dist < dist)
