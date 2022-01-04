# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from nerf.raymarcher import EmissionAbsorptionNeRFRaymarcher
from pytorch3d.renderer import EmissionAbsorptionRaymarcher


class TestRaymarcher(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_raymarcher(self):
        """
        Checks that the nerf raymarcher outputs are identical to the
        EmissionAbsorptionRaymarcher.
        """

        feat_dim = 3
        rays_densities = torch.rand(100, 10, 1)
        rays_features = torch.randn(100, 10, feat_dim)

        out, out_nerf = [
            raymarcher(rays_densities, rays_features)
            for raymarcher in (
                EmissionAbsorptionRaymarcher(),
                EmissionAbsorptionNeRFRaymarcher(),
            )
        ]

        self.assertTrue(
            torch.allclose(out[..., :feat_dim], out_nerf[0][..., :feat_dim])
        )
