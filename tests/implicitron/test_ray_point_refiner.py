# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from pytorch3d.implicitron.models.renderer.ray_point_refiner import RayPointRefiner
from pytorch3d.renderer import RayBundle


if os.environ.get("FB_TEST", False):
    from common_testing import TestCaseMixin
else:
    from tests.common_testing import TestCaseMixin


class TestRayPointRefiner(TestCaseMixin, unittest.TestCase):
    def test_simple(self):
        length = 15
        n_pts_per_ray = 10

        for add_input_samples in [False, True]:
            ray_point_refiner = RayPointRefiner(
                n_pts_per_ray=n_pts_per_ray,
                random_sampling=False,
                add_input_samples=add_input_samples,
            )
            lengths = torch.arange(length, dtype=torch.float32).expand(3, 25, length)
            bundle = RayBundle(lengths=lengths, origins=None, directions=None, xys=None)
            weights = torch.ones(3, 25, length)
            refined = ray_point_refiner(bundle, weights)

            self.assertIsNone(refined.directions)
            self.assertIsNone(refined.origins)
            self.assertIsNone(refined.xys)
            expected = torch.linspace(0.5, length - 1.5, n_pts_per_ray)
            expected = expected.expand(3, 25, n_pts_per_ray)
            if add_input_samples:
                full_expected = torch.cat((lengths, expected), dim=-1).sort()[0]
            else:
                full_expected = expected
            self.assertClose(refined.lengths, full_expected)

            ray_point_refiner_random = RayPointRefiner(
                n_pts_per_ray=n_pts_per_ray,
                random_sampling=True,
                add_input_samples=add_input_samples,
            )
            refined_random = ray_point_refiner_random(bundle, weights)
            lengths_random = refined_random.lengths
            self.assertEqual(lengths_random.shape, full_expected.shape)
            if not add_input_samples:
                self.assertGreater(lengths_random.min().item(), 0.5)
                self.assertLess(lengths_random.max().item(), length - 1.5)

            # Check sorted
            self.assertTrue(
                (lengths_random[..., 1:] - lengths_random[..., :-1] > 0).all()
            )
