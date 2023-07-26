# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import torch

from pytorch3d.implicitron.models.renderer.ray_point_refiner import (
    apply_blurpool_on_weights,
    RayPointRefiner,
)
from pytorch3d.implicitron.models.renderer.ray_sampler import ImplicitronRayBundle
from tests.common_testing import TestCaseMixin


class TestRayPointRefiner(TestCaseMixin, unittest.TestCase):
    def test_simple(self):
        length = 15
        n_pts_per_ray = 10

        for add_input_samples, use_blurpool in product([False, True], [False, True]):
            ray_point_refiner = RayPointRefiner(
                n_pts_per_ray=n_pts_per_ray,
                random_sampling=False,
                add_input_samples=add_input_samples,
                blurpool_weights=use_blurpool,
            )
            lengths = torch.arange(length, dtype=torch.float32).expand(3, 25, length)
            bundle = ImplicitronRayBundle(
                lengths=lengths,
                origins=None,
                directions=None,
                xys=None,
                camera_ids=None,
                camera_counts=None,
            )
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
                blurpool_weights=use_blurpool,
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

    def test_simple_use_bins(self):
        """
        Same spirit than test_simple but use bins in the ImplicitronRayBunle.
        It has been duplicated to avoid cognitive overload while reading the
        test (lot of if else).
        """
        length = 15
        n_pts_per_ray = 10

        for add_input_samples, use_blurpool in product([False, True], [False, True]):
            ray_point_refiner = RayPointRefiner(
                n_pts_per_ray=n_pts_per_ray,
                random_sampling=False,
                add_input_samples=add_input_samples,
            )

            bundle = ImplicitronRayBundle(
                lengths=None,
                bins=torch.arange(length + 1, dtype=torch.float32).expand(
                    3, 25, length + 1
                ),
                origins=None,
                directions=None,
                xys=None,
                camera_ids=None,
                camera_counts=None,
            )
            weights = torch.ones(3, 25, length)
            refined = ray_point_refiner(bundle, weights, blurpool_weights=use_blurpool)

            self.assertIsNone(refined.directions)
            self.assertIsNone(refined.origins)
            self.assertIsNone(refined.xys)
            expected_bins = torch.linspace(0, length, n_pts_per_ray + 1)
            expected_bins = expected_bins.expand(3, 25, n_pts_per_ray + 1)
            if add_input_samples:
                expected_bins = torch.cat((bundle.bins, expected_bins), dim=-1).sort()[
                    0
                ]
            full_expected = torch.lerp(
                expected_bins[..., :-1], expected_bins[..., 1:], 0.5
            )

            self.assertClose(refined.lengths, full_expected)

            ray_point_refiner_random = RayPointRefiner(
                n_pts_per_ray=n_pts_per_ray,
                random_sampling=True,
                add_input_samples=add_input_samples,
            )

            refined_random = ray_point_refiner_random(
                bundle, weights, blurpool_weights=use_blurpool
            )
            lengths_random = refined_random.lengths
            self.assertEqual(lengths_random.shape, full_expected.shape)
            if not add_input_samples:
                self.assertGreater(lengths_random.min().item(), 0)
                self.assertLess(lengths_random.max().item(), length)

            # Check sorted
            self.assertTrue(
                (lengths_random[..., 1:] - lengths_random[..., :-1] > 0).all()
            )

    def test_apply_blurpool_on_weights(self):
        weights = torch.tensor(
            [
                [0.5, 0.6, 0.7],
                [0.5, 0.3, 0.9],
            ]
        )
        expected_weights = 0.5 * torch.tensor(
            [
                [0.5 + 0.6, 0.6 + 0.7, 0.7 + 0.7],
                [0.5 + 0.5, 0.5 + 0.9, 0.9 + 0.9],
            ]
        )
        out_weights = apply_blurpool_on_weights(weights)
        self.assertTrue(torch.allclose(out_weights, expected_weights))

    def test_shapes_apply_blurpool_on_weights(self):
        weights = torch.randn((5, 4, 3, 2, 1))
        out_weights = apply_blurpool_on_weights(weights)
        self.assertEqual((5, 4, 3, 2, 1), out_weights.shape)
