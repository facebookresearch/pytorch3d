# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np

import torch

from pytorch3d.implicitron.models.renderer.base import (
    approximate_conical_frustum_as_gaussians,
    compute_3d_diagonal_covariance_gaussian,
    conical_frustum_to_gaussian,
    ImplicitronRayBundle,
)
from pytorch3d.implicitron.models.renderer.ray_sampler import AbstractMaskRaySampler

from tests.common_testing import TestCaseMixin


class TestRendererBase(TestCaseMixin, unittest.TestCase):
    def test_implicitron_from_bins(self) -> None:
        bins = torch.randn(2, 3, 4, 5)
        ray_bundle = ImplicitronRayBundle(
            origins=None,
            directions=None,
            lengths=None,
            xys=None,
            bins=bins,
        )
        self.assertClose(ray_bundle.lengths, 0.5 * (bins[..., 1:] + bins[..., :-1]))
        self.assertClose(ray_bundle.bins, bins)

    def test_implicitron_raise_value_error_bins_is_set_and_try_to_set_lengths(
        self,
    ) -> None:
        ray_bundle = ImplicitronRayBundle(
            origins=torch.rand(2, 3, 4, 3),
            directions=torch.rand(2, 3, 4, 3),
            lengths=None,
            xys=torch.rand(2, 3, 4, 2),
            bins=torch.rand(2, 3, 4, 14),
        )
        with self.assertRaisesRegex(
            ValueError,
            "If the bins attribute is not None you cannot set the lengths attribute.",
        ):
            ray_bundle.lengths = torch.empty(2)

    def test_implicitron_raise_value_error_if_bins_dim_equal_1(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "The last dim of bins must be at least superior or equal to 2."
        ):
            ImplicitronRayBundle(
                origins=torch.rand(2, 3, 4, 3),
                directions=torch.rand(2, 3, 4, 3),
                lengths=None,
                xys=torch.rand(2, 3, 4, 2),
                bins=torch.rand(2, 3, 4, 1),
            )

    def test_implicitron_raise_value_error_if_neither_bins_or_lengths_provided(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Please set either bins or lengths to initialize an ImplicitronRayBundle.",
        ):
            ImplicitronRayBundle(
                origins=torch.rand(2, 3, 4, 3),
                directions=torch.rand(2, 3, 4, 3),
                lengths=None,
                xys=torch.rand(2, 3, 4, 2),
                bins=None,
            )

    def test_conical_frustum_to_gaussian(self) -> None:
        origins = torch.zeros(3, 3, 3)
        directions = torch.tensor(
            [
                [[0, 0, 0], [1, 0, 0], [3, 0, 0]],
                [[0, 0.25, 0], [1, 0.25, 0], [3, 0.25, 0]],
                [[0, 1, 0], [1, 1, 0], [3, 1, 0]],
            ]
        )
        bins = torch.tensor(
            [
                [[0.5, 1.5], [0.3, 0.7], [0.3, 0.7]],
                [[0.5, 1.5], [0.3, 0.7], [0.3, 0.7]],
                [[0.5, 1.5], [0.3, 0.7], [0.3, 0.7]],
            ]
        )
        # see test_compute_pixel_radii_from_ray_direction
        radii = torch.tensor(
            [
                [1.25, 2.25, 2.25],
                [1.75, 2.75, 2.75],
                [1.75, 2.75, 2.75],
            ]
        )
        radii = radii[..., None] / 12**0.5

        # The expected mean and diagonal covariance have been computed
        # by hand from the official code of MipNerf.
        # https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/mip.py#L125
        # mean, cov_diag = cast_rays(length, origins, directions, radii, 'cone', diag=True)

        expected_mean = torch.tensor(
            [
                [
                    [[0.0, 0.0, 0.0]],
                    [[0.5506329, 0.0, 0.0]],
                    [[1.6518986, 0.0, 0.0]],
                ],
                [
                    [[0.0, 0.28846154, 0.0]],
                    [[0.5506329, 0.13765822, 0.0]],
                    [[1.6518986, 0.13765822, 0.0]],
                ],
                [
                    [[0.0, 1.1538461, 0.0]],
                    [[0.5506329, 0.5506329, 0.0]],
                    [[1.6518986, 0.5506329, 0.0]],
                ],
            ]
        )
        expected_diag_cov = torch.tensor(
            [
                [
                    [[0.04544772, 0.04544772, 0.04544772]],
                    [[0.01130973, 0.03317059, 0.03317059]],
                    [[0.10178753, 0.03317059, 0.03317059]],
                ],
                [
                    [[0.08907752, 0.00404956, 0.08907752]],
                    [[0.0142245, 0.04734321, 0.04955113]],
                    [[0.10212927, 0.04991625, 0.04955113]],
                ],
                [
                    [[0.08907752, 0.0647929, 0.08907752]],
                    [[0.03608529, 0.03608529, 0.04955113]],
                    [[0.10674264, 0.05590574, 0.04955113]],
                ],
            ]
        )

        ray = ImplicitronRayBundle(
            origins=origins,
            directions=directions,
            bins=bins,
            lengths=None,
            pixel_radii_2d=radii,
            xys=None,
        )
        mean, diag_cov = conical_frustum_to_gaussian(ray)

        self.assertClose(mean, expected_mean)
        self.assertClose(diag_cov, expected_diag_cov)

    def test_scale_conical_frustum_to_gaussian(self) -> None:
        origins = torch.zeros(2, 2, 3)
        directions = torch.Tensor(
            [
                [[0, 1, 0], [0, 0, 1]],
                [[0, 1, 0], [0, 0, 1]],
            ]
        )
        bins = torch.Tensor(
            [
                [[0.5, 1.5], [0.3, 0.7]],
                [[0.5, 1.5], [0.3, 0.7]],
            ]
        )
        radii = torch.ones(2, 2, 1)

        ray = ImplicitronRayBundle(
            origins=origins,
            directions=directions,
            bins=bins,
            pixel_radii_2d=radii,
            lengths=None,
            xys=None,
        )

        mean, diag_cov = conical_frustum_to_gaussian(ray)

        scaling_factor = 2.5
        ray = ImplicitronRayBundle(
            origins=origins,
            directions=directions,
            bins=bins * scaling_factor,
            pixel_radii_2d=radii,
            lengths=None,
            xys=None,
        )
        mean_scaled, diag_cov_scaled = conical_frustum_to_gaussian(ray)
        np.testing.assert_allclose(mean * scaling_factor, mean_scaled)
        np.testing.assert_allclose(
            diag_cov * scaling_factor**2, diag_cov_scaled, atol=1e-6
        )

    def test_approximate_conical_frustum_as_gaussian(self) -> None:
        """Ensure that the computation modularity in our function is well done."""
        bins = torch.Tensor([[0.5, 1.5], [0.3, 0.7]])
        radii = torch.Tensor([[1.0], [1.0]])
        t_mean, t_var, r_var = approximate_conical_frustum_as_gaussians(bins, radii)

        self.assertEqual(t_mean.shape, (2, 1))
        self.assertEqual(t_var.shape, (2, 1))
        self.assertEqual(r_var.shape, (2, 1))

        mu = np.array([[1.0], [0.5]])
        delta = np.array([[0.5], [0.2]])

        np.testing.assert_allclose(
            mu + (2 * mu * delta**2) / (3 * mu**2 + delta**2), t_mean.numpy()
        )
        np.testing.assert_allclose(
            (delta**2) / 3
            - (4 / 15)
            * (
                (delta**4 * (12 * mu**2 - delta**2))
                / (3 * mu**2 + delta**2) ** 2
            ),
            t_var.numpy(),
        )
        np.testing.assert_allclose(
            radii**2
            * (
                (mu**2) / 4
                + (5 / 12) * delta**2
                - 4 / 15 * (delta**4) / (3 * mu**2 + delta**2)
            ),
            r_var.numpy(),
        )

    def test_compute_3d_diagonal_covariance_gaussian(self) -> None:
        ray_directions = torch.Tensor([[0, 0, 1]])
        t_var = torch.Tensor([0.5, 0.5, 1])
        r_var = torch.Tensor([0.6, 0.3, 0.4])
        expected_diag_cov = np.array(
            [
                [
                    # t_cov_diag + xy_cov_diag
                    [0.0 + 0.6, 0.0 + 0.6, 0.5 + 0.0],
                    [0.0 + 0.3, 0.0 + 0.3, 0.5 + 0.0],
                    [0.0 + 0.4, 0.0 + 0.4, 1.0 + 0.0],
                ]
            ]
        )
        diag_cov = compute_3d_diagonal_covariance_gaussian(ray_directions, t_var, r_var)
        np.testing.assert_allclose(diag_cov.numpy(), expected_diag_cov)

    def test_conical_frustum_to_gaussian_raise_valueerror(self) -> None:
        lengths = torch.linspace(0, 1, steps=6)
        directions = torch.tensor([0, 0, 1])
        origins = torch.tensor([1, 1, 1])
        ray = ImplicitronRayBundle(
            origins=origins, directions=directions, lengths=lengths, xys=None
        )

        expected_error_message = (
            "RayBundle pixel_radii_2d or bins have not been provided."
            " Look at pytorch3d.renderer.implicit.renderer.ray_sampler::"
            "AbstractMaskRaySampler to see how to compute them. Have you forgot to set"
            "`cast_ray_bundle_as_cone` to True?"
        )

        with self.assertRaisesRegex(ValueError, expected_error_message):
            _ = conical_frustum_to_gaussian(ray)

        # Ensure message is coherent with AbstractMaskRaySampler
        class FakeRaySampler(AbstractMaskRaySampler):
            def _get_min_max_depth_bounds(self, *args):
                return None

        message_assertion = (
            "If cast_ray_bundle_as_cone has been removed please update the doc"
            "conical_frustum_to_gaussian"
        )
        self.assertIsNotNone(
            getattr(FakeRaySampler(), "cast_ray_bundle_as_cone", None),
            message_assertion,
        )
