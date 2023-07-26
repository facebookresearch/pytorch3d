# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product
from typing import Tuple

from unittest.mock import patch

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.implicitron.models.renderer.base import EvaluationMode
from pytorch3d.implicitron.models.renderer.ray_sampler import (
    AdaptiveRaySampler,
    compute_radii,
    NearFarRaySampler,
)

from pytorch3d.renderer.cameras import (
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
)
from pytorch3d.renderer.implicit.utils import HeterogeneousRayBundle
from tests.common_camera_utils import init_random_cameras

from tests.common_testing import TestCaseMixin

CAMERA_TYPES = (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    OrthographicCameras,
    PerspectiveCameras,
)


def unproject_xy_grid_from_ndc_to_world_coord(
    cameras: CamerasBase, xy_grid: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    Unproject a xy_grid from NDC coordinates to world coordinates.

    Args:
        cameras: CamerasBase.
        xy_grid: A tensor of shape `(..., H*W, 2)` representing the
            x, y coords.

    Returns:
        A tensor of shape `(..., H*W, 3)` representing the
    """

    batch_size = xy_grid.shape[0]
    n_rays_per_image = xy_grid.shape[1:-1].numel()
    xy = xy_grid.view(batch_size, -1, 2)
    xyz = torch.cat([xy, xy_grid.new_ones(batch_size, n_rays_per_image, 1)], dim=-1)
    plane_at_depth1 = cameras.unproject_points(xyz, from_ndc=True)
    return plane_at_depth1.view(*xy_grid.shape[:-1], 3)


class TestRaysampler(TestCaseMixin, unittest.TestCase):
    def test_ndc_raysampler_n_ray_total_is_none(self):
        sampler = NearFarRaySampler()
        message = (
            "If you introduce the support of `n_rays_total` for {0}, please handle the "
            "packing and unpacking logic for the radii and lengths computation."
        )
        self.assertIsNone(
            sampler._training_raysampler._n_rays_total, message.format(type(sampler))
        )
        self.assertIsNone(
            sampler._evaluation_raysampler._n_rays_total, message.format(type(sampler))
        )

        sampler = AdaptiveRaySampler()
        self.assertIsNone(
            sampler._training_raysampler._n_rays_total, message.format(type(sampler))
        )
        self.assertIsNone(
            sampler._evaluation_raysampler._n_rays_total, message.format(type(sampler))
        )

    def test_catch_heterogeneous_exception(self):
        cameras = init_random_cameras(FoVPerspectiveCameras, 1, random_z=True)

        class FakeSampler:
            def __init__(self):
                self.min_x, self.max_x = 1, 2
                self.min_y, self.max_y = 1, 2

            def __call__(self, **kwargs):
                return HeterogeneousRayBundle(
                    torch.rand(3), torch.rand(3), torch.rand(3), torch.rand(1)
                )

        with patch(
            "pytorch3d.implicitron.models.renderer.ray_sampler.NDCMultinomialRaysampler",
            return_value=FakeSampler(),
        ):
            for sampler in [
                AdaptiveRaySampler(cast_ray_bundle_as_cone=True),
                NearFarRaySampler(cast_ray_bundle_as_cone=True),
            ]:
                with self.assertRaises(TypeError):
                    _ = sampler(cameras, EvaluationMode.TRAINING)
            for sampler in [
                AdaptiveRaySampler(cast_ray_bundle_as_cone=False),
                NearFarRaySampler(cast_ray_bundle_as_cone=False),
            ]:
                _ = sampler(cameras, EvaluationMode.TRAINING)

    def test_compute_radii(self):
        batch_size = 1
        image_height, image_width = 20, 10
        min_y, max_y, min_x, max_x = -1.0, 1.0, -1.0, 1.0
        y, x = meshgrid_ij(
            torch.linspace(min_y, max_y, image_height, dtype=torch.float32),
            torch.linspace(min_x, max_x, image_width, dtype=torch.float32),
        )
        xy_grid = torch.stack([x, y], dim=-1).view(-1, 2)
        pixel_width = (max_x - min_x) / (image_width - 1)
        pixel_height = (max_y - min_y) / (image_height - 1)

        for cam_type in CAMERA_TYPES:
            # init a batch of random cameras
            cameras = init_random_cameras(cam_type, batch_size, random_z=True)
            # This method allow us to compute the radii whithout having
            # access to the full grid. Raysamplers during the training
            # will sample random rays from the grid.
            radii = compute_radii(
                cameras, xy_grid, pixel_hw_ndc=(pixel_height, pixel_width)
            )
            plane_at_depth1 = unproject_xy_grid_from_ndc_to_world_coord(
                cameras, xy_grid
            )
            # This method absolutely needs the full grid to work.
            expected_radii = compute_pixel_radii_from_grid(
                plane_at_depth1.reshape(1, image_height, image_width, 3)
            )
            self.assertClose(expected_radii.reshape(-1, 1), radii)

    def test_forward(self):
        n_rays_per_image = 16
        image_height, image_width = 20, 20
        kwargs = {
            "image_width": image_width,
            "image_height": image_height,
            "n_pts_per_ray_training": 32,
            "n_pts_per_ray_evaluation": 32,
            "n_rays_per_image_sampled_from_mask": n_rays_per_image,
            "cast_ray_bundle_as_cone": False,
        }

        batch_size = 2
        samplers = [NearFarRaySampler(**kwargs), AdaptiveRaySampler(**kwargs)]
        evaluation_modes = [EvaluationMode.TRAINING, EvaluationMode.EVALUATION]

        for cam_type, sampler, evaluation_mode in product(
            CAMERA_TYPES, samplers, evaluation_modes
        ):
            cameras = init_random_cameras(cam_type, batch_size, random_z=True)
            ray_bundle = sampler(cameras, evaluation_mode)

            shape_out = (
                (batch_size, image_width, image_height)
                if evaluation_mode == EvaluationMode.EVALUATION
                else (batch_size, n_rays_per_image, 1)
            )
            n_pts_per_ray = (
                kwargs["n_pts_per_ray_evaluation"]
                if evaluation_mode == EvaluationMode.EVALUATION
                else kwargs["n_pts_per_ray_training"]
            )
            self.assertIsNone(ray_bundle.bins)
            self.assertIsNone(ray_bundle.pixel_radii_2d)
            self.assertEqual(
                ray_bundle.lengths.shape,
                (*shape_out, n_pts_per_ray),
            )
            self.assertEqual(ray_bundle.directions.shape, (*shape_out, 3))
            self.assertEqual(ray_bundle.origins.shape, (*shape_out, 3))

    def test_forward_with_use_bins(self):
        n_rays_per_image = 16
        image_height, image_width = 20, 20
        kwargs = {
            "image_width": image_width,
            "image_height": image_height,
            "n_pts_per_ray_training": 32,
            "n_pts_per_ray_evaluation": 32,
            "n_rays_per_image_sampled_from_mask": n_rays_per_image,
            "cast_ray_bundle_as_cone": True,
        }

        batch_size = 1
        samplers = [NearFarRaySampler(**kwargs), AdaptiveRaySampler(**kwargs)]
        evaluation_modes = [EvaluationMode.TRAINING, EvaluationMode.EVALUATION]
        for cam_type, sampler, evaluation_mode in product(
            CAMERA_TYPES, samplers, evaluation_modes
        ):
            cameras = init_random_cameras(cam_type, batch_size, random_z=True)
            ray_bundle = sampler(cameras, evaluation_mode)

            lengths = 0.5 * (ray_bundle.bins[..., :-1] + ray_bundle.bins[..., 1:])

            self.assertClose(ray_bundle.lengths, lengths)
            shape_out = (
                (batch_size, image_width, image_height)
                if evaluation_mode == EvaluationMode.EVALUATION
                else (batch_size, n_rays_per_image, 1)
            )
            self.assertEqual(ray_bundle.pixel_radii_2d.shape, (*shape_out, 1))
            self.assertEqual(ray_bundle.directions.shape, (*shape_out, 3))
            self.assertEqual(ray_bundle.origins.shape, (*shape_out, 3))


# Helper to test compute_radii
def compute_pixel_radii_from_grid(pixel_grid: torch.Tensor) -> torch.Tensor:
    """
    Compute the radii of a conical frustum given the pixel grid.

    To compute the radii we first compute the translation from a pixel
    to its neighbors along the x and y axis. Then, we compute the norm
    of each translation along the x and y axis.
    The radii are then obtained by the following formula:

    (dx_norm + dy_norm) * 0.5 * 2 / 12**0.5

    where 2/12**0.5 is a scaling factor to match
    the variance of the pixel’s footprint.

    Args:
        pixel_grid: A tensor of shape `(..., H, W, dim)` representing the
            full grid of rays pixel_grid.

    Returns:
        The radiis for each pixels and shape `(..., H, W, 1)`.
    """
    # [B, H, W - 1, 3]
    x_translation = torch.diff(pixel_grid, dim=-2)
    # [B, H - 1, W, 3]
    y_translation = torch.diff(pixel_grid, dim=-3)
    # [B, H, W - 1, 1]
    dx_norm = torch.linalg.norm(x_translation, dim=-1, keepdim=True)
    # [B, H - 1, W, 1]
    dy_norm = torch.linalg.norm(y_translation, dim=-1, keepdim=True)

    # Fill the missing value [B, H, W, 1]
    dx_norm = torch.concatenate([dx_norm, dx_norm[..., -1:, :]], -2)
    dy_norm = torch.concatenate([dy_norm, dy_norm[..., -1:, :, :]], -3)

    # Cut the distance in half to obtain the base radius: (dx_norm + dy_norm) * 0.5
    # and multiply it by the scaling factor: * 2 / 12**0.5
    radii = (dx_norm + dy_norm) / 12**0.5
    return radii


class TestRadiiComputationOnFullGrid(TestCaseMixin, unittest.TestCase):
    def test_compute_pixel_radii_from_grid(self):
        pixel_grid = torch.tensor(
            [
                [[0.0, 0, 0], [1.0, 0.0, 0], [3.0, 0.0, 0.0]],
                [[0.0, 0.25, 0], [1.0, 0.25, 0], [3.0, 0.25, 0]],
                [[0.0, 1, 0], [1.0, 1.0, 0], [3.0000, 1.0, 0]],
            ]
        )

        expected_y_norm = torch.tensor(
            [
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [0.75, 0.75, 0.75],  # duplicated from previous row
            ]
        )
        expected_x_norm = torch.tensor(
            [
                # 3rd column is duplicated from 2nd
                [1.0, 2.0, 2.0],
                [1.0, 2.0, 2.0],
                [1.0, 2.0, 2.0],
            ]
        )
        expected_radii = (expected_x_norm + expected_y_norm) / 12**0.5
        radii = compute_pixel_radii_from_grid(pixel_grid)
        self.assertClose(radii, expected_radii[..., None])
