# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from nerf.raysampler import NeRFRaysampler, ProbabilisticRaysampler
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.transforms.rotation_conversions import random_rotations


class TestRaysampler(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_raysampler_caching(self, batch_size=10):
        """
        Tests the consistency of the NeRF raysampler caching.
        """

        raysampler = NeRFRaysampler(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            n_pts_per_ray=10,
            min_depth=0.1,
            max_depth=10.0,
            n_rays_per_image=12,
            image_width=10,
            image_height=10,
            stratified=False,
            stratified_test=False,
            invert_directions=True,
        )

        raysampler.eval()

        cameras, rays = [], []

        for _ in range(batch_size):
            R = random_rotations(1)
            T = torch.randn(1, 3)
            focal_length = torch.rand(1, 2) + 0.5
            principal_point = torch.randn(1, 2)

            camera = PerspectiveCameras(
                focal_length=focal_length,
                principal_point=principal_point,
                R=R,
                T=T,
            )

            cameras.append(camera)
            rays.append(raysampler(camera))

        raysampler.precache_rays(cameras, list(range(batch_size)))

        for cam_index, rays_ in enumerate(rays):
            rays_cached_ = raysampler(
                cameras=cameras[cam_index],
                chunksize=None,
                chunk_idx=0,
                camera_hash=cam_index,
                caching=False,
            )

            for v, v_cached in zip(rays_, rays_cached_):
                self.assertTrue(torch.allclose(v, v_cached))

    def test_probabilistic_raysampler(self, batch_size=1, n_pts_per_ray=60):
        """
        Check that the probabilistic ray sampler does not crash for various
        settings.
        """

        raysampler_grid = NeRFRaysampler(
            min_x=0.0,
            max_x=10.0,
            min_y=0.0,
            max_y=10.0,
            n_pts_per_ray=n_pts_per_ray,
            min_depth=1.0,
            max_depth=10.0,
            n_rays_per_image=12,
            image_width=10,
            image_height=10,
            stratified=False,
            stratified_test=False,
            invert_directions=True,
        )

        R = random_rotations(batch_size)
        T = torch.randn(batch_size, 3)
        focal_length = torch.rand(batch_size, 2) + 0.5
        principal_point = torch.randn(batch_size, 2)
        camera = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
        )

        raysampler_grid.eval()

        ray_bundle = raysampler_grid(cameras=camera)

        ray_weights = torch.rand_like(ray_bundle.lengths)

        # Just check that we dont crash for all possible settings.
        for stratified_test in (True, False):
            for stratified in (True, False):
                raysampler_prob = ProbabilisticRaysampler(
                    n_pts_per_ray=n_pts_per_ray,
                    stratified=stratified,
                    stratified_test=stratified_test,
                    add_input_samples=True,
                )
                for mode in ("train", "eval"):
                    getattr(raysampler_prob, mode)()
                    for _ in range(10):
                        raysampler_prob(ray_bundle, ray_weights)
