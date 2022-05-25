# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.renderer import AbsorptionOnlyRaymarcher, EmissionAbsorptionRaymarcher

from .common_testing import TestCaseMixin


class TestRaymarching(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    @staticmethod
    def _init_random_rays(
        n_rays=10, n_pts_per_ray=9, device="cuda", dtype=torch.float32
    ):
        """
        Generate a batch of ray points with features, densities, and z-coordinates
        such that their EmissionAbsorption renderring results in
        feature renders `features_gt`, depth renders `depths_gt`,
        and opacity renders `opacities_gt`.
        """

        # generate trivial ray z-coordinates of sampled points coinciding with
        # each point's order along a ray.
        rays_z = torch.arange(n_pts_per_ray, dtype=dtype, device=device)[None].repeat(
            n_rays, 1
        )

        # generate ground truth depth values of the underlying surface.
        depths_gt = torch.randint(
            low=1, high=n_pts_per_ray + 2, size=(n_rays,)
        ).type_as(rays_z)

        # compute ideal densities that are 0 before the surface and 1 after
        # the corresponding ground truth depth value
        rays_densities = (rays_z >= depths_gt[..., None]).type_as(rays_z)[..., None]
        opacities_gt = (depths_gt < n_pts_per_ray).type_as(rays_z)

        # generate random per-ray features
        rays_features = torch.rand(
            (n_rays, n_pts_per_ray, 3), device=rays_z.device, dtype=rays_z.dtype
        )

        # infer the expected feature render "features_gt"
        gt_surface = ((rays_z - depths_gt[..., None]).abs() <= 1e-4).type_as(rays_z)
        features_gt = (rays_features * gt_surface[..., None]).sum(dim=-2)

        return (
            rays_z,
            rays_densities,
            rays_features,
            depths_gt,
            features_gt,
            opacities_gt,
        )

    @staticmethod
    def raymarcher(
        raymarcher_type=EmissionAbsorptionRaymarcher, n_rays=10, n_pts_per_ray=10
    ):
        (
            rays_z,
            rays_densities,
            rays_features,
            depths_gt,
            features_gt,
            opacities_gt,
        ) = TestRaymarching._init_random_rays(
            n_rays=n_rays, n_pts_per_ray=n_pts_per_ray
        )

        raymarcher = raymarcher_type()

        def run_raymarcher():
            raymarcher(
                rays_densities=rays_densities,
                rays_features=rays_features,
                rays_z=rays_z,
            )
            torch.cuda.synchronize()

        return run_raymarcher

    def test_emission_absorption_inputs(self):
        """
        Test the checks of validity of the inputs to `EmissionAbsorptionRaymarcher`.
        """

        # init the EA raymarcher
        raymarcher_ea = EmissionAbsorptionRaymarcher()

        # bad ways of passing densities and features
        # [rays_densities, rays_features, rays_z]
        bad_inputs = [
            [torch.rand(10, 5, 4), None],
            [torch.Tensor(3)[0], torch.rand(10, 5, 4)],
            [1.0, torch.rand(10, 5, 4)],
            [torch.rand(10, 5, 4), 1.0],
            [torch.rand(10, 5, 4), None],
            [torch.rand(10, 5, 4), torch.rand(10, 5, 4)],
            [torch.rand(10, 5, 4), torch.rand(10, 5, 4, 3)],
            [torch.rand(10, 5, 4, 3), torch.rand(10, 5, 4, 3)],
        ]

        for bad_input in bad_inputs:
            with self.assertRaises(ValueError):
                raymarcher_ea(*bad_input)

    def test_absorption_only_inputs(self):
        """
        Test the checks of validity of the inputs to `AbsorptionOnlyRaymarcher`.
        """

        # init the AO raymarcher
        raymarcher_ao = AbsorptionOnlyRaymarcher()

        # bad ways of passing densities and features
        # [rays_densities, rays_features, rays_z]
        bad_inputs = [[torch.Tensor(3)[0]]]

        for bad_input in bad_inputs:
            with self.assertRaises(ValueError):
                raymarcher_ao(*bad_input)

    def test_emission_absorption(self):
        """
        Test the EA raymarching algorithm.
        """
        (
            rays_z,
            rays_densities,
            rays_features,
            depths_gt,
            features_gt,
            opacities_gt,
        ) = TestRaymarching._init_random_rays(
            n_rays=1000, n_pts_per_ray=9, device=None, dtype=torch.float32
        )

        # init the EA raymarcher
        raymarcher_ea = EmissionAbsorptionRaymarcher()

        # allow gradients for a differentiability check
        rays_densities.requires_grad = True
        rays_features.requires_grad = True

        # render the features first and check with gt
        data_render = raymarcher_ea(rays_densities, rays_features)
        features_render, opacities_render = data_render[..., :-1], data_render[..., -1]
        self.assertClose(opacities_render, opacities_gt)
        self.assertClose(
            features_render * opacities_render[..., None],
            features_gt * opacities_gt[..., None],
        )

        # get the depth map by rendering the ray z components and check with gt
        depths_render = raymarcher_ea(rays_densities, rays_z[..., None])[..., 0]
        self.assertClose(depths_render * opacities_render, depths_gt * opacities_gt)

        # check differentiability
        loss = features_render.mean()
        loss.backward()
        for field in (rays_densities, rays_features):
            self.assertTrue(torch.isfinite(field.grad.data).all())

    def test_absorption_only(self):
        """
        Test the AO raymarching algorithm.
        """
        (
            rays_z,
            rays_densities,
            rays_features,
            depths_gt,
            features_gt,
            opacities_gt,
        ) = TestRaymarching._init_random_rays(
            n_rays=1000, n_pts_per_ray=9, device=None, dtype=torch.float32
        )

        # init the AO raymarcher
        raymarcher_ao = AbsorptionOnlyRaymarcher()

        # allow gradients for a differentiability check
        rays_densities.requires_grad = True

        # render opacities, check with gt and check that returned features are None
        opacities_render = raymarcher_ao(rays_densities)[..., 0]
        self.assertClose(opacities_render, opacities_gt)

        # check differentiability
        loss = opacities_render.mean()
        loss.backward()
        self.assertTrue(torch.isfinite(rays_densities.grad.data).all())
