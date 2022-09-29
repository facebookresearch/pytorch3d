# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from math import pi

import torch
from pytorch3d.implicitron.tools.circle_fitting import (
    _signed_area,
    fit_circle_in_2d,
    fit_circle_in_3d,
    get_rotation_to_best_fit_xy,
)
from pytorch3d.transforms import random_rotation, random_rotations
from tests.common_testing import TestCaseMixin


class TestCircleFitting(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def _assertParallel(self, a, b, **kwargs):
        """
        Given a and b of shape (..., 3) each containing 3D vectors,
        assert that correspnding vectors are parallel. Changed sign is ok.
        """
        self.assertClose(torch.cross(a, b, dim=-1), torch.zeros_like(a), **kwargs)

    def test_plane_levelling(self):
        device = torch.device("cuda:0")
        B = 16
        N = 1024
        random = torch.randn((B, N, 3), device=device)

        # first, check that we always return a vaild rotation
        rot = get_rotation_to_best_fit_xy(random)
        self.assertClose(rot.det(), torch.ones_like(rot[:, 0, 0]))
        self.assertClose(rot.norm(dim=-1), torch.ones_like(rot[:, 0]))

        # then, check the result is what we expect
        z_squeeze = 0.1
        random[..., -1] *= z_squeeze
        rot_gt = random_rotations(B, device=device)
        rotated = random @ rot_gt.transpose(-1, -2)
        rot_hat = get_rotation_to_best_fit_xy(rotated)
        self.assertClose(rot.det(), torch.ones_like(rot[:, 0, 0]))
        self.assertClose(rot.norm(dim=-1), torch.ones_like(rot[:, 0]))
        # covariance matrix of the levelled points is by design diag(1, 1, z_squeeze²)
        self.assertClose(
            (rotated @ rot_hat)[..., -1].std(dim=-1),
            torch.ones_like(rot_hat[:, 0, 0]) * z_squeeze,
            rtol=0.1,
        )

    def test_simple_3d(self):
        device = torch.device("cuda:0")
        for _ in range(7):
            radius = 10 * torch.rand(1, device=device)[0]
            center = 10 * torch.rand(3, device=device)
            rot = random_rotation(device=device)
            offset = torch.rand(3, device=device)
            up = torch.rand(3, device=device)
            self._simple_3d_test(radius, center, rot, offset, up)

    def _simple_3d_test(self, radius, center, rot, offset, up):
        # angles are increasing so the points move in a well defined direction.
        angles = torch.cumsum(torch.rand(17, device=rot.device), dim=0)
        many = torch.stack(
            [torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)], dim=1
        )
        source_points = (many * radius) @ rot + center[None]

        # case with no generation
        result = fit_circle_in_3d(source_points)
        self.assertClose(result.radius, radius)
        self.assertClose(result.center, center)
        self._assertParallel(result.normal, rot[2], atol=1e-5)
        self.assertEqual(result.generated_points.shape, (0, 3))

        # Generate 5 points around the circle
        n_new_points = 5
        result2 = fit_circle_in_3d(source_points, n_points=n_new_points)
        self.assertClose(result2.radius, radius)
        self.assertClose(result2.center, center)
        self.assertClose(result2.normal, result.normal)
        self.assertEqual(result2.generated_points.shape, (5, 3))

        observed_points = result2.generated_points
        self.assertClose(observed_points[0], observed_points[4], atol=1e-4)
        self.assertClose(observed_points[0], source_points[0], atol=1e-5)
        observed_normal = torch.cross(
            observed_points[0] - observed_points[2],
            observed_points[1] - observed_points[3],
            dim=-1,
        )
        self._assertParallel(observed_normal, result.normal, atol=1e-4)
        diameters = observed_points[:2] - observed_points[2:4]
        self.assertClose(
            torch.norm(diameters, dim=1), diameters.new_full((2,), 2 * radius)
        )

        # Regenerate the input points
        result3 = fit_circle_in_3d(source_points, angles=angles - angles[0])
        self.assertClose(result3.radius, radius)
        self.assertClose(result3.center, center)
        self.assertClose(result3.normal, result.normal)
        self.assertClose(result3.generated_points, source_points, atol=1e-5)

        # Test with offset
        result4 = fit_circle_in_3d(
            source_points, angles=angles - angles[0], offset=offset, up=up
        )
        self.assertClose(result4.radius, radius)
        self.assertClose(result4.center, center)
        self.assertClose(result4.normal, result.normal)
        observed_offsets = result4.generated_points - source_points

        # observed_offset is constant
        self.assertClose(
            observed_offsets.min(0).values, observed_offsets.max(0).values, atol=1e-5
        )
        # observed_offset has the right length
        self.assertClose(observed_offsets[0].norm(), offset.norm())

        self.assertClose(result.normal.norm(), torch.ones(()))
        # component of observed_offset along normal
        component = torch.dot(observed_offsets[0], result.normal)
        self.assertClose(component.abs(), offset[2].abs(), atol=1e-5)
        agree_normal = torch.dot(result.normal, up) > 0
        agree_signs = component * offset[2] > 0
        self.assertEqual(agree_normal, agree_signs)

    def test_simple_2d(self):
        radius = 7.0
        center = torch.tensor([9, 2.5])
        angles = torch.cumsum(torch.rand(17), dim=0)
        many = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        source_points = (many * radius) + center[None]

        result = fit_circle_in_2d(source_points)
        self.assertClose(result.radius, torch.tensor(radius))
        self.assertClose(result.center, center)
        self.assertEqual(result.generated_points.shape, (0, 2))

        # Generate 5 points around the circle
        n_new_points = 5
        result2 = fit_circle_in_2d(source_points, n_points=n_new_points)
        self.assertClose(result2.radius, torch.tensor(radius))
        self.assertClose(result2.center, center)
        self.assertEqual(result2.generated_points.shape, (5, 2))

        observed_points = result2.generated_points
        self.assertClose(observed_points[0], observed_points[4])
        self.assertClose(observed_points[0], source_points[0], atol=1e-5)
        diameters = observed_points[:2] - observed_points[2:4]
        self.assertClose(torch.norm(diameters, dim=1), torch.full((2,), 2 * radius))

        # Regenerate the input points
        result3 = fit_circle_in_2d(source_points, angles=angles - angles[0])
        self.assertClose(result3.radius, torch.tensor(radius))
        self.assertClose(result3.center, center)
        self.assertClose(result3.generated_points, source_points, atol=1e-5)

    def test_minimum_inputs(self):
        fit_circle_in_3d(torch.rand(3, 3), n_points=10)

        with self.assertRaisesRegex(
            ValueError, "2 points are not enough to determine a circle"
        ):
            fit_circle_in_3d(torch.rand(2, 3))

    def test_signed_area(self):
        n_points = 1001
        angles = torch.linspace(0, 2 * pi, n_points)
        radius = 0.85
        center = torch.rand(2)
        circle = center + radius * torch.stack(
            [torch.cos(angles), torch.sin(angles)], dim=1
        )
        circle_area = torch.tensor(pi * radius * radius)
        self.assertClose(_signed_area(circle), circle_area)
        # clockwise is negative
        self.assertClose(_signed_area(circle.flip(0)), -circle_area)

        # Semicircles
        self.assertClose(_signed_area(circle[: (n_points + 1) // 2]), circle_area / 2)
        self.assertClose(_signed_area(circle[n_points // 2 :]), circle_area / 2)

        # A straight line bounds no area
        self.assertClose(_signed_area(torch.rand(2, 2)), torch.tensor(0.0))

        # Letter 'L' written anticlockwise.
        L_shape = [[0, 1], [0, 0], [1, 0]]
        # Triangle area is 0.5 * b * h.
        self.assertClose(_signed_area(torch.tensor(L_shape)), torch.tensor(0.5))
