# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import itertools
import math
import unittest
from distutils.version import LooseVersion
from typing import Optional, Union

import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    euler_angles_to_matrix,
    matrix_to_axis_angle,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    matrix_to_rotation_6d,
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
    random_quaternions,
    random_rotation,
    random_rotations,
    rotation_6d_to_matrix,
)

from .common_testing import TestCaseMixin


class TestRandomRotation(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_random_rotation_invariant(self):
        """The image of the x-axis isn't biased among quadrants."""
        N = 1000
        base = random_rotation()
        quadrants = list(itertools.product([False, True], repeat=3))

        matrices = random_rotations(N)
        transformed = torch.matmul(base, matrices)
        transformed2 = torch.matmul(matrices, base)

        for k, results in enumerate([matrices, transformed, transformed2]):
            counts = {i: 0 for i in quadrants}
            for j in range(N):
                counts[tuple(i.item() > 0 for i in results[j, 0])] += 1
            average = N / 8.0
            counts_tensor = torch.tensor(list(counts.values()))
            chisquare_statistic = torch.sum(
                (counts_tensor - average) * (counts_tensor - average) / average
            )
            # The 0.1 significance level for chisquare(8-1) is
            # scipy.stats.chi2(7).ppf(0.9) == 12.017.
            self.assertLess(chisquare_statistic, 12, (counts, chisquare_statistic, k))


class TestRotationConversion(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_from_quat(self):
        """quat -> mtx -> quat"""
        data = random_quaternions(13, dtype=torch.float64)
        mdata = matrix_to_quaternion(quaternion_to_matrix(data))
        self._assert_quaternions_close(data, mdata)

    def test_to_quat(self):
        """mtx -> quat -> mtx"""
        data = random_rotations(13, dtype=torch.float64)
        mdata = quaternion_to_matrix(matrix_to_quaternion(data))
        self.assertClose(data, mdata)

    def test_quat_grad_exists(self):
        """Quaternion calculations are differentiable."""
        rotation = random_rotation()
        rotation.requires_grad = True
        modified = quaternion_to_matrix(matrix_to_quaternion(rotation))
        [g] = torch.autograd.grad(modified.sum(), rotation)
        self.assertTrue(torch.isfinite(g).all())

    def _tait_bryan_conventions(self):
        return map("".join, itertools.permutations("XYZ"))

    def _proper_euler_conventions(self):
        letterpairs = itertools.permutations("XYZ", 2)
        return (l0 + l1 + l0 for l0, l1 in letterpairs)

    def _all_euler_angle_conventions(self):
        return itertools.chain(
            self._tait_bryan_conventions(), self._proper_euler_conventions()
        )

    def test_conventions(self):
        """The conventions listings have the right length."""
        all = list(self._all_euler_angle_conventions())
        self.assertEqual(len(all), 12)
        self.assertEqual(len(set(all)), 12)

    def test_from_euler(self):
        """euler -> mtx -> euler"""
        n_repetitions = 10
        # tolerance is how much we keep the middle angle away from the extreme
        # allowed values which make the calculation unstable (Gimbal lock).
        tolerance = 0.04
        half_pi = math.pi / 2
        data = torch.zeros(n_repetitions, 3)
        data.uniform_(-math.pi, math.pi)

        data[:, 1].uniform_(-half_pi + tolerance, half_pi - tolerance)
        for convention in self._tait_bryan_conventions():
            matrices = euler_angles_to_matrix(data, convention)
            mdata = matrix_to_euler_angles(matrices, convention)
            self.assertClose(data, mdata)

        data[:, 1] += half_pi
        for convention in self._proper_euler_conventions():
            matrices = euler_angles_to_matrix(data, convention)
            mdata = matrix_to_euler_angles(matrices, convention)
            self.assertClose(data, mdata)

    def test_to_euler(self):
        """mtx -> euler -> mtx"""
        data = random_rotations(13, dtype=torch.float64)
        for convention in self._all_euler_angle_conventions():
            euler_angles = matrix_to_euler_angles(data, convention)
            mdata = euler_angles_to_matrix(euler_angles, convention)
            self.assertClose(data, mdata)

    def test_euler_grad_exists(self):
        """Euler angle calculations are differentiable."""
        rotation = random_rotation(dtype=torch.float64)
        rotation.requires_grad = True
        for convention in self._all_euler_angle_conventions():
            euler_angles = matrix_to_euler_angles(rotation, convention)
            mdata = euler_angles_to_matrix(euler_angles, convention)
            [g] = torch.autograd.grad(mdata.sum(), rotation)
            self.assertTrue(torch.isfinite(g).all())

    def test_quaternion_multiplication(self):
        """Quaternion and matrix multiplication are equivalent."""
        a = random_quaternions(15, torch.float64).reshape((3, 5, 4))
        b = random_quaternions(21, torch.float64).reshape((7, 3, 1, 4))
        ab = quaternion_multiply(a, b)
        self.assertEqual(ab.shape, (7, 3, 5, 4))
        a_matrix = quaternion_to_matrix(a)
        b_matrix = quaternion_to_matrix(b)
        ab_matrix = torch.matmul(a_matrix, b_matrix)
        ab_from_matrix = matrix_to_quaternion(ab_matrix)
        self._assert_quaternions_close(ab, ab_from_matrix)

    def test_matrix_to_quaternion_corner_case(self):
        """Check no bad gradients from sqrt(0)."""
        matrix = torch.eye(3, requires_grad=True)
        target = torch.Tensor([0.984808, 0, 0.174, 0])

        optimizer = torch.optim.Adam([matrix], lr=0.05)
        optimizer.zero_grad()
        q = matrix_to_quaternion(matrix)
        loss = torch.sum((q - target) ** 2)
        loss.backward()
        optimizer.step()

        self.assertClose(matrix, matrix, msg="Result has non-finite values")
        delta = 1e-2
        self.assertLess(
            matrix.trace(),
            3.0 - delta,
            msg="Identity initialisation unchanged by a gradient step",
        )

    def test_matrix_to_quaternion_by_pi(self):
        # We check that rotations by pi around each of the 26
        # nonzero vectors containing nothing but 0, 1 and -1
        # are mapped to the right quaternions.
        # This is representative across the directions.
        options = [0.0, -1.0, 1.0]
        axes = [
            torch.tensor(vec)
            for vec in itertools.islice(  # exclude [0, 0, 0]
                itertools.product(options, options, options), 1, None
            )
        ]

        axes = torch.nn.functional.normalize(torch.stack(axes), dim=-1)
        # Rotation by pi around unit vector x is given by
        # the matrix 2 x x^T - Id.
        R = 2 * torch.matmul(axes[..., None], axes[..., None, :]) - torch.eye(3)
        quats_hat = matrix_to_quaternion(R)
        R_hat = quaternion_to_matrix(quats_hat)
        self.assertClose(R, R_hat, atol=1e-3)

    def test_from_axis_angle(self):
        """axis_angle -> mtx -> axis_angle"""
        n_repetitions = 20
        data = torch.rand(n_repetitions, 3)
        matrices = axis_angle_to_matrix(data)
        self.assertClose(data, matrix_to_axis_angle(matrices), atol=2e-6)
        self.assertClose(data, matrix_to_axis_angle(matrices, fast=True), atol=2e-6)
        matrices = axis_angle_to_matrix(data, fast=True)
        mdata = matrix_to_axis_angle(matrices)
        self.assertClose(data, mdata, atol=2e-6)

    def test_from_axis_angle_has_grad(self):
        n_repetitions = 20
        data = torch.rand(n_repetitions, 3, requires_grad=True)
        matrices = axis_angle_to_matrix(data)
        mdata = matrix_to_axis_angle(matrices)
        quats = axis_angle_to_quaternion(data)
        mdata2 = quaternion_to_axis_angle(quats)
        (grad,) = torch.autograd.grad(mdata.sum() + mdata2.sum(), data)
        self.assertTrue(torch.isfinite(grad).all())

    def test_to_axis_angle(self):
        """mtx -> axis_angle -> mtx"""
        data = random_rotations(13, dtype=torch.float64)
        euler_angles = matrix_to_axis_angle(data)
        euler_angles_fast = matrix_to_axis_angle(data)
        self.assertClose(data, axis_angle_to_matrix(euler_angles))
        self.assertClose(data, axis_angle_to_matrix(euler_angles_fast))
        self.assertClose(data, axis_angle_to_matrix(euler_angles, fast=True))

    def test_quaternion_application(self):
        """Applying a quaternion is the same as applying the matrix."""
        quaternions = random_quaternions(3, torch.float64)
        quaternions.requires_grad = True
        matrices = quaternion_to_matrix(quaternions)
        points = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        transform1 = quaternion_apply(quaternions, points)
        transform2 = torch.matmul(matrices, points[..., None])[..., 0]
        self.assertClose(transform1, transform2)

        [p, q] = torch.autograd.grad(transform1.sum(), [points, quaternions])
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.isfinite(q).all())

    def test_6d(self):
        """Converting to 6d and back"""
        r = random_rotations(13, dtype=torch.float64)

        # 6D representation is not unique,
        # but we implement it by taking the first two rows of the matrix
        r6d = matrix_to_rotation_6d(r)
        self.assertClose(r6d, r[:, :2, :].reshape(-1, 6))

        # going to 6D and back should not change the matrix
        r_hat = rotation_6d_to_matrix(r6d)
        self.assertClose(r_hat, r)

        # moving the second row R2 in the span of (R1, R2) should not matter
        r6d[:, 3:] += 2 * r6d[:, :3]
        r6d[:, :3] *= 3.0
        r_hat = rotation_6d_to_matrix(r6d)
        self.assertClose(r_hat, r)

        # check that we map anything to a valid rotation
        r6d = torch.rand(13, 6)
        r6d[:4, :] *= 3.0
        r6d[4:8, :] -= 0.5
        r = rotation_6d_to_matrix(r6d)
        self.assertClose(
            torch.matmul(r, r.permute(0, 2, 1)), torch.eye(3).expand_as(r), atol=1e-6
        )

    @unittest.skipIf(LooseVersion(torch.__version__) < "1.9", "recent torchscript only")
    def test_scriptable(self):
        torch.jit.script(axis_angle_to_matrix)
        torch.jit.script(axis_angle_to_quaternion)
        torch.jit.script(euler_angles_to_matrix)
        torch.jit.script(matrix_to_axis_angle)
        torch.jit.script(matrix_to_euler_angles)
        torch.jit.script(matrix_to_quaternion)
        torch.jit.script(matrix_to_rotation_6d)
        torch.jit.script(quaternion_apply)
        torch.jit.script(quaternion_multiply)
        torch.jit.script(quaternion_to_matrix)
        torch.jit.script(quaternion_to_axis_angle)
        torch.jit.script(random_quaternions)
        torch.jit.script(random_rotation)
        torch.jit.script(random_rotations)
        torch.jit.script(random_quaternions)
        torch.jit.script(rotation_6d_to_matrix)

    def _assert_quaternions_close(
        self,
        input: Union[torch.Tensor, np.ndarray],
        other: Union[torch.Tensor, np.ndarray],
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        msg: Optional[str] = None,
    ):
        self.assertEqual(np.shape(input), np.shape(other))
        dot = (input * other).sum(-1)
        ones = torch.ones_like(dot)
        self.assertClose(
            dot.abs(), ones, rtol=rtol, atol=atol, equal_nan=equal_nan, msg=msg
        )
