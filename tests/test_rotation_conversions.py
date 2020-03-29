# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import itertools
import math
import unittest

import torch
from pytorch3d.transforms.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_apply,
    quaternion_multiply,
    quaternion_to_matrix,
    random_quaternions,
    random_rotation,
    random_rotations,
)


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


class TestRotationConversion(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_from_quat(self):
        """quat -> mtx -> quat"""
        data = random_quaternions(13, dtype=torch.float64)
        mdata = matrix_to_quaternion(quaternion_to_matrix(data))
        self.assertTrue(torch.allclose(data, mdata))

    def test_to_quat(self):
        """mtx -> quat -> mtx"""
        data = random_rotations(13, dtype=torch.float64)
        mdata = quaternion_to_matrix(matrix_to_quaternion(data))
        self.assertTrue(torch.allclose(data, mdata))

    def test_quat_grad_exists(self):
        """Quaternion calculations are differentiable."""
        rotation = random_rotation(requires_grad=True)
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
            self.assertTrue(torch.allclose(data, mdata))

        data[:, 1] += half_pi
        for convention in self._proper_euler_conventions():
            matrices = euler_angles_to_matrix(data, convention)
            mdata = matrix_to_euler_angles(matrices, convention)
            self.assertTrue(torch.allclose(data, mdata))

    def test_to_euler(self):
        """mtx -> euler -> mtx"""
        data = random_rotations(13, dtype=torch.float64)
        for convention in self._all_euler_angle_conventions():
            euler_angles = matrix_to_euler_angles(data, convention)
            mdata = euler_angles_to_matrix(euler_angles, convention)
            self.assertTrue(torch.allclose(data, mdata))

    def test_euler_grad_exists(self):
        """Euler angle calculations are differentiable."""
        rotation = random_rotation(dtype=torch.float64, requires_grad=True)
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
        self.assertEqual(ab.shape, ab_from_matrix.shape)
        self.assertTrue(torch.allclose(ab, ab_from_matrix))

    def test_quaternion_application(self):
        """Applying a quaternion is the same as applying the matrix."""
        quaternions = random_quaternions(3, torch.float64, requires_grad=True)
        matrices = quaternion_to_matrix(quaternions)
        points = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        transform1 = quaternion_apply(quaternions, points)
        transform2 = torch.matmul(matrices, points[..., None])[..., 0]
        self.assertTrue(torch.allclose(transform1, transform2))

        [p, q] = torch.autograd.grad(transform1.sum(), [points, quaternions])
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.isfinite(q).all())
