# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import numpy as np
import torch
from pytorch3d.transforms.so3 import (
    hat,
    so3_exponential_map,
    so3_log_map,
    so3_relative_angle,
)


class TestSO3(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    @staticmethod
    def init_log_rot(batch_size: int = 10):
        """
        Initialize a list of `batch_size` 3-dimensional vectors representing
        randomly generated logarithms of rotation matrices.
        """
        device = torch.device("cuda:0")
        log_rot = torch.randn((batch_size, 3), dtype=torch.float32, device=device)
        return log_rot

    @staticmethod
    def init_rot(batch_size: int = 10):
        """
        Randomly generate a batch of `batch_size` 3x3 rotation matrices.
        """
        device = torch.device("cuda:0")

        # TODO(dnovotny): replace with random_rotation from random_rotation.py
        rot = []
        for _ in range(batch_size):
            r = torch.qr(torch.randn((3, 3), device=device))[0]
            f = torch.randint(2, (3,), device=device, dtype=torch.float32)
            if f.sum() % 2 == 0:
                f = 1 - f
            rot.append(r * (2 * f - 1).float())
        rot = torch.stack(rot)

        return rot

    def test_determinant(self):
        """
        Tests whether the determinants of 3x3 rotation matrices produced
        by `so3_exponential_map` are (almost) equal to 1.
        """
        log_rot = TestSO3.init_log_rot(batch_size=30)
        Rs = so3_exponential_map(log_rot)
        for R in Rs:
            det = np.linalg.det(R.cpu().numpy())
            self.assertAlmostEqual(float(det), 1.0, 5)

    def test_cross(self):
        """
        For a pair of randomly generated 3-dimensional vectors `a` and `b`,
        tests whether a matrix product of `hat(a)` and `b` equals the result
        of a cross product between `a` and `b`.
        """
        device = torch.device("cuda:0")
        a, b = torch.randn((2, 100, 3), dtype=torch.float32, device=device)
        hat_a = hat(a)
        cross = torch.bmm(hat_a, b[:, :, None])[:, :, 0]
        torch_cross = torch.cross(a, b, dim=1)
        max_df = (cross - torch_cross).abs().max()
        self.assertAlmostEqual(float(max_df), 0.0, 5)

    def test_bad_so3_input_value_err(self):
        """
        Tests whether `so3_exponential_map` and `so3_log_map` correctly return
        a ValueError if called with an argument of incorrect shape or, in case
        of `so3_exponential_map`, unexpected trace.
        """
        device = torch.device("cuda:0")
        log_rot = torch.randn(size=[5, 4], device=device)
        with self.assertRaises(ValueError) as err:
            so3_exponential_map(log_rot)
        self.assertTrue("Input tensor shape has to be Nx3." in str(err.exception))

        rot = torch.randn(size=[5, 3, 5], device=device)
        with self.assertRaises(ValueError) as err:
            so3_log_map(rot)
        self.assertTrue("Input has to be a batch of 3x3 Tensors." in str(err.exception))

        # trace of rot definitely bigger than 3 or smaller than -1
        rot = torch.cat(
            (
                torch.rand(size=[5, 3, 3], device=device) + 4.0,
                torch.rand(size=[5, 3, 3], device=device) - 3.0,
            )
        )
        with self.assertRaises(ValueError) as err:
            so3_log_map(rot)
        self.assertTrue(
            "A matrix has trace outside valid range [-1-eps,3+eps]."
            in str(err.exception)
        )

    def test_so3_exp_singularity(self, batch_size: int = 100):
        """
        Tests whether the `so3_exponential_map` is robust to the input vectors
        the norms of which are close to the numerically unstable region
        (vectors with low l2-norms).
        """
        # generate random log-rotations with a tiny angle
        log_rot = TestSO3.init_log_rot(batch_size=batch_size)
        log_rot_small = log_rot * 1e-6
        R = so3_exponential_map(log_rot_small)
        # tests whether all outputs are finite
        R_sum = float(R.sum())
        self.assertEqual(R_sum, R_sum)

    def test_so3_log_singularity(self, batch_size: int = 100):
        """
        Tests whether the `so3_log_map` is robust to the input matrices
        who's rotation angles are close to the numerically unstable region
        (i.e. matrices with low rotation angles).
        """
        # generate random rotations with a tiny angle
        device = torch.device("cuda:0")
        r = torch.eye(3, device=device)[None].repeat((batch_size, 1, 1))
        r += torch.randn((batch_size, 3, 3), device=device) * 1e-3
        r = torch.stack([torch.qr(r_)[0] for r_ in r])
        # the log of the rotation matrix r
        r_log = so3_log_map(r)
        # tests whether all outputs are finite
        r_sum = float(r_log.sum())
        self.assertEqual(r_sum, r_sum)

    def test_so3_log_to_exp_to_log(self, batch_size: int = 100):
        """
        Check that `so3_log_map(so3_exponential_map(log_rot))==log_rot` for
        a randomly generated batch of rotation matrix logarithms `log_rot`.
        """
        log_rot = TestSO3.init_log_rot(batch_size=batch_size)
        log_rot_ = so3_log_map(so3_exponential_map(log_rot))
        max_df = (log_rot - log_rot_).abs().max()
        self.assertAlmostEqual(float(max_df), 0.0, 4)

    def test_so3_exp_to_log_to_exp(self, batch_size: int = 100):
        """
        Check that `so3_exponential_map(so3_log_map(R))==R` for
        a batch of randomly generated rotation matrices `R`.
        """
        rot = TestSO3.init_rot(batch_size=batch_size)
        rot_ = so3_exponential_map(so3_log_map(rot))
        angles = so3_relative_angle(rot, rot_)
        max_angle = angles.max()
        # a lot of precision lost here :(
        # TODO: fix this test??
        self.assertTrue(np.allclose(float(max_angle), 0.0, atol=0.1))

    def test_so3_cos_angle(self, batch_size: int = 100):
        """
        Check that `so3_relative_angle(R1, R2, cos_angle=False).cos()`
        is the same as `so3_relative_angle(R1, R2, cos_angle=True)`
        batches of randomly generated rotation matrices `R1` and `R2`.
        """
        rot1 = TestSO3.init_rot(batch_size=batch_size)
        rot2 = TestSO3.init_rot(batch_size=batch_size)
        angles = so3_relative_angle(rot1, rot2, cos_angle=False).cos()
        angles_ = so3_relative_angle(rot1, rot2, cos_angle=True)
        self.assertTrue(torch.allclose(angles, angles_))

    @staticmethod
    def so3_expmap(batch_size: int = 10):
        log_rot = TestSO3.init_log_rot(batch_size=batch_size)
        torch.cuda.synchronize()

        def compute_rots():
            so3_exponential_map(log_rot)
            torch.cuda.synchronize()

        return compute_rots

    @staticmethod
    def so3_logmap(batch_size: int = 10):
        log_rot = TestSO3.init_rot(batch_size=batch_size)
        torch.cuda.synchronize()

        def compute_logs():
            so3_log_map(log_rot)
            torch.cuda.synchronize()

        return compute_logs
