# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import math
import unittest

import numpy as np
import torch
from common_testing import TestCaseMixin
from pytorch3d.transforms.so3 import (
    hat,
    so3_exponential_map,
    so3_log_map,
    so3_relative_angle,
)


class TestSO3(TestCaseMixin, unittest.TestCase):
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
        dets = torch.det(Rs)
        self.assertClose(dets, torch.ones_like(dets), atol=1e-4)

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
        self.assertClose(torch_cross, cross, atol=1e-4)

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
        identity = torch.eye(3, device=device)
        rot180 = identity * torch.tensor([[1.0, -1.0, -1.0]], device=device)
        r = [identity, rot180]
        r.extend(
            [
                torch.qr(identity + torch.randn_like(identity) * 1e-4)[0]
                for _ in range(batch_size - 2)
            ]
        )
        r = torch.stack(r)
        # the log of the rotation matrix r
        r_log = so3_log_map(r)
        # tests whether all outputs are finite
        r_sum = float(r_log.sum())
        self.assertEqual(r_sum, r_sum)

    def test_so3_log_to_exp_to_log_to_exp(self, batch_size: int = 100):
        """
        Check that
        `so3_exponential_map(so3_log_map(so3_exponential_map(log_rot)))
        == so3_exponential_map(log_rot)`
        for a randomly generated batch of rotation matrix logarithms `log_rot`.
        Unlike `test_so3_log_to_exp_to_log`, this test allows to check the
        correctness of converting `log_rot` which contains values > math.pi.
        """
        log_rot = 2.0 * TestSO3.init_log_rot(batch_size=batch_size)
        # check also the singular cases where rot. angle = {0, pi, 2pi, 3pi}
        log_rot[:3] = 0
        log_rot[1, 0] = math.pi
        log_rot[2, 0] = 2.0 * math.pi
        log_rot[3, 0] = 3.0 * math.pi
        rot = so3_exponential_map(log_rot, eps=1e-8)
        rot_ = so3_exponential_map(so3_log_map(rot, eps=1e-8), eps=1e-8)
        angles = so3_relative_angle(rot, rot_)
        self.assertClose(angles, torch.zeros_like(angles), atol=0.01)

    def test_so3_log_to_exp_to_log(self, batch_size: int = 100):
        """
        Check that `so3_log_map(so3_exponential_map(log_rot))==log_rot` for
        a randomly generated batch of rotation matrix logarithms `log_rot`.
        """
        log_rot = TestSO3.init_log_rot(batch_size=batch_size)
        # check also the singular cases where rot. angle = 0
        log_rot[:1] = 0
        log_rot_ = so3_log_map(so3_exponential_map(log_rot))
        self.assertClose(log_rot, log_rot_, atol=1e-4)

    def test_so3_exp_to_log_to_exp(self, batch_size: int = 100):
        """
        Check that `so3_exponential_map(so3_log_map(R))==R` for
        a batch of randomly generated rotation matrices `R`.
        """
        rot = TestSO3.init_rot(batch_size=batch_size)
        rot_ = so3_exponential_map(so3_log_map(rot, eps=1e-8), eps=1e-8)
        angles = so3_relative_angle(rot, rot_)
        # TODO: a lot of precision lost here ...
        self.assertClose(angles, torch.zeros_like(angles), atol=0.1)

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
        self.assertClose(angles, angles_)

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
