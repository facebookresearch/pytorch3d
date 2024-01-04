# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import math
import unittest
from distutils.version import LooseVersion

import numpy as np
import torch
from pytorch3d.transforms.so3 import (
    hat,
    so3_exp_map,
    so3_log_map,
    so3_relative_angle,
    so3_rotation_angle,
)

from .common_testing import TestCaseMixin


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
            r = torch.linalg.qr(torch.randn((3, 3), device=device))[0]
            f = torch.randint(2, (3,), device=device, dtype=torch.float32)
            if f.sum() % 2 == 0:
                f = 1 - f
            rot.append(r * (2 * f - 1).float())
        rot = torch.stack(rot)

        return rot

    def test_determinant(self):
        """
        Tests whether the determinants of 3x3 rotation matrices produced
        by `so3_exp_map` are (almost) equal to 1.
        """
        log_rot = TestSO3.init_log_rot(batch_size=30)
        Rs = so3_exp_map(log_rot)
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
        Tests whether `so3_exp_map` and `so3_log_map` correctly return
        a ValueError if called with an argument of incorrect shape or, in case
        of `so3_exp_map`, unexpected trace.
        """
        device = torch.device("cuda:0")
        log_rot = torch.randn(size=[5, 4], device=device)
        with self.assertRaises(ValueError) as err:
            so3_exp_map(log_rot)
        self.assertTrue("Input tensor shape has to be Nx3." in str(err.exception))

        rot = torch.randn(size=[5, 3, 5], device=device)
        with self.assertRaises(ValueError) as err:
            so3_log_map(rot)
        self.assertTrue("Input has to be a batch of 3x3 Tensors." in str(err.exception))

    def test_so3_exp_singularity(self, batch_size: int = 100):
        """
        Tests whether the `so3_exp_map` is robust to the input vectors
        the norms of which are close to the numerically unstable region
        (vectors with low l2-norms).
        """
        # generate random log-rotations with a tiny angle
        log_rot = TestSO3.init_log_rot(batch_size=batch_size)
        log_rot_small = log_rot * 1e-6
        log_rot_small.requires_grad = True
        R = so3_exp_map(log_rot_small)
        # tests whether all outputs are finite
        self.assertTrue(torch.isfinite(R).all())
        # tests whether the gradient is not None and all finite
        loss = R.sum()
        loss.backward()
        self.assertIsNotNone(log_rot_small.grad)
        self.assertTrue(torch.isfinite(log_rot_small.grad).all())

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
        # add random rotations and random almost orthonormal matrices
        r.extend(
            [
                torch.linalg.qr(identity + torch.randn_like(identity) * 1e-4)[0]
                + float(i > batch_size // 2) * (0.5 - torch.rand_like(identity)) * 1e-3
                # this adds random noise to the second half
                # of the random orthogonal matrices to generate
                # near-orthogonal matrices
                for i in range(batch_size - 2)
            ]
        )
        r = torch.stack(r)
        r.requires_grad = True
        # the log of the rotation matrix r
        r_log = so3_log_map(r, cos_bound=1e-4, eps=1e-2)
        # tests whether all outputs are finite
        self.assertTrue(torch.isfinite(r_log).all())
        # tests whether the gradient is not None and all finite
        loss = r.sum()
        loss.backward()
        self.assertIsNotNone(r.grad)
        self.assertTrue(torch.isfinite(r.grad).all())

    def test_so3_log_to_exp_to_log_to_exp(self, batch_size: int = 100):
        """
        Check that
        `so3_exp_map(so3_log_map(so3_exp_map(log_rot)))
        == so3_exp_map(log_rot)`
        for a randomly generated batch of rotation matrix logarithms `log_rot`.
        Unlike `test_so3_log_to_exp_to_log`, this test checks the
        correctness of converting a `log_rot` which contains values > math.pi.
        """
        log_rot = 2.0 * TestSO3.init_log_rot(batch_size=batch_size)
        # check also the singular cases where rot. angle = {0, 2pi}
        log_rot[:2] = 0
        log_rot[1, 0] = 2.0 * math.pi - 1e-6
        rot = so3_exp_map(log_rot, eps=1e-4)
        rot_ = so3_exp_map(so3_log_map(rot, eps=1e-4, cos_bound=1e-6), eps=1e-6)
        self.assertClose(rot, rot_, atol=0.01)
        angles = so3_relative_angle(rot, rot_, cos_bound=1e-6)
        self.assertClose(angles, torch.zeros_like(angles), atol=0.01)

    def test_so3_log_to_exp_to_log(self, batch_size: int = 100):
        """
        Check that `so3_log_map(so3_exp_map(log_rot))==log_rot` for
        a randomly generated batch of rotation matrix logarithms `log_rot`.
        """
        log_rot = TestSO3.init_log_rot(batch_size=batch_size)
        # check also the singular cases where rot. angle = 0
        log_rot[:1] = 0
        log_rot_ = so3_log_map(so3_exp_map(log_rot))
        self.assertClose(log_rot, log_rot_, atol=1e-4)

    def test_so3_exp_to_log_to_exp(self, batch_size: int = 100):
        """
        Check that `so3_exp_map(so3_log_map(R))==R` for
        a batch of randomly generated rotation matrices `R`.
        """
        rot = TestSO3.init_rot(batch_size=batch_size)
        non_singular = (so3_rotation_angle(rot) - math.pi).abs() > 1e-2
        rot = rot[non_singular]
        rot_ = so3_exp_map(so3_log_map(rot, eps=1e-8, cos_bound=1e-8), eps=1e-8)
        self.assertClose(rot_, rot, atol=0.1)
        angles = so3_relative_angle(rot, rot_, cos_bound=1e-4)
        self.assertClose(angles, torch.zeros_like(angles), atol=0.1)

    def test_so3_cos_relative_angle(self, batch_size: int = 100):
        """
        Check that `so3_relative_angle(R1, R2, cos_angle=False).cos()`
        is the same as `so3_relative_angle(R1, R2, cos_angle=True)` for
        batches of randomly generated rotation matrices `R1` and `R2`.
        """
        rot1 = TestSO3.init_rot(batch_size=batch_size)
        rot2 = TestSO3.init_rot(batch_size=batch_size)
        angles = so3_relative_angle(rot1, rot2, cos_angle=False).cos()
        angles_ = so3_relative_angle(rot1, rot2, cos_angle=True)
        self.assertClose(angles, angles_, atol=1e-4)

    def test_so3_cos_angle(self, batch_size: int = 100):
        """
        Check that `so3_rotation_angle(R, cos_angle=False).cos()`
        is the same as `so3_rotation_angle(R, cos_angle=True)` for
        a batch of randomly generated rotation matrices `R`.
        """
        rot = TestSO3.init_rot(batch_size=batch_size)
        angles = so3_rotation_angle(rot, cos_angle=False).cos()
        angles_ = so3_rotation_angle(rot, cos_angle=True)
        self.assertClose(angles, angles_, atol=1e-4)

    def test_so3_cos_bound(self, batch_size: int = 100):
        """
        Checks that for an identity rotation `R=I`, the so3_rotation_angle returns
        non-finite gradients when `cos_bound=None` and finite gradients
        for `cos_bound > 0.0`.
        """
        # generate random rotations with a tiny angle to generate cases
        # with the gradient singularity
        device = torch.device("cuda:0")
        identity = torch.eye(3, device=device)
        rot180 = identity * torch.tensor([[1.0, -1.0, -1.0]], device=device)
        r = [identity, rot180]
        r.extend(
            [
                torch.linalg.qr(identity + torch.randn_like(identity) * 1e-4)[0]
                for _ in range(batch_size - 2)
            ]
        )
        r = torch.stack(r)
        r.requires_grad = True
        for is_grad_finite in (True, False):
            # clear the gradients and decide the cos_bound:
            #     for is_grad_finite we run so3_rotation_angle with cos_bound
            #     set to a small float, otherwise we set to 0.0
            r.grad = None
            cos_bound = 1e-4 if is_grad_finite else 0.0
            # compute the angles of r
            angles = so3_rotation_angle(r, cos_bound=cos_bound)
            # tests whether all outputs are finite in both cases
            self.assertTrue(torch.isfinite(angles).all())
            # compute the gradients
            loss = angles.sum()
            loss.backward()
            # tests whether the gradient is not None for both cases
            self.assertIsNotNone(r.grad)
            if is_grad_finite:
                # all grad values have to be finite
                self.assertTrue(torch.isfinite(r.grad).all())

    @unittest.skipIf(LooseVersion(torch.__version__) < "1.9", "recent torchscript only")
    def test_scriptable(self):
        torch.jit.script(so3_exp_map)
        torch.jit.script(so3_log_map)

    @staticmethod
    def so3_expmap(batch_size: int = 10):
        log_rot = TestSO3.init_log_rot(batch_size=batch_size)
        torch.cuda.synchronize()

        def compute_rots():
            so3_exp_map(log_rot)
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
