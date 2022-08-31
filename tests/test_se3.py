# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import random_rotations
from pytorch3d.transforms.se3 import se3_exp_map, se3_log_map
from pytorch3d.transforms.so3 import so3_exp_map, so3_log_map, so3_rotation_angle

from .common_testing import TestCaseMixin


class TestSE3(TestCaseMixin, unittest.TestCase):
    precomputed_log_transform = torch.tensor(
        [
            [0.1900, 2.1600, -0.1700, 0.8500, -1.9200, 0.6500],
            [-0.6500, -0.8200, 0.5300, -1.2800, -1.6600, -0.3000],
            [-0.0900, 0.2000, -1.1200, 1.8600, -0.7100, 0.6900],
            [0.8000, -0.0300, 1.4900, -0.5200, -0.2500, 1.4700],
            [-0.3300, -1.1600, 2.3600, -0.6900, 0.1800, -1.1800],
            [-1.8000, -1.5800, 0.8400, 1.4200, 0.6500, 0.4300],
            [-1.5900, 0.6200, 1.6900, -0.6600, 0.9400, 0.0800],
            [0.0800, -0.1400, 0.3300, -0.5900, -1.0700, 0.1000],
            [-0.3300, -0.5300, -0.8800, 0.3900, 0.1600, -0.2000],
            [1.0100, -1.3500, -0.3500, -0.6400, 0.4500, -0.5400],
        ],
        dtype=torch.float32,
    )

    precomputed_transform = torch.tensor(
        [
            [
                [-0.3496, -0.2966, 0.8887, 0.0000],
                [-0.7755, 0.6239, -0.0968, 0.0000],
                [-0.5258, -0.7230, -0.4481, 0.0000],
                [-0.7392, 1.9119, 0.3122, 1.0000],
            ],
            [
                [0.0354, 0.5992, 0.7998, 0.0000],
                [0.8413, 0.4141, -0.3475, 0.0000],
                [-0.5395, 0.6852, -0.4894, 0.0000],
                [-0.9902, -0.4840, 0.1226, 1.0000],
            ],
            [
                [0.6664, -0.1679, 0.7264, 0.0000],
                [-0.7309, -0.3394, 0.5921, 0.0000],
                [0.1471, -0.9255, -0.3489, 0.0000],
                [-0.0815, 0.8719, -0.4516, 1.0000],
            ],
            [
                [0.1010, 0.9834, -0.1508, 0.0000],
                [-0.8783, 0.0169, -0.4779, 0.0000],
                [-0.4674, 0.1807, 0.8654, 0.0000],
                [0.2375, 0.7043, 1.4159, 1.0000],
            ],
            [
                [0.3935, -0.8930, 0.2184, 0.0000],
                [0.7873, 0.2047, -0.5817, 0.0000],
                [0.4747, 0.4009, 0.7836, 0.0000],
                [-0.3476, -0.0424, 2.5408, 1.0000],
            ],
            [
                [0.7572, 0.6342, -0.1567, 0.0000],
                [0.1039, 0.1199, 0.9873, 0.0000],
                [0.6449, -0.7638, 0.0249, 0.0000],
                [-1.2885, -2.0666, -0.1137, 1.0000],
            ],
            [
                [0.6020, -0.2140, -0.7693, 0.0000],
                [-0.3409, 0.8024, -0.4899, 0.0000],
                [0.7221, 0.5572, 0.4101, 0.0000],
                [-0.7550, 1.1928, 1.8480, 1.0000],
            ],
            [
                [0.4913, 0.3548, 0.7954, 0.0000],
                [0.2013, 0.8423, -0.5000, 0.0000],
                [-0.8474, 0.4058, 0.3424, 0.0000],
                [-0.1003, -0.0406, 0.3295, 1.0000],
            ],
            [
                [0.9678, -0.1622, -0.1926, 0.0000],
                [0.2235, 0.9057, 0.3603, 0.0000],
                [0.1160, -0.3917, 0.9128, 0.0000],
                [-0.4417, -0.3111, -0.9227, 1.0000],
            ],
            [
                [0.7710, -0.5957, -0.2250, 0.0000],
                [0.3288, 0.6750, -0.6605, 0.0000],
                [0.5454, 0.4352, 0.7163, 0.0000],
                [0.5623, -1.5886, -0.0182, 1.0000],
            ],
        ],
        dtype=torch.float32,
    )

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    @staticmethod
    def init_log_transform(batch_size: int = 10):
        """
        Initialize a list of `batch_size` 6-dimensional vectors representing
        randomly generated logarithms of SE(3) transforms.
        """
        device = torch.device("cuda:0")
        log_rot = torch.randn((batch_size, 6), dtype=torch.float32, device=device)
        return log_rot

    @staticmethod
    def init_transform(batch_size: int = 10):
        """
        Initialize a list of `batch_size` 4x4 SE(3) transforms.
        """
        device = torch.device("cuda:0")
        transform = torch.zeros(batch_size, 4, 4, dtype=torch.float32, device=device)
        transform[:, :3, :3] = random_rotations(
            batch_size, dtype=torch.float32, device=device
        )
        transform[:, 3, :3] = torch.randn(
            (batch_size, 3), dtype=torch.float32, device=device
        )
        transform[:, 3, 3] = 1.0
        return transform

    def test_se3_exp_output_format(self, batch_size: int = 100):
        """
        Check that the output of `se3_exp_map` is a valid SE3 matrix.
        """
        transform = se3_exp_map(TestSE3.init_log_transform(batch_size=batch_size))
        R = transform[:, :3, :3]
        T = transform[:, 3, :3]
        rest = transform[:, :, 3]
        Rdet = R.det()

        # check det(R)==1
        self.assertClose(Rdet, torch.ones_like(Rdet), atol=1e-4)

        # check that the translation is a finite vector
        self.assertTrue(torch.isfinite(T).all())

        # check last column == [0,0,0,1]
        last_col = rest.new_zeros(batch_size, 4)
        last_col[:, -1] = 1.0
        self.assertClose(rest, last_col)

    def test_compare_with_precomputed(self):
        """
        Compare the outputs against precomputed results.
        """
        self.assertClose(
            se3_log_map(self.precomputed_transform),
            self.precomputed_log_transform,
            atol=1e-4,
        )
        self.assertClose(
            self.precomputed_transform,
            se3_exp_map(self.precomputed_log_transform),
            atol=1e-4,
        )

    def test_se3_exp_singularity(self, batch_size: int = 100):
        """
        Tests whether the `se3_exp_map` is robust to the input vectors
        with low L2 norms, where the algorithm is numerically unstable.
        """
        # generate random log-rotations with a tiny angle
        log_rot = TestSE3.init_log_transform(batch_size=batch_size)
        log_rot_small = log_rot * 1e-6
        log_rot_small.requires_grad = True
        transforms = se3_exp_map(log_rot_small)
        # tests whether all outputs are finite
        self.assertTrue(torch.isfinite(transforms).all())
        # tests whether all gradients are finite and not None
        loss = transforms.sum()
        loss.backward()
        self.assertIsNotNone(log_rot_small.grad)
        self.assertTrue(torch.isfinite(log_rot_small.grad).all())

    def test_se3_log_singularity(self, batch_size: int = 100):
        """
        Tests whether the `se3_log_map` is robust to the input matrices
        whose rotation angles and translations are close to the numerically
        unstable region (i.e. matrices with low rotation angles
        and 0 translation).
        """
        # generate random rotations with a tiny angle
        device = torch.device("cuda:0")
        identity = torch.eye(3, device=device)
        rot180 = identity * torch.tensor([[1.0, -1.0, -1.0]], device=device)
        r = [identity, rot180]
        r.extend(
            [
                torch.linalg.qr(identity + torch.randn_like(identity) * 1e-6)[0]
                + float(i > batch_size // 2) * (0.5 - torch.rand_like(identity)) * 1e-8
                # this adds random noise to the second half
                # of the random orthogonal matrices to generate
                # near-orthogonal matrices
                for i in range(batch_size - 2)
            ]
        )
        r = torch.stack(r)
        # tiny translations
        t = torch.randn(batch_size, 3, dtype=r.dtype, device=device) * 1e-6
        # create the transform matrix
        transform = torch.zeros(batch_size, 4, 4, dtype=torch.float32, device=device)
        transform[:, :3, :3] = r
        transform[:, 3, :3] = t
        transform[:, 3, 3] = 1.0
        transform.requires_grad = True
        # the log of the transform
        log_transform = se3_log_map(transform, eps=1e-4, cos_bound=1e-4)
        # tests whether all outputs are finite
        self.assertTrue(torch.isfinite(log_transform).all())
        # tests whether all gradients are finite and not None
        loss = log_transform.sum()
        loss.backward()
        self.assertIsNotNone(transform.grad)
        self.assertTrue(torch.isfinite(transform.grad).all())

    def test_se3_exp_zero_translation(self, batch_size: int = 100):
        """
        Check that `se3_exp_map` with zero translation gives
        the same result as corresponding `so3_exp_map`.
        """
        log_transform = TestSE3.init_log_transform(batch_size=batch_size)
        log_transform[:, :3] *= 0.0
        transform = se3_exp_map(log_transform, eps=1e-8)
        transform_so3 = so3_exp_map(log_transform[:, 3:], eps=1e-8)
        self.assertClose(
            transform[:, :3, :3], transform_so3.permute(0, 2, 1), atol=1e-4
        )
        self.assertClose(
            transform[:, 3, :3], torch.zeros_like(transform[:, :3, 3]), atol=1e-4
        )

    def test_se3_log_zero_translation(self, batch_size: int = 100):
        """
        Check that `se3_log_map` with zero translation gives
        the same result as corresponding `so3_log_map`.
        """
        transform = TestSE3.init_transform(batch_size=batch_size)
        transform[:, 3, :3] *= 0.0
        log_transform = se3_log_map(transform, eps=1e-8, cos_bound=1e-4)
        log_transform_so3 = so3_log_map(transform[:, :3, :3], eps=1e-8, cos_bound=1e-4)
        self.assertClose(log_transform[:, 3:], -log_transform_so3, atol=1e-4)
        self.assertClose(
            log_transform[:, :3], torch.zeros_like(log_transform[:, :3]), atol=1e-4
        )

    def test_se3_exp_to_log_to_exp(self, batch_size: int = 10000):
        """
        Check that `se3_exp_map(se3_log_map(A))==A` for
        a batch of randomly generated SE(3) matrices `A`.
        """
        transform = TestSE3.init_transform(batch_size=batch_size)
        # Limit test transforms to those not around the singularity where
        # the rotation angle~=pi.
        nonsingular = so3_rotation_angle(transform[:, :3, :3]) < 3.134
        transform = transform[nonsingular]
        transform_ = se3_exp_map(
            se3_log_map(transform, eps=1e-8, cos_bound=0.0), eps=1e-8
        )
        self.assertClose(transform, transform_, atol=0.02)

    def test_se3_log_to_exp_to_log(self, batch_size: int = 100):
        """
        Check that `se3_log_map(se3_exp_map(log_transform))==log_transform`
        for a randomly generated batch of SE(3) matrix logarithms `log_transform`.
        """
        log_transform = TestSE3.init_log_transform(batch_size=batch_size)
        log_transform_ = se3_log_map(se3_exp_map(log_transform, eps=1e-8), eps=1e-8)
        self.assertClose(log_transform, log_transform_, atol=1e-1)

    def test_bad_se3_input_value_err(self):
        """
        Tests whether `se3_exp_map` and `se3_log_map` correctly return
        a ValueError if called with an argument of incorrect shape, or with
        an tensor containing illegal values.
        """
        device = torch.device("cuda:0")

        for size in ([5, 4], [3, 4, 5], [3, 5, 6]):
            log_transform = torch.randn(size=size, device=device)
            with self.assertRaises(ValueError):
                se3_exp_map(log_transform)

        for size in ([5, 4], [3, 4, 5], [3, 5, 6], [2, 2, 3, 4]):
            transform = torch.randn(size=size, device=device)
            with self.assertRaises(ValueError):
                se3_log_map(transform)

        # Test the case where transform[:, :, :3] != 0.
        transform = torch.rand(size=[5, 4, 4], device=device) + 0.1
        with self.assertRaises(ValueError):
            se3_log_map(transform)

    @staticmethod
    def se3_expmap(batch_size: int = 10):
        log_transform = TestSE3.init_log_transform(batch_size=batch_size)
        torch.cuda.synchronize()

        def compute_transforms():
            se3_exp_map(log_transform)
            torch.cuda.synchronize()

        return compute_transforms

    @staticmethod
    def se3_logmap(batch_size: int = 10):
        log_transform = TestSE3.init_transform(batch_size=batch_size)
        torch.cuda.synchronize()

        def compute_logs():
            se3_log_map(log_transform)
            torch.cuda.synchronize()

        return compute_logs
