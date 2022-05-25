# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch

from .common_testing import TestCaseMixin


class TestOpsUtils(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    def test_all_close(self):
        device = torch.device("cuda:0")
        n_points = 20
        noise_std = 1e-3
        msg = "tratata"

        # test absolute tolerance
        x = torch.rand(n_points, 3, device=device)
        x_noise = x + noise_std * torch.rand(n_points, 3, device=device)
        assert torch.allclose(x, x_noise, atol=10 * noise_std)
        assert not torch.allclose(x, x_noise, atol=0.1 * noise_std)
        self.assertClose(x, x_noise, atol=10 * noise_std)
        with self.assertRaises(AssertionError) as context:
            self.assertClose(x, x_noise, atol=0.1 * noise_std, msg=msg)
        self.assertTrue(msg in str(context.exception))

        # test numpy
        def to_np(t):
            return t.data.cpu().numpy()

        self.assertClose(to_np(x), to_np(x_noise), atol=10 * noise_std)
        with self.assertRaises(AssertionError) as context:
            self.assertClose(to_np(x), to_np(x_noise), atol=0.1 * noise_std, msg=msg)
        self.assertIn(msg, str(context.exception))
        self.assertIn("Not close", str(context.exception))

        # test relative tolerance
        assert torch.allclose(x, x_noise, rtol=100 * noise_std)
        assert not torch.allclose(x, x_noise, rtol=noise_std)
        self.assertClose(x, x_noise, rtol=100 * noise_std)
        with self.assertRaises(AssertionError) as context:
            self.assertClose(x, x_noise, rtol=noise_std, msg=msg)
        self.assertTrue(msg in str(context.exception))

        # test norm aggregation
        # if one of the spatial dimensions is small, norm aggregation helps
        x_noise[:, 0] = x_noise[:, 0] - x[:, 0]
        x[:, 0] = 0.0
        assert not torch.allclose(x, x_noise, rtol=100 * noise_std)
        self.assertNormsClose(
            x, x_noise, rtol=100 * noise_std, norm_fn=lambda t: t.norm(dim=-1)
        )
