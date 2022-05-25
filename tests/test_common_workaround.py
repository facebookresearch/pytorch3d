# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import torch
from pytorch3d.common.workaround import _safe_det_3x3

from .common_testing import TestCaseMixin


class TestSafeDet3x3(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    def _test_det_3x3(self, batch_size, device):
        t = torch.rand((batch_size, 3, 3), dtype=torch.float32, device=device)
        actual_det = _safe_det_3x3(t)
        expected_det = t.det()
        self.assertClose(actual_det, expected_det, atol=1e-7)

    def test_empty_batch(self):
        self._test_det_3x3(0, torch.device("cpu"))
        self._test_det_3x3(0, torch.device("cuda:0"))

    def test_manual(self):
        t = torch.Tensor(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                [[2, -5, 3], [0, 7, -2], [-1, 4, 1]],
                [[6, 1, 1], [4, -2, 5], [2, 8, 7]],
            ]
        ).to(dtype=torch.float32)
        expected_det = torch.Tensor([1, 41, -306]).to(dtype=torch.float32)
        self.assertClose(_safe_det_3x3(t), expected_det)

        device_cuda = torch.device("cuda:0")
        self.assertClose(
            _safe_det_3x3(t.to(device=device_cuda)), expected_det.to(device=device_cuda)
        )

    def test_regression(self):
        tries = 32
        device_cpu = torch.device("cpu")
        device_cuda = torch.device("cuda:0")
        batch_sizes = np.random.randint(low=1, high=128, size=tries)

        for batch_size in batch_sizes:
            self._test_det_3x3(batch_size, device_cpu)
            self._test_det_3x3(batch_size, device_cuda)
