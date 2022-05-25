# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.common.linear_with_repeat import LinearWithRepeat

from .common_testing import TestCaseMixin


class TestLinearWithRepeat(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    def test_simple(self):
        x = torch.rand(4, 6, 7, 3)
        y = torch.rand(4, 6, 4)

        linear = torch.nn.Linear(7, 8)
        torch.nn.init.xavier_uniform_(linear.weight.data)
        linear.bias.data.uniform_()
        equivalent = torch.cat([x, y.unsqueeze(-2).expand(4, 6, 7, 4)], dim=-1)
        expected = linear.forward(equivalent)

        linear_with_repeat = LinearWithRepeat(7, 8)
        linear_with_repeat.load_state_dict(linear.state_dict())
        actual = linear_with_repeat.forward((x, y))
        self.assertClose(actual, expected, rtol=1e-4)
