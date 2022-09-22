# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.utils import checkerboard

from .common_testing import TestCaseMixin


class TestCheckerboard(TestCaseMixin, unittest.TestCase):
    def test_simple(self):
        board = checkerboard(5)
        verts = board.verts_packed()
        expect = torch.tensor([5.0, 5.0, 0])
        self.assertClose(verts.min(dim=0).values, -expect)
        self.assertClose(verts.max(dim=0).values, expect)
