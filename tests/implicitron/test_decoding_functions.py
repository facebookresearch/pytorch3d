# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch

from pytorch3d.implicitron.models.implicit_function.decoding_functions import (
    IdentityDecoder,
    MLPDecoder,
)
from pytorch3d.implicitron.tools.config import expand_args_fields

from tests.common_testing import TestCaseMixin


class TestVoxelGrids(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        expand_args_fields(IdentityDecoder)
        expand_args_fields(MLPDecoder)

    def test_identity_function(self, in_shape=(33, 4, 1), n_tests=2):
        """
        Test that identity function returns its input
        """
        func = IdentityDecoder()
        for _ in range(n_tests):
            _in = torch.randn(in_shape)
            assert torch.allclose(func(_in), _in)
