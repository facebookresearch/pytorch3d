# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from pytorch3d.implicitron import eval_demo

from tests.common_testing import interactive_testing_requested

from .common_resources import CO3D_MANIFOLD_PATH

"""
This test runs a single sequence eval_demo, useful for debugging datasets.
It only runs interactively.
"""


class TestEvalDemo(unittest.TestCase):
    def test_a(self):
        if not interactive_testing_requested():
            return

        os.environ["CO3D_DATASET_ROOT"] = CO3D_MANIFOLD_PATH

        eval_demo.evaluate_dbir_for_category("donut", single_sequence_id=0)
