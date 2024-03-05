# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import unittest

from .. import visualize_reconstruction
from .utils import interactive_testing_requested

internal = os.environ.get("FB_TEST", False)


class TestVisualize(unittest.TestCase):
    def test_from_defaults(self):
        if not interactive_testing_requested():
            return
        checkpoint_dir = os.environ["exp_dir"]
        argv = [
            f"exp_dir={checkpoint_dir}",
            "n_eval_cameras=40",
            "render_size=[64,64]",
            "video_size=[256,256]",
        ]
        visualize_reconstruction.main(argv)
