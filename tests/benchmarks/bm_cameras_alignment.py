# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from fvcore.common.benchmark import benchmark
from tests.test_cameras_alignment import TestCamerasAlignment


def bm_cameras_alignment() -> None:
    case_grid = {
        "batch_size": [10, 100, 1000],
        "mode": ["centers", "extrinsics"],
        "estimate_scale": [False, True],
    }
    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]

    benchmark(
        TestCamerasAlignment.corresponding_cameras_alignment,
        "CORRESPONDING_CAMERAS_ALIGNMENT",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_cameras_alignment()
