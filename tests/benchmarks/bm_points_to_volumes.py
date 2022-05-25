# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from fvcore.common.benchmark import benchmark
from tests.test_points_to_volumes import TestPointsToVolumes


def bm_points_to_volumes() -> None:
    case_grid = {
        "device": ["cpu", "cuda:0"],
        "batch_size": [10, 100],
        "interp_mode": ["trilinear", "nearest"],
        "volume_size": [[25, 25, 25], [101, 111, 121]],
        "n_points": [1000, 10000, 100000],
    }
    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]

    benchmark(
        TestPointsToVolumes.add_points_to_volumes,
        "ADD_POINTS_TO_VOLUMES",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_points_to_volumes()
