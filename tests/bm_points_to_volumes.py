# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools

from fvcore.common.benchmark import benchmark
from test_points_to_volumes import TestPointsToVolumes


def bm_points_to_volumes() -> None:
    case_grid = {
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
