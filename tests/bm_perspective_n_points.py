# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import itertools

from fvcore.common.benchmark import benchmark
from test_perspective_n_points import TestPerspectiveNPoints


def bm_perspective_n_points() -> None:
    case_grid = {
        "batch_size": [1, 10, 100],
        "num_pts": [100, 100000],
        "skip_q": [False, True],
    }

    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]

    test = TestPerspectiveNPoints()
    benchmark(
        test.case_with_gaussian_points,
        "PerspectiveNPoints",
        kwargs_list,
        warmup_iters=1,
    )
