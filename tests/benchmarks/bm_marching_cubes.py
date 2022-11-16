# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from fvcore.common.benchmark import benchmark
from tests.test_marching_cubes import TestMarchingCubes


def bm_marching_cubes() -> None:
    case_grid = {
        "algo_type": [
            "naive",
            "extension",
        ],
        "batch_size": [1, 2],
        "V": [5, 10, 20, 100, 512],
        "device": ["cpu", "cuda:0"],
    }
    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]

    benchmark(
        TestMarchingCubes.marching_cubes_with_init,
        "MARCHING_CUBES",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_marching_cubes()
