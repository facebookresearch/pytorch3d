# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

from fvcore.common.benchmark import benchmark
from tests.test_perspective_n_points import TestPerspectiveNPoints


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


if __name__ == "__main__":
    bm_perspective_n_points()
