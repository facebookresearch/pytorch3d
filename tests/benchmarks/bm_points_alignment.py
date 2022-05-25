# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_points_alignment import TestCorrespondingPointsAlignment, TestICP


def bm_iterative_closest_point() -> None:

    case_grid = {
        "batch_size": [1, 10],
        "dim": [3, 20],
        "n_points_X": [100, 1000],
        "n_points_Y": [100, 1000],
        "use_pointclouds": [False],
    }

    test_args = sorted(case_grid.keys())
    test_cases = product(*case_grid.values())
    kwargs_list = [dict(zip(test_args, case)) for case in test_cases]

    # add the use_pointclouds=True test cases whenever we have dim==3
    kwargs_to_add = []
    for entry in kwargs_list:
        if entry["dim"] == 3:
            entry_add = deepcopy(entry)
            entry_add["use_pointclouds"] = True
            kwargs_to_add.append(entry_add)
    kwargs_list.extend(kwargs_to_add)

    benchmark(
        TestICP.iterative_closest_point,
        "IterativeClosestPoint",
        kwargs_list,
        warmup_iters=1,
    )


def bm_corresponding_points_alignment() -> None:

    case_grid = {
        "allow_reflection": [True, False],
        "batch_size": [1, 10, 100],
        "dim": [3, 20],
        "estimate_scale": [True, False],
        "n_points": [100, 10000],
        "random_weights": [False, True],
        "use_pointclouds": [False],
    }

    test_args = sorted(case_grid.keys())
    test_cases = product(*case_grid.values())
    kwargs_list = [dict(zip(test_args, case)) for case in test_cases]

    # add the use_pointclouds=True test cases whenever we have dim==3
    kwargs_to_add = []
    for entry in kwargs_list:
        if entry["dim"] == 3:
            entry_add = deepcopy(entry)
            entry_add["use_pointclouds"] = True
            kwargs_to_add.append(entry_add)
    kwargs_list.extend(kwargs_to_add)

    benchmark(
        TestCorrespondingPointsAlignment.corresponding_points_alignment,
        "CorrespodingPointsAlignment",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_corresponding_points_alignment()
    bm_iterative_closest_point()
