# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fvcore.common.benchmark import benchmark
from test_marching_cubes import TestMarchingCubes


def bm_marching_cubes() -> None:
    kwargs_list = [
        {"batch_size": 1, "V": 5},
        {"batch_size": 1, "V": 10},
        {"batch_size": 1, "V": 20},
        {"batch_size": 1, "V": 40},
        {"batch_size": 5, "V": 5},
        {"batch_size": 20, "V": 20},
    ]
    benchmark(
        TestMarchingCubes.marching_cubes_with_init,
        "MARCHING_CUBES",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_marching_cubes()
