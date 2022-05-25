# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fvcore.common.benchmark import benchmark
from tests.test_se3 import TestSE3


def bm_se3() -> None:
    kwargs_list = [
        {"batch_size": 1},
        {"batch_size": 10},
        {"batch_size": 100},
        {"batch_size": 1000},
    ]
    benchmark(TestSE3.se3_expmap, "SE3_EXP", kwargs_list, warmup_iters=1)
    benchmark(TestSE3.se3_logmap, "SE3_LOG", kwargs_list, warmup_iters=1)


if __name__ == "__main__":
    bm_se3()
