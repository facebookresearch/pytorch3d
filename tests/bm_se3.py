# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from fvcore.common.benchmark import benchmark
from test_se3 import TestSE3


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
