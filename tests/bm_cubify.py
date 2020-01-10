#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from fvcore.common.benchmark import benchmark

from test_cubify import TestCubify


def bm_cubify() -> None:
    kwargs_list = [
        {"batch_size": 32, "V": 16},
        {"batch_size": 64, "V": 16},
        {"batch_size": 16, "V": 32},
    ]
    benchmark(
        TestCubify.cubify_with_init, "CUBIFY", kwargs_list, warmup_iters=1
    )
