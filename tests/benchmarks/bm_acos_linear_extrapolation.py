# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fvcore.common.benchmark import benchmark
from tests.test_acos_linear_extrapolation import TestAcosLinearExtrapolation


def bm_acos_linear_extrapolation() -> None:
    kwargs_list = [
        {"batch_size": 1},
        {"batch_size": 100},
        {"batch_size": 10000},
        {"batch_size": 1000000},
    ]
    benchmark(
        TestAcosLinearExtrapolation.acos_linear_extrapolation,
        "ACOS_LINEAR_EXTRAPOLATION",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_acos_linear_extrapolation()
