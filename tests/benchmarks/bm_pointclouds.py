# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_pointclouds import TestPointclouds


def bm_compute_packed_padded_pointclouds() -> None:
    kwargs_list = []
    num_clouds = [32, 128]
    max_p = [100, 10000]
    feats = [1, 10, 300]
    test_cases = product(num_clouds, max_p, feats)
    for case in test_cases:
        n, p, f = case
        kwargs_list.append({"num_clouds": n, "max_p": p, "features": f})
    benchmark(
        TestPointclouds.compute_packed_with_init,
        "COMPUTE_PACKED",
        kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestPointclouds.compute_padded_with_init,
        "COMPUTE_PADDED",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_compute_packed_padded_pointclouds()
