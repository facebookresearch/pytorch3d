# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_sample_farthest_points import TestFPS


def bm_fps() -> None:
    kwargs_list = []
    backends = ["cpu", "cuda:0"]
    Ns = [8, 32]
    Ps = [64, 256]
    Ds = [3]
    Ks = [24]
    test_cases = product(Ns, Ps, Ds, Ks, backends)
    for case in test_cases:
        N, P, D, K, d = case
        kwargs_list.append({"N": N, "P": P, "D": D, "K": K, "device": d})

    benchmark(
        TestFPS.sample_farthest_points_naive,
        "FPS_NAIVE_PYTHON",
        kwargs_list,
        warmup_iters=1,
    )

    # Add some larger batch sizes and pointcloud sizes
    Ns = [32]
    Ps = [2048, 8192, 18384]
    Ds = [3, 9]
    Ks = [24, 48]
    test_cases = product(Ns, Ps, Ds, Ks, backends)
    for case in test_cases:
        N, P, D, K, d = case
        kwargs_list.append({"N": N, "P": P, "D": D, "K": K, "device": d})

    benchmark(TestFPS.sample_farthest_points, "FPS", kwargs_list, warmup_iters=1)


if __name__ == "__main__":
    bm_fps()
