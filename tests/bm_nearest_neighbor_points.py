# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from itertools import product
import torch
from fvcore.common.benchmark import benchmark

from test_nearest_neighbor_points import TestNearestNeighborPoints


def bm_nn_points() -> None:
    kwargs_list = []

    N = [1, 4, 32]
    D = [3, 4]
    P1 = [1, 128]
    P2 = [32, 128]
    test_cases = product(N, D, P1, P2)
    for case in test_cases:
        n, d, p1, p2 = case
        kwargs_list.append({"N": n, "D": d, "P1": p1, "P2": p2})

    benchmark(
        TestNearestNeighborPoints.bm_nn_points_python_with_init,
        "NN_PYTHON",
        kwargs_list,
        warmup_iters=1,
    )

    benchmark(
        TestNearestNeighborPoints.bm_nn_points_cpu_with_init,
        "NN_CPU",
        kwargs_list,
        warmup_iters=1,
    )

    if torch.cuda.is_available():
        benchmark(
            TestNearestNeighborPoints.bm_nn_points_cuda_with_init,
            "NN_CUDA",
            kwargs_list,
            warmup_iters=1,
        )
