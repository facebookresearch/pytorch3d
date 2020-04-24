# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from test_chamfer import TestChamfer


def bm_chamfer() -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")

    kwargs_list_naive = []
    batch_size = [1, 32]
    return_normals = [True, False]
    test_cases = product(batch_size, return_normals, devices)

    for case in test_cases:
        b, n, d = case
        kwargs_list_naive.append(
            {"batch_size": b, "P1": 32, "P2": 64, "return_normals": n, "device": d}
        )

    benchmark(
        TestChamfer.chamfer_naive_with_init,
        "CHAMFER_NAIVE",
        kwargs_list_naive,
        warmup_iters=1,
    )

    if torch.cuda.is_available():
        device = "cuda:0"
        kwargs_list = []
        batch_size = [1, 32]
        P1 = [32, 1000, 10000]
        P2 = [64, 3000, 30000]
        return_normals = [True, False]
        homogeneous = [True, False]
        test_cases = product(batch_size, P1, P2, return_normals, homogeneous)

        for case in test_cases:
            b, p1, p2, n, h = case
            kwargs_list.append(
                {
                    "batch_size": b,
                    "P1": p1,
                    "P2": p2,
                    "return_normals": n,
                    "homogeneous": h,
                    "device": device,
                }
            )
        benchmark(TestChamfer.chamfer_with_init, "CHAMFER", kwargs_list, warmup_iters=1)
