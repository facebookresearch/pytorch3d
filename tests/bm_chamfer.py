#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import torch
from fvcore.common.benchmark import benchmark

from test_chamfer import TestChamfer


def bm_chamfer() -> None:
    kwargs_list_naive = [
        {"batch_size": 1, "P1": 32, "P2": 64, "return_normals": False},
        {"batch_size": 1, "P1": 32, "P2": 64, "return_normals": True},
        {"batch_size": 32, "P1": 32, "P2": 64, "return_normals": False},
    ]
    benchmark(
        TestChamfer.chamfer_naive_with_init,
        "CHAMFER_NAIVE",
        kwargs_list_naive,
        warmup_iters=1,
    )

    if torch.cuda.is_available():
        kwargs_list = kwargs_list_naive + [
            {"batch_size": 1, "P1": 1000, "P2": 3000, "return_normals": False},
            {"batch_size": 1, "P1": 1000, "P2": 30000, "return_normals": True},
        ]
        benchmark(
            TestChamfer.chamfer_with_init,
            "CHAMFER",
            kwargs_list,
            warmup_iters=1,
        )
