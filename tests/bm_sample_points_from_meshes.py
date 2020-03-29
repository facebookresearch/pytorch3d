# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from test_sample_points_from_meshes import TestSamplePoints


def bm_sample_points() -> None:

    backend = ["cpu"]
    if torch.cuda.is_available():
        backend.append("cuda:0")
    kwargs_list = []
    num_meshes = [2, 10, 32]
    num_verts = [100, 1000]
    num_faces = [300, 3000]
    num_samples = [5000, 10000]
    test_cases = product(num_meshes, num_verts, num_faces, num_samples, backend)
    for case in test_cases:
        n, v, f, s, b = case
        kwargs_list.append(
            {
                "num_meshes": n,
                "num_verts": v,
                "num_faces": f,
                "num_samples": s,
                "device": b,
            }
        )
    benchmark(
        TestSamplePoints.sample_points_with_init,
        "SAMPLE_MESH",
        kwargs_list,
        warmup_iters=1,
    )
