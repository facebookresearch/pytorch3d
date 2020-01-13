#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product
import torch
from fvcore.common.benchmark import benchmark

from test_sample_points_from_meshes import TestSamplePoints


def bm_sample_points() -> None:
    if torch.cuda.is_available():
        device = "cuda:0"
        kwargs_list = []
        num_meshes = [2, 10, 32]
        num_verts = [100, 1000]
        num_faces = [300, 3000]
        num_samples = [5000, 10000]
        test_cases = product(num_meshes, num_verts, num_faces, num_samples)
        for case in test_cases:
            n, v, f, s = case
            kwargs_list.append(
                {
                    "num_meshes": n,
                    "num_verts": v,
                    "num_faces": f,
                    "num_samples": s,
                    "device": device,
                }
            )
        benchmark(
            TestSamplePoints.sample_points_with_init,
            "SAMPLE_MESH",
            kwargs_list,
            warmup_iters=1,
        )

    kwargs_list = []
    backend_cuda = ["False"]
    if torch.cuda.is_available():
        backend_cuda.append("True")

    num_meshes = [2, 10, 32]
    num_verts = [100, 1000]
    num_faces = [300, 3000]

    test_cases = product(num_meshes, num_verts, num_faces, backend_cuda)
    for case in test_cases:
        n, v, f, c = case
        kwargs_list.append(
            {"num_meshes": n, "num_verts": v, "num_faces": f, "cuda": c}
        )
    benchmark(
        TestSamplePoints.face_areas_with_init,
        "FACE_AREAS",
        kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestSamplePoints.packed_to_padded_with_init,
        "PACKED_TO_PADDED",
        kwargs_list,
        warmup_iters=1,
    )
