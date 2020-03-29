# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from test_packed_to_padded import TestPackedToPadded


def bm_packed_to_padded() -> None:
    kwargs_list = []
    backend = ["cpu"]
    if torch.cuda.is_available():
        backend.append("cuda:0")

    num_meshes = [2, 10, 32]
    num_verts = [100, 1000]
    num_faces = [300, 3000]
    num_ds = [0, 1, 16]

    test_cases = product(num_meshes, num_verts, num_faces, num_ds, backend)
    for case in test_cases:
        n, v, f, d, b = case
        kwargs_list.append(
            {"num_meshes": n, "num_verts": v, "num_faces": f, "num_d": d, "device": b}
        )
    benchmark(
        TestPackedToPadded.packed_to_padded_with_init,
        "PACKED_TO_PADDED",
        kwargs_list,
        warmup_iters=1,
    )

    benchmark(
        TestPackedToPadded.packed_to_padded_with_init_torch,
        "PACKED_TO_PADDED_TORCH",
        kwargs_list,
        warmup_iters=1,
    )
