# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from test_vert_align import TestVertAlign


def bm_vert_align() -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    kwargs_list = []
    num_meshes = [2, 10, 32]
    num_verts = [100, 1000]
    num_faces = [300, 3000]
    test_cases = product(num_meshes, num_verts, num_faces, devices)
    for case in test_cases:
        n, v, f, d = case
        kwargs_list.append(
            {"num_meshes": n, "num_verts": v, "num_faces": f, "device": d}
        )

    benchmark(
        TestVertAlign.vert_align_with_init, "VERT_ALIGN", kwargs_list, warmup_iters=1
    )
