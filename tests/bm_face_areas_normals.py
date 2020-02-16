#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product
import torch
from fvcore.common.benchmark import benchmark

from test_face_areas_normals import TestFaceAreasNormals


def bm_face_areas_normals() -> None:
    kwargs_list = []
    backend_cuda = [False]
    if torch.cuda.is_available():
        backend_cuda.append(True)

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
        TestFaceAreasNormals.face_areas_normals_with_init,
        "FACE_AREAS_NORMALS",
        kwargs_list,
        warmup_iters=1,
    )

    benchmark(
        TestFaceAreasNormals.face_areas_normals_with_init_torch,
        "FACE_AREAS_NORMALS_TORCH",
        kwargs_list,
        warmup_iters=1,
    )
