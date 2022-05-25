# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from tests.test_face_areas_normals import TestFaceAreasNormals


def bm_face_areas_normals() -> None:
    kwargs_list = []
    backend = ["cpu"]
    if torch.cuda.is_available():
        backend.append("cuda:0")

    num_meshes = [2, 10, 32]
    num_verts = [100, 1000]
    num_faces = [300, 3000]

    test_cases = product(num_meshes, num_verts, num_faces, backend)
    for case in test_cases:
        n, v, f, d = case
        kwargs_list.append(
            {"num_meshes": n, "num_verts": v, "num_faces": f, "device": d}
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


if __name__ == "__main__":
    bm_face_areas_normals()
