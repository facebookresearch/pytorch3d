# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from tests.test_mesh_laplacian_smoothing import TestLaplacianSmoothing


def bm_mesh_laplacian_smoothing() -> None:
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
        TestLaplacianSmoothing.laplacian_smoothing_with_init,
        "MESH_LAPLACIAN_SMOOTHING",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_mesh_laplacian_smoothing()
