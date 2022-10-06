# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from tests.test_rasterize_meshes import TestRasterizeMeshes

BM_RASTERIZE_MESHES_N_THREADS = os.getenv("BM_RASTERIZE_MESHES_N_THREADS", 1)
torch.set_num_threads(int(BM_RASTERIZE_MESHES_N_THREADS))

# ico levels:
# 0: (12 verts, 20 faces)
# 1: (42 verts, 80 faces)
# 3: (642 verts, 1280 faces)
# 4: (2562 verts, 5120 faces)
# 5: (10242 verts, 20480 faces)
# 6: (40962 verts, 81920 faces)


def bm_rasterize_meshes() -> None:
    kwargs_list = [
        {
            "num_meshes": 1,
            "ico_level": 0,
            "image_size": 10,  # very slow with large image size
            "blur_radius": 0.0,
            "faces_per_pixel": 3,
        }
    ]
    benchmark(
        TestRasterizeMeshes.rasterize_meshes_python_with_init,
        "RASTERIZE_MESHES",
        kwargs_list,
        warmup_iters=1,
    )

    kwargs_list = []
    num_meshes = [1]
    ico_level = [1]
    image_size = [64, 128, 512]
    blur = [1e-6]
    faces_per_pixel = [3, 50]
    test_cases = product(num_meshes, ico_level, image_size, blur, faces_per_pixel)
    for case in test_cases:
        n, ic, im, b, f = case
        kwargs_list.append(
            {
                "num_meshes": n,
                "ico_level": ic,
                "image_size": im,
                "blur_radius": b,
                "faces_per_pixel": f,
            }
        )
    benchmark(
        TestRasterizeMeshes.rasterize_meshes_cpu_with_init,
        "RASTERIZE_MESHES",
        kwargs_list,
        warmup_iters=1,
    )

    if torch.cuda.is_available():
        kwargs_list = []
        num_meshes = [8, 16]
        ico_level = [4, 5, 6]
        # Square and non square cases
        image_size = [64, 128, 512, (512, 256), (256, 512)]
        blur = [1e-6]
        faces_per_pixel = [40]
        test_cases = product(num_meshes, ico_level, image_size, blur, faces_per_pixel)

        for case in test_cases:
            n, ic, im, b, f = case
            kwargs_list.append(
                {
                    "num_meshes": n,
                    "ico_level": ic,
                    "image_size": im,
                    "blur_radius": b,
                    "faces_per_pixel": f,
                }
            )
        benchmark(
            TestRasterizeMeshes.rasterize_meshes_cuda_with_init,
            "RASTERIZE_MESHES_CUDA",
            kwargs_list,
            warmup_iters=1,
        )

        # Test a subset of the cases with the
        # image plane intersecting the mesh.
        kwargs_list = []
        num_meshes = [8, 16]
        # Square and non square cases
        image_size = [64, 128, 512, (512, 256), (256, 512)]
        dist = [3, 0.8, 0.5]
        test_cases = product(num_meshes, dist, image_size)

        for case in test_cases:
            n, d, im = case
            kwargs_list.append(
                {
                    "num_meshes": n,
                    "ico_level": 4,
                    "image_size": im,
                    "blur_radius": 1e-6,
                    "faces_per_pixel": 40,
                    "dist": d,
                }
            )

        benchmark(
            TestRasterizeMeshes.bm_rasterize_meshes_with_clipping,
            "RASTERIZE_MESHES_CUDA_CLIPPING",
            kwargs_list,
            warmup_iters=1,
        )


if __name__ == "__main__":
    bm_rasterize_meshes()
