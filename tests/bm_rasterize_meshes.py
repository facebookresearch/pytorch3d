# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from test_rasterize_meshes import TestRasterizeMeshes


# ico levels:
# 0: (12 verts, 20 faces)
# 1: (42 verts, 80 faces)
# 3: (642 verts, 1280 faces)
# 4: (2562 verts, 5120 faces)


def bm_rasterize_meshes() -> None:
    kwargs_list = [
        {
            "num_meshes": 1,
            "ico_level": 0,
            "image_size": 10,  # very slow with large image size
            "blur_radius": 0.0,
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
    image_size = [64, 128]
    blur = [0.0, 1e-8, 1e-4]
    test_cases = product(num_meshes, ico_level, image_size, blur)
    for case in test_cases:
        n, ic, im, b = case
        kwargs_list.append(
            {"num_meshes": n, "ico_level": ic, "image_size": im, "blur_radius": b}
        )
    benchmark(
        TestRasterizeMeshes.rasterize_meshes_cpu_with_init,
        "RASTERIZE_MESHES",
        kwargs_list,
        warmup_iters=1,
    )

    if torch.cuda.is_available():
        kwargs_list = []
        num_meshes = [1, 8]
        ico_level = [0, 1, 3, 4]
        image_size = [64, 128, 512]
        blur = [0.0, 1e-8, 1e-4]
        bin_size = [0, 8, 32]
        test_cases = product(num_meshes, ico_level, image_size, blur, bin_size)
        # only keep cases where bin_size == 0 or image_size / bin_size < 16
        test_cases = [
            elem for elem in test_cases if (elem[-1] == 0 or elem[-3] / elem[-1] < 16)
        ]
        for case in test_cases:
            n, ic, im, b, bn = case
            kwargs_list.append(
                {
                    "num_meshes": n,
                    "ico_level": ic,
                    "image_size": im,
                    "blur_radius": b,
                    "bin_size": bn,
                    "max_faces_per_bin": 200,
                }
            )
        benchmark(
            TestRasterizeMeshes.rasterize_meshes_cuda_with_init,
            "RASTERIZE_MESHES_CUDA",
            kwargs_list,
            warmup_iters=1,
        )
