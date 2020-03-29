# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product

from fvcore.common.benchmark import benchmark
from test_blending import TestBlending


def bm_blending() -> None:
    devices = ["cpu", "cuda"]
    kwargs_list = []
    num_meshes = [16]
    image_size = [128, 256]
    faces_per_pixel = [50, 100]
    test_cases = product(num_meshes, image_size, faces_per_pixel, devices)

    for case in test_cases:
        n, s, k, d = case
        kwargs_list.append(
            {"num_meshes": n, "image_size": s, "faces_per_pixel": k, "device": d}
        )

    benchmark(
        TestBlending.bm_sigmoid_alpha_blending,
        "SIGMOID_ALPHA_BLENDING_PYTORCH",
        kwargs_list,
        warmup_iters=1,
    )

    benchmark(
        TestBlending.bm_softmax_blending,
        "SOFTMAX_BLENDING_PYTORCH",
        kwargs_list,
        warmup_iters=1,
    )
