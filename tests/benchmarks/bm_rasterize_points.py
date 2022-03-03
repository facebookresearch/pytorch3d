# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from pytorch3d.renderer.points.rasterize_points import (
    rasterize_points,
    rasterize_points_python,
)
from pytorch3d.structures.pointclouds import Pointclouds


def _bm_python_with_init(N, P, img_size=32, radius=0.1, pts_per_pxl=3):
    torch.manual_seed(231)
    points = torch.randn(N, P, 3)
    pointclouds = Pointclouds(points=points)
    args = (pointclouds, img_size, radius, pts_per_pxl)
    return lambda: rasterize_points_python(*args)


def _bm_rasterize_points_with_init(
    N, P, img_size=32, radius=0.1, pts_per_pxl=3, device="cpu", expand_radius=False
):
    torch.manual_seed(231)
    device = torch.device(device)
    points = torch.randn(N, P, 3, device=device)
    pointclouds = Pointclouds(points=points)

    if expand_radius:
        points_padded = pointclouds.points_padded()
        radius = torch.full((N, P), fill_value=radius).type_as(points_padded)

    args = (pointclouds, img_size, radius, pts_per_pxl)
    if device == "cuda":
        torch.cuda.synchronize(device)

    def fn():
        rasterize_points(*args)
        if device == "cuda":
            torch.cuda.synchronize(device)

    return fn


def bm_python_vs_cpu_vs_cuda() -> None:
    kwargs_list = []
    num_meshes = [1]
    num_points = [10000, 2000]
    image_size = [128, 256]
    radius = [1e-3, 0.01]
    pts_per_pxl = [50, 100]
    expand = [True, False]
    test_cases = product(
        num_meshes, num_points, image_size, radius, pts_per_pxl, expand
    )
    for case in test_cases:
        n, p, im, r, pts, e = case
        kwargs_list.append(
            {
                "N": n,
                "P": p,
                "img_size": im,
                "radius": r,
                "pts_per_pxl": pts,
                "device": "cpu",
                "expand_radius": e,
            }
        )

    benchmark(
        _bm_rasterize_points_with_init, "RASTERIZE_CPU", kwargs_list, warmup_iters=1
    )
    kwargs_list += [
        {"N": 32, "P": 100000, "img_size": 128, "radius": 0.01, "pts_per_pxl": 50},
        {"N": 8, "P": 200000, "img_size": 512, "radius": 0.01, "pts_per_pxl": 50},
        {"N": 8, "P": 200000, "img_size": 256, "radius": 0.01, "pts_per_pxl": 50},
        {
            "N": 8,
            "P": 200000,
            "img_size": (512, 256),
            "radius": 0.01,
            "pts_per_pxl": 50,
        },
        {
            "N": 8,
            "P": 200000,
            "img_size": (256, 512),
            "radius": 0.01,
            "pts_per_pxl": 50,
        },
    ]
    for k in kwargs_list:
        k["device"] = "cuda"
    benchmark(
        _bm_rasterize_points_with_init, "RASTERIZE_CUDA", kwargs_list, warmup_iters=1
    )


if __name__ == "__main__":
    bm_python_vs_cpu_vs_cuda()
