# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


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


def _bm_cpu_with_init(N, P, img_size=32, radius=0.1, pts_per_pxl=3):
    torch.manual_seed(231)
    points = torch.randn(N, P, 3)
    pointclouds = Pointclouds(points=points)
    args = (pointclouds, img_size, radius, pts_per_pxl)
    return lambda: rasterize_points(*args)


def _bm_cuda_with_init(N, P, img_size=32, radius=0.1, pts_per_pxl=3):
    torch.manual_seed(231)
    device = torch.device("cuda:0")
    points = torch.randn(N, P, 3, device=device)
    pointclouds = Pointclouds(points=points)
    args = (pointclouds, img_size, radius, pts_per_pxl)
    torch.cuda.synchronize(device)

    def fn():
        rasterize_points(*args)
        torch.cuda.synchronize(device)

    return fn


def bm_python_vs_cpu() -> None:
    kwargs_list = [
        {"N": 1, "P": 32, "img_size": 32, "radius": 0.1, "pts_per_pxl": 3},
        {"N": 2, "P": 32, "img_size": 32, "radius": 0.1, "pts_per_pxl": 3},
    ]
    benchmark(_bm_python_with_init, "RASTERIZE_PYTHON", kwargs_list, warmup_iters=1)
    benchmark(_bm_cpu_with_init, "RASTERIZE_CPU", kwargs_list, warmup_iters=1)
    kwargs_list = [
        {"N": 2, "P": 32, "img_size": 32, "radius": 0.1, "pts_per_pxl": 3},
        {"N": 4, "P": 1024, "img_size": 128, "radius": 0.05, "pts_per_pxl": 5},
    ]
    benchmark(_bm_cpu_with_init, "RASTERIZE_CPU", kwargs_list, warmup_iters=1)
    kwargs_list += [
        {"N": 32, "P": 10000, "img_size": 128, "radius": 0.01, "pts_per_pxl": 50},
        {"N": 32, "P": 100000, "img_size": 128, "radius": 0.01, "pts_per_pxl": 50},
        {"N": 8, "P": 200000, "img_size": 512, "radius": 0.01, "pts_per_pxl": 50},
    ]
    benchmark(_bm_cuda_with_init, "RASTERIZE_CUDA", kwargs_list, warmup_iters=1)
