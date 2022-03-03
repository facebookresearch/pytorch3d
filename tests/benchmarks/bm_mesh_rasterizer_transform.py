# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer
from pytorch3d.utils.ico_sphere import ico_sphere


def rasterize_transform_with_init(num_meshes: int, ico_level: int = 5, device="cuda"):
    # Init meshes
    sphere_meshes = ico_sphere(ico_level, device).extend(num_meshes)
    # Init transform
    R, T = look_at_view_transform(1.0, 0.0, 0.0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    # Init rasterizer
    rasterizer = MeshRasterizer(cameras=cameras)

    torch.cuda.synchronize()

    def raster_fn():
        rasterizer.transform(sphere_meshes)
        torch.cuda.synchronize()

    return raster_fn


def bm_mesh_rasterizer_transform() -> None:
    if torch.cuda.is_available():
        kwargs_list = []
        num_meshes = [1, 8]
        ico_level = [0, 1, 3, 4]
        test_cases = product(num_meshes, ico_level)
        for case in test_cases:
            n, ic = case
            kwargs_list.append({"num_meshes": n, "ico_level": ic})
        benchmark(
            rasterize_transform_with_init,
            "MESH_RASTERIZER",
            kwargs_list,
            warmup_iters=1,
        )


if __name__ == "__main__":
    bm_mesh_rasterizer_transform()
