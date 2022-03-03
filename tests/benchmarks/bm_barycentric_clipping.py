# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.mesh.rasterizer import (
    Fragments,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.renderer.mesh.utils import (
    _clip_barycentric_coordinates,
    _interpolate_zbuf,
)
from pytorch3d.utils.ico_sphere import ico_sphere


def baryclip_cuda(
    num_meshes: int = 8,
    ico_level: int = 5,
    image_size: int = 64,
    faces_per_pixel: int = 50,
    device="cuda",
):
    # Init meshes
    sphere_meshes = ico_sphere(ico_level, device).extend(num_meshes)
    # Init transform
    R, T = look_at_view_transform(1.0, 0.0, 0.0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    # Init rasterizer
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-4,
        faces_per_pixel=faces_per_pixel,
        clip_barycentric_coords=True,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    torch.cuda.synchronize()

    def raster_fn():
        rasterizer(sphere_meshes)
        torch.cuda.synchronize()

    return raster_fn


def baryclip_pytorch(
    num_meshes: int = 8,
    ico_level: int = 5,
    image_size: int = 64,
    faces_per_pixel: int = 50,
    device="cuda",
):
    # Init meshes
    sphere_meshes = ico_sphere(ico_level, device).extend(num_meshes)
    # Init transform
    R, T = look_at_view_transform(1.0, 0.0, 0.0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    # Init rasterizer
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-4,
        faces_per_pixel=faces_per_pixel,
        clip_barycentric_coords=False,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    torch.cuda.synchronize()

    def raster_fn():
        fragments = rasterizer(sphere_meshes)

        # Clip bary and reinterpolate
        clipped_bary_coords = _clip_barycentric_coordinates(fragments.bary_coords)
        clipped_zbuf = _interpolate_zbuf(
            fragments.pix_to_face, clipped_bary_coords, sphere_meshes
        )
        fragments = Fragments(
            bary_coords=clipped_bary_coords,
            zbuf=clipped_zbuf,
            dists=fragments.dists,
            pix_to_face=fragments.pix_to_face,
        )
        torch.cuda.synchronize()

    return raster_fn


def bm_barycentric_clip() -> None:
    if torch.cuda.is_available():
        kwargs_list = []
        num_meshes = [1, 8]
        ico_level = [0, 4]
        image_size = [64, 128, 256]
        faces_per_pixel = [10, 75, 100]
        test_cases = product(num_meshes, ico_level, image_size, faces_per_pixel)
        for case in test_cases:
            n, ic, im, nf = case
            kwargs_list.append(
                {
                    "num_meshes": n,
                    "ico_level": ic,
                    "image_size": im,
                    "faces_per_pixel": nf,
                }
            )

        benchmark(baryclip_cuda, "BARY_CLIP_CUDA", kwargs_list, warmup_iters=1)
        benchmark(baryclip_pytorch, "BARY_CLIP_PYTORCH", kwargs_list, warmup_iters=1)


if __name__ == "__main__":
    bm_barycentric_clip()
