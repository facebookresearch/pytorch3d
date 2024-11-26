# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import cast, Optional, Tuple

import torch
import torch.nn.functional as Fu
from pytorch3d.renderer import (
    AlphaCompositor,
    NDCMultinomialRaysampler,
    PointsRasterizationSettings,
    PointsRasterizer,
    ray_bundle_to_ray_points,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds


def get_rgbd_point_cloud(
    camera: CamerasBase,
    image_rgb: torch.Tensor,
    depth_map: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mask_thr: float = 0.5,
    *,
    euclidean: bool = False,
) -> Pointclouds:
    """
    Given a batch of images, depths, masks and cameras, generate a single colored
    point cloud by unprojecting depth maps and coloring with the source
    pixel colors.

    Arguments:
        camera: Batch of N cameras
        image_rgb: Batch of N images of shape (N, C, H, W).
            For RGB images C=3.
        depth_map: Batch of N depth maps of shape (N, 1, H', W').
            Only positive values here are used to generate points.
            If euclidean=False (default) this contains perpendicular distances
            from each point to the camera plane (z-values).
            If euclidean=True, this contains distances from each point to
            the camera center.
        mask: If provided, batch of N masks of the same shape as depth_map.
            If provided, values in depth_map are ignored if the corresponding
            element of mask is smaller than mask_thr.
        mask_thr: used in interpreting mask
        euclidean: used in interpreting depth_map.

    Returns:
        Pointclouds object containing one point cloud.
    """
    imh, imw = depth_map.shape[2:]

    # convert the depth maps to point clouds using the grid ray sampler
    pts_3d = ray_bundle_to_ray_points(
        NDCMultinomialRaysampler(
            image_width=imw,
            image_height=imh,
            n_pts_per_ray=1,
            min_depth=1.0,
            max_depth=1.0,
            unit_directions=euclidean,
        )(camera)._replace(lengths=depth_map[:, 0, ..., None])
    )

    pts_mask = depth_map > 0.0
    if mask is not None:
        pts_mask *= mask > mask_thr
    pts_mask = pts_mask.reshape(-1)

    pts_3d = pts_3d.reshape(-1, 3)[pts_mask]

    pts_colors = torch.nn.functional.interpolate(
        image_rgb,
        size=[imh, imw],
        mode="bilinear",
        align_corners=False,
    )
    pts_colors = pts_colors.permute(0, 2, 3, 1).reshape(-1, image_rgb.shape[1])[
        pts_mask
    ]

    return Pointclouds(points=pts_3d[None], features=pts_colors[None])


def render_point_cloud_pytorch3d(
    camera,
    point_cloud,
    render_size: Tuple[int, int],
    point_radius: float = 0.03,
    topk: int = 10,
    eps: float = 1e-2,
    bg_color=None,
    bin_size: Optional[int] = None,
    **kwargs,
):
    # feature dimension
    featdim = point_cloud.features_packed().shape[-1]

    # move to the camera coordinates; using identity cameras in the renderer
    point_cloud = _transform_points(camera, point_cloud, eps, **kwargs)
    camera_trivial = camera.clone()
    camera_trivial.R[:] = torch.eye(3)
    camera_trivial.T *= 0.0

    bin_size = (
        bin_size
        if bin_size is not None
        else (64 if int(max(render_size)) > 1024 else None)
    )
    rasterizer = PointsRasterizer(
        cameras=camera_trivial,
        raster_settings=PointsRasterizationSettings(
            image_size=render_size,
            radius=point_radius,
            points_per_pixel=topk,
            bin_size=bin_size,
        ),
    )

    fragments = rasterizer(point_cloud, **kwargs)

    # Construct weights based on the distance of a point to the true point.
    # However, this could be done differently: e.g. predicted as opposed
    # to a function of the weights.
    r = rasterizer.raster_settings.radius

    # set up the blending weights
    dists2 = fragments.dists
    weights = 1 - dists2 / (r * r)
    ok = cast(torch.BoolTensor, (fragments.idx >= 0)).float()

    weights = weights * ok

    fragments_prm = fragments.idx.long().permute(0, 3, 1, 2)
    weights_prm = weights.permute(0, 3, 1, 2)
    images = AlphaCompositor()(
        fragments_prm,
        weights_prm,
        point_cloud.features_packed().permute(1, 0),
        background_color=bg_color if bg_color is not None else [0.0] * featdim,
        **kwargs,
    )

    # get the depths ...
    # weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
    # cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])
    cumprod = torch.cumprod(1 - weights, dim=-1)
    cumprod = torch.cat((torch.ones_like(cumprod[..., :1]), cumprod[..., :-1]), dim=-1)
    depths = (weights * cumprod * fragments.zbuf).sum(dim=-1)
    # add the rendering mask
    render_mask = -torch.prod(1.0 - weights, dim=-1) + 1.0

    # cat depths and render mask
    rendered_blob = torch.cat((images, depths[:, None], render_mask[:, None]), dim=1)

    # reshape back
    rendered_blob = Fu.interpolate(
        rendered_blob,
        size=tuple(render_size),
        mode="bilinear",
        align_corners=False,
    )

    data_rendered, depth_rendered, render_mask = rendered_blob.split(
        [rendered_blob.shape[1] - 2, 1, 1],
        dim=1,
    )

    return data_rendered, render_mask, depth_rendered


def _signed_clamp(x, eps):
    sign = x.sign() + (x == 0.0).type_as(x)
    x_clamp = sign * torch.clamp(x.abs(), eps)
    return x_clamp


def _transform_points(cameras, point_clouds, eps, **kwargs):
    pts_world = point_clouds.points_padded()
    pts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
        pts_world, eps=eps
    )
    # it is crucial to actually clamp the points as well ...
    pts_view = torch.cat(
        (pts_view[..., :-1], _signed_clamp(pts_view[..., -1:], eps)), dim=-1
    )
    point_clouds = point_clouds.update_padded(pts_view)
    return point_clouds
