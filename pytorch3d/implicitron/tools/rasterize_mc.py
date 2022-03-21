# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Pointclouds

from .point_cloud_utils import render_point_cloud_pytorch3d


def rasterize_mc_samples(
    xys: torch.Tensor,
    feats: torch.Tensor,
    image_size_hw: Tuple[int, int],
    radius: float = 0.03,
    topk: int = 5,
    masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rasterizes Monte-Carlo sampled features back onto the image.

    Specifically, the code uses the PyTorch3D point rasterizer to render
    a z-flat point cloud composed of the xy MC locations and their features.

    Args:
        xys: B x N x 2 2D point locations in PyTorch3D NDC convention
        feats: B x N x dim tensor containing per-point rendered features.
        image_size_hw: Tuple[image_height, image_width] containing
            the size of rasterized image.
        radius: Rasterization point radius.
        topk: The maximum z-buffer size for the PyTorch3D point cloud rasterizer.
        masks: B x N x 1 tensor containing the alpha mask of the
            rendered features.
    """

    if masks is None:
        masks = torch.ones_like(xys[..., :1])

    feats = torch.cat((feats, masks), dim=-1)
    pointclouds = Pointclouds(
        points=torch.cat([xys, torch.ones_like(xys[..., :1])], dim=-1),
        features=feats,
    )

    data_rendered, render_mask, _ = render_point_cloud_pytorch3d(
        PerspectiveCameras(device=feats.device),
        pointclouds,
        render_size=image_size_hw,
        point_radius=radius,
        topk=topk,
    )

    data_rendered, masks_pt = data_rendered.split(
        [data_rendered.shape[1] - 1, 1], dim=1
    )
    render_mask = masks_pt * render_mask

    return data_rendered, render_mask
