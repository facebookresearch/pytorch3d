# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, Dict, List

import torch
from pytorch3d.implicitron.dataset.utils import is_known_frame
from pytorch3d.implicitron.evaluation.evaluate_new_view_synthesis import (
    NewViewSynthesisPrediction,
)
from pytorch3d.implicitron.tools.point_cloud_utils import (
    get_rgbd_point_cloud,
    render_point_cloud_pytorch3d,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds


class ModelDBIR(torch.nn.Module):
    """
    A simple depth-based image rendering model.
    """

    def __init__(
        self,
        image_size: int = 256,
        bg_color: float = 0.0,
        max_points: int = -1,
    ):
        """
        Initializes a simple DBIR model.

        Args:
            image_size: The size of the rendered rectangular images.
            bg_color: The color of the background.
            max_points: Maximum number of points in the point cloud
                formed by unprojecting all source view depths.
                If more points are present, they are randomly subsampled
                to #max_size points without replacement.
        """

        super().__init__()
        self.image_size = image_size
        self.bg_color = bg_color
        self.max_points = max_points

    def forward(
        self,
        camera: CamerasBase,
        image_rgb: torch.Tensor,
        depth_map: torch.Tensor,
        fg_probability: torch.Tensor,
        frame_type: List[str],
        **kwargs,
    ) -> Dict[str, Any]:  # TODO: return a namedtuple or dataclass
        """
        Given a set of input source cameras images and depth maps, unprojects
        all RGBD maps to a colored point cloud and renders into the target views.

        Args:
            camera: A batch of `N` PyTorch3D cameras.
            image_rgb: A batch of `N` images of shape `(N, 3, H, W)`.
            depth_map: A batch of `N` depth maps of shape `(N, 1, H, W)`.
            fg_probability: A batch of `N` foreground probability maps
                of shape `(N, 1, H, W)`.
            frame_type: A list of `N` strings containing frame type indicators
                which specify target and source views.

        Returns:
            preds: A dict with the following fields:
                nvs_prediction: The rendered colors, depth and mask
                    of the target views.
                point_cloud: The point cloud of the scene. It's renders are
                    stored in `nvs_prediction`.
        """

        is_known = is_known_frame(frame_type)
        is_known_idx = torch.where(is_known)[0]

        mask_fg = (fg_probability > 0.5).type_as(image_rgb)

        point_cloud = get_rgbd_point_cloud(
            camera[is_known_idx],
            image_rgb[is_known_idx],
            depth_map[is_known_idx],
            mask_fg[is_known_idx],
        )

        pcl_size = int(point_cloud.num_points_per_cloud())
        if (self.max_points > 0) and (pcl_size > self.max_points):
            prm = torch.randperm(pcl_size)[: self.max_points]
            point_cloud = Pointclouds(
                point_cloud.points_padded()[:, prm, :],
                # pyre-fixme[16]: Optional type has no attribute `__getitem__`.
                features=point_cloud.features_padded()[:, prm, :],
            )

        is_target_idx = torch.where(~is_known)[0]

        depth_render, image_render, mask_render = [], [], []

        # render into target frames in a for loop to save memory
        for tgt_idx in is_target_idx:
            _image_render, _mask_render, _depth_render = render_point_cloud_pytorch3d(
                camera[int(tgt_idx)],
                point_cloud,
                render_size=(self.image_size, self.image_size),
                point_radius=1e-2,
                topk=10,
                bg_color=self.bg_color,
            )
            _image_render = _image_render.clamp(0.0, 1.0)
            # the mask is the set of pixels with opacity bigger than eps
            _mask_render = (_mask_render > 1e-4).float()

            depth_render.append(_depth_render)
            image_render.append(_image_render)
            mask_render.append(_mask_render)

        nvs_prediction = NewViewSynthesisPrediction(
            **{
                k: torch.cat(v, dim=0)
                for k, v in zip(
                    ["depth_render", "image_render", "mask_render"],
                    [depth_render, image_render, mask_render],
                )
            }
        )

        preds = {
            "nvs_prediction": nvs_prediction,
            "point_cloud": point_cloud,
        }

        return preds
