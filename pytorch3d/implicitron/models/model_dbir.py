# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch3d.implicitron.dataset.utils import is_known_frame
from pytorch3d.implicitron.tools.config import registry
from pytorch3d.implicitron.tools.point_cloud_utils import (
    get_rgbd_point_cloud,
    render_point_cloud_pytorch3d,
)
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds

from .base_model import ImplicitronModelBase, ImplicitronRender
from .renderer.base import EvaluationMode


@registry.register
class ModelDBIR(ImplicitronModelBase):
    """
    A simple depth-based image rendering model.

    Args:
        render_image_width: The width of the rendered rectangular images.
        render_image_height: The height of the rendered rectangular images.
        bg_color: The color of the background.
        max_points: Maximum number of points in the point cloud
            formed by unprojecting all source view depths.
            If more points are present, they are randomly subsampled
            to this number of points without replacement.
    """

    render_image_width: int = 256
    render_image_height: int = 256
    bg_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max_points: int = -1

    # pyre-fixme[14]: `forward` overrides method defined in `ImplicitronModelBase`
    #  inconsistently.
    def forward(
        self,
        *,  # force keyword-only arguments
        image_rgb: Optional[torch.Tensor],
        camera: CamerasBase,
        fg_probability: Optional[torch.Tensor],
        mask_crop: Optional[torch.Tensor],
        depth_map: Optional[torch.Tensor],
        sequence_name: Optional[List[str]],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
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
                implicitron_render: The rendered colors, depth and mask
                    of the target views.
                point_cloud: The point cloud of the scene. It's renders are
                    stored in `implicitron_render`.
        """

        if image_rgb is None:
            raise ValueError("ModelDBIR needs image input")

        if fg_probability is None:
            raise ValueError("ModelDBIR needs foreground mask input")

        if depth_map is None:
            raise ValueError("ModelDBIR needs depth map input")

        is_known = is_known_frame(frame_type)
        is_known_idx = torch.where(is_known)[0]

        mask_fg = (fg_probability > 0.5).type_as(image_rgb)

        point_cloud = get_rgbd_point_cloud(
            # pyre-fixme[6]: For 1st param expected `Union[List[int], int,
            #  LongTensor]` but got `Tensor`.
            camera[is_known_idx],
            image_rgb[is_known_idx],
            depth_map[is_known_idx],
            mask_fg[is_known_idx],
        )

        pcl_size = point_cloud.num_points_per_cloud().item()
        if (self.max_points > 0) and (pcl_size > self.max_points):
            # pyre-fixme[6]: For 1st param expected `int` but got `Union[bool,
            #  float, int]`.
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
                render_size=(self.render_image_height, self.render_image_width),
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

        implicitron_render = ImplicitronRender(
            **{
                k: torch.cat(v, dim=0)
                for k, v in zip(
                    ["depth_render", "image_render", "mask_render"],
                    [depth_render, image_render, mask_render],
                )
            }
        )

        preds = {
            "implicitron_render": implicitron_render,
            "point_cloud": point_cloud,
        }

        return preds
