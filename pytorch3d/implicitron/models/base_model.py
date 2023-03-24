# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pytorch3d.implicitron.models.renderer.base import EvaluationMode
from pytorch3d.implicitron.tools.config import ReplaceableBase
from pytorch3d.renderer.cameras import CamerasBase


@dataclass
class ImplicitronRender:
    """
    Holds the tensors that describe a result of rendering.
    """

    depth_render: Optional[torch.Tensor] = None
    image_render: Optional[torch.Tensor] = None
    mask_render: Optional[torch.Tensor] = None
    camera_distance: Optional[torch.Tensor] = None

    def clone(self) -> "ImplicitronRender":
        def safe_clone(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.detach().clone() if t is not None else None

        return ImplicitronRender(
            depth_render=safe_clone(self.depth_render),
            image_render=safe_clone(self.image_render),
            mask_render=safe_clone(self.mask_render),
            camera_distance=safe_clone(self.camera_distance),
        )


class ImplicitronModelBase(ReplaceableBase, torch.nn.Module):
    """
    Replaceable abstract base for all image generation / rendering models.
    `forward()` method produces a render with a depth map. Derives from Module
    so we can rely on basic functionality provided to torch for model
    optimization.
    """

    # The keys from `preds` (output of ImplicitronModelBase.forward) to be logged in
    # the training loop.
    log_vars: List[str] = field(default_factory=lambda: ["objective"])

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
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Args:
            image_rgb: A tensor of shape `(B, 3, H, W)` containing a batch of rgb images;
                the first `min(B, n_train_target_views)` images are considered targets and
                are used to supervise the renders; the rest corresponding to the source
                viewpoints from which features will be extracted.
            camera: An instance of CamerasBase containing a batch of `B` cameras corresponding
                to the viewpoints of target images, from which the rays will be sampled,
                and source images, which will be used for intersecting with target rays.
            fg_probability: A tensor of shape `(B, 1, H, W)` containing a batch of
                foreground masks.
            mask_crop: A binary tensor of shape `(B, 1, H, W)` deonting valid
                regions in the input images (i.e. regions that do not correspond
                to, e.g., zero-padding). When the `RaySampler`'s sampling mode is set to
                "mask_sample", rays  will be sampled in the non zero regions.
            depth_map: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.
            sequence_name: A list of `B` strings corresponding to the sequence names
                from which images `image_rgb` were extracted. They are used to match
                target frames with relevant source frames.
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering.

        Returns:
            preds: A dictionary containing all outputs of the forward pass. All models should
                output an instance of `ImplicitronRender` in `preds["implicitron_render"]`.
        """
        raise NotImplementedError()
