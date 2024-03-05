# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Any, Dict, Optional

import torch
from pytorch3d.implicitron.tools.config import ReplaceableBase


class FeatureExtractorBase(ReplaceableBase, torch.nn.Module):
    """
    Base class for an extractor of a set of features from images.
    """

    def get_feat_dims(self) -> int:
        """
        Returns:
            total number of feature dimensions of the output.
            (i.e. sum_i(dim_i))
        """
        raise NotImplementedError

    def forward(
        self,
        imgs: Optional[torch.Tensor],
        masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[Any, torch.Tensor]:
        """
        Args:
            imgs: A batch of input images of shape `(B, 3, H, W)`.
            masks: A batch of input masks of shape `(B, 3, H, W)`.

        Returns:
            out_feats: A dict `{f_i: t_i}` keyed by predicted feature names `f_i`
                and their corresponding tensors `t_i` of shape `(B, dim_i, H_i, W_i)`.
        """
        raise NotImplementedError
