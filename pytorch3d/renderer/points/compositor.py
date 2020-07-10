# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from ..compositing import alpha_composite, norm_weighted_sum


# A compositor should take as input 3D points and some corresponding information.
# Given this information, the compositor can:
#     - blend colors across the top K vertices at a pixel


class AlphaCompositor(nn.Module):
    """
    Accumulate points using alpha compositing.
    """

    def __init__(self):
        super().__init__()

    def forward(self, fragments, alphas, ptclds, **kwargs) -> torch.Tensor:
        images = alpha_composite(fragments, alphas, ptclds)
        return images


class NormWeightedCompositor(nn.Module):
    """
    Accumulate points using a normalized weighted sum.
    """

    def __init__(self):
        super().__init__()

    def forward(self, fragments, alphas, ptclds, **kwargs) -> torch.Tensor:
        images = norm_weighted_sum(fragments, alphas, ptclds)
        return images
