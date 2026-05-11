# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch

from pytorch3d import _C

from .compositor import AlphaCompositor, NormWeightedCompositor
from .rasterize_points import rasterize_points
from .rasterizer import PointsRasterizationSettings, PointsRasterizer
from .renderer import PointsRenderer


# PulsarPointsRenderer wraps the native pulsar renderer, which is not yet
# available on ROCm builds. Only expose the wrapper when the native class is
# present so that ``import pytorch3d.renderer`` succeeds on AMD GPUs.
if hasattr(_C, "PulsarRenderer"):
    from .pulsar.unified import PulsarPointsRenderer  # noqa: F401


__all__ = [k for k in globals().keys() if not k.startswith("_")]
