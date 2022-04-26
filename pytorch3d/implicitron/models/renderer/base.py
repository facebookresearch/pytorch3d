# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import torch
from pytorch3d.implicitron.tools.config import ReplaceableBase


class EvaluationMode(Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class RenderSamplingMode(Enum):
    MASK_SAMPLE = "mask_sample"
    FULL_GRID = "full_grid"


@dataclass
class RendererOutput:
    """
    A structure for storing the output of a renderer.

    Args:
        features: rendered features (usually RGB colors), (B, ..., C) tensor.
        depth: rendered ray-termination depth map, in NDC coordinates, (B, ..., 1) tensor.
        mask: rendered object mask, values in [0, 1], (B, ..., 1) tensor.
        prev_stage: for multi-pass renderers (e.g. in NeRF),
            a reference to the output of the previous stage.
        normals: surface normals, for renderers that estimate them; (B, ..., 3) tensor.
        points: ray-termination points in the world coordinates, (B, ..., 3) tensor.
        aux: dict for implementation-specific renderer outputs.
    """

    features: torch.Tensor
    depths: torch.Tensor
    masks: torch.Tensor
    prev_stage: Optional[RendererOutput] = None
    normals: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None  # TODO: redundant with depths
    aux: Dict[str, Any] = field(default_factory=lambda: {})


class ImplicitFunctionWrapper(torch.nn.Module):
    def __init__(self, fn: torch.nn.Module):
        super().__init__()
        self._fn = fn
        self.bound_args = {}

    def bind_args(self, **bound_args):
        self.bound_args = bound_args
        self._fn.on_bind_args()

    def unbind_args(self):
        self.bound_args = {}

    def forward(self, *args, **kwargs):
        return self._fn(*args, **{**kwargs, **self.bound_args})


class BaseRenderer(ABC, ReplaceableBase):
    """
    Base class for all Renderer implementations.
    """

    def __init__(self) -> None:
        super().__init__()

    def requires_object_mask(self) -> bool:
        """
        Whether `forward` needs the object_mask.
        """
        return False

    @abstractmethod
    def forward(
        self,
        ray_bundle,
        implicit_functions: List[ImplicitFunctionWrapper],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs
    ) -> RendererOutput:
        """
        Each Renderer should implement its own forward function
        that returns an instance of RendererOutput.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape (minibatch, ..., 3) denoting
                    the origins of the rendering rays.
                directions: A tensor of shape (minibatch, ..., 3)
                    containing the direction vectors of rendering rays.
                lengths: A tensor of shape
                    (minibatch, ..., num_points_per_ray)containing the
                    lengths at which the ray points are sampled.
                    The coordinates of the points on the rays are thus computed
                    as `origins + lengths * directions`.
                xys: A tensor of shape
                    (minibatch, ..., 2) containing the
                    xy locations of each ray's pixel in the NDC screen space.
            implicit_functions: List of ImplicitFunctionWrappers which define the
                implicit function methods to be used. Most Renderers only allow
                a single implicit function. Currently, only the MultiPassEARenderer
                allows specifying mulitple values in the list.
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering.
            **kwargs: In addition to the name args, custom keyword args can be specified.
                For example in the SignedDistanceFunctionRenderer, an object_mask is
                required which needs to be passed via the kwargs.

        Returns:
            instance of RendererOutput
        """
        pass
