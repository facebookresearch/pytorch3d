# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch3d.implicitron.tools.config import ReplaceableBase
from pytorch3d.ops import packed_to_padded


class EvaluationMode(Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class RenderSamplingMode(Enum):
    MASK_SAMPLE = "mask_sample"
    FULL_GRID = "full_grid"


@dataclasses.dataclass
class ImplicitronRayBundle:
    """
    Parametrizes points along projection rays by storing ray `origins`,
    `directions` vectors and `lengths` at which the ray-points are sampled.
    Furthermore, the xy-locations (`xys`) of the ray pixels are stored as well.
    Note that `directions` don't have to be normalized; they define unit vectors
    in the respective 1D coordinate systems; see documentation for
    :func:`ray_bundle_to_ray_points` for the conversion formula.

    Ray bundle may represent rays from multiple cameras. In that case, cameras
    are stored in the packed form (i.e. rays from the same camera are stored in
    the consecutive elements). The following indices will be set:
        camera_ids: A tensor of shape (N, ) which indicates which camera
            was used to sample the rays. `N` is the number of different
            sampled cameras.
        camera_counts: A tensor of shape (N, ) which how many times the
            coresponding camera in `camera_ids` was sampled.
            `sum(camera_counts) == minibatch`, where `minibatch = origins.shape[0]`.
    """

    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor
    camera_ids: Optional[torch.LongTensor] = None
    camera_counts: Optional[torch.LongTensor] = None

    def is_packed(self) -> bool:
        """
        Returns whether the ImplicitronRayBundle carries data in packed state
        """
        return self.camera_ids is not None and self.camera_counts is not None

    def get_padded_xys(self) -> Tuple[torch.Tensor, torch.LongTensor, int]:
        """
        For a packed ray bundle, returns padded rays. Assumes the input bundle is packed
        (i.e. `camera_ids` and `camera_counts` are set).

        Returns:
            - xys: Tensor of shape (N, max_size, ...) containing the padded
                representation of the pixel coordinated;
                where max_size is max of `camera_counts`. The values for camera id `i`
                will be copied to `xys[i, :]`, with zeros padding out the extra inputs.
            - first_idxs: cumulative sum of `camera_counts` defininf the boundaries
                between cameras in the packed representation
            - num_inputs: the number of cameras in the bundle.
        """
        if not self.is_packed():
            raise ValueError("get_padded_xys can be called only on a packed bundle")

        camera_counts = self.camera_counts
        assert camera_counts is not None

        cumsum = torch.cumsum(camera_counts, dim=0, dtype=torch.long)
        first_idxs = torch.cat(
            (camera_counts.new_zeros((1,), dtype=torch.long), cumsum[:-1])
        )
        num_inputs = camera_counts.sum().item()
        max_size = torch.max(camera_counts).item()
        xys = packed_to_padded(self.xys, first_idxs, max_size)
        # pyre-ignore [7] pytorch typeshed inaccuracy
        return xys, first_idxs, num_inputs


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
    weights: Optional[torch.Tensor] = None
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
        ray_bundle: ImplicitronRayBundle,
        implicit_functions: List[ImplicitFunctionWrapper],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs,
    ) -> RendererOutput:
        """
        Each Renderer should implement its own forward function
        that returns an instance of RendererOutput.

        Args:
            ray_bundle: An ImplicitronRayBundle object containing the following variables:
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
                camera_ids: A tensor of shape (N, ) which indicates which camera
                    was used to sample the rays. `N` is the number of different
                    sampled cameras.
                camera_counts: A tensor of shape (N, ) which how many times the
                    coresponding camera in `camera_ids` was sampled.
                    `sum(camera_counts)==minibatch`
            implicit_functions: List of ImplicitFunctionWrappers which define the
                implicit function methods to be used. Most Renderers only allow
                a single implicit function. Currently, only the
                MultiPassEmissionAbsorptionRenderer allows specifying mulitple
                values in the list.
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
