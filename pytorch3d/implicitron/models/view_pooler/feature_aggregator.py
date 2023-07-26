# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from pytorch3d.implicitron.models.view_pooler.view_sampler import (
    cameras_points_cartesian_product,
)
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from pytorch3d.ops import wmean
from pytorch3d.renderer.cameras import CamerasBase


class ReductionFunction(Enum):
    AVG = "avg"  # simple average
    MAX = "max"  # maximum
    STD = "std"  # standard deviation
    STD_AVG = "std_avg"  # average of per-dimension standard deviations


class FeatureAggregatorBase(ABC, ReplaceableBase):
    """
    Base class for aggregating features.

    Typically, the aggregated features and their masks are output by `ViewSampler`
    which samples feature tensors extracted from a set of source images.

    Settings:
        exclude_target_view: If `True`/`False`, enables/disables pooling
            from target view to itself.
        exclude_target_view_mask_features: If `True`,
            mask the features from the target view before aggregation
        concatenate_output: If `True`,
            concatenate the aggregated features into a single tensor,
            otherwise return a dictionary mapping feature names to tensors.
    """

    exclude_target_view: bool = True
    exclude_target_view_mask_features: bool = True
    concatenate_output: bool = True

    @abstractmethod
    def forward(
        self,
        feats_sampled: Dict[str, torch.Tensor],
        masks_sampled: torch.Tensor,
        camera: Optional[CamerasBase] = None,
        pts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feats_sampled: A `dict` of sampled feature tensors `{f_i: t_i}`,
                where each `t_i` is a tensor of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
            masks_sampled: A binary mask represented as a tensor of shape
                `(minibatch, n_source_views, n_samples, 1)` denoting valid
                sampled features.
            camera: A batch of `n_source_views` `CamerasBase` objects corresponding
                to the source view cameras.
            pts: A tensor of shape `(minibatch, n_samples, 3)` denoting the
                3D points whose 2D projections to source views were sampled in
                order to generate `feats_sampled` and `masks_sampled`.

        Returns:
            feats_aggregated: If `concatenate_output==True`, a tensor
                of shape `(minibatch, reduce_dim, n_samples, sum(dim_1, ... dim_N))`
                containing the concatenation of the aggregated features `feats_sampled`.
                `reduce_dim` depends on the specific feature aggregator
                implementation and typically equals 1 or `n_source_views`.
                If `concatenate_output==False`, the aggregator does not concatenate
                the aggregated features and returns a dictionary of per-feature
                aggregations `{f_i: t_i_aggregated}` instead. Each `t_i_aggregated`
                is of shape `(minibatch, reduce_dim, n_samples, aggr_dim_i)`.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_aggregated_feature_dim(
        self, feats_or_feats_dim: Union[Dict[str, torch.Tensor], int]
    ):
        """
        Returns the final dimensionality of the output aggregated features.

        Args:
            feats_or_feats_dim: Either a `dict` of sampled features `{f_i: t_i}` corresponding
                to the `feats_sampled` argument of `forward`,
                or an `int` representing the sum of dimensionalities of each `t_i`.

        Returns:
            aggregated_feature_dim: The final dimensionality of the output
                aggregated features.
        """
        raise NotImplementedError()

    def has_aggregation(self) -> bool:
        """
        Specifies whether the aggregator reduces the output `reduce_dim` dimension to 1.

        Returns:
            has_aggregation: `True` if `reduce_dim==1`, else `False`.
        """
        return hasattr(self, "reduction_functions")


@registry.register
class IdentityFeatureAggregator(torch.nn.Module, FeatureAggregatorBase):
    """
    This aggregator does not perform any feature aggregation. Depending on the
    settings the aggregator allows to mask target view features and concatenate
    the outputs.
    """

    def get_aggregated_feature_dim(
        self, feats_or_feats_dim: Union[Dict[str, torch.Tensor], int]
    ):
        return _get_reduction_aggregator_feature_dim(feats_or_feats_dim, [])

    def forward(
        self,
        feats_sampled: Dict[str, torch.Tensor],
        masks_sampled: torch.Tensor,
        camera: Optional[CamerasBase] = None,
        pts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feats_sampled: A `dict` of sampled feature tensors `{f_i: t_i}`,
                where each `t_i` is a tensor of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
            masks_sampled: A binary mask represented as a tensor of shape
                `(minibatch, n_source_views, n_samples, 1)` denoting valid
                sampled features.
            camera: A batch of `n_source_views` `CamerasBase` objects
                corresponding to the source view cameras.
            pts: A tensor of shape `(minibatch, n_samples, 3)` denoting the
                3D points whose 2D projections to source views were sampled in
                order to generate `feats_sampled` and `masks_sampled`.

        Returns:
            feats_aggregated: If `concatenate_output==True`, a tensor
                of shape `(minibatch, 1, n_samples, sum(dim_1, ... dim_N))`.
                If `concatenate_output==False`, a dictionary `{f_i: t_i_aggregated}`
                with each `t_i_aggregated` of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
        """
        if self.exclude_target_view_mask_features:
            feats_sampled = _mask_target_view_features(feats_sampled)
        feats_aggregated = feats_sampled
        if self.concatenate_output:
            feats_aggregated = torch.cat(tuple(feats_aggregated.values()), dim=-1)
        return feats_aggregated


@registry.register
class ReductionFeatureAggregator(torch.nn.Module, FeatureAggregatorBase):
    """
    Aggregates using a set of predefined `reduction_functions` and concatenates
    the results of each aggregation function along the
    channel dimension. The reduction functions singularize the second dimension
    of the sampled features which stacks the source views.

    Settings:
        reduction_functions: A list of `ReductionFunction`s` that reduce the
            the stack of source-view-specific features to a single feature.
    """

    reduction_functions: Tuple[ReductionFunction, ...] = (
        ReductionFunction.AVG,
        ReductionFunction.STD,
    )

    def get_aggregated_feature_dim(
        self, feats_or_feats_dim: Union[Dict[str, torch.Tensor], int]
    ):
        return _get_reduction_aggregator_feature_dim(
            feats_or_feats_dim, self.reduction_functions
        )

    def forward(
        self,
        feats_sampled: Dict[str, torch.Tensor],
        masks_sampled: torch.Tensor,
        camera: Optional[CamerasBase] = None,
        pts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feats_sampled: A `dict` of sampled feature tensors `{f_i: t_i}`,
                where each `t_i` is a tensor of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
            masks_sampled: A binary mask represented as a tensor of shape
                `(minibatch, n_source_views, n_samples, 1)` denoting valid
                sampled features.
            camera: A batch of `n_source_views` `CamerasBase` objects corresponding
                to the source view cameras.
            pts: A tensor of shape `(minibatch, n_samples, 3)` denoting the
                3D points whose 2D projections to source views were sampled in
                order to generate `feats_sampled` and `masks_sampled`.

        Returns:
            feats_aggregated: If `concatenate_output==True`, a tensor
                of shape `(minibatch, 1, n_samples, sum(dim_1, ... dim_N))`.
                If `concatenate_output==False`, a dictionary `{f_i: t_i_aggregated}`
                with each `t_i_aggregated` of shape `(minibatch, 1, n_samples, aggr_dim_i)`.
        """

        pts_batch, n_cameras = masks_sampled.shape[:2]
        if self.exclude_target_view_mask_features:
            feats_sampled = _mask_target_view_features(feats_sampled)
        sampling_mask = _get_view_sampling_mask(
            n_cameras,
            pts_batch,
            masks_sampled.device,
            self.exclude_target_view,
        )
        aggr_weigths = masks_sampled[..., 0] * sampling_mask[..., None]
        feats_aggregated = {
            k: _avgmaxstd_reduction_function(
                f,
                aggr_weigths,
                dim=1,
                reduction_functions=self.reduction_functions,
            )
            for k, f in feats_sampled.items()
        }
        if self.concatenate_output:
            feats_aggregated = torch.cat(tuple(feats_aggregated.values()), dim=-1)
        return feats_aggregated


@registry.register
class AngleWeightedReductionFeatureAggregator(torch.nn.Module, FeatureAggregatorBase):
    """
    Performs a weighted aggregation using a set of predefined `reduction_functions`
    and concatenates the results of each aggregation function along the
    channel dimension. The weights are proportional to the cosine of the
    angle between the target ray and the source ray::

        weight = (
            dot(target_ray, source_ray) * 0.5 + 0.5 + self.min_ray_angle_weight
        )**self.weight_by_ray_angle_gamma

    The reduction functions singularize the second dimension
    of the sampled features which stacks the source views.

    Settings:
        reduction_functions: A list of `ReductionFunction`s that reduce the
            the stack of source-view-specific features to a single feature.
        min_ray_angle_weight: The minimum possible aggregation weight
            before rasising to the power of `self.weight_by_ray_angle_gamma`.
        weight_by_ray_angle_gamma: The exponent of the cosine of the ray angles
            used when calculating the angle-based aggregation weights.
    """

    reduction_functions: Tuple[ReductionFunction, ...] = (
        ReductionFunction.AVG,
        ReductionFunction.STD,
    )
    weight_by_ray_angle_gamma: float = 1.0
    min_ray_angle_weight: float = 0.1

    def get_aggregated_feature_dim(
        self, feats_or_feats_dim: Union[Dict[str, torch.Tensor], int]
    ):
        return _get_reduction_aggregator_feature_dim(
            feats_or_feats_dim, self.reduction_functions
        )

    def forward(
        self,
        feats_sampled: Dict[str, torch.Tensor],
        masks_sampled: torch.Tensor,
        camera: Optional[CamerasBase] = None,
        pts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feats_sampled: A `dict` of sampled feature tensors `{f_i: t_i}`,
                where each `t_i` is a tensor of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
            masks_sampled: A binary mask represented as a tensor of shape
                `(minibatch, n_source_views, n_samples, 1)` denoting valid
                sampled features.
            camera: A batch of `n_source_views` `CamerasBase` objects
                corresponding to the source view cameras.
            pts: A tensor of shape `(minibatch, n_samples, 3)` denoting the
                3D points whose 2D projections to source views were sampled in
                order to generate `feats_sampled` and `masks_sampled`.

        Returns:
            feats_aggregated: If `concatenate_output==True`, a tensor
                of shape `(minibatch, 1, n_samples, sum(dim_1, ... dim_N))`.
                If `concatenate_output==False`, a dictionary `{f_i: t_i_aggregated}`
                with each `t_i_aggregated` of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
        """

        if camera is None:
            raise ValueError("camera cannot be None for angle weighted aggregation")

        if pts is None:
            raise ValueError("Points cannot be None for angle weighted aggregation")

        pts_batch, n_cameras = masks_sampled.shape[:2]
        if self.exclude_target_view_mask_features:
            feats_sampled = _mask_target_view_features(feats_sampled)
        view_sampling_mask = _get_view_sampling_mask(
            n_cameras,
            pts_batch,
            masks_sampled.device,
            self.exclude_target_view,
        )
        aggr_weights = _get_angular_reduction_weights(
            view_sampling_mask,
            masks_sampled,
            camera,
            pts,
            self.min_ray_angle_weight,
            self.weight_by_ray_angle_gamma,
        )
        assert torch.isfinite(aggr_weights).all()
        feats_aggregated = {
            k: _avgmaxstd_reduction_function(
                f,
                aggr_weights,
                dim=1,
                reduction_functions=self.reduction_functions,
            )
            for k, f in feats_sampled.items()
        }
        if self.concatenate_output:
            feats_aggregated = torch.cat(tuple(feats_aggregated.values()), dim=-1)
        return feats_aggregated


@registry.register
class AngleWeightedIdentityFeatureAggregator(torch.nn.Module, FeatureAggregatorBase):
    """
    This aggregator does not perform any feature aggregation. It only weights
    the features by the weights proportional to the cosine of the
    angle between the target ray and the source ray::

        weight = (
            dot(target_ray, source_ray) * 0.5 + 0.5 + self.min_ray_angle_weight
        )**self.weight_by_ray_angle_gamma

    Settings:
        min_ray_angle_weight: The minimum possible aggregation weight
            before rasising to the power of `self.weight_by_ray_angle_gamma`.
        weight_by_ray_angle_gamma: The exponent of the cosine of the ray angles
            used when calculating the angle-based aggregation weights.

    Additionally the aggregator allows to mask target view features and to concatenate
    the outputs.
    """

    weight_by_ray_angle_gamma: float = 1.0
    min_ray_angle_weight: float = 0.1

    def get_aggregated_feature_dim(
        self, feats_or_feats_dim: Union[Dict[str, torch.Tensor], int]
    ):
        return _get_reduction_aggregator_feature_dim(feats_or_feats_dim, [])

    def forward(
        self,
        feats_sampled: Dict[str, torch.Tensor],
        masks_sampled: torch.Tensor,
        camera: Optional[CamerasBase] = None,
        pts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            feats_sampled: A `dict` of sampled feature tensors `{f_i: t_i}`,
                where each `t_i` is a tensor of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
            masks_sampled: A binary mask represented as a tensor of shape
                `(minibatch, n_source_views, n_samples, 1)` denoting valid
                sampled features.
            camera: A batch of `n_source_views` `CamerasBase` objects corresponding
                to the source view cameras.
            pts: A tensor of shape `(minibatch, n_samples, 3)` denoting the
                3D points whose 2D projections to source views were sampled in
                order to generate `feats_sampled` and `masks_sampled`.

        Returns:
            feats_aggregated: If `concatenate_output==True`, a tensor
                of shape `(minibatch, n_source_views, n_samples, sum(dim_1, ... dim_N))`.
                If `concatenate_output==False`, a dictionary `{f_i: t_i_aggregated}`
                with each `t_i_aggregated` of shape
                `(minibatch, n_source_views, n_samples, dim_i)`.
        """

        if camera is None:
            raise ValueError("camera cannot be None for angle weighted aggregation")

        if pts is None:
            raise ValueError("Points cannot be None for angle weighted aggregation")

        pts_batch, n_cameras = masks_sampled.shape[:2]
        if self.exclude_target_view_mask_features:
            feats_sampled = _mask_target_view_features(feats_sampled)
        view_sampling_mask = _get_view_sampling_mask(
            n_cameras,
            pts_batch,
            masks_sampled.device,
            self.exclude_target_view,
        )
        aggr_weights = _get_angular_reduction_weights(
            view_sampling_mask,
            masks_sampled,
            camera,
            pts,
            self.min_ray_angle_weight,
            self.weight_by_ray_angle_gamma,
        )
        feats_aggregated = {
            k: f * aggr_weights[..., None] for k, f in feats_sampled.items()
        }
        if self.concatenate_output:
            feats_aggregated = torch.cat(tuple(feats_aggregated.values()), dim=-1)
        return feats_aggregated


def _get_reduction_aggregator_feature_dim(
    feats_or_feats_dim: Union[Dict[str, torch.Tensor], int],
    reduction_functions: Sequence[ReductionFunction],
) -> int:
    if isinstance(feats_or_feats_dim, int):
        feat_dim = feats_or_feats_dim
    else:
        feat_dim = int(sum(f.shape[1] for f in feats_or_feats_dim.values()))
    if len(reduction_functions) == 0:
        return feat_dim
    return sum(
        _get_reduction_function_output_dim(
            reduction_function,
            feat_dim,
        )
        for reduction_function in reduction_functions
    )


def _get_reduction_function_output_dim(
    reduction_function: ReductionFunction,
    feat_dim: int,
) -> int:
    if reduction_function == ReductionFunction.STD_AVG:
        return 1
    else:
        return feat_dim


def _get_view_sampling_mask(
    n_cameras: int,
    pts_batch: int,
    device: Union[str, torch.device],
    exclude_target_view: bool,
):
    return (
        -torch.eye(n_cameras, device=device, dtype=torch.float32)
        * float(exclude_target_view)
        + 1.0
    )[:pts_batch]


def _mask_target_view_features(
    feats_sampled: Dict[str, torch.Tensor],
):
    # mask out the sampled features to be sure we dont use them
    # anywhere later
    one_feature_sampled = next(iter(feats_sampled.values()))
    pts_batch, n_cameras = one_feature_sampled.shape[:2]
    view_sampling_mask = _get_view_sampling_mask(
        n_cameras,
        pts_batch,
        one_feature_sampled.device,
        True,
    )
    view_sampling_mask = view_sampling_mask.view(
        pts_batch, n_cameras, *([1] * (one_feature_sampled.ndim - 2))
    )
    return {k: f * view_sampling_mask for k, f in feats_sampled.items()}


def _get_angular_reduction_weights(
    view_sampling_mask: torch.Tensor,
    masks_sampled: torch.Tensor,
    camera: CamerasBase,
    pts: torch.Tensor,
    min_ray_angle_weight: float,
    weight_by_ray_angle_gamma: float,
):
    aggr_weights = masks_sampled.clone()[..., 0]
    assert not any(v is None for v in [camera, pts])
    angle_weight = _get_ray_angle_weights(
        camera,
        pts,
        min_ray_angle_weight,
        weight_by_ray_angle_gamma,
    )
    assert torch.isfinite(angle_weight).all()
    # multiply the final aggr weights with ray angles
    view_sampling_mask = view_sampling_mask.view(
        *view_sampling_mask.shape[:2], *([1] * (aggr_weights.ndim - 2))
    )
    aggr_weights = (
        aggr_weights * angle_weight.reshape_as(aggr_weights) * view_sampling_mask
    )
    return aggr_weights


def _get_ray_dir_dot_prods(camera: CamerasBase, pts: torch.Tensor):
    n_cameras = camera.R.shape[0]
    pts_batch = pts.shape[0]

    camera_rep, pts_rep = cameras_points_cartesian_product(camera, pts)

    # does not produce nans randomly unlike get_camera_center() below
    cam_centers_rep = -torch.bmm(
        camera_rep.T[:, None],
        camera_rep.R.permute(0, 2, 1),
    ).reshape(-1, *([1] * (pts.ndim - 2)), 3)
    # cam_centers_rep = camera_rep.get_camera_center().reshape(
    #     -1, *([1]*(pts.ndim - 2)), 3
    # )

    ray_dirs = F.normalize(pts_rep - cam_centers_rep, dim=-1)
    # camera_rep = [                 pts_rep = [
    #     camera[0]                      pts[0],
    #     camera[0]                      pts[1],
    #     camera[0]                      ...,
    #     ...                            pts[batch_pts-1],
    #     camera[1]                      pts[0],
    #     camera[1]                      pts[1],
    #     camera[1]                      ...,
    #     ...                            pts[batch_pts-1],
    #     ...                            ...,
    #     camera[n_cameras-1]            pts[0],
    #     camera[n_cameras-1]            pts[1],
    #     camera[n_cameras-1]            ...,
    #     ...                            pts[batch_pts-1],
    # ]                              ]

    ray_dirs_reshape = ray_dirs.view(n_cameras, pts_batch, -1, 3)
    # [
    #   [pts_0 in cam_0, pts_1 in cam_0, ..., pts_m in cam_0],
    #   [pts_0 in cam_1, pts_1 in cam_1, ..., pts_m in cam_1],
    #   ...
    #   [pts_0 in cam_n, pts_1 in cam_n, ..., pts_m in cam_n],
    # ]

    ray_dirs_pts = torch.stack([ray_dirs_reshape[i, i] for i in range(pts_batch)])
    ray_dir_dot_prods = (ray_dirs_pts[None] * ray_dirs_reshape).sum(
        dim=-1
    )  # pts_batch x n_cameras x n_pts

    return ray_dir_dot_prods.transpose(0, 1)


def _get_ray_angle_weights(
    camera: CamerasBase,
    pts: torch.Tensor,
    min_ray_angle_weight: float,
    weight_by_ray_angle_gamma: float,
):
    ray_dir_dot_prods = _get_ray_dir_dot_prods(
        camera, pts
    )  # pts_batch x n_cameras x ... x 3
    angle_weight_01 = ray_dir_dot_prods * 0.5 + 0.5  # [-1, 1] to [0, 1]
    angle_weight = (angle_weight_01 + min_ray_angle_weight) ** weight_by_ray_angle_gamma
    return angle_weight


def _avgmaxstd_reduction_function(
    x: torch.Tensor,
    w: torch.Tensor,
    reduction_functions: Sequence[ReductionFunction],
    dim: int = 1,
):
    """
    Args:
        x: Features to aggreagate. Tensor of shape `(batch, n_views, ..., dim)`.
        w: Aggregation weights. Tensor of shape `(batch, n_views, ...,)`.
        dim: the dimension along which to aggregate.
        reduction_functions: The set of reduction functions.

    Returns:
        x_aggr: Aggregation of `x` to a tensor of shape `(batch, 1, ..., dim_aggregate)`.
    """

    pooled_features = []

    mu = None
    std = None

    if ReductionFunction.AVG in reduction_functions:
        # average pool
        mu = _avg_reduction_function(x, w, dim=dim)
        pooled_features.append(mu)

    if ReductionFunction.STD in reduction_functions:
        # standard-dev pool
        std = _std_reduction_function(x, w, dim=dim, mu=mu)
        pooled_features.append(std)

    if ReductionFunction.STD_AVG in reduction_functions:
        # average-of-standard-dev pool
        stdavg = _std_avg_reduction_function(x, w, dim=dim, mu=mu, std=std)
        pooled_features.append(stdavg)

    if ReductionFunction.MAX in reduction_functions:
        max_ = _max_reduction_function(x, w, dim=dim)
        pooled_features.append(max_)

    # cat all results along the feature dimension (the last dim)
    x_aggr = torch.cat(pooled_features, dim=-1)

    # zero out features that were all masked out
    # pyre-fixme[16]: `bool` has no attribute `type_as`.
    any_active = (w.max(dim=dim, keepdim=True).values > 1e-4).type_as(x_aggr)
    x_aggr = x_aggr * any_active[..., None]

    # some asserts to check that everything was done right
    assert torch.isfinite(x_aggr).all()
    assert x_aggr.shape[1] == 1

    return x_aggr


def _avg_reduction_function(
    x: torch.Tensor,
    w: torch.Tensor,
    dim: int = 1,
):
    mu = wmean(x, w, dim=dim, eps=1e-2)
    return mu


def _std_reduction_function(
    x: torch.Tensor,
    w: torch.Tensor,
    dim: int = 1,
    mu: Optional[torch.Tensor] = None,  # pre-computed mean
):
    if mu is None:
        mu = _avg_reduction_function(x, w, dim=dim)
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    std = wmean((x - mu) ** 2, w, dim=dim, eps=1e-2).clamp(1e-4).sqrt()
    # FIXME: somehow this is extremely heavy in mem?
    return std


def _std_avg_reduction_function(
    x: torch.Tensor,
    w: torch.Tensor,
    dim: int = 1,
    mu: Optional[torch.Tensor] = None,  # pre-computed mean
    std: Optional[torch.Tensor] = None,  # pre-computed std
):
    if std is None:
        std = _std_reduction_function(x, w, dim=dim, mu=mu)
    stdmean = std.mean(dim=-1, keepdim=True)
    return stdmean


def _max_reduction_function(
    x: torch.Tensor,
    w: torch.Tensor,
    dim: int = 1,
    big_M_factor: float = 10.0,
):
    big_M = x.max(dim=dim, keepdim=True).values.abs() * big_M_factor
    max_ = (x * w - ((1 - w) * big_M)).max(dim=dim, keepdim=True).values
    return max_
