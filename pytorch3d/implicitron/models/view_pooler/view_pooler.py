# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

import torch
from pytorch3d.implicitron.tools.config import Configurable, run_auto_creation
from pytorch3d.renderer.cameras import CamerasBase

from .feature_aggregator import FeatureAggregatorBase
from .view_sampler import ViewSampler


# pyre-ignore: 13
class ViewPooler(Configurable, torch.nn.Module):
    """
    Implements sampling of image-based features at the 2d projections of a set
    of 3D points, and a subsequent aggregation of the resulting set of features
    per-point.

    Args:
        view_sampler: An instance of ViewSampler which is used for sampling of
            image-based features at the 2D projections of a set
            of 3D points.
        feature_aggregator_class_type: The name of the feature aggregator class which
            is available in the global registry.
        feature_aggregator: A feature aggregator class which inherits from
            FeatureAggregatorBase. Typically, the aggregated features and their
            masks are output by a `ViewSampler` which samples feature tensors extracted
            from a set of source images. FeatureAggregator executes step (4) above.
    """

    view_sampler: ViewSampler
    feature_aggregator_class_type: str = "AngleWeightedReductionFeatureAggregator"
    feature_aggregator: FeatureAggregatorBase

    def __post_init__(self):
        run_auto_creation(self)

    def get_aggregated_feature_dim(self, feats: Union[Dict[str, torch.Tensor], int]):
        """
        Returns the final dimensionality of the output aggregated features.

        Args:
            feats: Either a `dict` of sampled features `{f_i: t_i}` corresponding
                to the `feats_sampled` argument of `feature_aggregator,forward`,
                or an `int` representing the sum of dimensionalities of each `t_i`.

        Returns:
            aggregated_feature_dim: The final dimensionality of the output
                aggregated features.
        """
        return self.feature_aggregator.get_aggregated_feature_dim(feats)

    def has_aggregation(self):
        """
        Specifies whether the `feature_aggregator` reduces the output `reduce_dim`
        dimension to 1.

        Returns:
            has_aggregation: `True` if `reduce_dim==1`, else `False`.
        """
        return self.feature_aggregator.has_aggregation()

    def forward(
        self,
        *,  # force kw args
        pts: torch.Tensor,
        seq_id_pts: Union[List[int], List[str], torch.LongTensor],
        camera: CamerasBase,
        seq_id_camera: Union[List[int], List[str], torch.LongTensor],
        feats: Dict[str, torch.Tensor],
        masks: Optional[torch.Tensor],
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Project each point cloud from a batch of point clouds to corresponding
        input cameras, sample features at the 2D projection locations in a batch
        of source images, and aggregate the pointwise sampled features.

        Args:
            pts: A tensor of shape `[pts_batch x n_pts x 3]` in world coords.
            seq_id_pts: LongTensor of shape `[pts_batch]` denoting the ids of the scenes
                from which `pts` were extracted, or a list of string names.
            camera: 'n_cameras' cameras, each coresponding to a batch element of `feats`.
            seq_id_camera: LongTensor of shape `[n_cameras]` denoting the ids of the scenes
                corresponding to cameras in `camera`, or a list of string names.
            feats: a dict of tensors of per-image features `{feat_i: T_i}`.
                Each tensor `T_i` is of shape `[n_cameras x dim_i x H_i x W_i]`.
            masks: `[n_cameras x 1 x H x W]`, define valid image regions
                for sampling `feats`.
        Returns:
            feats_aggregated: If `feature_aggregator.concatenate_output==True`, a tensor
                of shape `(pts_batch, reduce_dim, n_pts, sum(dim_1, ... dim_N))`
                containing the aggregated features. `reduce_dim` depends on
                the specific feature aggregator implementation and typically
                equals 1 or `n_cameras`.
                If `feature_aggregator.concatenate_output==False`, the aggregator
                does not concatenate the aggregated features and returns a dictionary
                of per-feature aggregations `{f_i: t_i_aggregated}` instead.
                Each `t_i_aggregated` is of shape
                `(pts_batch, reduce_dim, n_pts, aggr_dim_i)`.
        """

        # (1) Sample features and masks at the ray points
        sampled_feats, sampled_masks = self.view_sampler(
            pts=pts,
            seq_id_pts=seq_id_pts,
            camera=camera,
            seq_id_camera=seq_id_camera,
            feats=feats,
            masks=masks,
        )

        # (2) Aggregate features from multiple views
        # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
        feats_aggregated = self.feature_aggregator(  # noqa: E731
            sampled_feats,
            sampled_masks,
            pts=pts,
            camera=camera,
        )  # TODO: do we need to pass a callback rather than compute here?

        return feats_aggregated
