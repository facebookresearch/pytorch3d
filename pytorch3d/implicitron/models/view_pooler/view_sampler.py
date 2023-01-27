# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, Union

import torch
from pytorch3d.implicitron.tools.config import Configurable
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.utils import ndc_grid_sample


class ViewSampler(Configurable, torch.nn.Module):
    """
    Implements sampling of image-based features at the 2d projections of a set
    of 3D points.

    Args:
        masked_sampling: If `True`, the `sampled_masks` output of `self.forward`
            contains the input `masks` sampled at the 2d projections. Otherwise,
            all entries of `sampled_masks` are set to 1.
        sampling_mode: Controls the mode of the `torch.nn.functional.grid_sample`
            function used to interpolate the sampled feature tensors at the
            locations of the 2d projections.
    """

    masked_sampling: bool = False
    sampling_mode: str = "bilinear"

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
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Project each point cloud from a batch of point clouds to corresponding
        input cameras and sample features at the 2D projection locations.

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
            sampled_feats: Dict of sampled features `{feat_i: sampled_T_i}`.
                Each `sampled_T_i` of shape `[pts_batch, n_cameras, n_pts, dim_i]`.
            sampled_masks: A tensor with  mask of the sampled features
                of shape `(pts_batch, n_cameras, n_pts, 1)`.
        """

        # convert sequence ids to long tensors
        seq_id_pts, seq_id_camera = [
            handle_seq_id(seq_id, pts.device) for seq_id in [seq_id_pts, seq_id_camera]
        ]

        if self.masked_sampling and masks is None:
            raise ValueError(
                "Masks have to be provided for `self.masked_sampling==True`"
            )

        # project pts to all cameras and sample feats from the locations of
        # the 2D projections
        sampled_feats_all_cams, sampled_masks_all_cams = project_points_and_sample(
            pts,
            feats,
            camera,
            masks if self.masked_sampling else None,
            sampling_mode=self.sampling_mode,
        )

        # generate the mask that invalidates features sampled from
        # non-corresponding cameras
        camera_pts_mask = (seq_id_camera[None] == seq_id_pts[:, None])[
            ..., None, None
        ].to(pts)

        # mask the sampled features and masks
        sampled_feats = {
            k: f * camera_pts_mask for k, f in sampled_feats_all_cams.items()
        }
        sampled_masks = sampled_masks_all_cams * camera_pts_mask

        return sampled_feats, sampled_masks


def project_points_and_sample(
    pts: torch.Tensor,
    feats: Dict[str, torch.Tensor],
    camera: CamerasBase,
    masks: Optional[torch.Tensor],
    eps: float = 1e-2,
    sampling_mode: str = "bilinear",
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Project each point cloud from a batch of point clouds to all input cameras
    and sample features at the 2D projection locations.

    Args:
        pts: `(pts_batch, n_pts, 3)` tensor containing a batch of 3D point clouds.
        feats: A dict `{feat_i: feat_T_i}` of features to sample,
            where each `feat_T_i` is a tensor of shape
            `(n_cameras, feat_i_dim, feat_i_H, feat_i_W)`
            of `feat_i_dim`-dimensional features extracted from `n_cameras`
            source views.
        camera: A batch of `n_cameras` cameras corresponding to their feature
            tensors `feat_T_i` from `feats`.
        masks: A tensor of shape `(n_cameras, 1, mask_H, mask_W)` denoting
            valid locations for sampling.
        eps: A small constant controlling the minimum depth of projections
            of `pts` to avoid divisons by zero in the projection operation.
        sampling_mode: Sampling mode of the grid sampler.

    Returns:
        sampled_feats: Dict of sampled features `{feat_i: sampled_T_i}`.
            Each `sampled_T_i` is of shape
            `(pts_batch, n_cameras, n_pts, feat_i_dim)`.
        sampled_masks: A tensor with the mask of the sampled features
            of shape `(pts_batch, n_cameras, n_pts, 1)`.
            If `masks` is `None`, the returned `sampled_masks` will be
            filled with 1s.
    """

    n_cameras = camera.R.shape[0]
    pts_batch = pts.shape[0]
    n_pts = pts.shape[1:-1]

    camera_rep, pts_rep = cameras_points_cartesian_product(camera, pts)

    # The eps here is super-important to avoid NaNs in backprop!
    proj_rep = camera_rep.transform_points(
        pts_rep.reshape(n_cameras * pts_batch, -1, 3), eps=eps
    )[..., :2]
    # [ pts1 in cam1, pts2 in cam1, pts3 in cam1,
    #   pts1 in cam2, pts2 in cam2, pts3 in cam2,
    #   pts1 in cam3, pts2 in cam3, pts3 in cam3 ]

    # reshape for the grid sampler
    sampling_grid_ndc = proj_rep.view(n_cameras, pts_batch, -1, 2)
    # [ [pts1 in cam1, pts2 in cam1, pts3 in cam1],
    #   [pts1 in cam2, pts2 in cam2, pts3 in cam2],
    #   [pts1 in cam3, pts2 in cam3, pts3 in cam3] ]
    #   n_cameras x pts_batch x n_pts x 2

    # sample both feats
    feats_sampled = {
        k: ndc_grid_sample(
            f,
            sampling_grid_ndc,
            mode=sampling_mode,
            align_corners=False,
        )
        .permute(2, 0, 3, 1)
        .reshape(pts_batch, n_cameras, *n_pts, -1)
        for k, f in feats.items()
    }  # {k: pts_batch x n_cameras x *n_pts x dim} for each feat type "k"

    if masks is not None:
        # sample masks
        masks_sampled = (
            ndc_grid_sample(
                masks,
                sampling_grid_ndc,
                mode=sampling_mode,
                align_corners=False,
            )
            .permute(2, 0, 3, 1)
            .reshape(pts_batch, n_cameras, *n_pts, 1)
        )
    else:
        masks_sampled = sampling_grid_ndc.new_ones(pts_batch, n_cameras, *n_pts, 1)

    return feats_sampled, masks_sampled


def handle_seq_id(
    seq_id: Union[torch.LongTensor, List[str], List[int]],
    device,
) -> torch.LongTensor:
    """
    Converts the input sequence id to a LongTensor.

    Args:
        seq_id: A sequence of sequence ids.
        device: The target device of the output.
    Returns
        long_seq_id: `seq_id` converted to a `LongTensor` and moved to `device`.
    """
    if not torch.is_tensor(seq_id):
        if isinstance(seq_id[0], str):
            seq_id = [hash(s) for s in seq_id]
        # pyre-fixme[9]: seq_id has type `Union[List[int], List[str], LongTensor]`;
        #  used as `Tensor`.
        seq_id = torch.tensor(seq_id, dtype=torch.long, device=device)
    # pyre-fixme[16]: Item `List` of `Union[List[int], List[str], LongTensor]` has
    #  no attribute `to`.
    return seq_id.to(device)


def cameras_points_cartesian_product(
    camera: CamerasBase, pts: torch.Tensor
) -> Tuple[CamerasBase, torch.Tensor]:
    """
    Generates all pairs of pairs of elements from 'camera' and 'pts' and returns
    `camera_rep` and `pts_rep` such that::

        camera_rep = [                 pts_rep = [
            camera[0]                      pts[0],
            camera[0]                      pts[1],
            camera[0]                      ...,
            ...                            pts[batch_pts-1],
            camera[1]                      pts[0],
            camera[1]                      pts[1],
            camera[1]                      ...,
            ...                            pts[batch_pts-1],
            ...                            ...,
            camera[n_cameras-1]            pts[0],
            camera[n_cameras-1]            pts[1],
            camera[n_cameras-1]            ...,
            ...                            pts[batch_pts-1],
        ]                              ]

    Args:
        camera: A batch of `n_cameras` cameras.
        pts: A batch of `batch_pts` points of shape `(batch_pts, ..., dim)`

    Returns:
        camera_rep: A batch of batch_pts*n_cameras cameras such that::

            camera_rep = [
                camera[0]
                camera[0]
                camera[0]
                ...
                camera[1]
                camera[1]
                camera[1]
                ...
                ...
                camera[n_cameras-1]
                camera[n_cameras-1]
                camera[n_cameras-1]
            ]


        pts_rep: Repeated `pts` of shape `(batch_pts*n_cameras, ..., dim)`,
            such that::

            pts_rep = [
                pts[0],
                pts[1],
                ...,
                pts[batch_pts-1],
                pts[0],
                pts[1],
                ...,
                pts[batch_pts-1],
                ...,
                pts[0],
                pts[1],
                ...,
                pts[batch_pts-1],
            ]

    """
    n_cameras = camera.R.shape[0]
    batch_pts = pts.shape[0]
    pts_rep = pts.repeat(n_cameras, *[1 for _ in pts.shape[1:]])
    idx_cams = (
        torch.arange(n_cameras)[:, None]
        .expand(
            n_cameras,
            batch_pts,
        )
        .reshape(batch_pts * n_cameras)
    )
    # pyre-fixme[6]: For 1st param expected `Union[List[int], int, LongTensor]` but
    #  got `Tensor`.
    camera_rep = camera[idx_cams]
    return camera_rep, pts_rep
