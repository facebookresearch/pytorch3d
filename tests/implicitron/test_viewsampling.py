# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import pytorch3d as pt3d
import torch
from pytorch3d.implicitron.models.view_pooler.view_sampler import ViewSampler
from pytorch3d.implicitron.tools.config import expand_args_fields


class TestViewsampling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        expand_args_fields(ViewSampler)

    def _init_view_sampler_problem(self, random_masks):
        """
        Generates a view-sampling problem:
        - 4 source views, 1st/2nd from the first sequence 'seq1', the rest from 'seq2'
        - 3 sets of 3D points from sequences 'seq1', 'seq2', 'seq2' respectively.
            - first 50 points in each batch correctly project to the source views,
                while the remaining 50 do not land in any projection plane.
        - each source view is labeled with image feature tensors of shape 7x100x50,
            where all elements of the n-th tensor are set to `n+1`.
        - the elements of the source view masks are either set to random binary number
            (if `random_masks==True`), or all set to 1 (`random_masks==False`).
        - the source view cameras are uniformly distributed on a unit circle
            in the x-z plane and look at (0,0,0).
        """
        seq_id_camera = ["seq1", "seq1", "seq2", "seq2"]
        seq_id_pts = ["seq1", "seq2", "seq2"]
        pts_batch = 3
        n_pts = 100
        n_views = 4
        fdim = 7
        H = 100
        W = 50

        # points that land into the projection planes of all cameras
        pts_inside = (
            torch.nn.functional.normalize(
                torch.randn(pts_batch, n_pts // 2, 3, device="cuda"),
                dim=-1,
            )
            * 0.1
        )

        # move the outside points far above the scene
        pts_outside = pts_inside.clone()
        pts_outside[:, :, 1] += 1e8
        pts = torch.cat([pts_inside, pts_outside], dim=1)

        R, T = pt3d.renderer.look_at_view_transform(
            dist=1.0,
            elev=0.0,
            azim=torch.linspace(0, 360, n_views + 1)[:n_views],
            degrees=True,
            device=pts.device,
        )
        focal_length = R.new_ones(n_views, 2)
        principal_point = R.new_zeros(n_views, 2)
        camera = pt3d.renderer.PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal_length,
            principal_point=principal_point,
            device=pts.device,
        )

        feats_map = torch.arange(n_views, device=pts.device, dtype=pts.dtype) + 1
        feats = {"feats": feats_map[:, None, None, None].repeat(1, fdim, H, W)}

        masks = (
            torch.rand(n_views, 1, H, W, device=pts.device, dtype=pts.dtype) > 0.5
        ).type_as(R)

        if not random_masks:
            masks[:] = 1.0

        return pts, camera, feats, masks, seq_id_camera, seq_id_pts

    def test_compare_with_naive(self):
        """
        Compares the outputs of the efficient ViewSampler module with a
        naive implementation.
        """

        (
            pts,
            camera,
            feats,
            masks,
            seq_id_camera,
            seq_id_pts,
        ) = self._init_view_sampler_problem(True)

        for masked_sampling in (True, False):
            feats_sampled_n, masks_sampled_n = _view_sample_naive(
                pts,
                seq_id_pts,
                camera,
                seq_id_camera,
                feats,
                masks,
                masked_sampling,
            )
            # make sure we generate the constructor for ViewSampler
            expand_args_fields(ViewSampler)
            view_sampler = ViewSampler(masked_sampling=masked_sampling)
            feats_sampled, masks_sampled = view_sampler(
                pts=pts,
                seq_id_pts=seq_id_pts,
                camera=camera,
                seq_id_camera=seq_id_camera,
                feats=feats,
                masks=masks,
            )
            for k in feats_sampled.keys():
                self.assertTrue(torch.allclose(feats_sampled[k], feats_sampled_n[k]))
            self.assertTrue(torch.allclose(masks_sampled, masks_sampled_n))

    def test_viewsampling(self):
        """
        Generates a viewsampling problem with predictable outcome, and compares
        the ViewSampler's output to the expected result.
        """

        (
            pts,
            camera,
            feats,
            masks,
            seq_id_camera,
            seq_id_pts,
        ) = self._init_view_sampler_problem(False)

        expand_args_fields(ViewSampler)

        for masked_sampling in (True, False):

            view_sampler = ViewSampler(masked_sampling=masked_sampling)

            feats_sampled, masks_sampled = view_sampler(
                pts=pts,
                seq_id_pts=seq_id_pts,
                camera=camera,
                seq_id_camera=seq_id_camera,
                feats=feats,
                masks=masks,
            )

            n_views = camera.R.shape[0]
            n_pts = pts.shape[1]
            feat_dim = feats["feats"].shape[1]
            pts_batch = pts.shape[0]
            n_pts_away = n_pts // 2

            for pts_i in range(pts_batch):
                for view_i in range(n_views):
                    if seq_id_pts[pts_i] != seq_id_camera[view_i]:
                        # points / cameras come from different sequences
                        gt_masks = pts.new_zeros(n_pts, 1)
                        gt_feats = pts.new_zeros(n_pts, feat_dim)
                    else:
                        gt_masks = pts.new_ones(n_pts, 1)
                        gt_feats = pts.new_ones(n_pts, feat_dim) * (view_i + 1)
                        gt_feats[n_pts_away:] = 0.0
                        if masked_sampling:
                            gt_masks[n_pts_away:] = 0.0

                    for k in feats_sampled:
                        self.assertTrue(
                            torch.allclose(
                                feats_sampled[k][pts_i, view_i],
                                gt_feats,
                            )
                        )
                    self.assertTrue(
                        torch.allclose(
                            masks_sampled[pts_i, view_i],
                            gt_masks,
                        )
                    )


def _view_sample_naive(
    pts,
    seq_id_pts,
    camera,
    seq_id_camera,
    feats,
    masks,
    masked_sampling,
):
    """
    A naive implementation of the forward pass of ViewSampler.
    Refer to ViewSampler's docstring for description of the arguments.
    """

    pts_batch = pts.shape[0]
    n_views = camera.R.shape[0]
    n_pts = pts.shape[1]

    feats_sampled = [[[] for _ in range(n_views)] for _ in range(pts_batch)]
    masks_sampled = [[[] for _ in range(n_views)] for _ in range(pts_batch)]

    for pts_i in range(pts_batch):
        for view_i in range(n_views):
            if seq_id_pts[pts_i] != seq_id_camera[view_i]:
                # points/cameras come from different sequences
                feats_sampled_ = {
                    k: f.new_zeros(n_pts, f.shape[1]) for k, f in feats.items()
                }
                masks_sampled_ = masks.new_zeros(n_pts, 1)
            else:
                # same sequence of pts and cameras -> sample
                feats_sampled_, masks_sampled_ = _sample_one_view_naive(
                    camera[view_i],
                    pts[pts_i],
                    {k: f[view_i] for k, f in feats.items()},
                    masks[view_i],
                    masked_sampling,
                    sampling_mode="bilinear",
                )
            feats_sampled[pts_i][view_i] = feats_sampled_
            masks_sampled[pts_i][view_i] = masks_sampled_

    masks_sampled_cat = torch.stack([torch.stack(m) for m in masks_sampled])
    feats_sampled_cat = {}
    for k in feats_sampled[0][0].keys():
        feats_sampled_cat[k] = torch.stack(
            [torch.stack([f_[k] for f_ in f]) for f in feats_sampled]
        )
    return feats_sampled_cat, masks_sampled_cat


def _sample_one_view_naive(
    camera,
    pts,
    feats,
    masks,
    masked_sampling,
    sampling_mode="bilinear",
):
    """
    Sample a single source view.
    """
    proj_ndc = camera.transform_points(pts[None])[None, ..., :-1]  # 1 x 1 x n_pts x 2
    feats_sampled = {
        k: pt3d.renderer.ndc_grid_sample(f[None], proj_ndc, mode=sampling_mode).permute(
            0, 3, 1, 2
        )[0, :, :, 0]
        for k, f in feats.items()
    }  # n_pts x dim
    if not masked_sampling:
        n_pts = pts.shape[0]
        masks_sampled = proj_ndc.new_ones(n_pts, 1)
    else:
        masks_sampled = pt3d.renderer.ndc_grid_sample(
            masks[None],
            proj_ndc,
            mode=sampling_mode,
            align_corners=False,
        )[0, 0, 0, :][:, None]
    return feats_sampled, masks_sampled
