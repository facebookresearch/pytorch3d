# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import pytorch3d as pt3d
import torch
from pytorch3d.implicitron.models.view_pooler.feature_aggregator import (
    IdentityFeatureAggregator,
    ReductionFeatureAggregator,
    ReductionFunction,
)
from pytorch3d.implicitron.models.view_pooler.view_sampler import (
    project_points_and_sample,
    ViewSampler,
)
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    ndc_grid_sample,
    OrthographicCameras,
    PerspectiveCameras,
)

_CAMERA_TYPES = (
    PerspectiveCameras,
    OrthographicCameras,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
)


def _test_devices():
    return ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]


class TestViewsampling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        expand_args_fields(ViewSampler)

    def test_nonpositive_camera_depth(self):
        for device in _test_devices():
            pts = torch.tensor(
                [
                    [
                        [0.0, 0.0, 2.0],
                        [0.0, 0.0, 1e-4],
                        [0.0, 0.0, -2.0],
                        [0.0, 0.0, 0.0],
                    ]
                ],
                device=device,
            )
            feats = {"features": torch.full((1, 2, 8, 12), 3.0, device=device)}
            masks = torch.full((1, 1, 5, 9), 0.25, device=device)
            for camera_type in _CAMERA_TYPES:
                camera = camera_type(device=device)
                for mode in ("nearest", "bilinear", "bicubic"):
                    for masked in (False, True):
                        with self.subTest(
                            device=device,
                            camera=camera_type.__name__,
                            mode=mode,
                            masked=masked,
                        ):
                            sampled, sampled_masks = project_points_and_sample(
                                pts,
                                feats,
                                camera,
                                masks if masked else None,
                                sampling_mode=mode,
                            )
                            expected_feats = torch.zeros_like(sampled["features"])
                            expected_feats[..., :2, :] = 3.0
                            expected_masks = torch.zeros_like(sampled_masks)
                            expected_masks[..., :2, :] = 0.25 if masked else 1.0
                            torch.testing.assert_close(
                                sampled["features"], expected_feats
                            )
                            torch.testing.assert_close(sampled_masks, expected_masks)

    def test_transformed_cameras_and_point_grid(self):
        for device in _test_devices():
            R = torch.stack(
                (torch.eye(3), torch.diag(torch.tensor([-1.0, 1.0, -1.0])))
            ).to(device)
            T = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=device)
            pts = torch.zeros(2, 2, 3, 3, device=device)
            pts[0, ..., 2] = torch.arange(-2, 4, device=device).reshape(2, 3)
            pts[1, ..., 2] = -pts[0, ..., 2]
            feats = {
                "rgb": torch.ones(2, 3, 8, 12, device=device),
                "embedding": torch.full((2, 2, 5, 9), 2.0, device=device),
            }
            masks = torch.full((2, 1, 7, 11), 0.5, device=device)
            # Independent row-vector world-to-view calculation, including the
            # camera-major/point-batch-major order and a multidimensional grid.
            view_pts = torch.einsum("bpqi,cij->bcpqj", pts, R) + T[None, :, None, None]
            valid = (view_pts[..., 2] > 0)[..., None]
            for camera_type in _CAMERA_TYPES:
                for masked in (False, True):
                    with self.subTest(
                        device=device, camera=camera_type.__name__, masked=masked
                    ):
                        sampled, sampled_masks = project_points_and_sample(
                            pts,
                            feats,
                            camera_type(R=R, T=T, device=device),
                            masks if masked else None,
                        )
                        for name, channels, value in (
                            ("rgb", 3, 1.0),
                            ("embedding", 2, 2.0),
                        ):
                            self.assertEqual(
                                sampled[name].shape, (2, 2, 2, 3, channels)
                            )
                            torch.testing.assert_close(
                                sampled[name],
                                valid.expand_as(sampled[name]).to(pts) * value,
                            )
                        torch.testing.assert_close(
                            sampled_masks,
                            valid.to(pts) * (0.5 if masked else 1.0),
                        )

    def test_depth_and_sequence_masks(self):
        for device in _test_devices():
            pts = torch.tensor(
                [[[0.0, 0.0, 2.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]]], device=device
            ).repeat(2, 1, 1)
            R = torch.eye(3, device=device)[None].repeat(2, 1, 1)
            feats = {"features": torch.ones(2, 2, 8, 12, device=device)}
            masks = torch.full((2, 1, 5, 9), 0.5, device=device)
            for camera_type in _CAMERA_TYPES:
                for masked in (False, True):
                    with self.subTest(
                        device=device, camera=camera_type.__name__, masked=masked
                    ):
                        sampled, sampled_masks = ViewSampler(masked_sampling=masked)(
                            pts=pts,
                            seq_id_pts=["a", "b"],
                            camera=camera_type(R=R, device=device),
                            seq_id_camera=["a", "b"],
                            feats=feats,
                            masks=masks,
                        )
                        expected = torch.zeros_like(sampled_masks)
                        expected[0, 0, 0] = 1.0
                        expected[1, 1, 0] = 1.0
                        torch.testing.assert_close(
                            sampled["features"], expected.expand(-1, -1, -1, 2)
                        )
                        torch.testing.assert_close(
                            sampled_masks, expected * (0.5 if masked else 1.0)
                        )

    def test_depth_gradients(self):
        for device in _test_devices():
            for camera_type in _CAMERA_TYPES:
                for mode in ("nearest", "bilinear", "bicubic"):
                    for masked in (False, True):
                        with self.subTest(
                            device=device,
                            camera=camera_type.__name__,
                            mode=mode,
                            masked=masked,
                        ):
                            pts = torch.tensor(
                                [
                                    [
                                        [0.13, -0.09, 2.0],
                                        [0.12, 0.17, -2.0],
                                        [0.11, -0.08, 0.0],
                                    ]
                                ],
                                device=device,
                                requires_grad=True,
                            )
                            R = torch.eye(3, device=device)[None].requires_grad_()
                            T = torch.zeros(1, 3, device=device, requires_grad=True)
                            feat_map = (
                                torch.linspace(0.1, 2.0, 48, device=device)
                                .reshape(1, 2, 4, 6)
                                .requires_grad_()
                            )
                            mask_map = (
                                torch.linspace(0.2, 0.9, 24, device=device)
                                .reshape(1, 1, 4, 6)
                                .requires_grad_()
                            )
                            camera = camera_type(R=R, T=T, device=device)
                            sampled, sampled_masks = project_points_and_sample(
                                pts,
                                {"features": feat_map},
                                camera,
                                mask_map if masked else None,
                                sampling_mode=mode,
                            )
                            inputs = [pts, R, T, feat_map] + (
                                [mask_map] if masked else []
                            )
                            invalid_loss = (
                                sampled["features"][..., 1:, :].sum()
                                + sampled_masks[..., 1:, :].sum()
                            )
                            for grad in torch.autograd.grad(
                                invalid_loss, inputs, retain_graph=True
                            ):
                                self.assertTrue(torch.isfinite(grad).all())
                                self.assertEqual(torch.count_nonzero(grad).item(), 0)

                            front_grid = camera.transform_points(pts[:, :1], eps=1e-2)[
                                ..., :2
                            ][:, None]
                            expected_feats = ndc_grid_sample(
                                feat_map, front_grid, mode=mode
                            ).permute(0, 2, 3, 1)
                            expected_masks = (
                                ndc_grid_sample(
                                    mask_map, front_grid, mode=mode
                                ).permute(0, 2, 3, 1)
                                if masked
                                else torch.ones_like(sampled_masks[..., :1, :])
                            )
                            torch.testing.assert_close(
                                sampled["features"][..., :1, :], expected_feats
                            )
                            torch.testing.assert_close(
                                sampled_masks[..., :1, :], expected_masks
                            )
                            weights = pts.new_tensor([1.0, 2.0])
                            loss = (
                                sampled["features"][..., :1, :] * weights
                            ).sum() + sampled_masks[..., :1, :].sum()
                            expected_loss = (
                                expected_feats * weights
                            ).sum() + expected_masks.sum()
                            grads = torch.autograd.grad(loss, inputs, retain_graph=True)
                            expected_grads = torch.autograd.grad(expected_loss, inputs)
                            for grad, expected_grad in zip(grads, expected_grads):
                                self.assertTrue(torch.isfinite(grad).all())
                                torch.testing.assert_close(grad, expected_grad)

    def test_depth_gradcheck(self):
        pts = torch.tensor(
            [[[0.13, -0.09, 1.7], [0.12, 0.17, -1.5]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        R = torch.eye(3, dtype=torch.float32)[None].requires_grad_()
        T = torch.zeros(1, 3, dtype=torch.float32, requires_grad=True)
        focal = torch.tensor([[1.1, 0.9]], dtype=torch.float32, requires_grad=True)
        feat_map = (
            torch.linspace(0.1, 2.0, 24, dtype=torch.float32)
            .reshape(1, 1, 4, 6)
            .requires_grad_()
        )
        mask_map = (
            torch.linspace(0.2, 0.9, 24, dtype=torch.float32)
            .reshape(1, 1, 4, 6)
            .requires_grad_()
        )

        def sample(pts, R, T, focal, feat_map, mask_map):
            camera = PerspectiveCameras(
                R=R, T=T, focal_length=focal, principal_point=torch.zeros_like(focal)
            )
            sampled, masks = project_points_and_sample(
                pts, {"features": feat_map}, camera, mask_map
            )
            return sampled["features"], masks

        # Camera transforms use float32, so finite differences need a larger step.
        self.assertTrue(
            torch.autograd.gradcheck(
                sample,
                (pts, R, T, focal, feat_map, mask_map),
                eps=1e-3,
                atol=1e-3,
                rtol=1e-2,
            )
        )

    def test_sampler_aggregator_backward(self):
        for aggregator_type in (IdentityFeatureAggregator, ReductionFeatureAggregator):
            expand_args_fields(aggregator_type)
        for device in _test_devices():
            for aggregator_type in (
                IdentityFeatureAggregator,
                ReductionFeatureAggregator,
            ):
                for masked in (False, True):
                    with self.subTest(
                        device=device,
                        aggregator=aggregator_type.__name__,
                        masked=masked,
                    ):
                        pts = torch.tensor(
                            [[[0.13, -0.09, 2.0], [0.12, 0.17, -2.0], [0.0, 0.0, 0.0]]],
                            device=device,
                            requires_grad=True,
                        )
                        R = (
                            torch.eye(3, device=device)[None]
                            .repeat(2, 1, 1)
                            .requires_grad_()
                        )
                        T = torch.tensor(
                            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                            device=device,
                            requires_grad=True,
                        )
                        feat_map = (
                            torch.linspace(0.1, 2.0, 96, device=device)
                            .reshape(2, 2, 4, 6)
                            .requires_grad_()
                        )
                        mask_map = torch.full(
                            (2, 1, 4, 6), 0.5, device=device, requires_grad=True
                        )
                        camera = PerspectiveCameras(R=R, T=T, device=device)
                        sampled, sampled_masks = ViewSampler(masked_sampling=masked)(
                            pts=pts,
                            seq_id_pts=["a"],
                            camera=camera,
                            seq_id_camera=["a", "a"],
                            feats={"features": feat_map},
                            masks=mask_map,
                        )
                        kwargs = dict(
                            exclude_target_view=False,
                            exclude_target_view_mask_features=False,
                        )
                        if aggregator_type is ReductionFeatureAggregator:
                            kwargs["reduction_functions"] = (ReductionFunction.AVG,)
                        aggregated = aggregator_type(**kwargs)(
                            sampled, sampled_masks, camera=camera, pts=pts
                        )
                        self.assertEqual(
                            torch.count_nonzero(aggregated[..., 1, :]).item(), 0
                        )
                        if aggregator_type is IdentityFeatureAggregator:
                            torch.testing.assert_close(aggregated, sampled["features"])
                        else:
                            weights = sampled_masks / sampled_masks.sum(
                                dim=1, keepdim=True
                            ).clamp_min(1e-2)
                            torch.testing.assert_close(
                                aggregated,
                                (sampled["features"] * weights).sum(
                                    dim=1, keepdim=True
                                ),
                            )
                        loss_weights = torch.arange(
                            1, aggregated.numel() + 1, device=device
                        ).reshape_as(aggregated)
                        (aggregated * loss_weights).sum().backward()
                        for tensor in (pts, R, T, feat_map) + (
                            (mask_map,)
                            if masked and aggregator_type is ReductionFeatureAggregator
                            else ()
                        ):
                            self.assertTrue(torch.isfinite(tensor.grad).all())
                        self.assertEqual(torch.count_nonzero(pts.grad[:, 1]).item(), 0)
                        self.assertGreater(torch.count_nonzero(feat_map.grad).item(), 0)

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
