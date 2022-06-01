# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.renderer.splatter_blend import (
    _compute_occlusion_layers,
    _compute_splatted_colors_and_weights,
    _compute_splatting_colors_and_weights,
    _get_splat_kernel_normalization,
    _normalize_and_compose_all_layers,
    _offset_splats,
    _precompute,
    _prepare_pixels_and_colors,
)

from .common_testing import TestCaseMixin

offsets = torch.tensor(
    [
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 0],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ],
    device=torch.device("cpu"),
)


def compute_splatting_colors_and_weights_naive(pixel_coords_screen, colors, sigma):
    normalizer = float(_get_splat_kernel_normalization(offsets))
    N, H, W, K, _ = colors.shape
    splat_weights_and_colors = torch.zeros((N, H, W, K, 9, 5))
    for n in range(N):
        for h in range(H):
            for w in range(W):
                for k in range(K):
                    q_xy = pixel_coords_screen[n, h, w, k]
                    q_to_px_center = torch.floor(q_xy) - q_xy + 0.5
                    color = colors[n, h, w, k]
                    alpha = colors[n, h, w, k, 3:4]
                    for d in range(9):
                        dist_p_q = torch.sum((q_to_px_center + offsets[d]) ** 2)
                        splat_weight = (
                            alpha * torch.exp(-dist_p_q / (2 * sigma**2)) * normalizer
                        )
                        splat_color = splat_weight * color
                        splat_weights_and_colors[n, h, w, k, d, :4] = splat_color
                        splat_weights_and_colors[n, h, w, k, d, 4:5] = splat_weight
    return splat_weights_and_colors


class TestPrecompute(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        self.results_cpu = _precompute((2, 3, 4, 5), torch.device("cpu"))
        self.results1_cpu = _precompute((1, 1, 1, 1), torch.device("cpu"))

    def test_offsets(self):
        self.assertClose(self.results_cpu[2].shape, offsets.shape, atol=0)
        self.assertClose(self.results_cpu[2], offsets, atol=0)

        # Offsets should be independent of input_size.
        self.assertClose(self.results_cpu[2], self.results1_cpu[2], atol=0)

    def test_crops_h(self):
        target_crops_h1 = torch.tensor(
            [
                # chennels being offset:
                # R  G  B  A  W(eight)
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
            ]
            * 3,  # 3 because we're aiming at (N, H, W+2, K, 9, 5) with W=1.
            device=torch.device("cpu"),
        ).reshape(1, 1, 3, 1, 9, 5)
        self.assertClose(self.results1_cpu[0], target_crops_h1, atol=0)

        target_crops_h_base = target_crops_h1[0, 0, 0]
        target_crops_h = torch.cat(
            [target_crops_h_base, target_crops_h_base + 1, target_crops_h_base + 2],
            dim=0,
        )

        # Check that we have the right shape, and (after broadcasting) it has the right
        # values. These should be repeated (tiled) for each n and k.
        self.assertClose(
            self.results_cpu[0].shape, torch.tensor([2, 3, 6, 5, 9, 5]), atol=0
        )
        for n in range(2):
            for w in range(6):
                for k in range(5):
                    self.assertClose(
                        self.results_cpu[0][n, :, w, k],
                        target_crops_h,
                    )

    def test_crops_w(self):
        target_crops_w1 = torch.tensor(
            [
                # chennels being offset:
                # R  G  B  A  W(eight)
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
                [2, 2, 2, 2, 2],
            ],
            device=torch.device("cpu"),
        ).reshape(1, 1, 1, 1, 9, 5)
        self.assertClose(self.results1_cpu[1], target_crops_w1)

        target_crops_w_base = target_crops_w1[0, 0, 0]
        target_crops_w = torch.cat(
            [
                target_crops_w_base,
                target_crops_w_base + 1,
                target_crops_w_base + 2,
                target_crops_w_base + 3,
            ],
            dim=0,
        )  # Each w value needs an increment.

        # Check that we have the right shape, and (after broadcasting) it has the right
        # values. These should be repeated (tiled) for each n and k.
        self.assertClose(self.results_cpu[1].shape, torch.tensor([2, 3, 4, 5, 9, 5]))
        for n in range(2):
            for h in range(3):
                for k in range(5):
                    self.assertClose(
                        self.results_cpu[1][n, h, :, k],
                        target_crops_w,
                        atol=0,
                    )


class TestPreparPixelsAndColors(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        N, H, W, K = 2, 3, 4, 5
        self.pixel_coords_cameras = torch.randn(
            (N, H, W, K, 3), device=self.device, requires_grad=True
        )
        self.colors_before = torch.rand((N, H, W, K, 3), device=self.device)
        self.cameras = FoVPerspectiveCameras(device=self.device)
        self.background_mask = torch.rand((N, H, W, K), device=self.device) < 0.5
        self.pixel_coords_screen, self.colors_after = _prepare_pixels_and_colors(
            self.pixel_coords_cameras,
            self.colors_before,
            self.cameras,
            self.background_mask,
        )

    def test_background_z(self):
        self.assertTrue(
            torch.all(self.pixel_coords_screen[..., 2][self.background_mask] == 1.0)
        )

    def test_background_alpha(self):
        self.assertTrue(
            torch.all(self.colors_after[..., 3][self.background_mask] == 0.0)
        )


class TestGetSplatKernelNormalization(TestCaseMixin, unittest.TestCase):
    def test_splat_kernel_normalization(self):
        self.assertAlmostEqual(
            float(_get_splat_kernel_normalization(offsets)), 0.6503, places=3
        )
        self.assertAlmostEqual(
            float(_get_splat_kernel_normalization(offsets, 0.01)), 1.05, places=3
        )
        with self.assertRaisesRegex(ValueError, "Only positive standard deviations"):
            _get_splat_kernel_normalization(offsets, 0)


class TestComputeOcclusionLayers(TestCaseMixin, unittest.TestCase):
    def test_single_layer(self):
        # If there's only one layer, all splats must be on the surface level.
        N, H, W, K = 2, 3, 4, 1
        q_depth = torch.rand(N, H, W, K)
        occlusion_layers = _compute_occlusion_layers(q_depth)
        self.assertClose(occlusion_layers, torch.zeros(N, H, W, 9).long(), atol=0.0)

    def test_all_equal(self):
        # If all q-vals are equal, then all splats must be on the surface level.
        N, H, W, K = 2, 3, 4, 5
        q_depth = torch.ones((N, H, W, K)) * 0.1234
        occlusion_layers = _compute_occlusion_layers(q_depth)
        self.assertClose(occlusion_layers, torch.zeros(N, H, W, 9).long(), atol=0.0)

    def test_mid_to_top_level_splatting(self):
        # Check that occlusion buffers get accumulated as expected when the splatting
        # and splatted pixels are co-surface on different intersection layers.
        # This test will make best sense with accompanying Fig. 4 from "Differentiable
        # Surface Rendering via Non-differentiable Sampling" by Cole et al.
        for direction, offset in enumerate(offsets):
            if direction == 4:
                continue  # Skip self-splatting which is always co-surface.

            depths = torch.zeros(1, 3, 3, 3)

            # This is our q, the pixel splatted onto, in the center of the image.
            depths[0, 1, 1] = torch.tensor([0.71, 0.8, 1.0])

            # This is our p, the splatting pixel.
            depths[0, offset[0] + 1, offset[1] + 1] = torch.tensor([0.5, 0.7, 0.9])

            occlusion_layers = _compute_occlusion_layers(depths)

            # Check that we computed that it is the middle layer of p that is co-
            # surface with q. (1, 1) is the id of q in the depth array, and offset_id
            # is the id of p's direction w.r.t. q.
            psurfaceid_onto_q = occlusion_layers[0, 1, 1, direction]
            self.assertEqual(int(psurfaceid_onto_q), 1)

            # Conversely, if we swap p and q, we have a top-level splatting onto
            # mid-level. offset + 1 is the id of p, and 8-offset_id is the id of
            # q's direction w.r.t. p (e.g. if p is [-1, -1] w.r.t. q, then q is
            # [1, 1] w.r.t. p; we use the ids of these two directions in the offsets
            # array).
            qsurfaceid_onto_p = occlusion_layers[
                0, offset[0] + 1, offset[1] + 1, 8 - direction
            ]
            self.assertEqual(int(qsurfaceid_onto_p), -1)


class TestComputeSplattingColorsAndWeights(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        self.N, self.H, self.W, self.K = 2, 3, 4, 5
        self.pixel_coords_screen = (
            torch.stack(
                meshgrid_ij(torch.arange(self.H), torch.arange(self.W)),
                dim=-1,
            )
            .reshape(1, self.H, self.W, 1, 2)
            .expand(self.N, self.H, self.W, self.K, 2)
            .float()
            + 0.5
        )
        self.colors = torch.ones((self.N, self.H, self.W, self.K, 4))

    def test_all_equal(self):
        # If all colors are equal and on a regular grid, all weights and reweighted
        # colors should be equal given a specific splatting direction.
        splatting_colors_and_weights = _compute_splatting_colors_and_weights(
            self.pixel_coords_screen, self.colors * 0.2345, sigma=0.5, offsets=offsets
        )

        # Splatting directly to the top/bottom/left/right should have the same strenght.
        non_diag_splats = splatting_colors_and_weights[
            :, :, :, :, torch.tensor([1, 3, 5, 7])
        ]

        # Same for diagonal splats.
        diag_splats = splatting_colors_and_weights[
            :, :, :, :, torch.tensor([0, 2, 6, 8])
        ]

        # And for self-splats.
        self_splats = splatting_colors_and_weights[:, :, :, :, torch.tensor([4])]

        for splats in non_diag_splats, diag_splats, self_splats:
            # Colors should be equal.
            self.assertTrue(torch.all(splats[..., :4] == splats[0, 0, 0, 0, 0, 0]))

            # Weights should be equal.
            self.assertTrue(torch.all(splats[..., 4] == splats[0, 0, 0, 0, 0, 4]))

        # Non-diagonal weights should be greater than diagonal weights.
        self.assertGreater(
            non_diag_splats[0, 0, 0, 0, 0, 0], diag_splats[0, 0, 0, 0, 0, 0]
        )

        # Self-splats should be strongest of all.
        self.assertGreater(
            self_splats[0, 0, 0, 0, 0, 0], non_diag_splats[0, 0, 0, 0, 0, 0]
        )

        # Splatting colors should be reweighted proportionally to their splat weights.
        diag_self_color_ratio = (
            diag_splats[0, 0, 0, 0, 0, 0] / self_splats[0, 0, 0, 0, 0, 0]
        )
        diag_self_weight_ratio = (
            diag_splats[0, 0, 0, 0, 0, 4] / self_splats[0, 0, 0, 0, 0, 4]
        )
        self.assertEqual(diag_self_color_ratio, diag_self_weight_ratio)

        non_diag_self_color_ratio = (
            non_diag_splats[0, 0, 0, 0, 0, 0] / self_splats[0, 0, 0, 0, 0, 0]
        )
        non_diag_self_weight_ratio = (
            non_diag_splats[0, 0, 0, 0, 0, 4] / self_splats[0, 0, 0, 0, 0, 4]
        )
        self.assertEqual(non_diag_self_color_ratio, non_diag_self_weight_ratio)

    def test_zero_alpha_zero_weight(self):
        # Pixels with zero alpha do no splatting, but should still be splatted on.
        colors = self.colors.clone()
        colors[0, 1, 1, 0, 3] = 0.0
        splatting_colors_and_weights = _compute_splatting_colors_and_weights(
            self.pixel_coords_screen, colors, sigma=0.5, offsets=offsets
        )

        # The transparent pixel should do no splatting.
        self.assertTrue(torch.all(splatting_colors_and_weights[0, 1, 1, 0] == 0.0))

        # Splatting *onto* the transparent pixel should be unaffected.
        reference_weights_colors = splatting_colors_and_weights[0, 1, 1, 1]
        for direction, offset in enumerate(offsets):
            if direction == 4:
                continue  # Ignore self-splats
            # We invert the direction to get the right (h, w, d) coordinate of each
            # pixel splatting *onto* the pixel with zero alpha.
            self.assertClose(
                splatting_colors_and_weights[
                    0, 1 + offset[0], 1 + offset[1], 0, 8 - direction
                ],
                reference_weights_colors[8 - direction],
                atol=0.001,
            )

    def test_random_inputs(self):
        pixel_coords_screen = (
            self.pixel_coords_screen
            + torch.randn((self.N, self.H, self.W, self.K, 2)) * 0.1
        )
        colors = torch.rand((self.N, self.H, self.W, self.K, 4))
        splatting_colors_and_weights = _compute_splatting_colors_and_weights(
            pixel_coords_screen, colors, sigma=0.5, offsets=offsets
        )
        naive_colors_and_weights = compute_splatting_colors_and_weights_naive(
            pixel_coords_screen, colors, sigma=0.5
        )

        self.assertClose(
            splatting_colors_and_weights, naive_colors_and_weights, atol=0.01
        )


class TestOffsetSplats(TestCaseMixin, unittest.TestCase):
    def test_offset(self):
        device = torch.device("cuda:0")
        N, H, W, K = 2, 3, 4, 5
        colors_and_weights = torch.rand((N, H, W, K, 9, 5), device=device)
        crop_ids_h, crop_ids_w, _ = _precompute((N, H, W, K), device=device)
        offset_colors_and_weights = _offset_splats(
            colors_and_weights, crop_ids_h, crop_ids_w
        )

        # Check each splatting direction individually, for clarity.
        # offset_x, offset_y = (-1, -1)
        direction = 0
        self.assertClose(
            offset_colors_and_weights[:, 1:, 1:, :, direction],
            colors_and_weights[:, :-1, :-1, :, direction],
            atol=0.001,
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, 0, :, :, direction] == 0.0)
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, :, 0, :, direction] == 0.0)
        )

        # offset_x, offset_y = (-1, 0)
        direction = 1
        self.assertClose(
            offset_colors_and_weights[:, :, 1:, :, direction],
            colors_and_weights[:, :, :-1, :, direction],
            atol=0.001,
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, :, 0, :, direction] == 0.0)
        )

        # offset_x, offset_y = (-1, 1)
        direction = 2
        self.assertClose(
            offset_colors_and_weights[:, :-1, 1:, :, direction],
            colors_and_weights[:, 1:, :-1, :, direction],
            atol=0.001,
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, -1, :, :, direction] == 0.0)
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, :, 0, :, direction] == 0.0)
        )

        # offset_x, offset_y = (0, -1)
        direction = 3
        self.assertClose(
            offset_colors_and_weights[:, 1:, :, :, direction],
            colors_and_weights[:, :-1, :, :, direction],
            atol=0.001,
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, 0, :, :, direction] == 0.0)
        )

        # self-splat
        direction = 4
        self.assertClose(
            offset_colors_and_weights[..., direction, :],
            colors_and_weights[..., direction, :],
            atol=0.001,
        )

        # offset_x, offset_y = (0, 1)
        direction = 5
        self.assertClose(
            offset_colors_and_weights[:, :-1, :, :, direction],
            colors_and_weights[:, 1:, :, :, direction],
            atol=0.001,
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, -1, :, :, direction] == 0.0)
        )

        # offset_x, offset_y = (1, -1)
        direction = 6
        self.assertClose(
            offset_colors_and_weights[:, 1:, :-1, :, direction],
            colors_and_weights[:, :-1, 1:, :, direction],
            atol=0.001,
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, 0, :, :, direction] == 0.0)
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, :, -1, :, direction] == 0.0)
        )

        # offset_x, offset_y = (1, 0)
        direction = 7
        self.assertClose(
            offset_colors_and_weights[:, :, :-1, :, direction],
            colors_and_weights[:, :, 1:, :, direction],
            atol=0.001,
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, :, -1, :, direction] == 0.0)
        )

        # offset_x, offset_y = (1, 1)
        direction = 8
        self.assertClose(
            offset_colors_and_weights[:, :-1, :-1, :, direction],
            colors_and_weights[:, 1:, 1:, :, direction],
            atol=0.001,
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, -1, :, :, direction] == 0.0)
        )
        self.assertTrue(
            torch.all(offset_colors_and_weights[:, :, -1, :, direction] == 0.0)
        )


class TestComputeSplattedColorsAndWeights(TestCaseMixin, unittest.TestCase):
    def test_accumulation_background(self):
        # Set occlusion_layers to all -1, so all splats are background splats.
        splat_colors_and_weights = torch.rand((1, 1, 1, 3, 9, 5))
        occlusion_layers = torch.zeros((1, 1, 1, 9)) - 1
        splatted_colors, splatted_weights = _compute_splatted_colors_and_weights(
            occlusion_layers, splat_colors_and_weights
        )

        # Foreground splats (there are none).
        self.assertClose(
            splatted_colors[0, 0, 0, :, 0],
            torch.zeros((4)),
            atol=0.001,
        )

        # Surface splats (there are none).
        self.assertClose(
            splatted_colors[0, 0, 0, :, 1],
            torch.zeros((4)),
            atol=0.001,
        )

        # Background splats.
        self.assertClose(
            splatted_colors[0, 0, 0, :, 2],
            splat_colors_and_weights[0, 0, 0, :, :, :4].sum(dim=0).sum(dim=0),
            atol=0.001,
        )

    def test_accumulation_middle(self):
        # Set occlusion_layers to all 0, so top splats are co-surface with splatted
        # pixels. Thus, the top splatting layer should be accumulated to surface, and
        # all other layers to background.
        splat_colors_and_weights = torch.rand((1, 1, 1, 3, 9, 5))
        occlusion_layers = torch.zeros((1, 1, 1, 9))
        splatted_colors, splatted_weights = _compute_splatted_colors_and_weights(
            occlusion_layers, splat_colors_and_weights
        )

        # Foreground splats (there are none).
        self.assertClose(
            splatted_colors[0, 0, 0, :, 0],
            torch.zeros((4)),
            atol=0.001,
        )

        # Surface splats
        self.assertClose(
            splatted_colors[0, 0, 0, :, 1],
            splat_colors_and_weights[0, 0, 0, 0, :, :4].sum(dim=0),
            atol=0.001,
        )

        # Background splats
        self.assertClose(
            splatted_colors[0, 0, 0, :, 2],
            splat_colors_and_weights[0, 0, 0, 1:, :, :4].sum(dim=0).sum(dim=0),
            atol=0.001,
        )

    def test_accumulation_foreground(self):
        # Set occlusion_layers to all 1. Then the top splatter is a foreground
        # splatter, mid splatter is surface, and bottom splatter is background.
        splat_colors_and_weights = torch.rand((1, 1, 1, 3, 9, 5))
        occlusion_layers = torch.zeros((1, 1, 1, 9)) + 1
        splatted_colors, splatted_weights = _compute_splatted_colors_and_weights(
            occlusion_layers, splat_colors_and_weights
        )

        # Foreground splats
        self.assertClose(
            splatted_colors[0, 0, 0, :, 0],
            splat_colors_and_weights[0, 0, 0, 0:1, :, :4].sum(dim=0).sum(dim=0),
            atol=0.001,
        )

        # Surface splats
        self.assertClose(
            splatted_colors[0, 0, 0, :, 1],
            splat_colors_and_weights[0, 0, 0, 1:2, :, :4].sum(dim=0).sum(dim=0),
            atol=0.001,
        )

        # Background splats
        self.assertClose(
            splatted_colors[0, 0, 0, :, 2],
            splat_colors_and_weights[0, 0, 0, 2:3, :, :4].sum(dim=0).sum(dim=0),
            atol=0.001,
        )


class TestNormalizeAndComposeAllLayers(TestCaseMixin, unittest.TestCase):
    def test_background_color(self):
        # Background should always have alpha=0, and the chosen RGB.
        N, H, W = 2, 3, 4
        # Make a mask with background in the zeroth row of the first image.
        bg_mask = torch.zeros([N, H, W, 1, 1])
        bg_mask[0, :, 0] = 1

        bg_color = torch.tensor([0.2, 0.3, 0.4])

        color_layers = torch.rand((N, H, W, 4, 3)) * (1 - bg_mask)
        color_weights = torch.rand((N, H, W, 1, 3)) * (1 - bg_mask)

        colors = _normalize_and_compose_all_layers(
            bg_color, color_layers, color_weights
        )

        # Background RGB should be .2, .3, .4, and alpha should be 0.
        self.assertClose(
            torch.masked_select(colors, bg_mask.bool()[..., 0]),
            torch.tensor([0.2, 0.3, 0.4, 0, 0.2, 0.3, 0.4, 0, 0.2, 0.3, 0.4, 0.0]),
            atol=0.001,
        )

    def test_compositing_opaque(self):
        # When all colors are opaque, only the foreground layer should be visible.
        N, H, W = 2, 3, 4
        color_layers = torch.rand((N, H, W, 4, 3))
        color_layers[..., 3, :] = 1.0
        color_weights = torch.ones((N, H, W, 1, 3))

        out_colors = _normalize_and_compose_all_layers(
            torch.tensor([0.0, 0.0, 0.0]), color_layers, color_weights
        )
        self.assertClose(out_colors, color_layers[..., 0], atol=0.001)

    def test_compositing_transparencies(self):
        # When foreground layer is transparent and surface and bg are semi-transparent,
        # we should return a  mix of the two latter.
        N, H, W = 2, 3, 4
        color_layers = torch.rand((N, H, W, 4, 3))
        color_layers[..., 3, 0] = 0.1  # fg
        color_layers[..., 3, 1] = 0.2  # surface
        color_layers[..., 3, 2] = 0.3  # bg
        color_weights = torch.ones((N, H, W, 1, 3))

        out_colors = _normalize_and_compose_all_layers(
            torch.tensor([0.0, 0.0, 0.0]), color_layers, color_weights
        )
        self.assertClose(
            out_colors,
            color_layers[..., 0]
            + 0.9 * (color_layers[..., 1] + 0.8 * color_layers[..., 2]),
        )
