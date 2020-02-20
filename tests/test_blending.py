#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import unittest
import torch

from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments


def sigmoid_blend_naive(colors, fragments, blend_params):
    """
    Naive for loop based implementation of distance based alpha calculation.
    Only for test purposes.
    """
    pix_to_face = fragments.pix_to_face
    dists = fragments.dists
    sigma = blend_params.sigma

    N, H, W, K = pix_to_face.shape
    device = pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=device)

    for n in range(N):
        for h in range(H):
            for w in range(W):
                alpha = 1.0

                # Loop over k faces and calculate 2D distance based probability
                # map.
                for k in range(K):
                    if pix_to_face[n, h, w, k] >= 0:
                        prob = torch.sigmoid(-dists[n, h, w, k] / sigma)
                        alpha *= 1.0 - prob  # cumulative product
                pixel_colors[n, h, w, :3] = colors[n, h, w, 0, :]
                pixel_colors[n, h, w, 3] = 1.0 - alpha

    pixel_colors = torch.clamp(pixel_colors, min=0, max=1.0)
    return torch.flip(pixel_colors, [1])


def softmax_blend_naive(colors, fragments, blend_params):
    """
    Naive for loop based implementation of softmax blending.
    Only for test purposes.
    """
    pix_to_face = fragments.pix_to_face
    dists = fragments.dists
    zbuf = fragments.zbuf
    sigma = blend_params.sigma
    gamma = blend_params.gamma

    N, H, W, K = pix_to_face.shape
    device = pix_to_face.device
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=device)

    # Near and far clipping planes
    zfar = 100.0
    znear = 1.0

    bk_color = blend_params.background_color
    if not torch.is_tensor(bk_color):
        bk_color = torch.tensor(bk_color, dtype=colors.dtype, device=device)

    # Background color component
    delta = np.exp(1e-10 / gamma) * 1e-10
    delta = torch.tensor(delta).to(device=device)

    for n in range(N):
        for h in range(H):
            for w in range(W):
                alpha = 1.0
                weights_k = torch.zeros(K)
                zmax = 0.0

                # Loop over K to find max z.
                for k in range(K):
                    if pix_to_face[n, h, w, k] >= 0:
                        zinv = (zfar - zbuf[n, h, w, k]) / (zfar - znear)
                        if zinv > zmax:
                            zmax = zinv

                # Loop over K faces to calculate 2D distance based probability
                # map and zbuf based weights for colors.
                for k in range(K):
                    if pix_to_face[n, h, w, k] >= 0:
                        zinv = (zfar - zbuf[n, h, w, k]) / (zfar - znear)
                        prob = torch.sigmoid(-dists[n, h, w, k] / sigma)
                        alpha *= 1.0 - prob  # cumulative product
                        weights_k[k] = prob * torch.exp((zinv - zmax) / gamma)

                denom = weights_k.sum() + delta
                weights = weights_k / denom
                cols = (weights[..., None] * colors[n, h, w, :, :]).sum(dim=0)
                pixel_colors[n, h, w, :3] = cols
                pixel_colors[n, h, w, :3] += (delta / denom) * bk_color
                pixel_colors[n, h, w, 3] = 1.0 - alpha

    pixel_colors = torch.clamp(pixel_colors, min=0, max=1.0)
    return torch.flip(pixel_colors, [1])


class TestBlending(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_hard_rgb_blend(self):
        N, H, W, K = 5, 10, 10, 20
        pix_to_face = torch.ones((N, H, W, K))
        bary_coords = torch.ones((N, H, W, K, 3))
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=bary_coords,
            zbuf=pix_to_face,  # dummy
            dists=pix_to_face,  # dummy
        )
        colors = bary_coords.clone()
        top_k = torch.randn((K, 3))
        colors[..., :, :] = top_k
        images = hard_rgb_blend(colors, fragments)
        expected_vals = torch.ones((N, H, W, 4))
        pix_cols = torch.ones_like(expected_vals[..., :3]) * top_k[0, :]
        expected_vals[..., :3] = pix_cols
        self.assertTrue(torch.allclose(images, expected_vals))

    def test_sigmoid_alpha_blend(self):
        """
        Test outputs of sigmoid alpha blend tensorised function match those of
        the naive iterative version. Also check gradients match.
        """

        # Create dummy outputs of rasterization simulating a cube in the centre
        # of the image with surrounding padded values.
        N, S, K = 1, 8, 2
        pix_to_face = -torch.ones((N, S, S, K), dtype=torch.int64)
        h = int(S / 2)
        pix_to_face_full = torch.randint(size=(N, h, h, K), low=0, high=100)
        s = int(S / 4)
        e = int(0.75 * S)
        pix_to_face[:, s:e, s:e, :] = pix_to_face_full
        bary_coords = torch.ones((N, S, S, K, 3))

        # randomly flip the sign of the distance
        # (-) means inside triangle, (+) means outside triangle.
        random_sign_flip = torch.rand((N, S, S, K))
        random_sign_flip[random_sign_flip > 0.5] *= -1.0
        dists = torch.randn(size=(N, S, S, K))
        dists1 = dists * random_sign_flip
        dists2 = dists1.clone()
        dists1.requires_grad = True
        dists2.requires_grad = True
        colors = torch.randn_like(bary_coords)
        fragments1 = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=bary_coords,  # dummy
            zbuf=pix_to_face,  # dummy
            dists=dists1,
        )
        fragments2 = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=bary_coords,  # dummy
            zbuf=pix_to_face,  # dummy
            dists=dists2,
        )
        blend_params = BlendParams(sigma=2e-1)
        images = sigmoid_alpha_blend(colors, fragments1, blend_params)
        images_naive = sigmoid_blend_naive(colors, fragments2, blend_params)
        self.assertTrue(torch.allclose(images, images_naive))

        torch.manual_seed(231)
        images.sum().backward()
        self.assertTrue(hasattr(dists1, "grad"))
        images_naive.sum().backward()
        self.assertTrue(hasattr(dists2, "grad"))

        self.assertTrue(torch.allclose(dists1.grad, dists2.grad, rtol=1e-5))

    def test_softmax_rgb_blend(self):
        # Create dummy outputs of rasterization simulating a cube in the centre
        # of the image with surrounding padded values.
        N, S, K = 1, 8, 2
        pix_to_face = -torch.ones((N, S, S, K), dtype=torch.int64)
        h = int(S / 2)
        pix_to_face_full = torch.randint(size=(N, h, h, K), low=0, high=100)
        s = int(S / 4)
        e = int(0.75 * S)
        pix_to_face[:, s:e, s:e, :] = pix_to_face_full
        bary_coords = torch.ones((N, S, S, K, 3))

        random_sign_flip = torch.rand((N, S, S, K))
        random_sign_flip[random_sign_flip > 0.5] *= -1.0
        zbuf1 = torch.randn(size=(N, S, S, K))

        # randomly flip the sign of the distance
        # (-) means inside triangle, (+) means outside triangle.
        dists1 = torch.randn(size=(N, S, S, K)) * random_sign_flip
        dists2 = dists1.clone()
        zbuf2 = zbuf1.clone()
        dists1.requires_grad = True
        dists2.requires_grad = True
        zbuf1.requires_grad = True
        zbuf2.requires_grad = True
        colors = torch.randn_like(bary_coords)
        fragments1 = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=bary_coords,  # dummy
            zbuf=zbuf1,
            dists=dists1,
        )
        fragments2 = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=bary_coords,  # dummy
            zbuf=zbuf2,
            dists=dists2,
        )
        blend_params = BlendParams(sigma=1e-1)
        images = softmax_rgb_blend(colors, fragments1, blend_params)
        images_naive = softmax_blend_naive(colors, fragments2, blend_params)
        self.assertTrue(torch.allclose(images, images_naive))

        # Check gradients.
        images.sum().backward()
        self.assertTrue(hasattr(dists1, "grad"))
        self.assertTrue(hasattr(zbuf1, "grad"))
        images_naive.sum().backward()
        self.assertTrue(hasattr(dists2, "grad"))
        self.assertTrue(hasattr(zbuf2, "grad"))

        self.assertTrue(torch.allclose(dists1.grad, dists2.grad, atol=2e-5))
        self.assertTrue(torch.allclose(zbuf1.grad, zbuf2.grad, atol=2e-5))

    def test_blend_params(self):
        """Test colour parameter of BlendParams().
        Assert default value is equal to same passed value (1.0, 1.0, 1.0).
        """
        bp1, bp2 = BlendParams(), BlendParams(background_color=(1.0, 1.0, 1.0))
        self.assertAlmostEqual(bp1.background_color, bp2.background_color)

        background_color = torch.tensor((0.5, 0.5, 0.5))
        bp3 = BlendParams(background_color=background_color)
