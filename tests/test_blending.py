# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.splatter_blend import SplatterBlender

from .common_testing import TestCaseMixin


def sigmoid_blend_naive_loop(colors, fragments, blend_params):
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

    return pixel_colors


def sigmoid_alpha_blend_vectorized(colors, fragments, blend_params) -> torch.Tensor:
    N, H, W, K = fragments.pix_to_face.shape
    pixel_colors = torch.ones((N, H, W, 4), dtype=colors.dtype, device=colors.device)
    mask = fragments.pix_to_face >= 0
    prob = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask
    pixel_colors[..., :3] = colors[..., 0, :]
    pixel_colors[..., 3] = 1.0 - torch.prod((1.0 - prob), dim=-1)
    return pixel_colors


def sigmoid_blend_naive_loop_backward(grad_images, images, fragments, blend_params):
    pix_to_face = fragments.pix_to_face
    dists = fragments.dists
    sigma = blend_params.sigma

    N, H, W, K = pix_to_face.shape
    device = pix_to_face.device
    grad_distances = torch.zeros((N, H, W, K), dtype=dists.dtype, device=device)

    for n in range(N):
        for h in range(H):
            for w in range(W):
                alpha = 1.0 - images[n, h, w, 3]
                grad_alpha = grad_images[n, h, w, 3]
                # Loop over k faces and calculate 2D distance based probability
                # map.
                for k in range(K):
                    if pix_to_face[n, h, w, k] >= 0:
                        prob = torch.sigmoid(-dists[n, h, w, k] / sigma)
                        grad_distances[n, h, w, k] = (
                            grad_alpha * (-1.0 / sigma) * prob * alpha
                        )
    return grad_distances


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
    eps = 1e-10

    bk_color = blend_params.background_color
    if not torch.is_tensor(bk_color):
        bk_color = torch.tensor(bk_color, dtype=colors.dtype, device=device)

    for n in range(N):
        for h in range(H):
            for w in range(W):
                alpha = 1.0
                weights_k = torch.zeros(K, device=device)
                zmax = torch.tensor(0.0, device=device)

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

                # Clamp to ensure delta is never 0
                delta = torch.exp((eps - zmax) / blend_params.gamma).clamp(min=eps)
                delta = delta.to(device)
                denom = weights_k.sum() + delta
                cols = (weights_k[..., None] * colors[n, h, w, :, :]).sum(dim=0)
                pixel_colors[n, h, w, :3] = cols + delta * bk_color
                pixel_colors[n, h, w, :3] /= denom
                pixel_colors[n, h, w, 3] = 1.0 - alpha

    return pixel_colors


class TestBlending(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def _compare_impls(
        self, fn1, fn2, args1, args2, grad_var1=None, grad_var2=None, compare_grads=True
    ):
        out1 = fn1(*args1)
        out2 = fn2(*args2)
        self.assertClose(out1.cpu()[..., 3], out2.cpu()[..., 3], atol=1e-7)

        # Check gradients
        if not compare_grads:
            return

        grad_out = torch.randn_like(out1)
        (out1 * grad_out).sum().backward()
        self.assertTrue(hasattr(grad_var1, "grad"))

        (out2 * grad_out).sum().backward()
        self.assertTrue(hasattr(grad_var2, "grad"))

        self.assertClose(grad_var1.grad.cpu(), grad_var2.grad.cpu(), atol=2e-5)

    def test_hard_rgb_blend(self):
        N, H, W, K = 5, 10, 10, 20
        pix_to_face = torch.randint(low=-1, high=100, size=(N, H, W, K))
        bary_coords = torch.ones((N, H, W, K, 3))
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=bary_coords,
            zbuf=pix_to_face,  # dummy
            dists=pix_to_face,  # dummy
        )
        colors = torch.randn((N, H, W, K, 3))
        blend_params = BlendParams(1e-4, 1e-4, (0.5, 0.5, 1))
        images = hard_rgb_blend(colors, fragments, blend_params)

        # Examine if the foreground colors are correct.
        is_foreground = pix_to_face[..., 0] >= 0
        self.assertClose(images[is_foreground][:, :3], colors[is_foreground][..., 0, :])

        # Examine if the background colors are correct.
        for i in range(3):  # i.e. RGB
            channel_color = blend_params.background_color[i]
            self.assertTrue(images[~is_foreground][..., i].eq(channel_color).all())

        # Examine the alpha channel
        self.assertClose(images[..., 3], (pix_to_face[..., 0] >= 0).float())

    def test_sigmoid_alpha_blend_manual_gradients(self):
        # Create dummy outputs of rasterization
        torch.manual_seed(231)
        F = 32  # number of faces in the mesh
        # The python loop version is really slow so only using small input sizes.
        N, S, K = 2, 3, 2
        device = torch.device("cuda")
        pix_to_face = torch.randint(F + 1, size=(N, S, S, K), device=device) - 1
        colors = torch.randn((N, S, S, K, 3), device=device)
        empty = torch.tensor([], device=device)

        # # randomly flip the sign of the distance
        # # (-) means inside triangle, (+) means outside triangle.
        random_sign_flip = torch.rand((N, S, S, K))
        random_sign_flip[random_sign_flip > 0.5] *= -1.0
        dists = torch.randn(size=(N, S, S, K), requires_grad=True, device=device)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=empty,  # dummy
            zbuf=empty,  # dummy
            dists=dists,
        )
        blend_params = BlendParams(sigma=1e-3)
        pix_cols = sigmoid_blend_naive_loop(colors, fragments, blend_params)
        grad_out = torch.randn_like(pix_cols)

        # Backward pass
        pix_cols.backward(grad_out)
        grad_dists = sigmoid_blend_naive_loop_backward(
            grad_out, pix_cols, fragments, blend_params
        )
        self.assertTrue(torch.allclose(dists.grad, grad_dists, atol=1e-7))

    def test_sigmoid_alpha_blend_python(self):
        """
        Test outputs of python tensorised function and python loop
        """

        # Create dummy outputs of rasterization
        torch.manual_seed(231)
        F = 32  # number of faces in the mesh
        # The python loop version is really slow so only using small input sizes.
        N, S, K = 1, 4, 1
        device = torch.device("cuda")
        pix_to_face = torch.randint(low=-1, high=F, size=(N, S, S, K), device=device)
        colors = torch.randn((N, S, S, K, 3), device=device)
        empty = torch.tensor([], device=device)

        dists1 = torch.randn(size=(N, S, S, K), device=device)
        dists2 = dists1.clone()
        dists1.requires_grad = True
        dists2.requires_grad = True

        fragments1 = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=empty,  # dummy
            zbuf=empty,  # dummy
            dists=dists1,
        )
        fragments2 = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=empty,  # dummy
            zbuf=empty,  # dummy
            dists=dists2,
        )

        blend_params = BlendParams(sigma=1e-2)
        args1 = (colors, fragments1, blend_params)
        args2 = (colors, fragments2, blend_params)

        self._compare_impls(
            sigmoid_alpha_blend,
            sigmoid_alpha_blend_vectorized,
            args1,
            args2,
            dists1,
            dists2,
            compare_grads=True,
        )

    def test_softmax_rgb_blend(self):
        # Create dummy outputs of rasterization simulating a cube in the center
        # of the image with surrounding padded values.
        N, S, K = 1, 8, 2
        device = torch.device("cuda")
        pix_to_face = torch.full(
            (N, S, S, K), fill_value=-1, dtype=torch.int64, device=device
        )
        h = int(S / 2)
        pix_to_face_full = torch.randint(
            size=(N, h, h, K), low=0, high=100, device=device
        )
        s = int(S / 4)
        e = int(0.75 * S)
        pix_to_face[:, s:e, s:e, :] = pix_to_face_full
        empty = torch.tensor([], device=device)

        random_sign_flip = torch.rand((N, S, S, K), device=device)
        random_sign_flip[random_sign_flip > 0.5] *= -1.0
        zbuf1 = torch.randn(size=(N, S, S, K), device=device)

        # randomly flip the sign of the distance
        # (-) means inside triangle, (+) means outside triangle.
        dists1 = torch.randn(size=(N, S, S, K), device=device) * random_sign_flip
        dists2 = dists1.clone()
        zbuf2 = zbuf1.clone()
        dists1.requires_grad = True
        dists2.requires_grad = True
        colors = torch.randn((N, S, S, K, 3), device=device)
        fragments1 = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=empty,  # dummy
            zbuf=zbuf1,
            dists=dists1,
        )
        fragments2 = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=empty,  # dummy
            zbuf=zbuf2,
            dists=dists2,
        )

        blend_params = BlendParams(sigma=1e-3)
        args1 = (colors, fragments1, blend_params)
        args2 = (colors, fragments2, blend_params)
        self._compare_impls(
            softmax_rgb_blend,
            softmax_blend_naive,
            args1,
            args2,
            dists1,
            dists2,
            compare_grads=True,
        )

    @staticmethod
    def bm_sigmoid_alpha_blending(
        num_meshes: int = 16,
        image_size: int = 128,
        faces_per_pixel: int = 100,
        device="cuda",
        backend: str = "pytorch",
    ):
        device = torch.device(device)
        torch.manual_seed(231)

        # Create dummy outputs of rasterization
        N, S, K = num_meshes, image_size, faces_per_pixel
        F = 32  # num faces in the mesh
        pix_to_face = torch.randint(
            low=-1, high=F + 1, size=(N, S, S, K), device=device
        )
        colors = torch.randn((N, S, S, K, 3), device=device)
        empty = torch.tensor([], device=device)

        dists1 = torch.randn(size=(N, S, S, K), requires_grad=True, device=device)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=empty,  # dummy
            zbuf=empty,  # dummy
            dists=dists1,
        )
        blend_params = BlendParams(sigma=1e-3)

        blend_fn = (
            sigmoid_alpha_blend_vectorized
            if backend == "pytorch"
            else sigmoid_alpha_blend
        )

        torch.cuda.synchronize()

        def fn():
            # test forward and backward pass
            images = blend_fn(colors, fragments, blend_params)
            images.sum().backward()
            torch.cuda.synchronize()

        return fn

    @staticmethod
    def bm_softmax_blending(
        num_meshes: int = 16,
        image_size: int = 128,
        faces_per_pixel: int = 100,
        device: str = "cpu",
        backend: str = "pytorch",
    ):
        if torch.cuda.is_available() and "cuda:" in device:
            # If a device other than the default is used, set the device explicity.
            torch.cuda.set_device(device)

        device = torch.device(device)
        torch.manual_seed(231)

        # Create dummy outputs of rasterization
        N, S, K = num_meshes, image_size, faces_per_pixel
        F = 32  # num faces in the mesh
        pix_to_face = torch.randint(
            low=-1, high=F + 1, size=(N, S, S, K), device=device
        )
        colors = torch.randn((N, S, S, K, 3), device=device)
        empty = torch.tensor([], device=device)

        dists1 = torch.randn(size=(N, S, S, K), requires_grad=True, device=device)
        zbuf = torch.randn(size=(N, S, S, K), requires_grad=True, device=device)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=empty,
            zbuf=zbuf,
            dists=dists1,  # dummy
        )
        blend_params = BlendParams(sigma=1e-3)

        torch.cuda.synchronize()

        def fn():
            # test forward and backward pass
            images = softmax_rgb_blend(colors, fragments, blend_params)
            images.sum().backward()
            torch.cuda.synchronize()

        return fn

    @staticmethod
    def bm_splatter_blending(
        num_meshes: int = 16,
        image_size: int = 128,
        faces_per_pixel: int = 2,
        use_jit: bool = False,
        device: str = "cpu",
        backend: str = "pytorch",
    ):
        if torch.cuda.is_available() and "cuda:" in device:
            # If a device other than the default is used, set the device explicity.
            torch.cuda.set_device(device)

        device = torch.device(device)
        torch.manual_seed(231)

        # Create dummy outputs of rasterization
        N, S, K = num_meshes, image_size, faces_per_pixel
        F = 32  # num faces in the mesh

        pixel_coords_camera = torch.randn(
            (N, S, S, K, 3), device=device, requires_grad=True
        )
        cameras = FoVPerspectiveCameras(device=device)
        colors = torch.randn((N, S, S, K, 3), device=device)
        background_mask = torch.randint(
            low=-1, high=F + 1, size=(N, S, S, K), device=device
        )
        background_mask = torch.full((N, S, S, K), False, dtype=bool, device=device)
        blend_params = BlendParams(sigma=0.5)

        torch.cuda.synchronize()
        splatter_blender = SplatterBlender((N, S, S, K), colors.device)

        def fn():
            # test forward and backward pass
            images = splatter_blender(
                colors,
                pixel_coords_camera,
                cameras,
                background_mask,
                blend_params,
            )
            images.sum().backward()
            torch.cuda.synchronize()

        return fn

    def test_blend_params(self):
        """Test color parameter of BlendParams().
        Assert passed value overrides default value.
        """
        bp_default = BlendParams()
        bp_new = BlendParams(background_color=(0.5, 0.5, 0.5))
        self.assertEqual(bp_new.background_color, (0.5, 0.5, 0.5))
        self.assertEqual(bp_default.background_color, (1.0, 1.0, 1.0))
