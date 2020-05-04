# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import numpy as np
import torch
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments


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
                weights_k = torch.zeros(K, device=device)
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

    return pixel_colors


class TestBlending(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def _compare_impls(
        self, fn1, fn2, args1, args2, grad_var1=None, grad_var2=None, compare_grads=True
    ):

        out1 = fn1(*args1)
        out2 = fn2(*args2)
        self.assertTrue(torch.allclose(out1.cpu(), out2.cpu(), atol=1e-7))

        # Check gradients
        if not compare_grads:
            return

        grad_out = torch.randn_like(out1)
        (out1 * grad_out).sum().backward()
        self.assertTrue(hasattr(grad_var1, "grad"))

        (out2 * grad_out).sum().backward()
        self.assertTrue(hasattr(grad_var2, "grad"))
        self.assertTrue(
            torch.allclose(grad_var1.grad.cpu(), grad_var2.grad.cpu(), atol=2e-5)
        )

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
        N, S, K = 2, 10, 5
        device = torch.device("cuda")
        pix_to_face = torch.randint(F + 1, size=(N, S, S, K), device=device) - 1
        colors = torch.randn((N, S, S, K, 3), device=device)
        empty = torch.tensor([], device=device)

        # # randomly flip the sign of the distance
        # # (-) means inside triangle, (+) means outside triangle.
        random_sign_flip = torch.rand((N, S, S, K))
        random_sign_flip[random_sign_flip > 0.5] *= -1.0
        dists1 = torch.randn(size=(N, S, S, K), requires_grad=True, device=device)
        dists2 = dists1.detach().clone()
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
            sigmoid_blend_naive_loop,
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
        device: str = "cpu",
    ):
        if torch.cuda.is_available() and "cuda:" in device:
            # If a device other than the default is used, set the device explicity.
            torch.cuda.set_device(device)

        device = torch.device(device)
        torch.manual_seed(231)

        # Create dummy outputs of rasterization
        N, S, K = num_meshes, image_size, faces_per_pixel
        F = 32  # num faces in the mesh
        pix_to_face = torch.randint(F + 1, size=(N, S, S, K), device=device) - 1
        colors = torch.randn((N, S, S, K, 3), device=device)
        empty = torch.tensor([], device=device)

        # # randomly flip the sign of the distance
        # # (-) means inside triangle, (+) means outside triangle.
        random_sign_flip = torch.rand((N, S, S, K), device=device)
        random_sign_flip[random_sign_flip > 0.5] *= -1.0
        dists1 = torch.randn(size=(N, S, S, K), requires_grad=True, device=device)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=empty,  # dummy
            zbuf=empty,  # dummy
            dists=dists1,
        )
        blend_params = BlendParams(sigma=1e-3)
        torch.cuda.synchronize()

        def fn():
            # test forward and backward pass
            images = sigmoid_alpha_blend(colors, fragments, blend_params)
            images.sum().backward()
            torch.cuda.synchronize()

        return fn

    @staticmethod
    def bm_softmax_blending(
        num_meshes: int = 16,
        image_size: int = 128,
        faces_per_pixel: int = 100,
        device: str = "cpu",
    ):
        if torch.cuda.is_available() and "cuda:" in device:
            # If a device other than the default is used, set the device explicity.
            torch.cuda.set_device(device)

        device = torch.device(device)
        torch.manual_seed(231)

        # Create dummy outputs of rasterization
        N, S, K = num_meshes, image_size, faces_per_pixel
        F = 32  # num faces in the mesh
        pix_to_face = torch.randint(F + 1, size=(N, S, S, K), device=device) - 1
        colors = torch.randn((N, S, S, K, 3), device=device)
        empty = torch.tensor([], device=device)

        # # randomly flip the sign of the distance
        # # (-) means inside triangle, (+) means outside triangle.
        random_sign_flip = torch.rand((N, S, S, K), device=device)
        random_sign_flip[random_sign_flip > 0.5] *= -1.0
        dists1 = torch.randn(size=(N, S, S, K), requires_grad=True, device=device)
        zbuf = torch.randn(size=(N, S, S, K), requires_grad=True, device=device)
        fragments = Fragments(
            pix_to_face=pix_to_face, bary_coords=empty, zbuf=zbuf, dists=dists1  # dummy
        )
        blend_params = BlendParams(sigma=1e-3)

        torch.cuda.synchronize()

        def fn():
            # test forward and backward pass
            images = softmax_rgb_blend(colors, fragments, blend_params)
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
