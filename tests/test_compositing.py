# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.renderer.compositing import (
    alpha_composite,
    norm_weighted_sum,
    weighted_sum,
)

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestAccumulatePoints(TestCaseMixin, unittest.TestCase):
    # NAIVE PYTHON IMPLEMENTATIONS (USED FOR TESTING)
    @staticmethod
    def accumulate_alphacomposite_python(points_idx, alphas, features):
        """
        Naive pure PyTorch implementation of alpha_composite.
        Inputs / Outputs: Same as function
        """

        B, K, H, W = points_idx.size()
        C = features.size(0)

        output = torch.zeros(B, C, H, W, dtype=alphas.dtype)

        for b in range(0, B):
            for c in range(0, C):
                for i in range(0, W):
                    for j in range(0, H):
                        t_alpha = 1
                        for k in range(0, K):
                            n_idx = points_idx[b, k, j, i]

                            if n_idx < 0:
                                continue

                            alpha = alphas[b, k, j, i]
                            output[b, c, j, i] += features[c, n_idx] * alpha * t_alpha
                            t_alpha = (1 - alpha) * t_alpha

        return output

    @staticmethod
    def accumulate_weightedsum_python(points_idx, alphas, features):
        """
        Naive pure PyTorch implementation of weighted_sum rasterization.
        Inputs / Outputs: Same as function
        """
        B, K, H, W = points_idx.size()
        C = features.size(0)

        output = torch.zeros(B, C, H, W, dtype=alphas.dtype)

        for b in range(0, B):
            for c in range(0, C):
                for i in range(0, W):
                    for j in range(0, H):
                        for k in range(0, K):
                            n_idx = points_idx[b, k, j, i]

                            if n_idx < 0:
                                continue

                            alpha = alphas[b, k, j, i]
                            output[b, c, j, i] += features[c, n_idx] * alpha

        return output

    @staticmethod
    def accumulate_weightedsumnorm_python(points_idx, alphas, features):
        """
        Naive pure PyTorch implementation of norm_weighted_sum.
        Inputs / Outputs: Same as function
        """

        B, K, H, W = points_idx.size()
        C = features.size(0)

        output = torch.zeros(B, C, H, W, dtype=alphas.dtype)

        for b in range(0, B):
            for c in range(0, C):
                for i in range(0, W):
                    for j in range(0, H):
                        t_alpha = 0
                        for k in range(0, K):
                            n_idx = points_idx[b, k, j, i]

                            if n_idx < 0:
                                continue

                            t_alpha += alphas[b, k, j, i]

                        t_alpha = max(t_alpha, 1e-4)

                        for k in range(0, K):
                            n_idx = points_idx[b, k, j, i]

                            if n_idx < 0:
                                continue

                            alpha = alphas[b, k, j, i]
                            output[b, c, j, i] += features[c, n_idx] * alpha / t_alpha

        return output

    def test_python(self):
        device = torch.device("cpu")
        self._simple_alphacomposite(self.accumulate_alphacomposite_python, device)
        self._simple_wsum(self.accumulate_weightedsum_python, device)
        self._simple_wsumnorm(self.accumulate_weightedsumnorm_python, device)

    def test_cpu(self):
        device = torch.device("cpu")
        self._simple_alphacomposite(alpha_composite, device)
        self._simple_wsum(weighted_sum, device)
        self._simple_wsumnorm(norm_weighted_sum, device)

    def test_cuda(self):
        device = get_random_cuda_device()
        self._simple_alphacomposite(alpha_composite, device)
        self._simple_wsum(weighted_sum, device)
        self._simple_wsumnorm(norm_weighted_sum, device)

    def test_python_vs_cpu_vs_cuda(self):
        self._python_vs_cpu_vs_cuda(
            self.accumulate_alphacomposite_python, alpha_composite
        )
        self._python_vs_cpu_vs_cuda(
            self.accumulate_weightedsumnorm_python, norm_weighted_sum
        )
        self._python_vs_cpu_vs_cuda(self.accumulate_weightedsum_python, weighted_sum)

    def _python_vs_cpu_vs_cuda(self, accumulate_func_python, accumulate_func):
        torch.manual_seed(231)
        device = torch.device("cpu")

        W = 8
        C = 3
        P = 32

        for d in ["cpu", get_random_cuda_device()]:
            # TODO(gkioxari) add torch.float64 to types after double precision
            # support is added to atomicAdd
            for t in [torch.float32]:
                device = torch.device(d)

                # Create values
                alphas = torch.rand(2, 4, W, W, dtype=t).to(device)
                alphas.requires_grad = True
                alphas_cpu = alphas.detach().cpu()
                alphas_cpu.requires_grad = True

                features = torch.randn(C, P, dtype=t).to(device)
                features.requires_grad = True
                features_cpu = features.detach().cpu()
                features_cpu.requires_grad = True

                inds = torch.randint(P + 1, size=(2, 4, W, W)).to(device) - 1
                inds_cpu = inds.detach().cpu()

                args_cuda = (inds, alphas, features)
                args_cpu = (inds_cpu, alphas_cpu, features_cpu)

                self._compare_impls(
                    accumulate_func_python,
                    accumulate_func,
                    args_cpu,
                    args_cuda,
                    (alphas_cpu, features_cpu),
                    (alphas, features),
                    compare_grads=True,
                )

    def _compare_impls(
        self, fn1, fn2, args1, args2, grads1, grads2, compare_grads=False
    ):
        res1 = fn1(*args1)
        res2 = fn2(*args2)

        self.assertClose(res1.cpu(), res2.cpu(), atol=1e-6)

        if not compare_grads:
            return

        # Compare gradients
        torch.manual_seed(231)
        grad_res = torch.randn_like(res1)
        loss1 = (res1 * grad_res).sum()
        loss1.backward()

        grads1 = [gradsi.grad.data.clone().cpu() for gradsi in grads1]
        grad_res = grad_res.to(res2)

        loss2 = (res2 * grad_res).sum()
        loss2.backward()
        grads2 = [gradsi.grad.data.clone().cpu() for gradsi in grads2]

        for i in range(0, len(grads1)):
            self.assertClose(grads1[i].cpu(), grads2[i].cpu(), atol=1e-6)

    def _simple_wsum(self, accum_func, device):
        # Initialise variables
        features = torch.Tensor([[0.1, 0.4, 0.6, 0.9], [0.1, 0.4, 0.6, 0.9]]).to(device)

        alphas = torch.Tensor(
            [
                [
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                ]
            ]
        ).to(device)

        points_idx = (
            torch.Tensor(
                [
                    [
                        # fmt: off
                        [
                            [0, 0, 0, 0],  # noqa: E241, E201
                            [0, -1, -1, -1],  # noqa: E241, E201
                            [0, 1, 1, 0],  # noqa: E241, E201
                            [0, 0, 0, 0],  # noqa: E241, E201
                        ],
                        [
                            [2, 2, 2, 2],  # noqa: E241, E201
                            [2, 3, 3, 2],  # noqa: E241, E201
                            [2, 3, 3, 2],  # noqa: E241, E201
                            [2, 2, -1, 2],  # noqa: E241, E201
                        ],
                        # fmt: on
                    ]
                ]
            )
            .long()
            .to(device)
        )

        result = accum_func(points_idx, alphas, features)

        self.assertTrue(result.shape == (1, 2, 4, 4))

        true_result = torch.Tensor(
            [
                [
                    [
                        [0.35, 0.35, 0.35, 0.35],
                        [0.35, 0.90, 0.90, 0.30],
                        [0.35, 1.30, 1.30, 0.35],
                        [0.35, 0.35, 0.05, 0.35],
                    ],
                    [
                        [0.35, 0.35, 0.35, 0.35],
                        [0.35, 0.90, 0.90, 0.30],
                        [0.35, 1.30, 1.30, 0.35],
                        [0.35, 0.35, 0.05, 0.35],
                    ],
                ]
            ]
        ).to(device)

        self.assertClose(result.cpu(), true_result.cpu(), rtol=1e-3)

    def _simple_wsumnorm(self, accum_func, device):
        # Initialise variables
        features = torch.Tensor([[0.1, 0.4, 0.6, 0.9], [0.1, 0.4, 0.6, 0.9]]).to(device)

        alphas = torch.Tensor(
            [
                [
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                ]
            ]
        ).to(device)

        # fmt: off
        points_idx = (
            torch.Tensor(
                [
                    [
                        [
                            [0,  0,  0,  0],  # noqa: E241, E201
                            [0, -1, -1, -1],  # noqa: E241, E201
                            [0,  1,  1,  0],  # noqa: E241, E201
                            [0,  0,  0,  0],  # noqa: E241, E201
                        ],
                        [
                            [2, 2,  2, 2],  # noqa: E241, E201
                            [2, 3,  3, 2],  # noqa: E241, E201
                            [2, 3,  3, 2],  # noqa: E241, E201
                            [2, 2, -1, 2],  # noqa: E241, E201
                        ],
                    ]
                ]
            )
            .long()
            .to(device)
        )
        # fmt: on

        result = accum_func(points_idx, alphas, features)

        self.assertTrue(result.shape == (1, 2, 4, 4))

        true_result = torch.Tensor(
            [
                [
                    [
                        [0.35, 0.35, 0.35, 0.35],
                        [0.35, 0.90, 0.90, 0.60],
                        [0.35, 0.65, 0.65, 0.35],
                        [0.35, 0.35, 0.10, 0.35],
                    ],
                    [
                        [0.35, 0.35, 0.35, 0.35],
                        [0.35, 0.90, 0.90, 0.60],
                        [0.35, 0.65, 0.65, 0.35],
                        [0.35, 0.35, 0.10, 0.35],
                    ],
                ]
            ]
        ).to(device)

        self.assertClose(result.cpu(), true_result.cpu(), rtol=1e-3)

    def _simple_alphacomposite(self, accum_func, device):
        # Initialise variables
        features = torch.Tensor([[0.1, 0.4, 0.6, 0.9], [0.1, 0.4, 0.6, 0.9]]).to(device)

        alphas = torch.Tensor(
            [
                [
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                    [
                        [0.5, 0.5, 0.5, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 1.0, 1.0, 0.5],
                        [0.5, 0.5, 0.5, 0.5],
                    ],
                ]
            ]
        ).to(device)

        # fmt: off
        points_idx = (
            torch.Tensor(
                [
                    [
                        [
                            [0,  0,  0,  0],  # noqa: E241, E201
                            [0, -1, -1, -1],  # noqa: E241, E201
                            [0,  1,  1,  0],  # noqa: E241, E201
                            [0,  0,  0,  0],  # noqa: E241, E201
                        ],
                        [
                            [2, 2,  2, 2],  # noqa: E241, E201
                            [2, 3,  3, 2],  # noqa: E241, E201
                            [2, 3,  3, 2],  # noqa: E241, E201
                            [2, 2, -1, 2],  # noqa: E241, E201
                        ],
                    ]
                ]
            )
            .long()
            .to(device)
        )
        # fmt: on

        result = accum_func(points_idx, alphas, features)

        self.assertTrue(result.shape == (1, 2, 4, 4))

        true_result = torch.Tensor(
            [
                [
                    [
                        [0.20, 0.20, 0.20, 0.20],
                        [0.20, 0.90, 0.90, 0.30],
                        [0.20, 0.40, 0.40, 0.20],
                        [0.20, 0.20, 0.05, 0.20],
                    ],
                    [
                        [0.20, 0.20, 0.20, 0.20],
                        [0.20, 0.90, 0.90, 0.30],
                        [0.20, 0.40, 0.40, 0.20],
                        [0.20, 0.20, 0.05, 0.20],
                    ],
                ]
            ]
        ).to(device)

        self.assertTrue((result == true_result).all().item())
