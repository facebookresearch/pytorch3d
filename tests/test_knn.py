# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import torch
from pytorch3d.ops.knn import _KNN, knn_gather, knn_points

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestKNN(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    @staticmethod
    def _knn_points_naive(
        p1, p2, lengths1, lengths2, K: int, norm: int = 2
    ) -> torch.Tensor:
        """
        Naive PyTorch implementation of K-Nearest Neighbors.
        Returns always sorted results
        """
        N, P1, D = p1.shape
        _N, P2, _D = p2.shape

        assert N == _N and D == _D

        if lengths1 is None:
            lengths1 = torch.full((N,), P1, dtype=torch.int64, device=p1.device)
        if lengths2 is None:
            lengths2 = torch.full((N,), P2, dtype=torch.int64, device=p1.device)

        dists = torch.zeros((N, P1, K), dtype=torch.float32, device=p1.device)
        idx = torch.zeros((N, P1, K), dtype=torch.int64, device=p1.device)

        for n in range(N):
            num1 = lengths1[n].item()
            num2 = lengths2[n].item()
            pp1 = p1[n, :num1].view(num1, 1, D)
            pp2 = p2[n, :num2].view(1, num2, D)
            diff = pp1 - pp2
            if norm == 2:
                diff = (diff * diff).sum(2)
            elif norm == 1:
                diff = diff.abs().sum(2)
            else:
                raise ValueError("No support for norm %d" % (norm))
            num2 = min(num2, K)
            for i in range(num1):
                dd = diff[i]
                srt_dd, srt_idx = dd.sort()

                dists[n, i, :num2] = srt_dd[:num2]
                idx[n, i, :num2] = srt_idx[:num2]

        return _KNN(dists=dists, idx=idx, knn=None)

    def _knn_vs_python_square_helper(self, device, return_sorted):
        Ns = [1, 4]
        Ds = [3, 5, 8]
        P1s = [8, 24]
        P2s = [8, 16, 32]
        Ks = [1, 3, 10]
        norms = [1, 2]
        versions = [0, 1, 2, 3]
        factors = [Ns, Ds, P1s, P2s, Ks, norms]
        for N, D, P1, P2, K, norm in product(*factors):
            for version in versions:
                if version == 3 and K > 4:
                    continue
                x = torch.randn(N, P1, D, device=device, requires_grad=True)
                x_cuda = x.clone().detach()
                x_cuda.requires_grad_(True)
                y = torch.randn(N, P2, D, device=device, requires_grad=True)
                y_cuda = y.clone().detach()
                y_cuda.requires_grad_(True)

                # forward
                out1 = self._knn_points_naive(
                    x, y, lengths1=None, lengths2=None, K=K, norm=norm
                )
                out2 = knn_points(
                    x_cuda,
                    y_cuda,
                    K=K,
                    norm=norm,
                    version=version,
                    return_sorted=return_sorted,
                )
                if K > 1 and not return_sorted:
                    # check out2 is not sorted
                    self.assertFalse(torch.allclose(out1[0], out2[0]))
                    self.assertFalse(torch.allclose(out1[1], out2[1]))
                    # now sort out2
                    dists, idx, _ = out2
                    if P2 < K:
                        dists[..., P2:] = float("inf")
                        dists, sort_idx = dists.sort(dim=2)
                        dists[..., P2:] = 0
                    else:
                        dists, sort_idx = dists.sort(dim=2)
                    idx = idx.gather(2, sort_idx)
                    out2 = _KNN(dists, idx, None)

                self.assertClose(out1[0], out2[0])
                self.assertTrue(torch.all(out1[1] == out2[1]))

                # backward
                grad_dist = torch.ones((N, P1, K), dtype=torch.float32, device=device)
                loss1 = (out1.dists * grad_dist).sum()
                loss1.backward()
                loss2 = (out2.dists * grad_dist).sum()
                loss2.backward()

                self.assertClose(x_cuda.grad, x.grad, atol=5e-6)
                self.assertClose(y_cuda.grad, y.grad, atol=5e-6)

    def test_knn_vs_python_square_cpu(self):
        device = torch.device("cpu")
        self._knn_vs_python_square_helper(device, return_sorted=True)

    def test_knn_vs_python_square_cuda(self):
        device = get_random_cuda_device()
        # Check both cases where the output is sorted and unsorted
        self._knn_vs_python_square_helper(device, return_sorted=True)
        self._knn_vs_python_square_helper(device, return_sorted=False)

    def _knn_vs_python_ragged_helper(self, device):
        Ns = [1, 4]
        Ds = [3, 5, 8]
        P1s = [8, 24]
        P2s = [8, 16, 32]
        Ks = [1, 3, 10]
        norms = [1, 2]
        factors = [Ns, Ds, P1s, P2s, Ks, norms]
        for N, D, P1, P2, K, norm in product(*factors):
            x = torch.rand((N, P1, D), device=device, requires_grad=True)
            y = torch.rand((N, P2, D), device=device, requires_grad=True)
            lengths1 = torch.randint(low=1, high=P1, size=(N,), device=device)
            lengths2 = torch.randint(low=1, high=P2, size=(N,), device=device)

            x_csrc = x.clone().detach()
            x_csrc.requires_grad_(True)
            y_csrc = y.clone().detach()
            y_csrc.requires_grad_(True)

            # forward
            out1 = self._knn_points_naive(
                x, y, lengths1=lengths1, lengths2=lengths2, K=K, norm=norm
            )
            out2 = knn_points(
                x_csrc, y_csrc, lengths1=lengths1, lengths2=lengths2, K=K, norm=norm
            )
            self.assertClose(out1[0], out2[0])
            self.assertTrue(torch.all(out1[1] == out2[1]))

            # backward
            grad_dist = torch.ones((N, P1, K), dtype=torch.float32, device=device)
            loss1 = (out1.dists * grad_dist).sum()
            loss1.backward()
            loss2 = (out2.dists * grad_dist).sum()
            loss2.backward()

            self.assertClose(x_csrc.grad, x.grad, atol=5e-6)
            self.assertClose(y_csrc.grad, y.grad, atol=5e-6)

    def test_knn_vs_python_ragged_cpu(self):
        device = torch.device("cpu")
        self._knn_vs_python_ragged_helper(device)

    def test_knn_vs_python_ragged_cuda(self):
        device = get_random_cuda_device()
        self._knn_vs_python_ragged_helper(device)

    def test_knn_gather(self):
        device = get_random_cuda_device()
        N, P1, P2, K, D = 4, 16, 12, 8, 3
        x = torch.rand((N, P1, D), device=device)
        y = torch.rand((N, P2, D), device=device)
        lengths1 = torch.randint(low=1, high=P1, size=(N,), device=device)
        lengths2 = torch.randint(low=1, high=P2, size=(N,), device=device)

        out = knn_points(x, y, lengths1=lengths1, lengths2=lengths2, K=K)
        y_nn = knn_gather(y, out.idx, lengths2)

        for n in range(N):
            for p1 in range(P1):
                for k in range(K):
                    if k < lengths2[n]:
                        self.assertClose(y_nn[n, p1, k], y[n, out.idx[n, p1, k]])
                    else:
                        self.assertTrue(torch.all(y_nn[n, p1, k] == 0.0))

    def test_knn_check_version(self):
        try:
            from pytorch3d._C import knn_check_version
        except ImportError:
            # knn_check_version will only be defined if we compiled with CUDA support
            return
        for D in range(-10, 10):
            for K in range(-10, 20):
                v0 = True
                v1 = 1 <= D <= 32
                v2 = 1 <= D <= 8 and 1 <= K <= 32
                v3 = 1 <= D <= 8 and 1 <= K <= 4
                all_expected = [v0, v1, v2, v3]
                for version in range(-10, 10):
                    actual = knn_check_version(version, D, K)
                    expected = False
                    if 0 <= version < len(all_expected):
                        expected = all_expected[version]
                    self.assertEqual(actual, expected)

    def test_invalid_norm(self):
        device = get_random_cuda_device()
        N, P1, P2, K, D = 4, 16, 12, 8, 3
        x = torch.rand((N, P1, D), device=device)
        y = torch.rand((N, P2, D), device=device)
        with self.assertRaisesRegex(ValueError, "Support for 1 or 2 norm."):
            knn_points(x, y, K=K, norm=3)

        with self.assertRaisesRegex(ValueError, "Support for 1 or 2 norm."):
            knn_points(x, y, K=K, norm=0)

    @staticmethod
    def knn_square(N: int, P1: int, P2: int, D: int, K: int, device: str):
        device = torch.device(device)
        pts1 = torch.randn(N, P1, D, device=device, requires_grad=True)
        pts2 = torch.randn(N, P2, D, device=device, requires_grad=True)
        grad_dists = torch.randn(N, P1, K, device=device)
        torch.cuda.synchronize()

        def output():
            out = knn_points(pts1, pts2, K=K)
            loss = (out.dists * grad_dists).sum()
            loss.backward()
            torch.cuda.synchronize()

        return output

    @staticmethod
    def knn_ragged(N: int, P1: int, P2: int, D: int, K: int, device: str):
        device = torch.device(device)
        pts1 = torch.rand((N, P1, D), device=device, requires_grad=True)
        pts2 = torch.rand((N, P2, D), device=device, requires_grad=True)
        lengths1 = torch.randint(low=1, high=P1, size=(N,), device=device)
        lengths2 = torch.randint(low=1, high=P2, size=(N,), device=device)
        grad_dists = torch.randn(N, P1, K, device=device)
        torch.cuda.synchronize()

        def output():
            out = knn_points(pts1, pts2, lengths1=lengths1, lengths2=lengths2, K=K)
            loss = (out.dists * grad_dists).sum()
            loss.backward()
            torch.cuda.synchronize()

        return output
