# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops.ball_query import ball_query
from pytorch3d.ops.knn import _KNN
from pytorch3d.utils import ico_sphere

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestBallQuery(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    @staticmethod
    def _ball_query_naive(
        p1, p2, lengths1, lengths2, K: int, radius: float
    ) -> torch.Tensor:
        """
        Naive PyTorch implementation of ball query.
        """
        N, P1, D = p1.shape
        _N, P2, _D = p2.shape

        assert N == _N and D == _D

        if lengths1 is None:
            lengths1 = torch.full((N,), P1, dtype=torch.int64, device=p1.device)
        if lengths2 is None:
            lengths2 = torch.full((N,), P2, dtype=torch.int64, device=p1.device)

        radius2 = radius * radius
        dists = torch.zeros((N, P1, K), dtype=torch.float32, device=p1.device)
        idx = torch.full((N, P1, K), fill_value=-1, dtype=torch.int64, device=p1.device)

        # Iterate through the batches
        for n in range(N):
            num1 = lengths1[n].item()
            num2 = lengths2[n].item()

            # Iterate through the points in the p1
            for i in range(num1):
                # Iterate through the points in the p2
                count = 0
                for j in range(num2):
                    dist = p2[n, j] - p1[n, i]
                    dist2 = (dist * dist).sum()
                    if dist2 < radius2 and count < K:
                        dists[n, i, count] = dist2
                        idx[n, i, count] = j
                        count += 1

        return _KNN(dists=dists, idx=idx, knn=None)

    def _ball_query_vs_python_square_helper(self, device):
        Ns = [1, 4]
        Ds = [3, 5, 8]
        P1s = [8, 24]
        P2s = [8, 16, 32]
        Ks = [1, 5]
        Rs = [3, 5]
        factors = [Ns, Ds, P1s, P2s, Ks, Rs]
        for N, D, P1, P2, K, R in product(*factors):
            x = torch.randn(N, P1, D, device=device, requires_grad=True)
            x_cuda = x.clone().detach()
            x_cuda.requires_grad_(True)
            y = torch.randn(N, P2, D, device=device, requires_grad=True)
            y_cuda = y.clone().detach()
            y_cuda.requires_grad_(True)

            # forward
            out1 = self._ball_query_naive(
                x, y, lengths1=None, lengths2=None, K=K, radius=R
            )
            out2 = ball_query(x_cuda, y_cuda, K=K, radius=R)

            # Check dists
            self.assertClose(out1.dists, out2.dists)
            # Check idx
            self.assertTrue(torch.all(out1.idx == out2.idx))

            # backward
            grad_dist = torch.ones((N, P1, K), dtype=torch.float32, device=device)
            loss1 = (out1.dists * grad_dist).sum()
            loss1.backward()
            loss2 = (out2.dists * grad_dist).sum()
            loss2.backward()

            self.assertClose(x_cuda.grad, x.grad, atol=5e-6)
            self.assertClose(y_cuda.grad, y.grad, atol=5e-6)

    def test_ball_query_vs_python_square_cpu(self):
        device = torch.device("cpu")
        self._ball_query_vs_python_square_helper(device)

    def test_ball_query_vs_python_square_cuda(self):
        device = get_random_cuda_device()
        self._ball_query_vs_python_square_helper(device)

    def _ball_query_vs_python_ragged_helper(self, device):
        Ns = [1, 4]
        Ds = [3, 5, 8]
        P1s = [8, 24]
        P2s = [8, 16, 32]
        Ks = [2, 3, 10]
        Rs = [1.4, 5]  # radius
        factors = [Ns, Ds, P1s, P2s, Ks, Rs]
        for N, D, P1, P2, K, R in product(*factors):
            x = torch.rand((N, P1, D), device=device, requires_grad=True)
            y = torch.rand((N, P2, D), device=device, requires_grad=True)
            lengths1 = torch.randint(low=1, high=P1, size=(N,), device=device)
            lengths2 = torch.randint(low=1, high=P2, size=(N,), device=device)

            x_csrc = x.clone().detach()
            x_csrc.requires_grad_(True)
            y_csrc = y.clone().detach()
            y_csrc.requires_grad_(True)

            # forward
            out1 = self._ball_query_naive(
                x, y, lengths1=lengths1, lengths2=lengths2, K=K, radius=R
            )
            out2 = ball_query(
                x_csrc,
                y_csrc,
                lengths1=lengths1,
                lengths2=lengths2,
                K=K,
                radius=R,
            )

            self.assertClose(out1.idx, out2.idx)
            self.assertClose(out1.dists, out2.dists)

            # backward
            grad_dist = torch.ones((N, P1, K), dtype=torch.float32, device=device)
            loss1 = (out1.dists * grad_dist).sum()
            loss1.backward()
            loss2 = (out2.dists * grad_dist).sum()
            loss2.backward()

            self.assertClose(x_csrc.grad, x.grad, atol=5e-6)
            self.assertClose(y_csrc.grad, y.grad, atol=5e-6)

    def test_ball_query_vs_python_ragged_cpu(self):
        device = torch.device("cpu")
        self._ball_query_vs_python_ragged_helper(device)

    def test_ball_query_vs_python_ragged_cuda(self):
        device = get_random_cuda_device()
        self._ball_query_vs_python_ragged_helper(device)

    def test_ball_query_output_simple(self):
        device = get_random_cuda_device()
        N, P1, P2, K = 5, 8, 16, 4
        sphere = ico_sphere(level=2, device=device).extend(N)
        points_1 = sample_points_from_meshes(sphere, P1)
        points_2 = sample_points_from_meshes(sphere, P2) * 5.0
        radius = 6.0

        naive_out = self._ball_query_naive(
            points_1, points_2, lengths1=None, lengths2=None, K=K, radius=radius
        )
        cuda_out = ball_query(points_1, points_2, K=K, radius=radius)

        # All points should have N sample neighbors as radius is large
        # Zero is a valid index but can only be present once (i.e. no zero padding)
        naive_out_zeros = (naive_out.idx == 0).sum(dim=-1).max()
        cuda_out_zeros = (cuda_out.idx == 0).sum(dim=-1).max()
        self.assertTrue(naive_out_zeros == 0 or naive_out_zeros == 1)
        self.assertTrue(cuda_out_zeros == 0 or cuda_out_zeros == 1)

        # All points should now have zero sample neighbors as radius is small
        radius = 0.5
        naive_out = self._ball_query_naive(
            points_1, points_2, lengths1=None, lengths2=None, K=K, radius=radius
        )
        cuda_out = ball_query(points_1, points_2, K=K, radius=radius)
        naive_out_allzeros = (naive_out.idx == -1).all()
        cuda_out_allzeros = (cuda_out.idx == -1).sum()
        self.assertTrue(naive_out_allzeros)
        self.assertTrue(cuda_out_allzeros)

    @staticmethod
    def ball_query_square(
        N: int, P1: int, P2: int, D: int, K: int, radius: float, device: str
    ):
        device = torch.device(device)
        pts1 = torch.randn(N, P1, D, device=device, requires_grad=True)
        pts2 = torch.randn(N, P2, D, device=device, requires_grad=True)
        grad_dists = torch.randn(N, P1, K, device=device)
        torch.cuda.synchronize()

        def output():
            out = ball_query(pts1, pts2, K=K, radius=radius)
            loss = (out.dists * grad_dists).sum()
            loss.backward()
            torch.cuda.synchronize()

        return output

    @staticmethod
    def ball_query_ragged(
        N: int, P1: int, P2: int, D: int, K: int, radius: float, device: str
    ):
        device = torch.device(device)
        pts1 = torch.rand((N, P1, D), device=device, requires_grad=True)
        pts2 = torch.rand((N, P2, D), device=device, requires_grad=True)
        lengths1 = torch.randint(low=1, high=P1, size=(N,), device=device)
        lengths2 = torch.randint(low=1, high=P2, size=(N,), device=device)
        grad_dists = torch.randn(N, P1, K, device=device)
        torch.cuda.synchronize()

        def output():
            out = ball_query(
                pts1, pts2, lengths1=lengths1, lengths2=lengths2, K=K, radius=radius
            )
            loss = (out.dists * grad_dists).sum()
            loss.backward()
            torch.cuda.synchronize()

        return output
