# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from itertools import product

import torch
from pytorch3d.ops.knn import _knn_points_idx_naive, knn_points_idx


class TestKNN(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def _check_knn_result(self, out1, out2, sorted):
        # When sorted=True, points should be sorted by distance and should
        # match between implementations. When sorted=False we we only want to
        # check that we got the same set of indices, so we sort the indices by
        # index value.
        idx1, dist1 = out1
        idx2, dist2 = out2
        if not sorted:
            idx1 = idx1.sort(dim=2).values
            idx2 = idx2.sort(dim=2).values
            dist1 = dist1.sort(dim=2).values
            dist2 = dist2.sort(dim=2).values
        if not torch.all(idx1 == idx2):
            print(idx1)
            print(idx2)
        self.assertTrue(torch.all(idx1 == idx2))
        self.assertTrue(torch.allclose(dist1, dist2))

    def test_knn_vs_python_cpu_square(self):
        """ Test CPU output vs PyTorch implementation """
        device = torch.device("cpu")
        Ns = [1, 4]
        Ds = [2, 3]
        P1s = [1, 10, 101]
        P2s = [10, 101]
        Ks = [1, 3, 10]
        sorts = [True, False]
        factors = [Ns, Ds, P1s, P2s, Ks, sorts]
        for N, D, P1, P2, K, sort in product(*factors):
            lengths1 = torch.full((N,), P1, dtype=torch.int64, device=device)
            lengths2 = torch.full((N,), P2, dtype=torch.int64, device=device)
            x = torch.randn(N, P1, D, device=device)
            y = torch.randn(N, P2, D, device=device)
            out1 = _knn_points_idx_naive(
                x, y, lengths1=lengths1, lengths2=lengths2, K=K
            )
            out2 = knn_points_idx(
                x, y, K=K, lengths1=lengths1, lengths2=lengths2, sorted=sort
            )
            self._check_knn_result(out1, out2, sort)

    def test_knn_vs_python_cuda_square(self):
        """ Test CUDA output vs PyTorch implementation """
        device = torch.device("cuda")
        Ns = [1, 4]
        Ds = [2, 3, 8]
        P1s = [1, 8, 64, 128, 1001]
        P2s = [32, 128, 513]
        Ks = [1, 3, 10]
        sorts = [True, False]
        versions = [0, 1, 2, 3]
        factors = [Ns, Ds, P1s, P2s, Ks, sorts]
        for N, D, P1, P2, K, sort in product(*factors):
            x = torch.randn(N, P1, D, device=device)
            y = torch.randn(N, P2, D, device=device)
            out1 = _knn_points_idx_naive(x, y, lengths1=None, lengths2=None, K=K)
            for version in versions:
                if version == 3 and K > 4:
                    continue
                out2 = knn_points_idx(x, y, K=K, sorted=sort, version=version)
                self._check_knn_result(out1, out2, sort)

    def test_knn_vs_python_cpu_ragged(self):
        device = torch.device("cpu")
        lengths1 = torch.tensor([10, 100, 10, 100], device=device, dtype=torch.int64)
        lengths2 = torch.tensor([10, 10, 100, 100], device=device, dtype=torch.int64)
        N = 4
        D = 3
        Ks = [1, 9, 10, 11, 101]
        sorts = [False, True]
        factors = [Ks, sorts]
        for K, sort in product(*factors):
            x = torch.randn(N, lengths1.max(), D, device=device)
            y = torch.randn(N, lengths2.max(), D, device=device)
            out1 = _knn_points_idx_naive(
                x, y, lengths1=lengths1, lengths2=lengths2, K=K
            )
            out2 = knn_points_idx(
                x, y, lengths1=lengths1, lengths2=lengths2, K=K, sorted=sort
            )
            self._check_knn_result(out1, out2, sort)

    def test_knn_vs_python_cuda_ragged(self):
        device = torch.device("cuda")
        lengths1 = torch.tensor([10, 100, 10, 100], device=device, dtype=torch.int64)
        lengths2 = torch.tensor([10, 10, 100, 100], device=device, dtype=torch.int64)
        N = 4
        D = 3
        Ks = [1, 9, 10, 11, 101]
        sorts = [True, False]
        versions = [0, 1, 2, 3]
        factors = [Ks, sorts]
        for K, sort in product(*factors):
            x = torch.randn(N, lengths1.max(), D, device=device)
            y = torch.randn(N, lengths2.max(), D, device=device)
            out1 = _knn_points_idx_naive(
                x, y, lengths1=lengths1, lengths2=lengths2, K=K
            )
            for version in versions:
                if version == 3 and K > 4:
                    continue
                out2 = knn_points_idx(
                    x, y, lengths1=lengths1, lengths2=lengths2, K=K, sorted=sort
                )
                self._check_knn_result(out1, out2, sort)
