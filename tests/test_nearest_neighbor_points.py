# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from itertools import product

import torch
from pytorch3d import _C


class TestNearestNeighborPoints(unittest.TestCase):
    @staticmethod
    def nn_points_idx_naive(x, y):
        """
        PyTorch implementation of nn_points_idx function.
        """
        N, P1, D = x.shape
        _N, P2, _D = y.shape
        assert N == _N and D == _D
        diffs = x.view(N, P1, 1, D) - y.view(N, 1, P2, D)
        dists2 = (diffs * diffs).sum(3)
        idx = dists2.argmin(2)
        return idx

    def _test_nn_helper(self, device):
        for D in [3, 4]:
            for N in [1, 4]:
                for P1 in [1, 8, 64, 128]:
                    for P2 in [32, 128]:
                        x = torch.randn(N, P1, D, device=device)
                        y = torch.randn(N, P2, D, device=device)

                        # _C.nn_points_idx should dispatch
                        # to the cpp or cuda versions of the function
                        # depending on the input type.
                        idx1 = _C.nn_points_idx(x, y)
                        idx2 = TestNearestNeighborPoints.nn_points_idx_naive(x, y)
                        self.assertTrue(idx1.size(1) == P1)
                        self.assertTrue(torch.all(idx1 == idx2))

    def test_nn_cuda(self):
        """
        Test cuda output vs naive python implementation.
        """
        device = torch.device("cuda:0")
        self._test_nn_helper(device)

    def test_nn_cpu(self):
        """
        Test cpu output vs naive python implementation
        """
        device = torch.device("cpu")
        self._test_nn_helper(device)

    @staticmethod
    def bm_nn_points_cpu_with_init(
        N: int = 4, D: int = 4, P1: int = 128, P2: int = 128
    ):
        device = torch.device("cpu")
        x = torch.randn(N, P1, D, device=device)
        y = torch.randn(N, P2, D, device=device)

        def nn_cpu():
            _C.nn_points_idx(x.contiguous(), y.contiguous())

        return nn_cpu

    @staticmethod
    def bm_nn_points_cuda_with_init(
        N: int = 4, D: int = 4, P1: int = 128, P2: int = 128
    ):
        device = torch.device("cuda:0")
        x = torch.randn(N, P1, D, device=device)
        y = torch.randn(N, P2, D, device=device)
        torch.cuda.synchronize()

        def nn_cpp():
            _C.nn_points_idx(x.contiguous(), y.contiguous())
            torch.cuda.synchronize()

        return nn_cpp

    @staticmethod
    def bm_nn_points_python_with_init(
        N: int = 4, D: int = 4, P1: int = 128, P2: int = 128
    ):
        x = torch.randn(N, P1, D)
        y = torch.randn(N, P2, D)

        def nn_python():
            TestNearestNeighborPoints.nn_points_idx_naive(x, y)

        return nn_python
