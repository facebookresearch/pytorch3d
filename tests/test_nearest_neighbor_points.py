#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
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

    def test_nn_cuda(self):
        """
        Test cuda output vs naive python implementation.
        """
        device = torch.device("cuda:0")
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
                        idx2 = TestNearestNeighborPoints.nn_points_idx_naive(
                            x, y
                        )
                        self.assertTrue(idx1.size(1) == P1)
                        self.assertTrue(torch.all(idx1 == idx2))

    def test_nn_cuda_error(self):
        """
        Check that nn_points_idx throws an error if cpu tensors
        are given as input.
        """
        x = torch.randn(1, 1, 3)
        y = torch.randn(1, 1, 3)
        with self.assertRaises(Exception) as err:
            _C.nn_points_idx(x, y)
        self.assertTrue("Not implemented on the CPU" in str(err.exception))

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
