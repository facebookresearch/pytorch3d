# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from common_testing import TestCaseMixin, get_random_cuda_device
from pytorch3d.ops.sample_farthest_points import (
    sample_farthest_points_naive,
    sample_farthest_points,
)
from pytorch3d.ops.utils import masked_gather


class TestFPS(TestCaseMixin, unittest.TestCase):
    def _test_simple(self, fps_func, device="cpu"):
        # fmt: off
        points = torch.tensor(
            [
                [
                    [-1.0, -1.0],  # noqa: E241, E201
                    [-1.3,  1.1],  # noqa: E241, E201
                    [ 0.2, -1.1],  # noqa: E241, E201
                    [ 0.0,  0.0],  # noqa: E241, E201
                    [ 1.3,  1.3],  # noqa: E241, E201
                    [ 1.0,  0.5],  # noqa: E241, E201
                    [-1.3,  0.2],  # noqa: E241, E201
                    [ 1.5, -0.5],  # noqa: E241, E201
                ],
                [
                    [-2.2, -2.4],  # noqa: E241, E201
                    [-2.1,  2.0],  # noqa: E241, E201
                    [ 2.2,  2.1],  # noqa: E241, E201
                    [ 2.1, -2.4],  # noqa: E241, E201
                    [ 0.4, -1.0],  # noqa: E241, E201
                    [ 0.3,  0.3],  # noqa: E241, E201
                    [ 1.2,  0.5],  # noqa: E241, E201
                    [ 4.5,  4.5],  # noqa: E241, E201
                ],
            ],
            dtype=torch.float32,
            device=device,
        )
        # fmt: on
        expected_inds = torch.tensor([[0, 4], [0, 7]], dtype=torch.int64, device=device)
        out_points, out_inds = fps_func(points, K=2)
        self.assertClose(out_inds, expected_inds)

        # Gather the points
        expected_inds = expected_inds[..., None].expand(-1, -1, points.shape[-1])
        self.assertClose(out_points, points.gather(dim=1, index=expected_inds))

        # Different number of points sampled for each pointcloud in the batch
        expected_inds = torch.tensor(
            [[0, 4, 1], [0, 7, -1]], dtype=torch.int64, device=device
        )
        out_points, out_inds = fps_func(points, K=[3, 2])
        self.assertClose(out_inds, expected_inds)

        # Gather the points
        expected_points = masked_gather(points, expected_inds)
        self.assertClose(out_points, expected_points)

    def _test_compare_random_heterogeneous(self, device="cpu"):
        N, P, D, K = 5, 20, 5, 8
        points = torch.randn((N, P, D), device=device, dtype=torch.float32)
        out_points_naive, out_idxs_naive = sample_farthest_points_naive(points, K=K)
        out_points, out_idxs = sample_farthest_points(points, K=K)
        self.assertTrue(out_idxs.min() >= 0)
        self.assertClose(out_idxs, out_idxs_naive)
        self.assertClose(out_points, out_points_naive)
        for n in range(N):
            self.assertEqual(out_idxs[n].ne(-1).sum(), K)

        # Test case where K > P
        K = 30
        points1 = torch.randn((N, P, D), dtype=torch.float32, device=device)
        points2 = points1.clone()
        points1.requires_grad = True
        points2.requires_grad = True
        lengths = torch.randint(low=1, high=P, size=(N,), device=device)
        out_points_naive, out_idxs_naive = sample_farthest_points_naive(
            points1, lengths, K=K
        )
        out_points, out_idxs = sample_farthest_points(points2, lengths, K=K)
        self.assertClose(out_idxs, out_idxs_naive)
        self.assertClose(out_points, out_points_naive)

        for n in range(N):
            # Check that for heterogeneous batches, the max number of
            # selected points is less than the length
            self.assertTrue(out_idxs[n].ne(-1).sum() <= lengths[n])
            self.assertTrue(out_idxs[n].max() <= lengths[n])

            # Check there are no duplicate indices
            val_mask = out_idxs[n].ne(-1)
            vals, counts = torch.unique(out_idxs[n][val_mask], return_counts=True)
            self.assertTrue(counts.le(1).all())

        # Check gradients
        grad_sampled_points = torch.ones((N, K, D), dtype=torch.float32, device=device)
        loss1 = (out_points_naive * grad_sampled_points).sum()
        loss1.backward()
        loss2 = (out_points * grad_sampled_points).sum()
        loss2.backward()
        self.assertClose(points1.grad, points2.grad, atol=5e-6)

    def _test_errors(self, fps_func, device="cpu"):
        N, P, D, K = 5, 40, 5, 8
        points = torch.randn((N, P, D), device=device)
        wrong_batch_dim = torch.randint(low=1, high=K, size=(K,), device=device)

        # K has diferent batch dimension to points
        with self.assertRaisesRegex(ValueError, "K and points must have"):
            sample_farthest_points_naive(points, K=wrong_batch_dim)

        # lengths has diferent batch dimension to points
        with self.assertRaisesRegex(ValueError, "points and lengths must have"):
            sample_farthest_points_naive(points, lengths=wrong_batch_dim, K=K)

    def _test_random_start(self, fps_func, device="cpu"):
        N, P, D, K = 5, 40, 5, 8
        points = torch.randn((N, P, D), device=device)
        out_points, out_idxs = sample_farthest_points_naive(
            points, K=K, random_start_point=True
        )
        # Check the first index is not 0 for all batch elements
        # when random_start_point = True
        self.assertTrue(out_idxs[:, 0].sum() > 0)

    def _test_gradcheck(self, fps_func, device="cpu"):
        N, P, D, K = 2, 5, 3, 2
        points = torch.randn(
            (N, P, D), dtype=torch.float32, device=device, requires_grad=True
        )
        torch.autograd.gradcheck(
            fps_func,
            (points, None, K),
            check_undefined_grad=False,
            eps=2e-3,
            atol=0.001,
        )

    def test_sample_farthest_points_naive(self):
        device = get_random_cuda_device()
        self._test_simple(sample_farthest_points_naive, device)
        self._test_errors(sample_farthest_points_naive, device)
        self._test_random_start(sample_farthest_points_naive, device)
        self._test_gradcheck(sample_farthest_points_naive, device)

    def test_sample_farthest_points_cpu(self):
        self._test_simple(sample_farthest_points, "cpu")
        self._test_errors(sample_farthest_points, "cpu")
        self._test_compare_random_heterogeneous("cpu")
        self._test_random_start(sample_farthest_points, "cpu")
        self._test_gradcheck(sample_farthest_points, "cpu")

    @staticmethod
    def sample_farthest_points_naive(N: int, P: int, D: int, K: int, device: str):
        device = torch.device(device)
        pts = torch.randn(
            N, P, D, dtype=torch.float32, device=device, requires_grad=True
        )
        grad_pts = torch.randn(N, K, D, dtype=torch.float32, device=device)
        torch.cuda.synchronize()

        def output():
            out_points, _ = sample_farthest_points_naive(pts, K=K)
            loss = (out_points * grad_pts).sum()
            loss.backward()
            torch.cuda.synchronize()

        return output

    @staticmethod
    def sample_farthest_points(N: int, P: int, D: int, K: int, device: str):
        device = torch.device(device)
        pts = torch.randn(
            N, P, D, dtype=torch.float32, device=device, requires_grad=True
        )
        grad_pts = torch.randn(N, K, D, dtype=torch.float32, device=device)
        torch.cuda.synchronize()

        def output():
            out_points, _ = sample_farthest_points(pts, K=K)
            loss = (out_points * grad_pts).sum()
            loss.backward()
            torch.cuda.synchronize()

        return output
