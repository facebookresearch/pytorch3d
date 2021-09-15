# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from common_testing import TestCaseMixin, get_random_cuda_device
from pytorch3d.ops.sample_farthest_points import sample_farthest_points_naive
from pytorch3d.ops.utils import masked_gather


class TestFPS(TestCaseMixin, unittest.TestCase):
    def test_simple(self):
        device = get_random_cuda_device()
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
        out_points, out_inds = sample_farthest_points_naive(points, K=2)
        self.assertClose(out_inds, expected_inds)

        # Gather the points
        expected_inds = expected_inds[..., None].expand(-1, -1, points.shape[-1])
        self.assertClose(out_points, points.gather(dim=1, index=expected_inds))

        # Different number of points sampled for each pointcloud in the batch
        expected_inds = torch.tensor(
            [[0, 4, 1], [0, 7, -1]], dtype=torch.int64, device=device
        )
        out_points, out_inds = sample_farthest_points_naive(points, K=[3, 2])
        self.assertClose(out_inds, expected_inds)

        # Gather the points
        expected_points = masked_gather(points, expected_inds)
        self.assertClose(out_points, expected_points)

    def test_random_heterogeneous(self):
        device = get_random_cuda_device()
        N, P, D, K = 5, 40, 5, 8
        points = torch.randn((N, P, D), device=device)
        out_points, out_idxs = sample_farthest_points_naive(points, K=K)
        self.assertTrue(out_idxs.min() >= 0)
        for n in range(N):
            self.assertEqual(out_idxs[n].ne(-1).sum(), K)

        lengths = torch.randint(low=1, high=P, size=(N,), device=device)
        out_points, out_idxs = sample_farthest_points_naive(points, lengths, K=50)

        for n in range(N):
            # Check that for heterogeneous batches, the max number of
            # selected points is less than the length
            self.assertTrue(out_idxs[n].ne(-1).sum() <= lengths[n])
            self.assertTrue(out_idxs[n].max() <= lengths[n])

            # Check there are no duplicate indices
            val_mask = out_idxs[n].ne(-1)
            vals, counts = torch.unique(out_idxs[n][val_mask], return_counts=True)
            self.assertTrue(counts.le(1).all())

    def test_errors(self):
        device = get_random_cuda_device()
        N, P, D, K = 5, 40, 5, 8
        points = torch.randn((N, P, D), device=device)
        wrong_batch_dim = torch.randint(low=1, high=K, size=(K,), device=device)

        # K has diferent batch dimension to points
        with self.assertRaisesRegex(ValueError, "K and points must have"):
            sample_farthest_points_naive(points, K=wrong_batch_dim)

        # lengths has diferent batch dimension to points
        with self.assertRaisesRegex(ValueError, "points and lengths must have"):
            sample_farthest_points_naive(points, lengths=wrong_batch_dim, K=K)

    def test_random_start(self):
        device = get_random_cuda_device()
        N, P, D, K = 5, 40, 5, 8
        points = torch.randn((N, P, D), device=device)
        out_points, out_idxs = sample_farthest_points_naive(
            points, K=K, random_start_point=True
        )
        # Check the first index is not 0 for all batch elements
        # when random_start_point = True
        self.assertTrue(out_idxs[:, 0].sum() > 0)
