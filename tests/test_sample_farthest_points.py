# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from pytorch3d.io import load_obj
from pytorch3d.ops.sample_farthest_points import (
    sample_farthest_points,
    sample_farthest_points_naive,
)
from pytorch3d.ops.utils import masked_gather

from .common_testing import (
    get_pytorch3d_dir,
    get_random_cuda_device,
    get_tests_dir,
    TestCaseMixin,
)


DATA_DIR = get_tests_dir() / "data"
TUTORIAL_DATA_DIR = get_pytorch3d_dir() / "docs/tutorials/data"
DEBUG = False


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
        points = torch.randn((N, P, D), dtype=torch.float32, device=device)
        out_points, out_idxs = fps_func(points, K=K, random_start_point=True)
        # Check the first index is not 0 or the same number for all batch elements
        # when random_start_point = True
        self.assertTrue(out_idxs[:, 0].sum() > 0)
        self.assertFalse(out_idxs[:, 0].eq(out_idxs[0, 0]).all())

    def _test_gradcheck(self, fps_func, device="cpu"):
        N, P, D, K = 2, 10, 3, 2
        points = torch.randn(
            (N, P, D), dtype=torch.float32, device=device, requires_grad=True
        )
        lengths = torch.randint(low=1, high=P, size=(N,), device=device)
        torch.autograd.gradcheck(
            fps_func,
            (points, lengths, K),
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

    def test_sample_farthest_points_cuda(self):
        device = get_random_cuda_device()
        self._test_simple(sample_farthest_points, device)
        self._test_errors(sample_farthest_points, device)
        self._test_compare_random_heterogeneous(device)
        self._test_random_start(sample_farthest_points, device)
        self._test_gradcheck(sample_farthest_points, device)

    def test_cuda_vs_cpu(self):
        """
        Compare cuda vs cpu on a complex object
        """
        obj_filename = TUTORIAL_DATA_DIR / "cow_mesh/cow.obj"
        K = 250

        # Run on CPU
        device = "cpu"
        points, _, _ = load_obj(obj_filename, device=device, load_textures=False)
        points = points[None, ...]
        out_points_cpu, out_idxs_cpu = sample_farthest_points(points, K=K)

        # Run on GPU
        device = get_random_cuda_device()
        points_cuda = points.to(device)
        out_points_cuda, out_idxs_cuda = sample_farthest_points(points_cuda, K=K)

        # Check that the indices from CUDA and CPU match
        self.assertClose(out_idxs_cpu, out_idxs_cuda.cpu())

        # Check there are no duplicate indices
        val_mask = out_idxs_cuda[0].ne(-1)
        vals, counts = torch.unique(out_idxs_cuda[0][val_mask], return_counts=True)
        self.assertTrue(counts.le(1).all())

        # Plot all results
        if DEBUG:
            # mplot3d is required for 3d projection plots
            import matplotlib.pyplot as plt
            from mpl_toolkits import mplot3d  # noqa: F401

            # Move to cpu and convert to numpy for plotting
            points = points.squeeze()
            out_points_cpu = out_points_cpu.squeeze().numpy()
            out_points_cuda = out_points_cuda.squeeze().cpu().numpy()

            # Farthest point sampling CPU
            fig = plt.figure(figsize=plt.figaspect(1.0 / 3))
            ax1 = fig.add_subplot(1, 3, 1, projection="3d")
            ax1.scatter(*points.T, alpha=0.1)
            ax1.scatter(*out_points_cpu.T, color="black")
            ax1.set_title("FPS CPU")

            # Farthest point sampling CUDA
            ax2 = fig.add_subplot(1, 3, 2, projection="3d")
            ax2.scatter(*points.T, alpha=0.1)
            ax2.scatter(*out_points_cuda.T, color="red")
            ax2.set_title("FPS CUDA")

            # Random Sampling
            random_points = np.random.permutation(points)[:K]
            ax3 = fig.add_subplot(1, 3, 3, projection="3d")
            ax3.scatter(*points.T, alpha=0.1)
            ax3.scatter(*random_points.T, color="green")
            ax3.set_title("Random")

            # Save image
            filename = "DEBUG_fps.jpg"
            filepath = DATA_DIR / filename
            plt.savefig(filepath)

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
