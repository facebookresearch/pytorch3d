# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import numpy as np
import torch
from common_testing import TestCaseMixin, get_random_cuda_device
from pytorch3d import _C
from pytorch3d.renderer.points.rasterize_points import (
    rasterize_points,
    rasterize_points_python,
)
from pytorch3d.structures.pointclouds import Pointclouds


class TestRasterizePoints(TestCaseMixin, unittest.TestCase):
    def test_python_simple_cpu(self):
        self._simple_test_case(
            rasterize_points_python, torch.device("cpu"), bin_size=-1
        )

    def test_naive_simple_cpu(self):
        device = torch.device("cpu")
        self._simple_test_case(rasterize_points, device)

    def test_naive_simple_cuda(self):
        device = get_random_cuda_device()
        self._simple_test_case(rasterize_points, device, bin_size=0)

    def test_python_behind_camera(self):
        self._test_behind_camera(
            rasterize_points_python, torch.device("cpu"), bin_size=-1
        )

    def test_cpu_behind_camera(self):
        self._test_behind_camera(rasterize_points, torch.device("cpu"))

    def test_cuda_behind_camera(self):
        device = get_random_cuda_device()
        self._test_behind_camera(rasterize_points, device, bin_size=0)

    def test_cpp_vs_naive_vs_binned(self):
        # Make sure that the backward pass runs for all pathways
        N = 2
        P = 1000
        image_size = 32
        radius = 0.1
        points_per_pixel = 3
        points1 = torch.randn(P, 3, requires_grad=True)
        points2 = torch.randn(int(P / 2), 3, requires_grad=True)
        pointclouds = Pointclouds(points=[points1, points2])
        grad_zbuf = torch.randn(N, image_size, image_size, points_per_pixel)
        grad_dists = torch.randn(N, image_size, image_size, points_per_pixel)

        # Option I: CPU, naive
        idx1, zbuf1, dists1 = rasterize_points(
            pointclouds, image_size, radius, points_per_pixel, bin_size=0
        )
        loss = (zbuf1 * grad_zbuf).sum() + (dists1 * grad_dists).sum()
        loss.backward()
        grad1 = points1.grad.data.clone()

        # Option II: CUDA, naive
        points1_cuda = points1.cuda().detach().clone().requires_grad_(True)
        points2_cuda = points2.cuda().detach().clone().requires_grad_(True)
        pointclouds = Pointclouds(points=[points1_cuda, points2_cuda])
        grad_zbuf = grad_zbuf.cuda()
        grad_dists = grad_dists.cuda()
        idx2, zbuf2, dists2 = rasterize_points(
            pointclouds, image_size, radius, points_per_pixel, bin_size=0
        )
        loss = (zbuf2 * grad_zbuf).sum() + (dists2 * grad_dists).sum()
        loss.backward()
        idx2 = idx2.data.cpu().clone()
        zbuf2 = zbuf2.data.cpu().clone()
        dists2 = dists2.data.cpu().clone()
        grad2 = points1_cuda.grad.data.cpu().clone()

        # Option III: CUDA, binned
        points1_cuda = points1.cuda().detach().clone().requires_grad_(True)
        points2_cuda = points2.cuda().detach().clone().requires_grad_(True)
        pointclouds = Pointclouds(points=[points1_cuda, points2_cuda])
        idx3, zbuf3, dists3 = rasterize_points(
            pointclouds, image_size, radius, points_per_pixel, bin_size=32
        )
        loss = (zbuf3 * grad_zbuf).sum() + (dists3 * grad_dists).sum()
        points1.grad.data.zero_()
        loss.backward()
        idx3 = idx3.data.cpu().clone()
        zbuf3 = zbuf3.data.cpu().clone()
        dists3 = dists3.data.cpu().clone()
        grad3 = points1_cuda.grad.data.cpu().clone()

        # Make sure everything was the same
        idx12_same = (idx1 == idx2).all().item()
        idx13_same = (idx1 == idx3).all().item()
        zbuf12_same = (zbuf1 == zbuf2).all().item()
        zbuf13_same = (zbuf1 == zbuf3).all().item()
        dists12_diff = (dists1 - dists2).abs().max().item()
        dists13_diff = (dists1 - dists3).abs().max().item()
        self.assertTrue(idx12_same)
        self.assertTrue(idx13_same)
        self.assertTrue(zbuf12_same)
        self.assertTrue(zbuf13_same)
        self.assertTrue(dists12_diff < 1e-6)
        self.assertTrue(dists13_diff < 1e-6)

        diff12 = (grad1 - grad2).abs().max().item()
        diff13 = (grad1 - grad3).abs().max().item()
        diff23 = (grad2 - grad3).abs().max().item()
        self.assertTrue(diff12 < 5e-6)
        self.assertTrue(diff13 < 5e-6)
        self.assertTrue(diff23 < 5e-6)

    def test_python_vs_cpu_naive(self):
        torch.manual_seed(231)
        image_size = 32
        radius = 0.1
        points_per_pixel = 3

        # Test a batch of homogeneous point clouds.
        N = 2
        P = 17
        points = torch.randn(N, P, 3, requires_grad=True)
        pointclouds = Pointclouds(points=points)
        args = (pointclouds, image_size, radius, points_per_pixel)
        self._compare_impls(
            rasterize_points_python,
            rasterize_points,
            args,
            args,
            points,
            points,
            compare_grads=True,
        )

        # Test a batch of heterogeneous point clouds.
        P2 = 10
        points1 = torch.randn(P, 3, requires_grad=True)
        points2 = torch.randn(P2, 3)
        pointclouds = Pointclouds(points=[points1, points2])
        args = (pointclouds, image_size, radius, points_per_pixel)
        self._compare_impls(
            rasterize_points_python,
            rasterize_points,
            args,
            args,
            points1,  # check gradients for first element in batch
            points1,
            compare_grads=True,
        )

    def test_cpu_vs_cuda_naive(self):
        torch.manual_seed(231)
        image_size = 64
        radius = 0.1
        points_per_pixel = 5

        # Test homogeneous point cloud batch.
        N = 2
        P = 1000
        bin_size = 0
        points_cpu = torch.rand(N, P, 3, requires_grad=True)
        points_cuda = points_cpu.cuda().detach().requires_grad_(True)
        pointclouds_cpu = Pointclouds(points=points_cpu)
        pointclouds_cuda = Pointclouds(points=points_cuda)
        args_cpu = (pointclouds_cpu, image_size, radius, points_per_pixel, bin_size)
        args_cuda = (pointclouds_cuda, image_size, radius, points_per_pixel, bin_size)
        self._compare_impls(
            rasterize_points,
            rasterize_points,
            args_cpu,
            args_cuda,
            points_cpu,
            points_cuda,
            compare_grads=True,
        )

    def _compare_impls(
        self,
        fn1,
        fn2,
        args1,
        args2,
        grad_var1=None,
        grad_var2=None,
        compare_grads=False,
    ):
        idx1, zbuf1, dist1 = fn1(*args1)
        torch.manual_seed(231)
        grad_zbuf = torch.randn_like(zbuf1)
        grad_dist = torch.randn_like(dist1)
        loss = (zbuf1 * grad_zbuf).sum() + (dist1 * grad_dist).sum()
        if compare_grads:
            loss.backward()
            grad_points1 = grad_var1.grad.data.clone().cpu()

        idx2, zbuf2, dist2 = fn2(*args2)
        grad_zbuf = grad_zbuf.to(zbuf2)
        grad_dist = grad_dist.to(dist2)
        loss = (zbuf2 * grad_zbuf).sum() + (dist2 * grad_dist).sum()
        if compare_grads:
            # clear points1.grad in case args1 and args2 reused the same tensor
            grad_var1.grad.data.zero_()
            loss.backward()
            grad_points2 = grad_var2.grad.data.clone().cpu()

        self.assertEqual((idx1.cpu() == idx2.cpu()).all().item(), 1)
        self.assertEqual((zbuf1.cpu() == zbuf2.cpu()).all().item(), 1)
        self.assertClose(dist1.cpu(), dist2.cpu())
        if compare_grads:
            self.assertClose(grad_points1, grad_points2, atol=2e-6)

    def test_bin_size_error(self):
        points = Pointclouds(points=torch.rand(5, 100, 3))
        image_size = 1024
        bin_size = 16
        with self.assertRaisesRegex(ValueError, "bin_size too small"):
            rasterize_points(points, image_size, 0.0, 2, bin_size=bin_size)

    def _test_behind_camera(self, rasterize_points_fn, device, bin_size=None):
        # Test case where all points are behind the camera -- nothing should
        # get rasterized
        N = 2
        P = 32
        xy = torch.randn(N, P, 2)
        z = torch.randn(N, P, 1).abs().mul(-1)  # Make them all negative
        points = torch.cat([xy, z], dim=2).to(device)
        image_size = 16
        points_per_pixel = 3
        radius = 0.2
        idx_expected = torch.full(
            (N, 16, 16, 3), fill_value=-1, dtype=torch.int32, device=device
        )
        zbuf_expected = torch.full(
            (N, 16, 16, 3), fill_value=-1, dtype=torch.float32, device=device
        )
        dists_expected = zbuf_expected.clone()
        pointclouds = Pointclouds(points=points)
        if bin_size == -1:
            # simple python case with no binning
            idx, zbuf, dists = rasterize_points_fn(
                pointclouds, image_size, radius, points_per_pixel
            )
        else:
            idx, zbuf, dists = rasterize_points_fn(
                pointclouds, image_size, radius, points_per_pixel, bin_size
            )
        idx_same = (idx == idx_expected).all().item() == 1
        zbuf_same = (zbuf == zbuf_expected).all().item() == 1

        self.assertTrue(idx_same)
        self.assertTrue(zbuf_same)
        self.assertClose(dists, dists_expected)

    def _simple_test_case(self, rasterize_points_fn, device, bin_size=0):
        # Create two pointclouds with different numbers of points.
        # fmt: off
        points1 = torch.tensor(
            [
                [0.0, 0.0,  0.0],  # noqa: E241
                [0.4, 0.0,  0.1],  # noqa: E241
                [0.0, 0.4,  0.2],  # noqa: E241
                [0.0, 0.0, -0.1],  # noqa: E241 Points with negative z should be skippped
            ],
            device=device,
        )
        points2 = torch.tensor(
            [
                [0.0, 0.0,  0.0],  # noqa: E241
                [0.4, 0.0,  0.1],  # noqa: E241
                [0.0, 0.4,  0.2],  # noqa: E241
                [0.0, 0.0, -0.1],  # noqa: E241 Points with negative z should be skippped
                [0.0, 0.0, -0.7],  # noqa: E241 Points with negative z should be skippped
            ],
            device=device,
        )
        # fmt: on
        pointclouds = Pointclouds(points=[points1, points2])

        image_size = 5
        points_per_pixel = 2
        radius = 0.5

        # The expected output values. Note that in the outputs, the world space
        # +Y is up, and the world space +X is left.
        idx1_expected = torch.full(
            (1, 5, 5, 2), fill_value=-1, dtype=torch.int32, device=device
        )
        # fmt: off
        idx1_expected[0, :, :, 0] = torch.tensor([
            [-1, -1,  2, -1, -1],  # noqa: E241
            [-1,  1,  0,  2, -1],  # noqa: E241
            [ 1,  0,  0,  0, -1],  # noqa: E241 E201
            [-1,  1,  0, -1, -1],  # noqa: E241
            [-1, -1, -1, -1, -1],  # noqa: E241
        ], device=device)
        idx1_expected[0, :, :, 1] = torch.tensor([
            [-1, -1, -1, -1, -1],  # noqa: E241
            [-1,  2,  2, -1, -1],  # noqa: E241
            [-1,  1,  1, -1, -1],  # noqa: E241
            [-1, -1, -1, -1, -1],  # noqa: E241
            [-1, -1, -1, -1, -1],  # noqa: E241
        ], device=device)
        # fmt: on

        zbuf1_expected = torch.full(
            (1, 5, 5, 2), fill_value=100, dtype=torch.float32, device=device
        )
        # fmt: off
        zbuf1_expected[0, :, :, 0] = torch.tensor([
            [-1.0, -1.0,  0.2, -1.0, -1.0],  # noqa: E241
            [-1.0,  0.1,  0.0,  0.2, -1.0],  # noqa: E241
            [ 0.1,  0.0,  0.0,  0.0, -1.0],  # noqa: E241 E201
            [-1.0,  0.1,  0.0, -1.0, -1.0],  # noqa: E241
            [-1.0, -1.0, -1.0, -1.0, -1.0]   # noqa: E241
        ], device=device)
        zbuf1_expected[0, :, :, 1] = torch.tensor([
            [-1.0, -1.0, -1.0, -1.0, -1.0],  # noqa: E241
            [-1.0,  0.2,  0.2, -1.0, -1.0],  # noqa: E241
            [-1.0,  0.1,  0.1, -1.0, -1.0],  # noqa: E241
            [-1.0, -1.0, -1.0, -1.0, -1.0],  # noqa: E241
            [-1.0, -1.0, -1.0, -1.0, -1.0],  # noqa: E241
        ], device=device)
        # fmt: on

        dists1_expected = torch.zeros((5, 5, 2), dtype=torch.float32, device=device)
        # fmt: off
        dists1_expected[:, :, 0] = torch.tensor([
            [-1.00, -1.00,  0.16, -1.00, -1.00],  # noqa: E241
            [-1.00,  0.16,  0.16,  0.16, -1.00],  # noqa: E241
            [ 0.16,  0.16,  0.00,  0.16, -1.00],  # noqa: E241 E201
            [-1.00,  0.16,  0.16, -1.00, -1.00],  # noqa: E241
            [-1.00, -1.00, -1.00, -1.00, -1.00],  # noqa: E241
        ], device=device)
        dists1_expected[:, :, 1] = torch.tensor([
            [-1.00, -1.00, -1.00, -1.00, -1.00],  # noqa: E241
            [-1.00,  0.16,  0.00, -1.00, -1.00],  # noqa: E241
            [-1.00,  0.00,  0.16, -1.00, -1.00],  # noqa: E241
            [-1.00, -1.00, -1.00, -1.00, -1.00],  # noqa: E241
            [-1.00, -1.00, -1.00, -1.00, -1.00],  # noqa: E241
        ], device=device)
        # fmt: on

        if bin_size == -1:
            # simple python case with no binning
            idx, zbuf, dists = rasterize_points_fn(
                pointclouds, image_size, radius, points_per_pixel
            )
        else:
            idx, zbuf, dists = rasterize_points_fn(
                pointclouds, image_size, radius, points_per_pixel, bin_size
            )

        # check first point cloud
        idx_same = (idx[0, ...] == idx1_expected).all().item() == 1
        if idx_same == 0:
            print(idx[0, :, :, 0])
            print(idx[0, :, :, 1])
        zbuf_same = (zbuf[0, ...] == zbuf1_expected).all().item() == 1
        self.assertClose(dists[0, ...], dists1_expected)
        self.assertTrue(idx_same)
        self.assertTrue(zbuf_same)

        # Check second point cloud - the indices in idx refer to points in the
        # pointclouds.points_packed() tensor. In the second point cloud,
        # two points are behind the screen - the expected indices are the same
        # the first pointcloud but offset by the number of points in the
        # first pointcloud.
        num_points_per_cloud = pointclouds.num_points_per_cloud()
        idx1_expected[idx1_expected >= 0] += num_points_per_cloud[0]

        idx_same = (idx[1, ...] == idx1_expected).all().item() == 1
        zbuf_same = (zbuf[1, ...] == zbuf1_expected).all().item() == 1
        self.assertTrue(idx_same)
        self.assertTrue(zbuf_same)
        self.assertClose(dists[1, ...], dists1_expected)

    def test_coarse_cpu(self):
        return self._test_coarse_rasterize(torch.device("cpu"))

    def test_coarse_cuda(self):
        device = get_random_cuda_device()
        return self._test_coarse_rasterize(device)

    def test_compare_coarse_cpu_vs_cuda(self):
        torch.manual_seed(231)
        N = 3
        max_P = 1000
        image_size = 64
        radius = 0.1
        bin_size = 16
        max_points_per_bin = 500

        # create heterogeneous point clouds
        points = []
        for _ in range(N):
            p = np.random.choice(max_P)
            points.append(torch.randn(p, 3))

        pointclouds = Pointclouds(points=points)
        points_packed = pointclouds.points_packed()
        cloud_to_packed_first_idx = pointclouds.cloud_to_packed_first_idx()
        num_points_per_cloud = pointclouds.num_points_per_cloud()
        args = (
            points_packed,
            cloud_to_packed_first_idx,
            num_points_per_cloud,
            image_size,
            radius,
            bin_size,
            max_points_per_bin,
        )
        bp_cpu = _C._rasterize_points_coarse(*args)

        device = get_random_cuda_device()
        pointclouds_cuda = pointclouds.to(device)
        points_packed = pointclouds_cuda.points_packed()
        cloud_to_packed_first_idx = pointclouds_cuda.cloud_to_packed_first_idx()
        num_points_per_cloud = pointclouds_cuda.num_points_per_cloud()
        args = (
            points_packed,
            cloud_to_packed_first_idx,
            num_points_per_cloud,
            image_size,
            radius,
            bin_size,
            max_points_per_bin,
        )
        bp_cuda = _C._rasterize_points_coarse(*args)

        # Bin points might not be the same: CUDA version might write them in
        # any order. But if we sort the non-(-1) elements of the CUDA output
        # then they should be the same.
        for n in range(N):
            for by in range(bp_cpu.shape[1]):
                for bx in range(bp_cpu.shape[2]):
                    K = (bp_cpu[n, by, bx] != -1).sum().item()
                    idxs_cpu = bp_cpu[n, by, bx].tolist()
                    idxs_cuda = bp_cuda[n, by, bx].tolist()
                    idxs_cuda[:K] = sorted(idxs_cuda[:K])
                    self.assertEqual(idxs_cpu, idxs_cuda)

    def _test_coarse_rasterize(self, device):
        #
        #
        #           |2                  (4)
        #           |
        #           |
        #           |
        #           |1
        #           |
        #           |    (1)
        #        (2)|
        # _________(5)___(0)_______________
        # -1        |           1         2
        #           |
        #           |            (3)
        #           |
        #           |-1
        #
        # Locations of the points are shown by o. The screen bounding box
        # is between [-1, 1] in both the x and y directions.
        #
        # These points are interesting because:
        # (0) Falls into two bins;
        # (1) and (2) fall into one bin;
        # (3) is out-of-bounds, but its disk is in-bounds;
        # (4) is out-of-bounds, and its entire disk is also out-of-bounds
        # (5) has a negative z-value, so it should be skipped
        # fmt: off
        points = torch.tensor(
            [
                [ 0.5,  0.0,  0.0],  # noqa: E241, E201
                [ 0.5,  0.5,  0.1],  # noqa: E241, E201
                [-0.3,  0.4,  0.0],  # noqa: E241
                [ 1.1, -0.5,  0.2],  # noqa: E241, E201
                [ 2.0,  2.0,  0.3],  # noqa: E241, E201
                [ 0.0,  0.0, -0.1],  # noqa: E241, E201
            ],
            device=device
        )
        # fmt: on
        image_size = 16
        radius = 0.2
        bin_size = 8
        max_points_per_bin = 5

        bin_points_expected = -1 * torch.ones(
            1, 2, 2, 5, dtype=torch.int32, device=device
        )
        # Note that the order is only deterministic here for CUDA if all points
        # fit in one chunk. This will the the case for this small example, but
        # to properly exercise coordianted writes among multiple chunks we need
        # to use a bigger test case.
        bin_points_expected[0, 0, 1, :2] = torch.tensor([0, 3])
        bin_points_expected[0, 1, 0, 0] = torch.tensor([2])
        bin_points_expected[0, 1, 1, :2] = torch.tensor([0, 1])

        pointclouds = Pointclouds(points=[points])
        args = (
            pointclouds.points_packed(),
            pointclouds.cloud_to_packed_first_idx(),
            pointclouds.num_points_per_cloud(),
            image_size,
            radius,
            bin_size,
            max_points_per_bin,
        )
        bin_points = _C._rasterize_points_coarse(*args)
        bin_points_same = (bin_points == bin_points_expected).all()

        self.assertTrue(bin_points_same.item() == 1)
