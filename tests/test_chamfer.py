# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures.pointclouds import Pointclouds

from .common_testing import get_random_cuda_device, TestCaseMixin


# Output of init_pointclouds
points_normals = namedtuple(
    "points_normals", "p1_lengths p2_lengths cloud1 cloud2 p1 p2 n1 n2 weights"
)


class TestChamfer(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    @staticmethod
    def init_pointclouds(
        N, P1, P2, device, requires_grad: bool = True, allow_empty: bool = True
    ):
        """
        Create 2 pointclouds object and associated padded points/normals tensors by
        starting from lists. The clouds and tensors have the same data. The
        leaf nodes for the clouds are a list of tensors. The padded tensor can be
        used directly as a leaf node.
        """
        low = 0 if allow_empty else 1
        p1_lengths = torch.randint(low, P1, size=(N,), dtype=torch.int64, device=device)
        p2_lengths = torch.randint(low, P2, size=(N,), dtype=torch.int64, device=device)
        P1 = p1_lengths.max().item()
        P2 = p2_lengths.max().item()
        weights = torch.rand((N,), dtype=torch.float32, device=device)

        # list of points and normals tensors
        p1 = torch.rand((N, P1, 3), dtype=torch.float32, device=device)
        p2 = torch.rand((N, P2, 3), dtype=torch.float32, device=device)
        n1 = torch.rand((N, P1, 3), dtype=torch.float32, device=device)
        n2 = torch.rand((N, P2, 3), dtype=torch.float32, device=device)
        n1 /= n1.norm(dim=-1, p=2, keepdim=True)
        n2 /= n2.norm(dim=-1, p=2, keepdim=True)

        p1_list = []
        p2_list = []
        n1_list = []
        n2_list = []
        for i in range(N):
            l1 = p1_lengths[i]
            l2 = p2_lengths[i]
            p1_list.append(p1[i, :l1].clone())
            p2_list.append(p2[i, :l2].clone())
            n1_list.append(n1[i, :l1].clone())
            n2_list.append(n2[i, :l2].clone())

        # Set requires_grad for all tensors in the lists and
        # padded tensors.
        if requires_grad:
            for p in p2_list + p1_list + n1_list + n2_list + [p1, p2, n1, n2]:
                p.requires_grad = True

        # Create pointclouds objects
        cloud1 = Pointclouds(points=p1_list, normals=n1_list)
        cloud2 = Pointclouds(points=p2_list, normals=n2_list)

        # Return pointclouds objects and padded tensors
        return points_normals(
            p1_lengths=p1_lengths,
            p2_lengths=p2_lengths,
            cloud1=cloud1,
            cloud2=cloud2,
            p1=p1,
            p2=p2,
            n1=n1,
            n2=n2,
            weights=weights,
        )

    @staticmethod
    def chamfer_distance_naive_pointclouds(
        p1, p2, norm: int = 2, device="cpu", abs_cosine=True
    ):
        """
        Naive iterative implementation of nearest neighbor and chamfer distance.
        x and y are assumed to be pointclouds objects with points and optionally normals.
        This functions supports heterogeneous pointclouds in a batch.
        Returns lists of the unreduced loss and loss_normals.
        """
        x = p1.points_padded()
        y = p2.points_padded()
        N, P1, D = x.shape
        P2 = y.size(1)
        x_lengths = p1.num_points_per_cloud()
        y_lengths = p2.num_points_per_cloud()
        x_normals = p1.normals_padded()
        y_normals = p2.normals_padded()

        return_normals = x_normals is not None and y_normals is not None

        # Initialize all distances to + inf
        dist = torch.ones((N, P1, P2), dtype=torch.float32, device=device) * np.inf

        x_mask = (
            torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
        )  # shape [N, P1]
        y_mask = (
            torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
        )  # shape [N, P2]

        is_x_heterogeneous = (x_lengths != P1).any()
        is_y_heterogeneous = (y_lengths != P2).any()
        # Only calculate the distances for the points which are not masked
        for n in range(N):
            for i1 in range(x_lengths[n]):
                for i2 in range(y_lengths[n]):
                    if norm == 2:
                        dist[n, i1, i2] = torch.sum((x[n, i1, :] - y[n, i2, :]) ** 2)
                    elif norm == 1:
                        dist[n, i1, i2] = torch.sum(
                            torch.abs(x[n, i1, :] - y[n, i2, :])
                        )
                    else:
                        raise ValueError("No support for norm %d" % (norm))

        x_dist = torch.min(dist, dim=2)[0]  # (N, P1)
        y_dist = torch.min(dist, dim=1)[0]  # (N, P2)

        if is_x_heterogeneous:
            x_dist[x_mask] = 0.0
        if is_y_heterogeneous:
            y_dist[y_mask] = 0.0

        loss = [x_dist, y_dist]

        lnorm = [x.new_zeros(()), x.new_zeros(())]

        if return_normals:
            x_index = dist.argmin(2).view(N, P1, 1).expand(N, P1, 3)
            y_index = dist.argmin(1).view(N, P2, 1).expand(N, P2, 3)
            cosine_sim1 = F.cosine_similarity(
                x_normals, y_normals.gather(1, x_index), dim=2, eps=1e-6
            )
            cosine_sim2 = F.cosine_similarity(
                y_normals, x_normals.gather(1, y_index), dim=2, eps=1e-6
            )

            if abs_cosine:
                lnorm1 = 1 - torch.abs(cosine_sim1)
                lnorm2 = 1 - torch.abs(cosine_sim2)
            else:
                lnorm1 = 1 - cosine_sim1
                lnorm2 = 1 - cosine_sim2

            if is_x_heterogeneous:
                lnorm1[x_mask] = 0.0
            if is_y_heterogeneous:
                lnorm2[y_mask] = 0.0

            lnorm = [lnorm1, lnorm2]  # [(N, P1), (N, P2)]

        return loss, lnorm

    @staticmethod
    def chamfer_distance_naive(
        x, y, x_normals=None, y_normals=None, norm: int = 2, abs_cosine=True
    ):
        """
        Naive iterative implementation of nearest neighbor and chamfer distance.
        Returns lists of the unreduced loss and loss_normals. This naive
        version only supports homogeneous pointcouds in a batch.
        """
        N, P1, D = x.shape
        P2 = y.size(1)
        device = x.device
        return_normals = x_normals is not None and y_normals is not None
        dist = torch.zeros((N, P1, P2), dtype=torch.float32, device=device)

        for n in range(N):
            for i1 in range(P1):
                for i2 in range(P2):
                    if norm == 2:
                        dist[n, i1, i2] = torch.sum((x[n, i1, :] - y[n, i2, :]) ** 2)
                    elif norm == 1:
                        dist[n, i1, i2] = torch.sum(
                            torch.abs(x[n, i1, :] - y[n, i2, :])
                        )
                    else:
                        raise ValueError("No support for norm %d" % (norm))

        loss = [
            torch.min(dist, dim=2)[0],  # (N, P1)
            torch.min(dist, dim=1)[0],  # (N, P2)
        ]
        lnorm = [x.new_zeros(()), x.new_zeros(())]

        if return_normals:
            x_index = dist.argmin(2).view(N, P1, 1).expand(N, P1, 3)
            y_index = dist.argmin(1).view(N, P2, 1).expand(N, P2, 3)

            cosine_sim1 = F.cosine_similarity(
                x_normals, y_normals.gather(1, x_index), dim=2, eps=1e-6
            )
            cosine_sim2 = F.cosine_similarity(
                y_normals, x_normals.gather(1, y_index), dim=2, eps=1e-6
            )

            if abs_cosine:
                lnorm1 = 1 - torch.abs(cosine_sim1)
                lnorm2 = 1 - torch.abs(cosine_sim2)
            else:
                lnorm1 = 1 - cosine_sim1
                lnorm2 = 1 - cosine_sim2

            lnorm = [lnorm1, lnorm2]  # [(N, P1), (N, P2)]

        return loss, lnorm

    def test_chamfer_point_batch_reduction_mean(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for the default settings (point_reduction = "mean" and batch_reduction = "mean")
        and no normals.
        This tests only uses homogeneous pointclouds.
        """
        N, max_P1, max_P2 = 7, 10, 18
        device = get_random_cuda_device()

        for norm in [1, 2]:
            points_normals = TestChamfer.init_pointclouds(N, max_P1, max_P2, device)
            p1 = points_normals.p1
            p2 = points_normals.p2
            weights = points_normals.weights
            p11 = p1.detach().clone()
            p22 = p2.detach().clone()
            p11.requires_grad = True
            p22.requires_grad = True
            P1 = p1.shape[1]
            P2 = p2.shape[1]

            pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
                p1, p2, norm=norm
            )

            # point_reduction = "mean".
            loss, loss_norm = chamfer_distance(p11, p22, weights=weights, norm=norm)
            pred_loss = pred_loss[0].sum(1) / P1 + pred_loss[1].sum(1) / P2
            pred_loss *= weights
            pred_loss = pred_loss.sum() / weights.sum()

            self.assertClose(loss, pred_loss)
            self.assertTrue(loss_norm is None)

            # Check gradients
            self._check_gradients(loss, None, pred_loss, None, p1, p11, p2, p22)

    def test_chamfer_vs_naive_pointcloud(self):
        """
        Test the default settings for chamfer_distance
        (point reduction = "mean" and batch_reduction="mean") but with heterogeneous
        pointclouds as input. Compare with the naive implementation of chamfer
        which supports heterogeneous pointcloud objects.
        """
        N, max_P1, max_P2 = 3, 70, 70
        device = get_random_cuda_device()

        for norm in [1, 2]:
            points_normals = TestChamfer.init_pointclouds(N, max_P1, max_P2, device)
            weights = points_normals.weights
            x_lengths = points_normals.p1_lengths
            y_lengths = points_normals.p2_lengths

            # Chamfer with tensors as input for heterogeneous pointclouds.
            cham_tensor, norm_tensor = chamfer_distance(
                points_normals.p1,
                points_normals.p2,
                x_normals=points_normals.n1,
                y_normals=points_normals.n2,
                x_lengths=points_normals.p1_lengths,
                y_lengths=points_normals.p2_lengths,
                weights=weights,
                norm=norm,
            )

            # Chamfer with pointclouds as input.
            pred_loss, pred_norm_loss = TestChamfer.chamfer_distance_naive_pointclouds(
                points_normals.cloud1, points_normals.cloud2, norm=norm, device=device
            )

            # Mean reduction point loss.
            pred_loss[0] *= weights.view(N, 1)
            pred_loss[1] *= weights.view(N, 1)
            pred_loss_mean = (
                pred_loss[0].sum(1) / x_lengths + pred_loss[1].sum(1) / y_lengths
            )
            pred_loss_mean = pred_loss_mean.sum()
            pred_loss_mean /= weights.sum()

            # Mean reduction norm loss.
            pred_norm_loss[0] *= weights.view(N, 1)
            pred_norm_loss[1] *= weights.view(N, 1)
            pred_norm_loss_mean = (
                pred_norm_loss[0].sum(1) / x_lengths
                + pred_norm_loss[1].sum(1) / y_lengths
            )
            pred_norm_loss_mean = pred_norm_loss_mean.sum() / weights.sum()

            self.assertClose(pred_loss_mean, cham_tensor)
            self.assertClose(pred_norm_loss_mean, norm_tensor)

            self._check_gradients(
                cham_tensor,
                norm_tensor,
                pred_loss_mean,
                pred_norm_loss_mean,
                points_normals.cloud1.points_list(),
                points_normals.p1,
                points_normals.cloud2.points_list(),
                points_normals.p2,
                points_normals.cloud1.normals_list(),
                points_normals.n1,
                points_normals.cloud2.normals_list(),
                points_normals.n2,
                x_lengths,
                y_lengths,
            )

    def test_single_directional_chamfer_vs_naive_pointcloud(self):
        """
        Test the single directional settings for chamfer_distance
        (point reduction = "mean" and batch_reduction="mean") but with heterogeneous
        pointclouds as input. Compare with the naive implementation of chamfer
        which supports heterogeneous pointcloud objects.
        """
        N, max_P1, max_P2 = 3, 70, 70
        device = get_random_cuda_device()

        for norm in [1, 2]:
            for abs_cosine in [True, False]:
                points_normals = TestChamfer.init_pointclouds(N, max_P1, max_P2, device)
                weights = points_normals.weights
                x_lengths = points_normals.p1_lengths
                y_lengths = points_normals.p2_lengths

                # Chamfer with tensors as input for heterogeneous pointclouds.
                cham_tensor, norm_tensor = chamfer_distance(
                    points_normals.p1,
                    points_normals.p2,
                    x_normals=points_normals.n1,
                    y_normals=points_normals.n2,
                    x_lengths=points_normals.p1_lengths,
                    y_lengths=points_normals.p2_lengths,
                    weights=weights,
                    norm=norm,
                    single_directional=True,
                    abs_cosine=abs_cosine,
                )

                # Chamfer with pointclouds as input.
                (
                    pred_loss,
                    pred_norm_loss,
                ) = TestChamfer.chamfer_distance_naive_pointclouds(
                    points_normals.cloud1,
                    points_normals.cloud2,
                    norm=norm,
                    device=device,
                    abs_cosine=abs_cosine,
                )

                # Mean reduction point loss.
                pred_loss[0] *= weights.view(N, 1)
                pred_loss_mean = pred_loss[0].sum(1) / x_lengths
                pred_loss_mean = pred_loss_mean.sum()
                pred_loss_mean /= weights.sum()

                # Mean reduction norm loss.
                pred_norm_loss[0] *= weights.view(N, 1)
                pred_norm_loss_mean = pred_norm_loss[0].sum(1) / x_lengths
                pred_norm_loss_mean = pred_norm_loss_mean.sum() / weights.sum()

                self.assertClose(pred_loss_mean, cham_tensor)
                self.assertClose(pred_norm_loss_mean, norm_tensor)

                self._check_gradients(
                    cham_tensor,
                    norm_tensor,
                    pred_loss_mean,
                    pred_norm_loss_mean,
                    points_normals.cloud1.points_list(),
                    points_normals.p1,
                    points_normals.cloud2.points_list(),
                    points_normals.p2,
                    points_normals.cloud1.normals_list(),
                    points_normals.n1,
                    points_normals.cloud2.normals_list(),
                    points_normals.n2,
                    x_lengths,
                    y_lengths,
                )

    def test_chamfer_pointcloud_object_withnormals(self):
        N = 5
        P1, P2 = 100, 100
        device = get_random_cuda_device()

        reductions = [
            ("sum", "sum"),
            ("mean", "sum"),
            ("sum", "mean"),
            ("mean", "mean"),
            ("sum", None),
            ("mean", None),
            (None, None),
        ]
        for point_reduction, batch_reduction in reductions:
            # Reinitialize all the tensors so that the
            # backward pass can be computed.
            points_normals = TestChamfer.init_pointclouds(
                N, P1, P2, device, allow_empty=False
            )

            # Chamfer with pointclouds as input.
            cham_cloud, norm_cloud = chamfer_distance(
                points_normals.cloud1,
                points_normals.cloud2,
                point_reduction=point_reduction,
                batch_reduction=batch_reduction,
            )

            # Chamfer with tensors as input.
            cham_tensor, norm_tensor = chamfer_distance(
                points_normals.p1,
                points_normals.p2,
                x_lengths=points_normals.p1_lengths,
                y_lengths=points_normals.p2_lengths,
                x_normals=points_normals.n1,
                y_normals=points_normals.n2,
                point_reduction=point_reduction,
                batch_reduction=batch_reduction,
            )

            if point_reduction is None:
                cham_tensor_bidirectional = torch.hstack(
                    [cham_tensor[0], cham_tensor[1]]
                )
                norm_tensor_bidirectional = torch.hstack(
                    [norm_tensor[0], norm_tensor[1]]
                )
                cham_cloud_bidirectional = torch.hstack([cham_cloud[0], cham_cloud[1]])
                norm_cloud_bidirectional = torch.hstack([norm_cloud[0], norm_cloud[1]])
                self.assertClose(cham_cloud_bidirectional, cham_tensor_bidirectional)
                self.assertClose(norm_cloud_bidirectional, norm_tensor_bidirectional)
                self._check_gradients(
                    cham_tensor_bidirectional,
                    norm_tensor_bidirectional,
                    cham_cloud_bidirectional,
                    norm_cloud_bidirectional,
                    points_normals.cloud1.points_list(),
                    points_normals.p1,
                    points_normals.cloud2.points_list(),
                    points_normals.p2,
                    points_normals.cloud1.normals_list(),
                    points_normals.n1,
                    points_normals.cloud2.normals_list(),
                    points_normals.n2,
                    points_normals.p1_lengths,
                    points_normals.p2_lengths,
                )
            else:
                self.assertClose(cham_cloud, cham_tensor)
                self.assertClose(norm_cloud, norm_tensor)
                self._check_gradients(
                    cham_tensor,
                    norm_tensor,
                    cham_cloud,
                    norm_cloud,
                    points_normals.cloud1.points_list(),
                    points_normals.p1,
                    points_normals.cloud2.points_list(),
                    points_normals.p2,
                    points_normals.cloud1.normals_list(),
                    points_normals.n1,
                    points_normals.cloud2.normals_list(),
                    points_normals.n2,
                    points_normals.p1_lengths,
                    points_normals.p2_lengths,
                )

    def test_chamfer_pointcloud_object_nonormals(self):
        N = 5
        P1, P2 = 100, 100
        device = get_random_cuda_device()

        reductions = [
            ("sum", "sum"),
            ("mean", "sum"),
            ("sum", "mean"),
            ("mean", "mean"),
            ("sum", None),
            ("mean", None),
            (None, None),
        ]
        for point_reduction, batch_reduction in reductions:
            # Reinitialize all the tensors so that the
            # backward pass can be computed.
            points_normals = TestChamfer.init_pointclouds(
                N, P1, P2, device, allow_empty=False
            )

            # Chamfer with pointclouds as input.
            cham_cloud, _ = chamfer_distance(
                points_normals.cloud1,
                points_normals.cloud2,
                point_reduction=point_reduction,
                batch_reduction=batch_reduction,
            )

            # Chamfer with tensors as input.
            cham_tensor, _ = chamfer_distance(
                points_normals.p1,
                points_normals.p2,
                x_lengths=points_normals.p1_lengths,
                y_lengths=points_normals.p2_lengths,
                point_reduction=point_reduction,
                batch_reduction=batch_reduction,
            )

            if point_reduction is None:
                cham_tensor_bidirectional = torch.hstack(
                    [cham_tensor[0], cham_tensor[1]]
                )
                cham_cloud_bidirectional = torch.hstack([cham_cloud[0], cham_cloud[1]])
                self.assertClose(cham_cloud_bidirectional, cham_tensor_bidirectional)
                self._check_gradients(
                    cham_tensor_bidirectional,
                    None,
                    cham_cloud_bidirectional,
                    None,
                    points_normals.cloud1.points_list(),
                    points_normals.p1,
                    points_normals.cloud2.points_list(),
                    points_normals.p2,
                    lengths1=points_normals.p1_lengths,
                    lengths2=points_normals.p2_lengths,
                )
            else:
                self.assertClose(cham_cloud, cham_tensor)
                self._check_gradients(
                    cham_tensor,
                    None,
                    cham_cloud,
                    None,
                    points_normals.cloud1.points_list(),
                    points_normals.p1,
                    points_normals.cloud2.points_list(),
                    points_normals.p2,
                    lengths1=points_normals.p1_lengths,
                    lengths2=points_normals.p2_lengths,
                )

    def test_chamfer_point_reduction_mean(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for point_reduction = "mean" and batch_reduction = None.
        """
        N, max_P1, max_P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, max_P1, max_P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        p1_normals = points_normals.n1
        p2_normals = points_normals.n2
        weights = points_normals.weights
        p11 = p1.detach().clone()
        p22 = p2.detach().clone()
        p11.requires_grad = True
        p22.requires_grad = True
        P1 = p1.shape[1]
        P2 = p2.shape[1]

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=p1_normals, y_normals=p2_normals
        )

        # point_reduction = "mean".
        loss, loss_norm = chamfer_distance(
            p11,
            p22,
            x_normals=p1_normals,
            y_normals=p2_normals,
            weights=weights,
            batch_reduction=None,
            point_reduction="mean",
        )
        pred_loss_mean = pred_loss[0].sum(1) / P1 + pred_loss[1].sum(1) / P2
        pred_loss_mean *= weights
        self.assertClose(loss, pred_loss_mean)

        pred_loss_norm_mean = (
            pred_loss_norm[0].sum(1) / P1 + pred_loss_norm[1].sum(1) / P2
        )
        pred_loss_norm_mean *= weights
        self.assertClose(loss_norm, pred_loss_norm_mean)

        # Check gradients
        self._check_gradients(
            loss, loss_norm, pred_loss_mean, pred_loss_norm_mean, p1, p11, p2, p22
        )

    def test_single_direction_chamfer_point_reduction_mean(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for point_reduction = "mean" and batch_reduction = None.
        """
        N, max_P1, max_P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, max_P1, max_P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        p1_normals = points_normals.n1
        p2_normals = points_normals.n2
        weights = points_normals.weights
        p11 = p1.detach().clone()
        p22 = p2.detach().clone()
        p11.requires_grad = True
        p22.requires_grad = True
        P1 = p1.shape[1]

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=p1_normals, y_normals=p2_normals
        )

        # point_reduction = "mean".
        loss, loss_norm = chamfer_distance(
            p11,
            p22,
            x_normals=p1_normals,
            y_normals=p2_normals,
            weights=weights,
            batch_reduction=None,
            point_reduction="mean",
            single_directional=True,
        )
        pred_loss_mean = pred_loss[0].sum(1) / P1
        pred_loss_mean *= weights
        self.assertClose(loss, pred_loss_mean)

        pred_loss_norm_mean = pred_loss_norm[0].sum(1) / P1
        pred_loss_norm_mean *= weights
        self.assertClose(loss_norm, pred_loss_norm_mean)

        # Check gradients
        self._check_gradients(
            loss, loss_norm, pred_loss_mean, pred_loss_norm_mean, p1, p11, p2, p22
        )

    def test_chamfer_point_reduction_sum(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for point_reduction = "sum" and batch_reduction = None.
        """
        N, P1, P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, P1, P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        p1_normals = points_normals.n1
        p2_normals = points_normals.n2
        weights = points_normals.weights
        p11 = p1.detach().clone()
        p22 = p2.detach().clone()
        p11.requires_grad = True
        p22.requires_grad = True

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=p1_normals, y_normals=p2_normals
        )

        loss, loss_norm = chamfer_distance(
            p11,
            p22,
            x_normals=p1_normals,
            y_normals=p2_normals,
            weights=weights,
            batch_reduction=None,
            point_reduction="sum",
        )
        pred_loss_sum = pred_loss[0].sum(1) + pred_loss[1].sum(1)
        pred_loss_sum *= weights
        self.assertClose(loss, pred_loss_sum)

        pred_loss_norm_sum = pred_loss_norm[0].sum(1) + pred_loss_norm[1].sum(1)
        pred_loss_norm_sum *= weights
        self.assertClose(loss_norm, pred_loss_norm_sum)

        # Check gradients
        self._check_gradients(
            loss, loss_norm, pred_loss_sum, pred_loss_norm_sum, p1, p11, p2, p22
        )

    def test_single_directional_chamfer_point_reduction_sum(self):
        """
        Compare output of vectorized single directional chamfer loss with naive implementation
        for point_reduction = "sum" and batch_reduction = None.
        """
        N, P1, P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, P1, P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        p1_normals = points_normals.n1
        p2_normals = points_normals.n2
        weights = points_normals.weights
        p11 = p1.detach().clone()
        p22 = p2.detach().clone()
        p11.requires_grad = True
        p22.requires_grad = True

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=p1_normals, y_normals=p2_normals
        )

        loss, loss_norm = chamfer_distance(
            p11,
            p22,
            x_normals=p1_normals,
            y_normals=p2_normals,
            weights=weights,
            batch_reduction=None,
            point_reduction="sum",
            single_directional=True,
        )
        pred_loss_sum = pred_loss[0].sum(1)
        pred_loss_sum *= weights
        self.assertClose(loss, pred_loss_sum)

        pred_loss_norm_sum = pred_loss_norm[0].sum(1)
        pred_loss_norm_sum *= weights
        self.assertClose(loss_norm, pred_loss_norm_sum)

        # Check gradients
        self._check_gradients(
            loss, loss_norm, pred_loss_sum, pred_loss_norm_sum, p1, p11, p2, p22
        )

    def test_chamfer_point_reduction_none(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for point_reduction = None and batch_reduction = None.
        """
        N, max_P1, max_P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, max_P1, max_P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        p1_normals = points_normals.n1
        p2_normals = points_normals.n2
        p11 = p1.detach().clone()
        p22 = p2.detach().clone()
        p11.requires_grad = True
        p22.requires_grad = True

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=p1_normals, y_normals=p2_normals
        )

        # point_reduction = None
        loss, loss_norm = chamfer_distance(
            p11,
            p22,
            x_normals=p1_normals,
            y_normals=p2_normals,
            batch_reduction=None,
            point_reduction=None,
        )

        loss_bidirectional = torch.hstack([loss[0], loss[1]])
        pred_loss_bidirectional = torch.hstack([pred_loss[0], pred_loss[1]])
        loss_norm_bidirectional = torch.hstack([loss_norm[0], loss_norm[1]])
        pred_loss_norm_bidirectional = torch.hstack(
            [pred_loss_norm[0], pred_loss_norm[1]]
        )

        self.assertClose(loss_bidirectional, pred_loss_bidirectional)
        self.assertClose(loss_norm_bidirectional, pred_loss_norm_bidirectional)

        # Check gradients
        self._check_gradients(
            loss_bidirectional,
            loss_norm_bidirectional,
            pred_loss_bidirectional,
            pred_loss_norm_bidirectional,
            p1,
            p11,
            p2,
            p22,
        )

    def test_single_direction_chamfer_point_reduction_none(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for point_reduction = None and batch_reduction = None.
        """
        N, max_P1, max_P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, max_P1, max_P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        p1_normals = points_normals.n1
        p2_normals = points_normals.n2
        p11 = p1.detach().clone()
        p22 = p2.detach().clone()
        p11.requires_grad = True
        p22.requires_grad = True

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=p1_normals, y_normals=p2_normals
        )

        # point_reduction = None
        loss, loss_norm = chamfer_distance(
            p11,
            p22,
            x_normals=p1_normals,
            y_normals=p2_normals,
            batch_reduction=None,
            point_reduction=None,
            single_directional=True,
        )

        self.assertClose(loss, pred_loss[0])
        self.assertClose(loss_norm, pred_loss_norm[0])

        # Check gradients
        self._check_gradients(
            loss, loss_norm, pred_loss[0], pred_loss_norm[0], p1, p11, p2, p22
        )

    def test_chamfer_point_reduction_max(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for point_reduction = "max" and batch_reduction = None.
        """
        N, P1, P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, P1, P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        weights = points_normals.weights
        p11 = p1.detach().clone()
        p22 = p2.detach().clone()
        p11.requires_grad = True
        p22.requires_grad = True

        pred_loss, unused_pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=None, y_normals=None
        )

        loss, loss_norm = chamfer_distance(
            p11,
            p22,
            x_normals=None,
            y_normals=None,
            weights=weights,
            batch_reduction=None,
            point_reduction="max",
        )
        pred_loss_max = torch.maximum(
            pred_loss[0].max(1).values, pred_loss[1].max(1).values
        )
        pred_loss_max *= weights
        self.assertClose(loss, pred_loss_max)

        self.assertIsNone(loss_norm)

        # Check gradients
        self._check_gradients(loss, loss_norm, pred_loss_max, None, p1, p11, p2, p22)

    def test_single_directional_chamfer_point_reduction_max(self):
        """
        Compare output of vectorized single directional chamfer loss with naive implementation
        for point_reduction = "max" and batch_reduction = None.
        """
        N, P1, P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, P1, P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        weights = points_normals.weights
        p11 = p1.detach().clone()
        p22 = p2.detach().clone()
        p11.requires_grad = True
        p22.requires_grad = True

        pred_loss, unused_pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=None, y_normals=None
        )

        loss, loss_norm = chamfer_distance(
            p11,
            p22,
            x_normals=None,
            y_normals=None,
            weights=weights,
            batch_reduction=None,
            point_reduction="max",
            single_directional=True,
        )
        pred_loss_max = pred_loss[0].max(1).values
        pred_loss_max *= weights
        self.assertClose(loss, pred_loss_max)

        self.assertIsNone(loss_norm)

        # Check gradients
        self._check_gradients(loss, loss_norm, pred_loss_max, None, p1, p11, p2, p22)

    def _check_gradients(
        self,
        loss,
        loss_norm,
        pred_loss,
        pred_loss_norm,
        x1,
        x2,
        y1,
        y2,
        xn1=None,  # normals
        xn2=None,  # normals
        yn1=None,  # normals
        yn2=None,  # normals
        lengths1=None,
        lengths2=None,
    ):
        """
        x1 and x2 can have different types based on the leaf node used in the calculation:
        e.g. x1 may be a list of tensors whereas x2 is a padded tensor.
        This also applies for the pairs: (y1, y2), (xn1, xn2), (yn1, yn2).
        """
        grad_loss = torch.rand(loss.shape, device=loss.device, dtype=loss.dtype)

        # Loss for normals is optional. Iniitalize to 0.
        norm_loss_term = pred_norm_loss_term = 0.0
        if loss_norm is not None and pred_loss_norm is not None:
            grad_normals = torch.rand(
                loss_norm.shape, device=loss.device, dtype=loss.dtype
            )
            norm_loss_term = loss_norm * grad_normals
            pred_norm_loss_term = pred_loss_norm * grad_normals

        l1 = (loss * grad_loss) + norm_loss_term
        l1.sum().backward()
        l2 = (pred_loss * grad_loss) + pred_norm_loss_term
        l2.sum().backward()

        self._check_grad_by_type(x1, x2, lengths1)
        self._check_grad_by_type(y1, y2, lengths2)

        # If leaf nodes for normals are passed in, check their gradients.
        if all(n is not None for n in [xn1, xn2, yn1, yn2]):
            self._check_grad_by_type(xn1, xn2, lengths1)
            self._check_grad_by_type(yn1, yn2, lengths2)

    def _check_grad_by_type(self, x1, x2, lengths=None):
        """
        x1 and x2 can be of different types e.g. list or tensor - compare appropriately
        based on the types.
        """
        error_msg = "All values for gradient checks must be tensors or lists of tensors"

        if all(isinstance(p, list) for p in [x1, x2]):
            # Lists of tensors
            for i in range(len(x1)):
                self.assertClose(x1[i].grad, x2[i].grad)
        elif isinstance(x1, list) and torch.is_tensor(x2):
            self.assertIsNotNone(lengths)  # lengths is required

            # List of tensors vs padded tensor
            for i in range(len(x1)):
                self.assertClose(x1[i].grad, x2.grad[i, : lengths[i]], atol=1e-7)
                self.assertTrue(x2.grad[i, lengths[i] :].sum().item() == 0.0)
        elif all(torch.is_tensor(p) for p in [x1, x2]):
            # Two tensors
            self.assertClose(x1.grad, x2.grad)
        else:
            raise ValueError(error_msg)

    def test_chamfer_joint_reduction(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        when batch_reduction in ["mean", "sum"] and
        point_reduction in ["mean", "sum"].
        """
        N, max_P1, max_P2 = 7, 10, 18
        device = get_random_cuda_device()

        points_normals = TestChamfer.init_pointclouds(N, max_P1, max_P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        p1_normals = points_normals.n1
        p2_normals = points_normals.n2
        weights = points_normals.weights

        P1 = p1.shape[1]
        P2 = p2.shape[1]

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, x_normals=p1_normals, y_normals=p2_normals
        )

        # batch_reduction = "sum", point_reduction = "sum".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            x_normals=p1_normals,
            y_normals=p2_normals,
            weights=weights,
            batch_reduction="sum",
            point_reduction="sum",
        )
        pred_loss[0] *= weights.view(N, 1)
        pred_loss[1] *= weights.view(N, 1)
        pred_loss_sum = pred_loss[0].sum(1) + pred_loss[1].sum(1)  # point sum
        pred_loss_sum = pred_loss_sum.sum()  # batch sum
        self.assertClose(loss, pred_loss_sum)

        pred_loss_norm[0] *= weights.view(N, 1)
        pred_loss_norm[1] *= weights.view(N, 1)
        pred_loss_norm_sum = pred_loss_norm[0].sum(1) + pred_loss_norm[1].sum(
            1
        )  # point sum.
        pred_loss_norm_sum = pred_loss_norm_sum.sum()  # batch sum
        self.assertClose(loss_norm, pred_loss_norm_sum)

        # batch_reduction = "mean", point_reduction = "sum".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            x_normals=p1_normals,
            y_normals=p2_normals,
            weights=weights,
            batch_reduction="mean",
            point_reduction="sum",
        )
        pred_loss_sum /= weights.sum()
        self.assertClose(loss, pred_loss_sum)

        pred_loss_norm_sum /= weights.sum()
        self.assertClose(loss_norm, pred_loss_norm_sum)

        # batch_reduction = "sum", point_reduction = "mean".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            x_normals=p1_normals,
            y_normals=p2_normals,
            weights=weights,
            batch_reduction="sum",
            point_reduction="mean",
        )
        pred_loss_mean = pred_loss[0].sum(1) / P1 + pred_loss[1].sum(1) / P2
        pred_loss_mean = pred_loss_mean.sum()
        self.assertClose(loss, pred_loss_mean)

        pred_loss_norm_mean = (
            pred_loss_norm[0].sum(1) / P1 + pred_loss_norm[1].sum(1) / P2
        )
        pred_loss_norm_mean = pred_loss_norm_mean.sum()
        self.assertClose(loss_norm, pred_loss_norm_mean)

        # batch_reduction = "mean", point_reduction = "mean". This is the default.
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            x_normals=p1_normals,
            y_normals=p2_normals,
            weights=weights,
            batch_reduction="mean",
            point_reduction="mean",
        )
        pred_loss_mean /= weights.sum()
        self.assertClose(loss, pred_loss_mean)

        pred_loss_norm_mean /= weights.sum()
        self.assertClose(loss_norm, pred_loss_norm_mean)

        # Error when batch_reduction is not in ["mean", "sum"] or None.
        with self.assertRaisesRegex(ValueError, "batch_reduction must be one of"):
            chamfer_distance(p1, p2, weights=weights, batch_reduction="max")

        # Error when point_reduction is not in ["mean", "sum", "max"] or None.
        with self.assertRaisesRegex(ValueError, "point_reduction must be one of"):
            chamfer_distance(p1, p2, weights=weights, point_reduction="min")

    def test_incorrect_weights(self):
        N, P1, P2 = 16, 64, 128
        device = get_random_cuda_device()
        p1 = torch.rand(
            (N, P1, 3), dtype=torch.float32, device=device, requires_grad=True
        )
        p2 = torch.rand(
            (N, P2, 3), dtype=torch.float32, device=device, requires_grad=True
        )

        weights = torch.zeros((N,), dtype=torch.float32, device=device)
        loss, loss_norm = chamfer_distance(
            p1, p2, weights=weights, batch_reduction="mean"
        )
        self.assertClose(loss.cpu(), torch.zeros(()))
        self.assertTrue(loss.requires_grad)
        self.assertClose(loss_norm.cpu(), torch.zeros(()))
        self.assertTrue(loss_norm.requires_grad)

        loss, loss_norm = chamfer_distance(
            p1, p2, weights=weights, batch_reduction=None
        )
        self.assertClose(loss.cpu(), torch.zeros((N, N)))
        self.assertTrue(loss.requires_grad)
        self.assertClose(loss_norm.cpu(), torch.zeros((N, N)))
        self.assertTrue(loss_norm.requires_grad)

        weights = torch.ones((N,), dtype=torch.float32, device=device) * -1
        with self.assertRaises(ValueError):
            loss, loss_norm = chamfer_distance(p1, p2, weights=weights)

        weights = torch.zeros((N - 1,), dtype=torch.float32, device=device)
        with self.assertRaises(ValueError):
            loss, loss_norm = chamfer_distance(p1, p2, weights=weights)

    def test_incorrect_inputs(self):
        N, P1, P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, P1, P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2
        p1_normals = points_normals.n1

        # Normals of wrong shape
        with self.assertRaisesRegex(ValueError, "Expected normals to be of shape"):
            chamfer_distance(p1, p2, x_normals=p1_normals[None])

        # Points of wrong shape
        with self.assertRaisesRegex(ValueError, "Expected points to be of shape"):
            chamfer_distance(p1[None], p2)

        # Lengths of wrong shape
        with self.assertRaisesRegex(ValueError, "Expected lengths to be of shape"):
            chamfer_distance(p1, p2, x_lengths=torch.tensor([1, 2, 3], device=device))

        # Points are not a tensor or Pointclouds
        with self.assertRaisesRegex(ValueError, "Pointclouds objects or torch.Tensor"):
            chamfer_distance(x=[1, 1, 1], y=[1, 1, 1])

    def test_invalid_norm(self):
        N, P1, P2 = 7, 10, 18
        device = get_random_cuda_device()
        points_normals = TestChamfer.init_pointclouds(N, P1, P2, device)
        p1 = points_normals.p1
        p2 = points_normals.p2

        with self.assertRaisesRegex(ValueError, "Support for 1 or 2 norm."):
            chamfer_distance(p1, p2, norm=0)

        with self.assertRaisesRegex(ValueError, "Support for 1 or 2 norm."):
            chamfer_distance(p1, p2, norm=3)

    def test_empty_clouds(self):
        # Check that point_reduction doesn't divide by zero
        points1 = Pointclouds(points=[torch.zeros(0, 3), torch.zeros(10, 3)])
        points2 = Pointclouds(points=torch.ones(2, 40, 3))
        loss, _ = chamfer_distance(points1, points2, batch_reduction=None)
        self.assertClose(loss, torch.tensor([0.0, 6.0]))

        # Check that batch_reduction doesn't divide by zero
        loss2, _ = chamfer_distance(Pointclouds([]), Pointclouds([]))
        self.assertClose(loss2, torch.tensor(0.0))

    @staticmethod
    def chamfer_with_init(
        batch_size: int,
        P1: int,
        P2: int,
        return_normals: bool,
        homogeneous: bool,
        device="cpu",
    ):
        points_normals = TestChamfer.init_pointclouds(batch_size, P1, P2, device=device)
        l1 = points_normals.p1_lengths
        l2 = points_normals.p2_lengths
        if homogeneous:
            # Set lengths to None so in Chamfer it assumes
            # there is no padding.
            l1 = l2 = None

        torch.cuda.synchronize()

        def loss():
            loss, loss_normals = chamfer_distance(
                points_normals.p1,
                points_normals.p2,
                x_lengths=l1,
                y_lengths=l2,
                x_normals=points_normals.n1,
                y_normals=points_normals.n2,
                weights=points_normals.weights,
            )
            torch.cuda.synchronize()

        return loss

    @staticmethod
    def chamfer_naive_with_init(
        batch_size: int, P1: int, P2: int, return_normals: bool, device="cpu"
    ):
        points_normals = TestChamfer.init_pointclouds(batch_size, P1, P2, device=device)
        torch.cuda.synchronize()

        def loss():
            loss, loss_normals = TestChamfer.chamfer_distance_naive(
                points_normals.p1,
                points_normals.p2,
                x_normals=points_normals.n1,
                y_normals=points_normals.n2,
            )
            torch.cuda.synchronize()

        return loss
