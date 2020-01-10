#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
import torch.nn.functional as F

from pytorch3d.loss import chamfer_distance


class TestChamfer(unittest.TestCase):
    @staticmethod
    def init_pointclouds(batch_size: int = 10, P1: int = 32, P2: int = 64):
        """
        Randomly initialize two batches of point clouds of sizes
        (N, P1, D) and (N, P2, D) and return random normal vectors for
        each batch of size (N, P1, 3) and (N, P2, 3).
        """
        device = torch.device("cuda:0")
        p1 = torch.rand((batch_size, P1, 3), dtype=torch.float32, device=device)
        p1_normals = torch.rand(
            (batch_size, P1, 3), dtype=torch.float32, device=device
        )
        p1_normals = p1_normals / p1_normals.norm(dim=2, p=2, keepdim=True)
        p2 = torch.rand((batch_size, P2, 3), dtype=torch.float32, device=device)
        p2_normals = torch.rand(
            (batch_size, P2, 3), dtype=torch.float32, device=device
        )
        p2_normals = p2_normals / p2_normals.norm(dim=2, p=2, keepdim=True)
        weights = torch.rand((batch_size,), dtype=torch.float32, device=device)

        return p1, p2, p1_normals, p2_normals, weights

    @staticmethod
    def chamfer_distance_naive(p1, p2, p1_normals=None, p2_normals=None):
        """
        Naive iterative implementation of nearest neighbor and chamfer distance.
        Returns lists of the unreduced loss and loss_normals.
        """
        N, P1, D = p1.shape
        P2 = p2.size(1)
        device = torch.device("cuda:0")
        return_normals = p1_normals is not None and p2_normals is not None
        dist = torch.zeros((N, P1, P2), dtype=torch.float32, device=device)

        for n in range(N):
            for i1 in range(P1):
                for i2 in range(P2):
                    dist[n, i1, i2] = torch.sum(
                        (p1[n, i1, :] - p2[n, i2, :]) ** 2
                    )

        loss = [
            torch.min(dist, dim=2)[0],  # (N, P1)
            torch.min(dist, dim=1)[0],  # (N, P2)
        ]

        lnorm = [p1.new_zeros(()), p1.new_zeros(())]

        if return_normals:
            p1_index = dist.argmin(2).view(N, P1, 1).expand(N, P1, 3)
            p2_index = dist.argmin(1).view(N, P2, 1).expand(N, P2, 3)
            lnorm1 = 1 - torch.abs(
                F.cosine_similarity(
                    p1_normals, p2_normals.gather(1, p1_index), dim=2, eps=1e-6
                )
            )
            lnorm2 = 1 - torch.abs(
                F.cosine_similarity(
                    p2_normals, p1_normals.gather(1, p2_index), dim=2, eps=1e-6
                )
            )
            lnorm = [lnorm1, lnorm2]  # [(N, P1), (N, P2)]

        return loss, lnorm

    def test_chamfer_default_no_normals(self):
        """
        Compare chamfer loss with naive implementation using default
        input values and no normals.
        """
        N, P1, P2 = 7, 10, 18
        p1, p2, _, _, weights = TestChamfer.init_pointclouds(N, P1, P2)
        pred_loss, _ = TestChamfer.chamfer_distance_naive(p1, p2)
        loss, loss_norm = chamfer_distance(p1, p2, weights=weights)
        pred_loss = pred_loss[0].sum(1) / P1 + pred_loss[1].sum(1) / P2
        pred_loss *= weights
        pred_loss = pred_loss.sum() / weights.sum()
        self.assertTrue(torch.allclose(loss, pred_loss))
        self.assertTrue(loss_norm is None)

    def test_chamfer_point_reduction(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for point_reduction in ["mean", "sum", "none"] and
        batch_reduction = "none".
        """
        N, P1, P2 = 7, 10, 18
        p1, p2, p1_normals, p2_normals, weights = TestChamfer.init_pointclouds(
            N, P1, P2
        )

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, p1_normals, p2_normals
        )

        # point_reduction = "mean".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            p1_normals,
            p2_normals,
            weights=weights,
            batch_reduction="none",
            point_reduction="mean",
        )
        pred_loss_mean = pred_loss[0].sum(1) / P1 + pred_loss[1].sum(1) / P2
        pred_loss_mean *= weights
        self.assertTrue(torch.allclose(loss, pred_loss_mean))

        pred_loss_norm_mean = (
            pred_loss_norm[0].sum(1) / P1 + pred_loss_norm[1].sum(1) / P2
        )
        pred_loss_norm_mean *= weights
        self.assertTrue(torch.allclose(loss_norm, pred_loss_norm_mean))

        # point_reduction = "sum".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            p1_normals,
            p2_normals,
            weights=weights,
            batch_reduction="none",
            point_reduction="sum",
        )
        pred_loss_sum = pred_loss[0].sum(1) + pred_loss[1].sum(1)
        pred_loss_sum *= weights
        self.assertTrue(torch.allclose(loss, pred_loss_sum))

        pred_loss_norm_sum = pred_loss_norm[0].sum(1) + pred_loss_norm[1].sum(1)
        pred_loss_norm_sum *= weights
        self.assertTrue(torch.allclose(loss_norm, pred_loss_norm_sum))

        # Error when point_reduction = "none" and batch_reduction = "none".
        with self.assertRaises(ValueError):
            chamfer_distance(
                p1,
                p2,
                weights=weights,
                batch_reduction="none",
                point_reduction="none",
            )

        # Error when batch_reduction is not in ["none", "mean", "sum"].
        with self.assertRaises(ValueError):
            chamfer_distance(p1, p2, weights=weights, batch_reduction="max")

    def test_chamfer_batch_reduction(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for batch_reduction in ["mean", "sum"] and point_reduction = "none".
        """
        N, P1, P2 = 7, 10, 18
        p1, p2, p1_normals, p2_normals, weights = TestChamfer.init_pointclouds(
            N, P1, P2
        )

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, p1_normals, p2_normals
        )

        # batch_reduction = "sum".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            p1_normals,
            p2_normals,
            weights=weights,
            batch_reduction="sum",
            point_reduction="none",
        )
        pred_loss[0] *= weights.view(N, 1)
        pred_loss[1] *= weights.view(N, 1)
        pred_loss = pred_loss[0].sum() + pred_loss[1].sum()
        self.assertTrue(torch.allclose(loss, pred_loss))

        pred_loss_norm[0] *= weights.view(N, 1)
        pred_loss_norm[1] *= weights.view(N, 1)
        pred_loss_norm = pred_loss_norm[0].sum() + pred_loss_norm[1].sum()
        self.assertTrue(torch.allclose(loss_norm, pred_loss_norm))

        # batch_reduction = "mean".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            p1_normals,
            p2_normals,
            weights=weights,
            batch_reduction="mean",
            point_reduction="none",
        )

        pred_loss /= weights.sum()
        self.assertTrue(torch.allclose(loss, pred_loss))

        pred_loss_norm /= weights.sum()
        self.assertTrue(torch.allclose(loss_norm, pred_loss_norm))

        # Error when point_reduction is not in ["none", "mean", "sum"].
        with self.assertRaises(ValueError):
            chamfer_distance(p1, p2, weights=weights, point_reduction="max")

    def test_chamfer_joint_reduction(self):
        """
        Compare output of vectorized chamfer loss with naive implementation
        for batch_reduction in ["mean", "sum"] and
        point_reduction in ["mean", "sum"].
        """
        N, P1, P2 = 7, 10, 18
        p1, p2, p1_normals, p2_normals, weights = TestChamfer.init_pointclouds(
            N, P1, P2
        )

        pred_loss, pred_loss_norm = TestChamfer.chamfer_distance_naive(
            p1, p2, p1_normals, p2_normals
        )

        # batch_reduction = "sum", point_reduction = "sum".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            p1_normals,
            p2_normals,
            weights=weights,
            batch_reduction="sum",
            point_reduction="sum",
        )
        pred_loss[0] *= weights.view(N, 1)
        pred_loss[1] *= weights.view(N, 1)
        pred_loss_sum = pred_loss[0].sum(1) + pred_loss[1].sum(1)  # point sum
        pred_loss_sum = pred_loss_sum.sum()  # batch sum
        self.assertTrue(torch.allclose(loss, pred_loss_sum))

        pred_loss_norm[0] *= weights.view(N, 1)
        pred_loss_norm[1] *= weights.view(N, 1)
        pred_loss_norm_sum = pred_loss_norm[0].sum(1) + pred_loss_norm[1].sum(
            1
        )  # point sum.
        pred_loss_norm_sum = pred_loss_norm_sum.sum()  # batch sum
        self.assertTrue(torch.allclose(loss_norm, pred_loss_norm_sum))

        # batch_reduction = "mean", point_reduction = "sum".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            p1_normals,
            p2_normals,
            weights=weights,
            batch_reduction="mean",
            point_reduction="sum",
        )
        pred_loss_sum /= weights.sum()
        self.assertTrue(torch.allclose(loss, pred_loss_sum))

        pred_loss_norm_sum /= weights.sum()
        self.assertTrue(torch.allclose(loss_norm, pred_loss_norm_sum))

        # batch_reduction = "sum", point_reduction = "mean".
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            p1_normals,
            p2_normals,
            weights=weights,
            batch_reduction="sum",
            point_reduction="mean",
        )
        pred_loss_mean = pred_loss[0].sum(1) / P1 + pred_loss[1].sum(1) / P2
        pred_loss_mean = pred_loss_mean.sum()
        self.assertTrue(torch.allclose(loss, pred_loss_mean))

        pred_loss_norm_mean = (
            pred_loss_norm[0].sum(1) / P1 + pred_loss_norm[1].sum(1) / P2
        )
        pred_loss_norm_mean = pred_loss_norm_mean.sum()
        self.assertTrue(torch.allclose(loss_norm, pred_loss_norm_mean))

        # batch_reduction = "mean", point_reduction = "mean". This is the default.
        loss, loss_norm = chamfer_distance(
            p1,
            p2,
            p1_normals,
            p2_normals,
            weights=weights,
            batch_reduction="mean",
            point_reduction="mean",
        )
        pred_loss_mean /= weights.sum()
        self.assertTrue(torch.allclose(loss, pred_loss_mean))

        pred_loss_norm_mean /= weights.sum()
        self.assertTrue(torch.allclose(loss_norm, pred_loss_norm_mean))

    def test_incorrect_weights(self):
        N, P1, P2 = 16, 64, 128
        device = torch.device("cuda:0")
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
        self.assertTrue(torch.allclose(loss.cpu(), torch.zeros((1,))))
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.allclose(loss_norm.cpu(), torch.zeros((1,))))
        self.assertTrue(loss_norm.requires_grad)

        loss, loss_norm = chamfer_distance(
            p1, p2, weights=weights, batch_reduction="none"
        )
        self.assertTrue(torch.allclose(loss.cpu(), torch.zeros((N,))))
        self.assertTrue(loss.requires_grad)
        self.assertTrue(torch.allclose(loss_norm.cpu(), torch.zeros((N,))))
        self.assertTrue(loss_norm.requires_grad)

        weights = torch.ones((N,), dtype=torch.float32, device=device) * -1
        with self.assertRaises(ValueError):
            loss, loss_norm = chamfer_distance(p1, p2, weights=weights)

        weights = torch.zeros((N - 1,), dtype=torch.float32, device=device)
        with self.assertRaises(ValueError):
            loss, loss_norm = chamfer_distance(p1, p2, weights=weights)

    @staticmethod
    def chamfer_with_init(
        batch_size: int, P1: int, P2: int, return_normals: bool
    ):
        p1, p2, p1_normals, p2_normals, weights = TestChamfer.init_pointclouds(
            batch_size, P1, P2
        )
        torch.cuda.synchronize()

        def loss():
            loss, loss_normals = chamfer_distance(
                p1, p2, p1_normals, p2_normals, weights=weights
            )
            torch.cuda.synchronize()

        return loss

    @staticmethod
    def chamfer_naive_with_init(
        batch_size: int, P1: int, P2: int, return_normals: bool
    ):
        p1, p2, p1_normals, p2_normals, weights = TestChamfer.init_pointclouds(
            batch_size, P1, P2
        )
        torch.cuda.synchronize()

        def loss():
            loss, loss_normals = TestChamfer.chamfer_distance_naive(
                p1, p2, p1_normals, p2_normals
            )
            torch.cuda.synchronize()

        return loss
