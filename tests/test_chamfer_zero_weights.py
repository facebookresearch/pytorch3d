# Copyright (c) 2026 Afloat16.
# This source code is licensed under the BSD-style license in the LICENSE file.

import unittest

import torch
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Pointclouds


class TestChamferZeroWeights(unittest.TestCase):
    def test_zero_weights_preserve_requested_shapes_and_optional_normals(self):
        reductions = [(None, None)] + [
            (points, batch)
            for points in ("mean", "sum", "max")
            for batch in (None, "mean", "sum")
        ]
        x = torch.arange(24).reshape(2, 4, 3).float() / 7
        y = torch.arange(42).reshape(2, 7, 3).float() / 11
        for points, batch in reductions:
            for single in (False, True):
                for normals in (False, True):
                    if points == "max" and normals:
                        continue
                    with self.subTest(
                        points=points, batch=batch, single=single, normals=normals
                    ):
                        distance, normal_loss = chamfer_distance(
                            x,
                            y,
                            weights=torch.zeros(2),
                            point_reduction=points,
                            batch_reduction=batch,
                            single_directional=single,
                            x_normals=torch.ones_like(x) if normals else None,
                            y_normals=torch.ones_like(y) if normals else None,
                        )
                        if points is None:
                            expected = (
                                torch.zeros((2, 4))
                                if single
                                else (torch.zeros((2, 4)), torch.zeros((2, 7)))
                            )
                        else:
                            expected = (
                                torch.zeros(2) if batch is None else torch.tensor(0.0)
                            )
                        torch.testing.assert_close(distance, expected)
                        if normals:
                            torch.testing.assert_close(normal_loss, expected)
                        else:
                            self.assertIsNone(normal_loss)

    def test_zero_weight_cloud_objects_attach_zero_gradients_to_actual_inputs(self):
        x = torch.ones(4, 3, requires_grad=True)
        y = torch.ones(7, 3, requires_grad=True)
        nx = torch.ones_like(x, requires_grad=True)
        ny = torch.ones_like(y, requires_grad=True)
        cloud_x = Pointclouds(points=[x], normals=[nx])
        cloud_y = Pointclouds(points=[y], normals=[ny])
        distance, normal_loss = chamfer_distance(
            cloud_x,
            cloud_y,
            weights=torch.zeros(1),
            batch_reduction=None,
            point_reduction=None,
        )
        torch.testing.assert_close(distance, (torch.zeros((1, 4)), torch.zeros((1, 7))))
        torch.testing.assert_close(
            normal_loss, (torch.zeros((1, 4)), torch.zeros((1, 7)))
        )
        sum(value.sum() for value in distance + normal_loss).backward()
        for tensor in (x, y, nx, ny):
            self.assertIsNotNone(tensor.grad)
            torch.testing.assert_close(
                tensor.grad, torch.zeros_like(tensor), atol=0, rtol=0
            )

    def test_nonzero_weight_distance_retains_pairwise_reference(self):
        x = torch.tensor([[[0.0, 0, 0], [2.0, 0, 0]], [[0.0, 0, 0], [0.0, 3, 0]]])
        y = torch.tensor([[[0.5, 0, 0], [3.0, 0, 0]], [[0.0, 1, 0], [0.0, 4, 0]]])
        weights = torch.tensor([0.0, 2.0])
        pairwise = (x[:, :, None] - y[:, None]).square().sum(-1)
        expected = (
            (pairwise.min(2).values.mean(1) + pairwise.min(1).values.mean(1)) * weights
        ).sum() / weights.sum()
        distance, normal_loss = chamfer_distance(x, y, weights=weights)
        torch.testing.assert_close(distance, expected)
        self.assertIsNone(normal_loss)
