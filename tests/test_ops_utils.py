# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from pytorch3d.ops import utils as oputil

from .common_testing import TestCaseMixin


class TestOpsUtils(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    def test_wmean(self):
        device = torch.device("cuda:0")
        n_points = 20

        x = torch.rand(n_points, 3, device=device)
        weight = torch.rand(n_points, device=device)
        x_np = x.cpu().data.numpy()
        weight_np = weight.cpu().data.numpy()

        # test unweighted
        mean = oputil.wmean(x, keepdim=False)
        mean_gt = np.average(x_np, axis=-2)
        self.assertClose(mean.cpu().data.numpy(), mean_gt)

        # test weighted
        mean = oputil.wmean(x, weight=weight, keepdim=False)
        mean_gt = np.average(x_np, axis=-2, weights=weight_np)
        self.assertClose(mean.cpu().data.numpy(), mean_gt)

        # test keepdim
        mean = oputil.wmean(x, weight=weight, keepdim=True)
        self.assertClose(mean[0].cpu().data.numpy(), mean_gt)

        # test binary weigths
        mean = oputil.wmean(x, weight=weight > 0.5, keepdim=False)
        mean_gt = np.average(x_np, axis=-2, weights=weight_np > 0.5)
        self.assertClose(mean.cpu().data.numpy(), mean_gt)

        # test broadcasting
        x = torch.rand(10, n_points, 3, device=device)
        x_np = x.cpu().data.numpy()
        mean = oputil.wmean(x, weight=weight, keepdim=False)
        mean_gt = np.average(x_np, axis=-2, weights=weight_np)
        self.assertClose(mean.cpu().data.numpy(), mean_gt)

        weight = weight[None, None, :].repeat(3, 1, 1)
        mean = oputil.wmean(x, weight=weight, keepdim=False)
        self.assertClose(mean[0].cpu().data.numpy(), mean_gt)

        # test failing broadcasting
        weight = torch.rand(x.shape[0], device=device)
        with self.assertRaises(ValueError) as context:
            oputil.wmean(x, weight=weight, keepdim=False)
        self.assertTrue("weights are not compatible" in str(context.exception))

        # test dim
        weight = torch.rand(x.shape[0], n_points, device=device)
        weight_np = np.tile(
            weight[:, :, None].cpu().data.numpy(), (1, 1, x_np.shape[-1])
        )
        mean = oputil.wmean(x, dim=0, weight=weight, keepdim=False)
        mean_gt = np.average(x_np, axis=0, weights=weight_np)
        self.assertClose(mean.cpu().data.numpy(), mean_gt)

        # test dim tuple
        mean = oputil.wmean(x, dim=(0, 1), weight=weight, keepdim=False)
        mean_gt = np.average(x_np, axis=(0, 1), weights=weight_np)
        self.assertClose(mean.cpu().data.numpy(), mean_gt)

    def test_masked_gather_errors(self):
        idx = torch.randint(0, 10, size=(5, 10, 4, 2))
        points = torch.randn(size=(5, 10, 3))
        with self.assertRaisesRegex(ValueError, "format is not supported"):
            oputil.masked_gather(points, idx)

        points = torch.randn(size=(2, 10, 3))
        with self.assertRaisesRegex(ValueError, "same batch dimension"):
            oputil.masked_gather(points, idx)
