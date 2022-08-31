# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import torch
from pytorch3d.transforms import acos_linear_extrapolation

from .common_testing import TestCaseMixin


class TestAcosLinearExtrapolation(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    @staticmethod
    def init_acos_boundary_values(batch_size: int = 10000):
        """
        Initialize a tensor containing values close to the bounds of the
        domain of `acos`, i.e. close to -1 or 1; and random values between (-1, 1).
        """
        device = torch.device("cuda:0")
        # one quarter are random values between -1 and 1
        x_rand = 2 * torch.rand(batch_size // 4, dtype=torch.float32, device=device) - 1
        x = [x_rand]
        for bound in [-1, 1]:
            for above_bound in [True, False]:
                for noise_std in [1e-4, 1e-2]:
                    n_generate = (batch_size - batch_size // 4) // 8
                    x_add = (
                        bound
                        + (2 * float(above_bound) - 1)
                        * torch.randn(
                            n_generate, device=device, dtype=torch.float32
                        ).abs()
                        * noise_std
                    )
                    x.append(x_add)
        x = torch.cat(x)
        return x

    @staticmethod
    def acos_linear_extrapolation(batch_size: int):
        x = TestAcosLinearExtrapolation.init_acos_boundary_values(batch_size)
        torch.cuda.synchronize()

        def compute_acos():
            acos_linear_extrapolation(x)
            torch.cuda.synchronize()

        return compute_acos

    def _test_acos_outside_bounds(self, x, y, dydx, bound):
        """
        Check that `acos_linear_extrapolation` yields points on a line with correct
        slope, and that the function is continuous around `bound`.
        """
        bound_t = torch.tensor(bound, device=x.device, dtype=x.dtype)
        # fit a line: slope * x + bias = y
        x_1 = torch.stack([x, torch.ones_like(x)], dim=-1)
        slope, bias = torch.linalg.lstsq(x_1, y[:, None]).solution.view(-1)[:2]
        desired_slope = (-1.0) / torch.sqrt(1.0 - bound_t**2)
        # test that the desired slope is the same as the fitted one
        self.assertClose(desired_slope.view(1), slope.view(1), atol=1e-2)
        # test that the autograd's slope is the same as the desired one
        self.assertClose(desired_slope.expand_as(dydx), dydx, atol=1e-2)
        # test that the value of the fitted line at x=bound equals
        # arccos(x), i.e. the function is continuous around the bound
        y_bound_lin = (slope * bound_t + bias).view(1)
        y_bound_acos = bound_t.acos().view(1)
        self.assertClose(y_bound_lin, y_bound_acos, atol=1e-2)

    def _one_acos_test(self, x: torch.Tensor, lower_bound: float, upper_bound: float):
        """
        Test that `acos_linear_extrapolation` returns correct values for
        `x` between/above/below `lower_bound`/`upper_bound`.
        """
        x.requires_grad = True
        x.grad = None
        y = acos_linear_extrapolation(x, [lower_bound, upper_bound])
        # compute the gradient of the acos w.r.t. x
        y.backward(torch.ones_like(y))
        dacos_dx = x.grad
        x_lower = x <= lower_bound
        x_upper = x >= upper_bound
        x_mid = (~x_lower) & (~x_upper)
        # test that between bounds, the function returns plain acos
        self.assertClose(x[x_mid].acos(), y[x_mid])
        # test that outside the bounds, the function is linear with the right
        # slope and continuous around the bound
        self._test_acos_outside_bounds(
            x[x_upper], y[x_upper], dacos_dx[x_upper], upper_bound
        )
        self._test_acos_outside_bounds(
            x[x_lower], y[x_lower], dacos_dx[x_lower], lower_bound
        )

    def test_acos(self, batch_size: int = 10000):
        """
        Tests whether the function returns correct outputs
        inside/outside the bounds.
        """
        x = TestAcosLinearExtrapolation.init_acos_boundary_values(batch_size)
        bounds = 1 - 10.0 ** torch.linspace(-1, -5, 5)
        for lower_bound in -bounds:
            for upper_bound in bounds:
                if upper_bound < lower_bound:
                    continue
                self._one_acos_test(x, float(lower_bound), float(upper_bound))

    def test_finite_gradient(self, batch_size: int = 10000):
        """
        Tests whether gradients stay finite close to the bounds.
        """
        x = TestAcosLinearExtrapolation.init_acos_boundary_values(batch_size)
        x.requires_grad = True
        bounds = 1 - 10.0 ** torch.linspace(-1, -5, 5)
        for lower_bound in -bounds:
            for upper_bound in bounds:
                if upper_bound < lower_bound:
                    continue
                x.grad = None
                y = acos_linear_extrapolation(
                    x,
                    [float(lower_bound), float(upper_bound)],
                )
                self.assertTrue(torch.isfinite(y).all())
                loss = y.mean()
                loss.backward()
                self.assertIsNotNone(x.grad)
                self.assertTrue(torch.isfinite(x.grad).all())
