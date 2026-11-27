# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import unittest

import torch
from pytorch3d.transforms import se3_exp_map, se3_log_map


def _matrix_exponential(log_transform):
    # Construct the Lie algebra directly instead of using the V coefficients.
    tx, ty, tz, x, y, z = log_transform.unbind(-1)
    zero = torch.zeros_like(x)
    algebra = torch.stack(
        (zero, -z, y, tx, z, zero, -x, ty, -y, x, zero, tz, zero, zero, zero, zero),
        dim=-1,
    ).reshape(log_transform.shape[:-1] + (4, 4))
    return torch.matrix_exp(algebra).transpose(-1, -2)


class TestSE3Precision(unittest.TestCase):
    def test_small_angle_translation_against_matrix_exponential(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            eps = 1e-4 if dtype in (torch.float16, torch.bfloat16) else 1e-8
            tolerance = max(8 * torch.finfo(dtype).eps, 1e-11)
            for angle in (0.001, 0.01, 0.03, 0.1, 0.49, 0.5, 0.51, 2.0):
                for axis in ([0.0, 0.0, 1.0], [1.0, 2.0, -3.0]):
                    with self.subTest(dtype=dtype, angle=angle, axis=axis):
                        omega = torch.tensor(axis, dtype=torch.float64)
                        omega = omega * angle / torch.linalg.vector_norm(omega)
                        log_transform = torch.cat(
                            (torch.tensor([0.0, 100.0, 20.0]), omega)
                        ).to(dtype)[None]
                        # The reference uses the actual quantized input.
                        expected = _matrix_exponential(log_transform.double())
                        actual = se3_exp_map(log_transform, eps=eps)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertTrue(torch.isfinite(actual).all())
                        torch.testing.assert_close(
                            actual.double(),
                            expected,
                            rtol=tolerance,
                            atol=tolerance,
                        )

    def test_clipping_parameter_is_preserved(self):
        angle = 0.001
        clipped_angle = 0.1
        log_transform = torch.tensor(
            [[0.0, 100.0, 20.0, 0.0, 0.0, angle]], dtype=torch.float64
        )
        transform = se3_exp_map(log_transform, eps=clipped_angle**2)
        b = (1 - math.cos(clipped_angle)) / clipped_angle**2
        c = (clipped_angle - math.sin(clipped_angle)) / clipped_angle**3
        expected = torch.tensor(
            [-100 * angle * b, 100 * (1 - angle**2 * c), 20],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            transform[0, 3, :3], expected, rtol=1e-12, atol=1e-12
        )

    def test_first_and_second_derivatives(self):
        for angle in (0.001, 0.49, 0.5, 0.51):
            with self.subTest(angle=angle):
                log_transform = torch.tensor(
                    [[0.2, -0.3, 0.4, 0.0, 0.0, angle]],
                    dtype=torch.float64,
                    requires_grad=True,
                )

                def function(value):
                    return se3_exp_map(value, eps=1e-8)

                self.assertTrue(torch.autograd.gradcheck(function, (log_transform,)))
                self.assertTrue(
                    torch.autograd.gradgradcheck(function, (log_transform,))
                )
                probe = torch.arange(1, 17, dtype=torch.float64).reshape(1, 4, 4)
                actual = torch.autograd.grad(
                    (function(log_transform) * probe).sum(),
                    log_transform,
                    create_graph=True,
                )[0]
                expected = torch.autograd.grad(
                    (_matrix_exponential(log_transform) * probe).sum(),
                    log_transform,
                    create_graph=True,
                )[0]
                torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
                actual_second = torch.autograd.grad(actual.sum(), log_transform)[0]
                expected_second = torch.autograd.grad(expected.sum(), log_transform)[0]
                torch.testing.assert_close(
                    actual_second, expected_second, rtol=1e-8, atol=1e-8
                )

    def test_input_and_parameter_gradients(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                layer = torch.nn.Linear(6, 6, bias=False, dtype=dtype)
                with torch.no_grad():
                    layer.weight.copy_(
                        torch.diag(
                            torch.tensor([1, 1, 1, 0.01, 0.01, 0.01], dtype=dtype)
                        )
                    )
                features = torch.tensor(
                    [0, 100, 20, 1, 2, -3], dtype=dtype, requires_grad=True
                )
                point = torch.tensor([0.5, -0.75, 1.25, 1], dtype=dtype)
                direction = torch.tensor([0.25, 0.5, -0.75, 0], dtype=dtype)
                transform = se3_exp_map(layer(features)[None])[0]
                loss = (point @ transform * direction).sum()
                actual = torch.autograd.grad(loss, (features, layer.weight))

                reference_features = features.detach().double().requires_grad_()
                reference_weights = layer.weight.detach().double().requires_grad_()
                reference_log = (reference_weights @ reference_features)[None]
                reference_transform = _matrix_exponential(reference_log)[0]
                reference_loss = (
                    point.double() @ reference_transform * direction.double()
                ).sum()
                expected = torch.autograd.grad(
                    reference_loss, (reference_features, reference_weights)
                )
                tolerance = 16 * torch.finfo(dtype).eps
                for gradient, reference in zip(actual, expected):
                    self.assertTrue(torch.isfinite(gradient).all())
                    torch.testing.assert_close(
                        gradient.double(),
                        reference,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_inactive_series_cannot_overflow_gradients(self):
        for dtype, angle in ((torch.float32, 1e5), (torch.float64, 1e40)):
            with self.subTest(dtype=dtype):
                log_transform = torch.tensor(
                    [[1, 2, 3, 0, 0, angle]], dtype=dtype, requires_grad=True
                )
                transform = se3_exp_map(log_transform)
                gradient = torch.autograd.grad(transform.sum(), log_transform)[0]
                self.assertTrue(torch.isfinite(transform).all())
                self.assertTrue(torch.isfinite(gradient).all())

    def test_zero_rotation_empty_batch_and_script(self):
        log_transform = torch.tensor(
            [[0.2, -0.3, 0.4, 0, 0, 0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        actual = se3_exp_map(log_transform)
        expected = _matrix_exponential(log_transform)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        gradient = torch.autograd.grad(actual.sum(), log_transform)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertEqual(se3_exp_map(torch.empty(0, 6)).shape, (0, 4, 4))
        scripted = torch.jit.script(se3_exp_map)
        torch.testing.assert_close(scripted(log_transform), actual)

    def test_logarithm_round_trip(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                log_transform = torch.tensor(
                    [
                        [0.0, 100.0, 20.0, 0.0, 0.0, 0.01],
                        [0.2, -0.3, 0.4, 0.1, -0.2, 0.3],
                    ],
                    dtype=dtype,
                )
                recovered = se3_log_map(se3_exp_map(log_transform, eps=1e-8), eps=1e-8)
                torch.testing.assert_close(
                    recovered, log_transform, rtol=2e-5, atol=2e-5
                )


if __name__ == "__main__":
    unittest.main()
