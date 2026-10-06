# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.transforms import quaternion_to_matrix


def _rotation_from_axis_angle(quaternion):
    # Independent exponential-map construction, evaluated in double precision.
    imaginary = quaternion[..., 1:]
    length = torch.linalg.vector_norm(imaginary, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(length, quaternion[..., :1])
    x, y, z = (imaginary * angle / length).unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    return torch.matrix_exp(skew.reshape(quaternion.shape[:-1] + (3, 3)))


class TestQuaternionPrecision(unittest.TestCase):
    def test_representable_nonzero_scales(self):
        # Every scaling below retains the exact same quaternion direction.
        scales = {
            torch.float16: (-24, -10, 0, 12),
            torch.bfloat16: (-133, -70, 0, 125),
            torch.float32: (-149, -70, 0, 125),
            torch.float64: (-1074, -520, 0, 1020),
        }
        expected = torch.tensor(
            [
                [-2 / 3, 2 / 15, 11 / 15],
                [2 / 3, -1 / 3, 2 / 3],
                [1 / 3, 14 / 15, 2 / 15],
            ],
            dtype=torch.float64,
        )
        for dtype, exponents in scales.items():
            for exponent in exponents:
                for sign in (-1, 1):
                    with self.subTest(dtype=dtype, exponent=exponent, sign=sign):
                        q = torch.tensor([1, 2, 3, 4], dtype=dtype)
                        q = q * (sign * 2.0**exponent)
                        original = q.clone()
                        self.assertTrue(torch.isfinite(q).all())
                        self.assertTrue((q != 0).all())
                        rotation = quaternion_to_matrix(q)
                        self.assertEqual(rotation.dtype, dtype)
                        self.assertTrue(torch.isfinite(rotation).all())
                        tolerance = 4 * torch.finfo(dtype).eps
                        torch.testing.assert_close(
                            rotation.double(),
                            expected,
                            rtol=tolerance,
                            atol=tolerance,
                        )
                        torch.testing.assert_close(q, original, rtol=0, atol=0)

    def test_batch_shapes_and_noncontiguous_input(self):
        data = torch.arange(1, 49, dtype=torch.float64).reshape(2, 3, 8)
        q = data[..., ::2]
        self.assertFalse(q.is_contiguous())
        rotation = quaternion_to_matrix(q)
        self.assertEqual(rotation.shape, (2, 3, 3, 3))
        torch.testing.assert_close(
            rotation, _rotation_from_axis_angle(q), rtol=1e-12, atol=1e-12
        )
        identity = torch.eye(3, dtype=q.dtype).expand(2, 3, 3, 3)
        torch.testing.assert_close(
            rotation @ rotation.transpose(-1, -2),
            identity,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            torch.linalg.det(rotation),
            torch.ones(2, 3, dtype=q.dtype),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(quaternion_to_matrix(torch.empty(2, 0, 4)).shape, (2, 0, 3, 3))

    def test_first_and_second_derivatives(self):
        for values in ([0.8, -0.1, 0.2, 0.3], [1.0, 1.0, 0.3, 0.1]):
            with self.subTest(values=values):
                q = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                self.assertTrue(torch.autograd.gradcheck(quaternion_to_matrix, (q,)))
                self.assertTrue(
                    torch.autograd.gradgradcheck(quaternion_to_matrix, (q,))
                )
                probe = torch.arange(1, 10, dtype=q.dtype).reshape(3, 3)
                native_gradient = torch.autograd.grad(
                    (quaternion_to_matrix(q) * probe).sum(), q, create_graph=True
                )[0]
                reference_gradient = torch.autograd.grad(
                    (_rotation_from_axis_angle(q) * probe).sum(),
                    q,
                    create_graph=True,
                )[0]
                torch.testing.assert_close(
                    native_gradient, reference_gradient, rtol=1e-10, atol=1e-10
                )
                native_second = torch.autograd.grad(native_gradient.sum(), q)[0]
                reference_second = torch.autograd.grad(reference_gradient.sum(), q)[0]
                torch.testing.assert_close(
                    native_second, reference_second, rtol=1e-9, atol=1e-9
                )

    def test_input_and_parameter_gradients(self):
        for dtype in (torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                layer = torch.nn.Linear(4, 4, bias=False, dtype=dtype)
                with torch.no_grad():
                    layer.weight.copy_(torch.eye(4, dtype=dtype) * 2.0**-10)
                features = torch.tensor([1, 2, 3, 4], dtype=dtype, requires_grad=True)
                point = torch.tensor(
                    [0.5, -0.75, 1.25], dtype=dtype, requires_grad=True
                )
                direction = torch.tensor([0.25, 0.5, -0.75], dtype=dtype)
                loss = (quaternion_to_matrix(layer(features)) @ point * direction).sum()
                actual = torch.autograd.grad(loss, (features, layer.weight, point))

                reference_features = features.detach().double().requires_grad_()
                reference_weights = layer.weight.detach().double().requires_grad_()
                reference_point = point.detach().double().requires_grad_()
                reference_q = reference_weights @ reference_features
                reference_loss = (
                    _rotation_from_axis_angle(reference_q)
                    @ reference_point
                    * direction.double()
                ).sum()
                expected = torch.autograd.grad(
                    reference_loss,
                    (reference_features, reference_weights, reference_point),
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

    def test_script_and_undefined_zero(self):
        scripted = torch.jit.script(quaternion_to_matrix)
        q = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64) * 2.0**-520
        torch.testing.assert_close(scripted(q), quaternion_to_matrix(q))
        self.assertFalse(torch.isfinite(quaternion_to_matrix(torch.zeros(4))).all())


if __name__ == "__main__":
    unittest.main()
