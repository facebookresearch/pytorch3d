# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import unittest
from unittest import mock

import torch
from pytorch3d.transforms import random_rotations
from pytorch3d.transforms.se3 import se3_log_map
from pytorch3d.transforms.so3 import so3_exp_map
from pytorch3d.transforms.transform3d import (
    Rotate,
    RotateAxisAngle,
    Scale,
    Transform3d,
    Translate,
)

from .common_testing import TestCaseMixin


class TestTransform(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_to(self):
        tr = Translate(torch.FloatTensor([[1.0, 2.0, 3.0]]))
        R = torch.FloatTensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        R = Rotate(R)
        t = Transform3d().compose(R, tr)

        cpu_device = torch.device("cpu")

        cpu_t = t.to("cpu")
        self.assertEqual(cpu_device, cpu_t.device)
        self.assertEqual(cpu_device, t.device)
        self.assertEqual(torch.float32, cpu_t.dtype)
        self.assertEqual(torch.float32, t.dtype)
        self.assertIs(t, cpu_t)

        cpu_t = t.to(cpu_device)
        self.assertEqual(cpu_device, cpu_t.device)
        self.assertEqual(cpu_device, t.device)
        self.assertEqual(torch.float32, cpu_t.dtype)
        self.assertEqual(torch.float32, t.dtype)
        self.assertIs(t, cpu_t)

        cpu_t = t.to(dtype=torch.float64, device=cpu_device)
        self.assertEqual(cpu_device, cpu_t.device)
        self.assertEqual(cpu_device, t.device)
        self.assertEqual(torch.float64, cpu_t.dtype)
        self.assertEqual(torch.float32, t.dtype)
        self.assertIsNot(t, cpu_t)

        cuda_device = torch.device("cuda:0")

        cuda_t = t.to("cuda:0")
        self.assertEqual(cuda_device, cuda_t.device)
        self.assertEqual(cpu_device, t.device)
        self.assertEqual(torch.float32, cuda_t.dtype)
        self.assertEqual(torch.float32, t.dtype)
        self.assertIsNot(t, cuda_t)

        cuda_t = t.to(cuda_device)
        self.assertEqual(cuda_device, cuda_t.device)
        self.assertEqual(cpu_device, t.device)
        self.assertEqual(torch.float32, cuda_t.dtype)
        self.assertEqual(torch.float32, t.dtype)
        self.assertIsNot(t, cuda_t)

        cuda_t = t.to(dtype=torch.float64, device=cuda_device)
        self.assertEqual(cuda_device, cuda_t.device)
        self.assertEqual(cpu_device, t.device)
        self.assertEqual(torch.float64, cuda_t.dtype)
        self.assertEqual(torch.float32, t.dtype)
        self.assertIsNot(t, cuda_t)

        cpu_points = torch.rand(9, 3)
        cuda_points = cpu_points.cuda()
        for _ in range(3):
            t = t.cpu()
            t.transform_points(cpu_points)
            t = t.cuda()
            t.transform_points(cuda_points)
            t = t.cuda()
            t = t.cpu()

    def test_dtype_propagation(self):
        """
        Check that a given dtype is correctly passed along to child
        transformations.
        """
        # Use at least two dtypes so we avoid only testing on the
        # default dtype.
        for dtype in [torch.float32, torch.float64]:
            R = torch.tensor(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                dtype=dtype,
            )
            tf = (
                Transform3d(dtype=dtype)
                .rotate(R)
                .rotate_axis_angle(
                    R[0],
                    "X",
                )
                .translate(3, 2, 1)
                .scale(0.5)
            )

            self.assertEqual(tf.dtype, dtype)
            for inner_tf in tf._transforms:
                self.assertEqual(inner_tf.dtype, dtype)

            transformed = tf.transform_points(R)
            self.assertEqual(transformed.dtype, dtype)

    def test_clone(self):
        """
        Check that cloned transformations contain different _matrix objects.
        Also, the clone of a composed translation and rotation has to be
        the same as composition of clones of translation and rotation.
        """
        tr = Translate(torch.FloatTensor([[1.0, 2.0, 3.0]]))
        R = torch.FloatTensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        R = Rotate(R)

        # check that the _matrix property of clones of
        # both transforms are different
        for t in (R, tr):
            self.assertTrue(t._matrix is not t.clone()._matrix)

        # check that the _transforms lists of composition of R, tr contain
        # different objects
        t1 = Transform3d().compose(R, tr)
        for t, t_clone in (t1._transforms, t1.clone()._transforms):
            self.assertTrue(t is not t_clone)
            self.assertTrue(t._matrix is not t_clone._matrix)

        # check that all composed transforms are numerically equivalent
        t2 = Transform3d().compose(R.clone(), tr.clone())
        t3 = t1.clone()
        for t_pair in ((t1, t2), (t1, t3), (t2, t3)):
            matrix1 = t_pair[0].get_matrix()
            matrix2 = t_pair[1].get_matrix()
            self.assertTrue(torch.allclose(matrix1, matrix2))

    def test_init_with_custom_matrix(self):
        for matrix in (torch.randn(10, 4, 4), torch.randn(4, 4)):
            t = Transform3d(matrix=matrix)
            self.assertTrue(t.device == matrix.device)
            self.assertTrue(t._matrix.dtype == matrix.dtype)
            self.assertTrue(torch.allclose(t._matrix, matrix.view(t._matrix.shape)))

    def test_init_with_custom_matrix_errors(self):
        bad_shapes = [[10, 5, 4], [3, 4], [10, 4, 4, 1], [10, 4, 4, 2], [4, 4, 4, 3]]
        for bad_shape in bad_shapes:
            matrix = torch.randn(*bad_shape).float()
            self.assertRaises(ValueError, Transform3d, matrix=matrix)

    def test_get_se3(self):
        N = 16
        random_rotations(N)
        tr = Translate(torch.rand((N, 3)))
        R = Rotate(random_rotations(N))
        transform = Transform3d().compose(R, tr)
        se3_log = transform.get_se3_log()
        gt_se3_log = se3_log_map(transform.get_matrix())
        self.assertClose(se3_log, gt_se3_log)

    def test_translate(self):
        t = Transform3d().translate(1, 2, 3)
        points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(
            1, 3, 3
        )
        normals = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ).view(1, 3, 3)
        points_out = t.transform_points(points)
        normals_out = t.transform_normals(normals)
        points_out_expected = torch.tensor(
            [[2.0, 2.0, 3.0], [1.0, 3.0, 3.0], [1.5, 2.5, 3.0]]
        ).view(1, 3, 3)
        normals_out_expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ).view(1, 3, 3)
        self.assertTrue(torch.allclose(points_out, points_out_expected))
        self.assertTrue(torch.allclose(normals_out, normals_out_expected))

    @mock.patch.dict(os.environ, {"PYTORCH3D_CHECK_ROTATION_MATRICES": "1"}, clear=True)
    def test_rotate_check_rot_valid_on(self):
        R = so3_exp_map(torch.randn((1, 3)))
        t = Transform3d().rotate(R)
        points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(
            1, 3, 3
        )
        normals = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ).view(1, 3, 3)
        points_out = t.transform_points(points)
        normals_out = t.transform_normals(normals)
        points_out_expected = torch.bmm(points, R)
        normals_out_expected = torch.bmm(normals, R)
        self.assertTrue(torch.allclose(points_out, points_out_expected))
        self.assertTrue(torch.allclose(normals_out, normals_out_expected))

    @mock.patch.dict(os.environ, {"PYTORCH3D_CHECK_ROTATION_MATRICES": "0"}, clear=True)
    def test_rotate_check_rot_valid_off(self):
        R = so3_exp_map(torch.randn((1, 3)))
        t = Transform3d().rotate(R)
        points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(
            1, 3, 3
        )
        normals = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ).view(1, 3, 3)
        points_out = t.transform_points(points)
        normals_out = t.transform_normals(normals)
        points_out_expected = torch.bmm(points, R)
        normals_out_expected = torch.bmm(normals, R)
        self.assertTrue(torch.allclose(points_out, points_out_expected))
        self.assertTrue(torch.allclose(normals_out, normals_out_expected))

    def test_scale(self):
        t = Transform3d().scale(2.0).scale(0.5, 0.25, 1.0)
        points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(
            1, 3, 3
        )
        normals = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ).view(1, 3, 3)
        points_out = t.transform_points(points)
        normals_out = t.transform_normals(normals)
        points_out_expected = torch.tensor(
            [[1.00, 0.00, 0.00], [0.00, 0.50, 0.00], [0.50, 0.25, 0.00]]
        ).view(1, 3, 3)
        normals_out_expected = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
        ).view(1, 3, 3)
        self.assertTrue(torch.allclose(points_out, points_out_expected))
        self.assertTrue(torch.allclose(normals_out, normals_out_expected))

    def test_scale_translate(self):
        t = Transform3d().scale(2, 1, 3).translate(1, 2, 3)
        points = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]).view(
            1, 3, 3
        )
        normals = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
        ).view(1, 3, 3)
        points_out = t.transform_points(points)
        normals_out = t.transform_normals(normals)
        points_out_expected = torch.tensor(
            [[3.0, 2.0, 3.0], [1.0, 3.0, 3.0], [2.0, 2.5, 3.0]]
        ).view(1, 3, 3)
        normals_out_expected = torch.tensor(
            [[0.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 1.0, 0.0]]
        ).view(1, 3, 3)
        self.assertTrue(torch.allclose(points_out, points_out_expected))
        self.assertTrue(torch.allclose(normals_out, normals_out_expected))

    def test_rotate_axis_angle(self):
        t = Transform3d().rotate_axis_angle(90.0, axis="Z")
        points = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 1.0]]).view(
            1, 3, 3
        )
        normals = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
        ).view(1, 3, 3)
        points_out = t.transform_points(points)
        normals_out = t.transform_normals(normals)
        points_out_expected = torch.tensor(
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 1.0]]
        ).view(1, 3, 3)
        normals_out_expected = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        ).view(1, 3, 3)
        self.assertTrue(torch.allclose(points_out, points_out_expected, atol=1e-7))
        self.assertTrue(torch.allclose(normals_out, normals_out_expected, atol=1e-7))

    def test_transform_points_fail(self):
        t1 = Scale(0.1, 0.1, 0.1)
        P = 7
        with self.assertRaises(ValueError):
            t1.transform_points(torch.randn(P))

    def test_compose_fail(self):
        # Only composing Transform3d objects is possible
        t1 = Scale(0.1, 0.1, 0.1)
        with self.assertRaises(ValueError):
            t1.compose(torch.randn(100))

    def test_transform_points_eps(self):
        t1 = Transform3d()
        persp_proj = [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ]
        t1._matrix = torch.FloatTensor(persp_proj)
        points = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1e-5], [-1.0, 0.0, 1e-5]]
        ).view(1, 3, 3)  # a set of points with z-coord very close to 0

        proj = t1.transform_points(points)
        proj_eps = t1.transform_points(points, eps=1e-4)

        self.assertTrue(not bool(torch.isfinite(proj.sum())))
        self.assertTrue(bool(torch.isfinite(proj_eps.sum())))

    def test_inverse(self, batch_size=5):
        device = torch.device("cuda:0")

        # generate a random chain of transforms
        for _ in range(10):  # 10 different tries
            # list of transform matrices
            ts = []

            for i in range(10):
                choice = float(torch.rand(1))
                if choice <= 1.0 / 3.0:
                    t_ = Translate(
                        torch.randn(
                            (batch_size, 3), dtype=torch.float32, device=device
                        ),
                        device=device,
                    )
                elif choice <= 2.0 / 3.0:
                    t_ = Rotate(
                        so3_exp_map(
                            torch.randn(
                                (batch_size, 3), dtype=torch.float32, device=device
                            )
                        ),
                        device=device,
                    )
                else:
                    rand_t = torch.randn(
                        (batch_size, 3), dtype=torch.float32, device=device
                    )
                    rand_t = rand_t.sign() * torch.clamp(rand_t.abs(), 0.2)
                    t_ = Scale(rand_t, device=device)
                ts.append(t_._matrix.clone())

                if i == 0:
                    t = t_
                else:
                    t = t.compose(t_)

            # generate the inverse transformation in several possible ways
            m1 = t.inverse(invert_composed=True).get_matrix()
            m2 = t.inverse(invert_composed=True)._matrix
            m3 = t.inverse(invert_composed=False).get_matrix()
            m4 = t.get_matrix().inverse()

            # compute the inverse explicitly ...
            m5 = torch.eye(4, dtype=torch.float32, device=device)
            m5 = m5[None].repeat(batch_size, 1, 1)
            for t_ in ts:
                m5 = torch.bmm(torch.inverse(t_), m5)

            # assert all same
            for m in (m1, m2, m3, m4):
                self.assertTrue(torch.allclose(m, m5, atol=1e-3))

    def _check_indexed_transforms(self, t3d, t3d_selected, indices):
        t3d_matrix = t3d.get_matrix()
        t3d_selected_matrix = t3d_selected.get_matrix()
        for order_index, selected_index in indices:
            self.assertClose(
                t3d_matrix[selected_index], t3d_selected_matrix[order_index]
            )

    def test_get_item(self, batch_size=5):
        device = torch.device("cuda:0")

        matrices = torch.randn(
            size=[batch_size, 4, 4], device=device, dtype=torch.float32
        )

        # init the Transforms3D class
        t3d = Transform3d(matrix=matrices)

        # int index
        index = 1
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 1)
        self._check_indexed_transforms(t3d, t3d_selected, [(0, 1)])

        # negative int index
        index = -1
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 1)
        self._check_indexed_transforms(t3d, t3d_selected, [(0, -1)])

        # list index
        index = [1, 2]
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), len(index))
        self._check_indexed_transforms(t3d, t3d_selected, enumerate(index))

        # empty list index
        index = []
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 0)
        self.assertEqual(t3d_selected.get_matrix().nelement(), 0)

        # slice index
        index = slice(0, 2, 1)
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 2)
        self._check_indexed_transforms(t3d, t3d_selected, [(0, 0), (1, 1)])

        # empty slice index
        index = slice(0, 0, 1)
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 0)
        self.assertEqual(t3d_selected.get_matrix().nelement(), 0)

        # bool tensor
        index = (torch.rand(batch_size) > 0.5).to(device)
        index[:2] = True  # make sure smth is selected
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), index.sum())
        self._check_indexed_transforms(
            t3d,
            t3d_selected,
            zip(
                torch.arange(index.sum()),
                torch.nonzero(index, as_tuple=False).squeeze(),
            ),
        )

        # all false bool tensor
        index = torch.zeros(batch_size).bool()
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 0)
        self.assertEqual(t3d_selected.get_matrix().nelement(), 0)

        # int tensor
        index = torch.tensor([1, 2], dtype=torch.int64, device=device)
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), index.numel())
        self._check_indexed_transforms(t3d, t3d_selected, enumerate(index.tolist()))

        # negative int tensor
        index = -(torch.tensor([1, 2], dtype=torch.int64, device=device))
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), index.numel())
        self._check_indexed_transforms(t3d, t3d_selected, enumerate(index.tolist()))

        # invalid index
        for invalid_index in (
            torch.tensor([1, 0, 1], dtype=torch.float32, device=device),  # float tensor
            1.2,  # float index
        ):
            with self.assertRaises(IndexError):
                t3d_selected = t3d[invalid_index]

    def test_stack(self):
        rotations = random_rotations(3)
        transform3 = Transform3d().rotate(rotations).translate(torch.full((3, 3), 0.3))
        transform1 = Scale(37)
        transform4 = transform1.stack(transform3)
        self.assertEqual(len(transform1), 1)
        self.assertEqual(len(transform3), 3)
        self.assertEqual(len(transform4), 4)
        self.assertClose(
            transform4.get_matrix(),
            torch.cat([transform1.get_matrix(), transform3.get_matrix()]),
        )
        points = torch.rand(4, 5, 3)
        new_points_expect = torch.cat(
            [
                transform1.transform_points(points[:1]),
                transform3.transform_points(points[1:]),
            ]
        )
        new_points = transform4.transform_points(points)
        self.assertClose(new_points, new_points_expect)


class TestTranslate(unittest.TestCase):
    def test_python_scalar(self):
        t = Translate(0.2, 0.3, 0.4)
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.2, 0.3, 0.4, 1],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_torch_scalar(self):
        x = torch.tensor(0.2)
        y = torch.tensor(0.3)
        z = torch.tensor(0.4)
        t = Translate(x, y, z)
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.2, 0.3, 0.4, 1],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_mixed_scalars(self):
        x = 0.2
        y = torch.tensor(0.3)
        z = 0.4
        t = Translate(x, y, z)
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.2, 0.3, 0.4, 1],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_torch_scalar_grads(self):
        # Make sure backprop works if we give torch scalars
        x = torch.tensor(0.2, requires_grad=True)
        y = torch.tensor(0.3, requires_grad=True)
        z = torch.tensor(0.4)
        t = Translate(x, y, z)
        t._matrix.sum().backward()
        self.assertTrue(hasattr(x, "grad"))
        self.assertTrue(hasattr(y, "grad"))
        self.assertTrue(torch.allclose(x.grad, x.new_ones(x.shape)))
        self.assertTrue(torch.allclose(y.grad, y.new_ones(y.shape)))

    def test_torch_vectors(self):
        x = torch.tensor([0.2, 2.0])
        y = torch.tensor([0.3, 3.0])
        z = torch.tensor([0.4, 4.0])
        t = Translate(x, y, z)
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.2, 0.3, 0.4, 1],
                ],
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [2.0, 3.0, 4.0, 1],
                ],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_vector_broadcast(self):
        x = torch.tensor([0.2, 2.0])
        y = torch.tensor([0.3, 3.0])
        z = torch.tensor([0.4])
        t = Translate(x, y, z)
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.2, 0.3, 0.4, 1],
                ],
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [2.0, 3.0, 0.4, 1],
                ],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_bad_broadcast(self):
        x = torch.tensor([0.2, 2.0, 20.0])
        y = torch.tensor([0.3, 3.0])
        z = torch.tensor([0.4])
        with self.assertRaises(ValueError):
            Translate(x, y, z)

    def test_mixed_broadcast(self):
        x = 0.2
        y = torch.tensor(0.3)
        z = torch.tensor([0.4, 4.0])
        t = Translate(x, y, z)
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.2, 0.3, 0.4, 1],
                ],
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.2, 0.3, 4.0, 1],
                ],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_mixed_broadcast_grad(self):
        x = 0.2
        y = torch.tensor(0.3, requires_grad=True)
        z = torch.tensor([0.4, 4.0], requires_grad=True)
        t = Translate(x, y, z)
        t._matrix.sum().backward()
        self.assertTrue(hasattr(y, "grad"))
        self.assertTrue(hasattr(z, "grad"))
        y_grad = torch.tensor(2.0)
        z_grad = torch.tensor([1.0, 1.0])
        self.assertEqual(y.grad.shape, y_grad.shape)
        self.assertEqual(z.grad.shape, z_grad.shape)
        self.assertTrue(torch.allclose(y.grad, y_grad))
        self.assertTrue(torch.allclose(z.grad, z_grad))

    def test_matrix(self):
        xyz = torch.tensor([[0.2, 0.3, 0.4], [2.0, 3.0, 4.0]])
        t = Translate(xyz)
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0.2, 0.3, 0.4, 1],
                ],
                [
                    [1.0, 0.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [2.0, 3.0, 4.0, 1],
                ],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_matrix_extra_args(self):
        xyz = torch.tensor([[0.2, 0.3, 0.4], [2.0, 3.0, 4.0]])
        with self.assertRaises(ValueError):
            Translate(xyz, xyz[:, 1], xyz[:, 2])

    def test_inverse(self):
        xyz = torch.tensor([[0.2, 0.3, 0.4], [2.0, 3.0, 4.0]])
        t = Translate(xyz)
        im = t.inverse()._matrix
        im_2 = t._matrix.inverse()
        im_comp = t.get_matrix().inverse()
        self.assertTrue(torch.allclose(im, im_comp))
        self.assertTrue(torch.allclose(im, im_2))

    def test_get_item(self, batch_size=5):
        device = torch.device("cuda:0")
        xyz = torch.randn(size=[batch_size, 3], device=device, dtype=torch.float32)
        t3d = Translate(xyz)
        index = 1
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 1)
        self.assertIsInstance(t3d_selected, Translate)


class TestScale(unittest.TestCase):
    def test_single_python_scalar(self):
        t = Scale(0.1)
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_single_torch_scalar(self):
        t = Scale(torch.tensor(0.1))
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_single_vector(self):
        t = Scale(torch.tensor([0.1, 0.2]))
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.2, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.2, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_single_matrix(self):
        xyz = torch.tensor([[0.1, 0.2, 0.3], [1.0, 2.0, 3.0]])
        t = Scale(xyz)
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_three_python_scalar(self):
        t = Scale(0.1, 0.2, 0.3)
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_three_torch_scalar(self):
        t = Scale(torch.tensor(0.1), torch.tensor(0.2), torch.tensor(0.3))
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_three_mixed_scalar(self):
        t = Scale(torch.tensor(0.1), 0.2, torch.tensor(0.3))
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_three_vector_broadcast(self):
        x = torch.tensor([0.1])
        y = torch.tensor([0.2, 2.0])
        z = torch.tensor([0.3, 3.0])
        t = Scale(x, y, z)
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_three_mixed_broadcast_grad(self):
        x = 0.1
        y = torch.tensor(0.2, requires_grad=True)
        z = torch.tensor([0.3, 3.0], requires_grad=True)
        t = Scale(x, y, z)
        matrix = torch.tensor(
            [
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 0.3, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.2, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))
        t._matrix.sum().backward()
        self.assertTrue(hasattr(y, "grad"))
        self.assertTrue(hasattr(z, "grad"))
        y_grad = torch.tensor(2.0)
        z_grad = torch.tensor([1.0, 1.0])
        self.assertTrue(torch.allclose(y.grad, y_grad))
        self.assertTrue(torch.allclose(z.grad, z_grad))

    def test_inverse(self):
        x = torch.tensor([0.1])
        y = torch.tensor([0.2, 2.0])
        z = torch.tensor([0.3, 3.0])
        t = Scale(x, y, z)
        im = t.inverse()._matrix
        im_2 = t._matrix.inverse()
        im_comp = t.get_matrix().inverse()
        self.assertTrue(torch.allclose(im, im_comp))
        self.assertTrue(torch.allclose(im, im_2))

    def test_get_item(self, batch_size=5):
        device = torch.device("cuda:0")
        s = torch.randn(size=[batch_size, 3], device=device, dtype=torch.float32)
        t3d = Scale(s)
        index = 1
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 1)
        self.assertIsInstance(t3d_selected, Scale)


class TestTransformBroadcast(unittest.TestCase):
    def test_broadcast_transform_points(self):
        t1 = Scale(0.1, 0.1, 0.1)
        N = 10
        P = 7
        M = 20
        x = torch.tensor([0.2] * N)
        y = torch.tensor([0.3] * N)
        z = torch.tensor([0.4] * N)
        tN = Translate(x, y, z)
        p1 = t1.transform_points(torch.randn(P, 3))
        self.assertTrue(p1.shape == (P, 3))
        p2 = t1.transform_points(torch.randn(1, P, 3))
        self.assertTrue(p2.shape == (1, P, 3))
        p3 = t1.transform_points(torch.randn(M, P, 3))
        self.assertTrue(p3.shape == (M, P, 3))
        p4 = tN.transform_points(torch.randn(P, 3))
        self.assertTrue(p4.shape == (N, P, 3))
        p5 = tN.transform_points(torch.randn(1, P, 3))
        self.assertTrue(p5.shape == (N, P, 3))

    def test_broadcast_transform_normals(self):
        t1 = Scale(0.1, 0.1, 0.1)
        N = 10
        P = 7
        M = 20
        x = torch.tensor([0.2] * N)
        y = torch.tensor([0.3] * N)
        z = torch.tensor([0.4] * N)
        tN = Translate(x, y, z)
        p1 = t1.transform_normals(torch.randn(P, 3))
        self.assertTrue(p1.shape == (P, 3))
        p2 = t1.transform_normals(torch.randn(1, P, 3))
        self.assertTrue(p2.shape == (1, P, 3))
        p3 = t1.transform_normals(torch.randn(M, P, 3))
        self.assertTrue(p3.shape == (M, P, 3))
        p4 = tN.transform_normals(torch.randn(P, 3))
        self.assertTrue(p4.shape == (N, P, 3))
        p5 = tN.transform_normals(torch.randn(1, P, 3))
        self.assertTrue(p5.shape == (N, P, 3))

    def test_broadcast_compose(self):
        t1 = Scale(0.1, 0.1, 0.1)
        N = 10
        scale_n = torch.tensor([0.3] * N)
        tN = Scale(scale_n)
        t1N = t1.compose(tN)
        self.assertTrue(t1._matrix.shape == (1, 4, 4))
        self.assertTrue(tN._matrix.shape == (N, 4, 4))
        self.assertTrue(t1N.get_matrix().shape == (N, 4, 4))
        t11 = t1.compose(t1)
        self.assertTrue(t11.get_matrix().shape == (1, 4, 4))

    def test_broadcast_compose_fail(self):
        # Cannot compose two transforms which have batch dimensions N and M
        # other than the case where either N or M is 1
        N = 10
        M = 20
        scale_n = torch.tensor([0.3] * N)
        tN = Scale(scale_n)
        x = torch.tensor([0.2] * M)
        y = torch.tensor([0.3] * M)
        z = torch.tensor([0.4] * M)
        tM = Translate(x, y, z)
        t = tN.compose(tM)
        with self.assertRaises(ValueError):
            t.get_matrix()

    def test_multiple_broadcast_compose(self):
        t1 = Scale(0.1, 0.1, 0.1)
        t2 = Scale(0.2, 0.2, 0.2)
        N = 10
        scale_n = torch.tensor([0.3] * N)
        tN = Scale(scale_n)
        t1N2 = t1.compose(tN.compose(t2))
        composed_mat = t1N2.get_matrix()
        self.assertTrue(composed_mat.shape == (N, 4, 4))
        expected_mat = torch.eye(3, dtype=torch.float32) * 0.3 * 0.2 * 0.1
        self.assertTrue(torch.allclose(composed_mat[0, :3, :3], expected_mat))


class TestRotate(unittest.TestCase):
    def test_single_matrix(self):
        R = torch.eye(3)
        t = Rotate(R)
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(torch.allclose(t._matrix, matrix))

    def test_invalid_dimensions(self):
        R = torch.eye(4)
        with self.assertRaises(ValueError):
            Rotate(R)

    def test_inverse(self, batch_size=5):
        device = torch.device("cuda:0")
        log_rot = torch.randn((batch_size, 3), dtype=torch.float32, device=device)
        R = so3_exp_map(log_rot)
        t = Rotate(R)
        im = t.inverse()._matrix
        im_2 = t._matrix.inverse()
        im_comp = t.get_matrix().inverse()
        self.assertTrue(torch.allclose(im, im_comp, atol=1e-4))
        self.assertTrue(torch.allclose(im, im_2, atol=1e-4))

    def test_get_item(self, batch_size=5):
        device = torch.device("cuda:0")
        r = random_rotations(batch_size, dtype=torch.float32, device=device)
        t3d = Rotate(r)
        index = 1
        t3d_selected = t3d[index]
        self.assertEqual(len(t3d_selected), 1)
        self.assertIsInstance(t3d_selected, Rotate)


class TestRotateAxisAngle(unittest.TestCase):
    def test_rotate_x_python_scalar(self):
        t = RotateAxisAngle(angle=90, axis="X")
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [1.0,  0.0, 0.0, 0.0],  # noqa: E241, E201
                    [0.0,  0.0, 1.0, 0.0],  # noqa: E241, E201
                    [0.0, -1.0, 0.0, 0.0],  # noqa: E241, E201
                    [0.0,  0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        points = torch.tensor([0.0, 1.0, 0.0])[None, None, :]  # (1, 1, 3)
        transformed_points = t.transform_points(points)
        expected_points = torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(
            torch.allclose(transformed_points.squeeze(), expected_points, atol=1e-7)
        )
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_x_torch_scalar(self):
        angle = torch.tensor(90.0)
        t = RotateAxisAngle(angle=angle, axis="X")
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [1.0,  0.0, 0.0, 0.0],  # noqa: E241, E201
                    [0.0,  0.0, 1.0, 0.0],  # noqa: E241, E201
                    [0.0, -1.0, 0.0, 0.0],  # noqa: E241, E201
                    [0.0,  0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        points = torch.tensor([0.0, 1.0, 0.0])[None, None, :]  # (1, 1, 3)
        transformed_points = t.transform_points(points)
        expected_points = torch.tensor([0.0, 0.0, 1.0])
        self.assertTrue(
            torch.allclose(transformed_points.squeeze(), expected_points, atol=1e-7)
        )
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_x_torch_tensor(self):
        angle = torch.tensor([0, 45.0, 90.0])  # (N)
        t = RotateAxisAngle(angle=angle, axis="X")
        r2_i = 1 / math.sqrt(2)
        r2_2 = math.sqrt(2) / 2
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0,   0.0,  0.0, 0.0],  # noqa: E241, E201
                    [0.0,  r2_2, r2_i, 0.0],  # noqa: E241, E201
                    [0.0, -r2_i, r2_2, 0.0],  # noqa: E241, E201
                    [0.0,   0.0,  0.0, 1.0],  # noqa: E241, E201
                ],
                [
                    [1.0,  0.0, 0.0,  0.0],   # noqa: E241, E201
                    [0.0,  0.0, 1.0,  0.0],   # noqa: E241, E201
                    [0.0, -1.0, 0.0,  0.0],   # noqa: E241, E201
                    [0.0,  0.0, 0.0,  1.0],   # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))
        angle = angle
        t = RotateAxisAngle(angle=angle, axis="X")
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_y_python_scalar(self):
        t = RotateAxisAngle(angle=90, axis="Y")
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [0.0, 0.0, -1.0, 0.0],  # noqa: E241, E201
                    [0.0, 1.0,  0.0, 0.0],  # noqa: E241, E201
                    [1.0, 0.0,  0.0, 0.0],  # noqa: E241, E201
                    [0.0, 0.0,  0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        points = torch.tensor([1.0, 0.0, 0.0])[None, None, :]  # (1, 1, 3)
        transformed_points = t.transform_points(points)
        expected_points = torch.tensor([0.0, 0.0, -1.0])
        self.assertTrue(
            torch.allclose(transformed_points.squeeze(), expected_points, atol=1e-7)
        )
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_y_torch_scalar(self):
        """
        Test rotation about Y axis. With a right hand coordinate system this
        should result in a vector pointing along the x-axis being rotated to
        point along the negative z axis.
        """
        angle = torch.tensor(90.0)
        t = RotateAxisAngle(angle=angle, axis="Y")
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [0.0, 0.0, -1.0, 0.0],  # noqa: E241, E201
                    [0.0, 1.0,  0.0, 0.0],  # noqa: E241, E201
                    [1.0, 0.0,  0.0, 0.0],  # noqa: E241, E201
                    [0.0, 0.0,  0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        points = torch.tensor([1.0, 0.0, 0.0])[None, None, :]  # (1, 1, 3)
        transformed_points = t.transform_points(points)
        expected_points = torch.tensor([0.0, 0.0, -1.0])
        self.assertTrue(
            torch.allclose(transformed_points.squeeze(), expected_points, atol=1e-7)
        )
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_y_torch_tensor(self):
        angle = torch.tensor([0, 45.0, 90.0])
        t = RotateAxisAngle(angle=angle, axis="Y")
        r2_i = 1 / math.sqrt(2)
        r2_2 = math.sqrt(2) / 2
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [r2_2,  0.0, -r2_i, 0.0],  # noqa: E241, E201
                    [ 0.0,  1.0,   0.0, 0.0],  # noqa: E241, E201
                    [r2_i,  0.0,  r2_2, 0.0],  # noqa: E241, E201
                    [ 0.0,  0.0,   0.0, 1.0],  # noqa: E241, E201
                ],
                [
                    [0.0,  0.0, -1.0, 0.0],  # noqa: E241, E201
                    [0.0,  1.0,  0.0, 0.0],  # noqa: E241, E201
                    [1.0,  0.0,  0.0, 0.0],  # noqa: E241, E201
                    [0.0,  0.0,  0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_z_python_scalar(self):
        t = RotateAxisAngle(angle=90, axis="Z")
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [ 0.0, 1.0, 0.0, 0.0],  # noqa: E241, E201
                    [-1.0, 0.0, 0.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 1.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        points = torch.tensor([1.0, 0.0, 0.0])[None, None, :]  # (1, 1, 3)
        transformed_points = t.transform_points(points)
        expected_points = torch.tensor([0.0, 1.0, 0.0])
        self.assertTrue(
            torch.allclose(transformed_points.squeeze(), expected_points, atol=1e-7)
        )
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_z_torch_scalar(self):
        angle = torch.tensor(90.0)
        t = RotateAxisAngle(angle=angle, axis="Z")
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [ 0.0, 1.0, 0.0, 0.0],  # noqa: E241, E201
                    [-1.0, 0.0, 0.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 1.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        points = torch.tensor([1.0, 0.0, 0.0])[None, None, :]  # (1, 1, 3)
        transformed_points = t.transform_points(points)
        expected_points = torch.tensor([0.0, 1.0, 0.0])
        self.assertTrue(
            torch.allclose(transformed_points.squeeze(), expected_points, atol=1e-7)
        )
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_z_torch_tensor(self):
        angle = torch.tensor([0, 45.0, 90.0])
        t = RotateAxisAngle(angle=angle, axis="Z")
        r2_i = 1 / math.sqrt(2)
        r2_2 = math.sqrt(2) / 2
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [ r2_2,   r2_i,  0.0, 0.0],  # noqa: E241, E201
                    [-r2_i,   r2_2,  0.0, 0.0],  # noqa: E241, E201
                    [  0.0,    0.0,  1.0, 0.0],  # noqa: E241, E201
                    [  0.0,    0.0,  0.0, 1.0],  # noqa: E241, E201
                ],
                [
                    [ 0.0,  1.0, 0.0, 0.0],  # noqa: E241, E201
                    [-1.0,  0.0, 0.0, 0.0],  # noqa: E241, E201
                    [ 0.0,  0.0, 1.0, 0.0],  # noqa: E241, E201
                    [ 0.0,  0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_rotate_compose_x_y_z(self):
        angle = torch.tensor(90.0)
        t1 = RotateAxisAngle(angle=angle, axis="X")
        t2 = RotateAxisAngle(angle=angle, axis="Y")
        t3 = RotateAxisAngle(angle=angle, axis="Z")
        t = t1.compose(t2, t3)
        # fmt: off
        matrix1 = torch.tensor(
            [
                [
                    [1.0,  0.0, 0.0, 0.0],  # noqa: E241, E201
                    [0.0,  0.0, 1.0, 0.0],  # noqa: E241, E201
                    [0.0, -1.0, 0.0, 0.0],  # noqa: E241, E201
                    [0.0,  0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        matrix2 = torch.tensor(
            [
                [
                    [0.0, 0.0, -1.0, 0.0],  # noqa: E241, E201
                    [0.0, 1.0,  0.0, 0.0],  # noqa: E241, E201
                    [1.0, 0.0,  0.0, 0.0],  # noqa: E241, E201
                    [0.0, 0.0,  0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        matrix3 = torch.tensor(
            [
                [
                    [ 0.0, 1.0, 0.0, 0.0],  # noqa: E241, E201
                    [-1.0, 0.0, 0.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 1.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        # order of transforms is t1 -> t2
        matrix = torch.matmul(matrix1, torch.matmul(matrix2, matrix3))
        composed_matrix = t.get_matrix()
        self.assertTrue(torch.allclose(composed_matrix, matrix, atol=1e-7))

    def test_rotate_angle_radians(self):
        t = RotateAxisAngle(angle=math.pi / 2, degrees=False, axis="Z")
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [ 0.0, 1.0, 0.0, 0.0],  # noqa: E241, E201
                    [-1.0, 0.0, 0.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 1.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_lower_case_axis(self):
        t = RotateAxisAngle(angle=90.0, axis="z")
        # fmt: off
        matrix = torch.tensor(
            [
                [
                    [ 0.0, 1.0, 0.0, 0.0],  # noqa: E241, E201
                    [-1.0, 0.0, 0.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 1.0, 0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, 0.0, 1.0],  # noqa: E241, E201
                ]
            ],
            dtype=torch.float32,
        )
        # fmt: on
        self.assertTrue(torch.allclose(t._matrix, matrix, atol=1e-7))

    def test_axis_fail(self):
        with self.assertRaises(ValueError):
            RotateAxisAngle(angle=90.0, axis="P")

    def test_rotate_angle_fail(self):
        angle = torch.tensor([[0, 45.0, 90.0], [0, 45.0, 90.0]])
        with self.assertRaises(ValueError):
            RotateAxisAngle(angle=angle, axis="X")
