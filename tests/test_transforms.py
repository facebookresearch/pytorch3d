# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import math
import unittest

import torch
from pytorch3d.transforms.so3 import so3_exponential_map
from pytorch3d.transforms.transform3d import (
    Rotate,
    RotateAxisAngle,
    Scale,
    Transform3d,
    Translate,
)


class TestTransform(unittest.TestCase):
    def test_to(self):
        tr = Translate(torch.FloatTensor([[1.0, 2.0, 3.0]]))
        R = torch.FloatTensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        R = Rotate(R)
        t = Transform3d().compose(R, tr)
        for _ in range(3):
            t.cpu()
            t.cuda()
            t.cuda()
            t.cpu()

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
        self.assertTrue(torch.allclose(points_out, points_out_expected))
        self.assertTrue(torch.allclose(normals_out, normals_out_expected))

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
        ).view(
            1, 3, 3
        )  # a set of points with z-coord very close to 0

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
                        so3_exponential_map(
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
        with self.assertRaises(ValueError):
            t = tN.compose(tM)
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
        R = so3_exponential_map(log_rot)
        t = Rotate(R)
        im = t.inverse()._matrix
        im_2 = t._matrix.inverse()
        im_comp = t.get_matrix().inverse()
        self.assertTrue(torch.allclose(im, im_comp, atol=1e-4))
        self.assertTrue(torch.allclose(im, im_2, atol=1e-4))


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
        self.assertTrue(torch.allclose(t._matrix, matrix))

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
