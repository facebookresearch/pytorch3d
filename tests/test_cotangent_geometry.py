# Copyright (c) 2026 Afloat16.
# This source code is licensed under the BSD-style license in the LICENSE file.

import unittest

import torch
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.ops import cot_laplacian
from pytorch3d.structures import Meshes


class TestCotangentGeometry(unittest.TestCase):
    def test_skinny_triangle_matches_analytic_cotangents_and_area(self):
        for dtype in (torch.float32,):
            for height in (0.1, 0.01, 0.0001, 0.00001, 0.00000001, 0.0):
                with self.subTest(dtype=dtype, height=height):
                    vertices = torch.tensor(
                        [[-1, 0, 0], [1, 0, 0], [0, height, 0]], dtype=dtype
                    )
                    actual_height = vertices[2, 1].item()
                    area = max(actual_height, 0.000001)
                    shallow = 1 / area
                    opposite = (actual_height**2 - 1) / (2 * area)
                    expected = torch.tensor(
                        [
                            [0, opposite, shallow],
                            [opposite, 0, shallow],
                            [shallow, shallow, 0],
                        ],
                        dtype=torch.float32,
                    )
                    faces = torch.tensor([[0, 1, 2]])
                    laplacian, inverse_areas = cot_laplacian(vertices, faces)
                    torch.testing.assert_close(
                        laplacian.to_dense(), expected, rtol=1e-5, atol=1e-5
                    )
                    torch.testing.assert_close(
                        inverse_areas,
                        torch.full((3, 1), 1 / area),
                        rtol=1e-5,
                        atol=1e-5,
                    )
                    # A cyclic coordinate permutation preserves the same geometry.
                    rotated, rotated_areas = cot_laplacian(
                        vertices[:, [2, 0, 1]], faces.flip(1)
                    )
                    torch.testing.assert_close(
                        rotated.to_dense(), expected, rtol=1e-5, atol=1e-5
                    )
                    torch.testing.assert_close(rotated_areas, inverse_areas)

    def test_inverse_area_gradients_match_analytic_triangle_derivative(self):
        for dtype in (torch.float32,):
            with self.subTest(dtype=dtype):
                vertices = torch.tensor(
                    [[-1, 0, 0], [1, 0, 0], [0, 0.0001, 0]],
                    dtype=dtype,
                    requires_grad=True,
                )
                height = vertices[2, 1].item()
                _, inverse_areas = cot_laplacian(vertices, torch.tensor([[0, 1, 2]]))
                gradient = torch.autograd.grad(
                    inverse_areas.sum() * height**2, vertices
                )[0]
                expected = torch.tensor(
                    [[1.5 * height, 1.5, 0], [-1.5 * height, 1.5, 0], [0, -3, 0]],
                    dtype=dtype,
                )
                torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-6)
                self.assertTrue(torch.isfinite(gradient).all())

    def test_cotangent_gradients_match_analytic_height_derivatives(self):
        for height in (0.1, 0.01, 0.0001):
            with self.subTest(height=height):
                vertices = torch.tensor(
                    [[-1, 0, 0], [1, 0, 0], [0, height, 0]],
                    dtype=torch.float32,
                    requires_grad=True,
                )
                actual_height = vertices[2, 1].item()
                laplacian, _ = cot_laplacian(vertices, torch.tensor([[0, 1, 2]]))
                matrix = laplacian.to_dense()
                shallow_gradient = torch.autograd.grad(
                    matrix[0, 2] * actual_height**2, vertices, retain_graph=True
                )[0]
                opposite_gradient = torch.autograd.grad(
                    matrix[0, 1] * actual_height**2, vertices
                )[0]
                torch.testing.assert_close(
                    shallow_gradient[2, 1], torch.tensor(-1.0), rtol=1e-5, atol=1e-6
                )
                torch.testing.assert_close(
                    opposite_gradient[2, 1],
                    torch.tensor(0.5 + 0.5 * actual_height**2),
                    rtol=1e-5,
                    atol=1e-6,
                )
                self.assertTrue(torch.isfinite(shallow_gradient).all())
                self.assertTrue(torch.isfinite(opposite_gradient).all())

    def test_cotangent_curvature_loss_matches_analytic_mesh_operator(self):
        height = 0.0001
        vertices = torch.tensor([[-1, 0, 0], [1, 0, 0], [0, height, 0]])
        height = vertices[2, 1].item()
        shallow = 1 / height
        opposite = (height**2 - 1) / (2 * height)
        matrix = torch.tensor(
            [[0, opposite, shallow], [opposite, 0, shallow], [shallow, shallow, 0]]
        )
        displacement = matrix @ vertices - matrix.sum(1, keepdim=True) * vertices
        expected = (displacement / (4 * height)).norm(dim=1).mean()
        mesh = Meshes(verts=[vertices], faces=[torch.tensor([[0, 1, 2]])])
        torch.testing.assert_close(
            mesh_laplacian_smoothing(mesh, "cotcurv"), expected, rtol=1e-5, atol=1e-5
        )
