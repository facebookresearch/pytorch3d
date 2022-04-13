# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import unittest

import torch
from common_testing import get_tests_dir, TestCaseMixin
from pytorch3d.ops.marching_cubes import marching_cubes_naive


USE_SCIKIT = False
DATA_DIR = get_tests_dir() / "data"


def convert_to_local(verts, volume_dim):
    return (2 * verts) / (volume_dim - 1) - 1


class TestCubeConfiguration(TestCaseMixin, unittest.TestCase):

    # Test single cubes. Each case corresponds to the corresponding
    # cube vertex configuration in each case here (0-indexed):
    # https://en.wikipedia.org/wiki/Marching_cubes#/media/File:MarchingCubes.svg

    def test_empty_volume(self):  # case 0
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor([])
        expected_faces = torch.tensor([], dtype=torch.int64)
        self.assertClose(verts, expected_verts)
        self.assertClose(faces, expected_faces)

    def test_case1(self):  # case 1
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5, 0, 0],
                [0, 0, 0.5],
                [0, 0.5, 0],
            ]
        )

        expected_faces = torch.tensor([[1, 2, 0]])
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case2(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0:2, 0, 0] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [1.0000, 0.0000, 0.5000],
                [0.0000, 0.0000, 0.5000],
                [1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )
        expected_faces = torch.tensor([[1, 2, 0], [3, 2, 1]])
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case3(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data[0, 1, 1, 0] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [1.0000, 1.0000, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )
        expected_faces = torch.tensor([[0, 1, 5], [4, 3, 2]])
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case4(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 1, 0, 0] = 0
        volume_data[0, 1, 0, 1] = 0
        volume_data[0, 0, 0, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [0.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 0.0000],
            ]
        )
        expected_faces = torch.tensor([[0, 2, 1], [0, 4, 2], [4, 3, 2]])
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case5(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0:2, 0, 0:2] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )

        expected_faces = torch.tensor([[1, 0, 2], [2, 0, 3]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case6(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 1, 0, 0] = 0
        volume_data[0, 1, 0, 1] = 0
        volume_data[0, 0, 0, 1] = 0
        volume_data[0, 0, 1, 0] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [0.0000, 1.0000, 0.5000],
                [0.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )
        expected_faces = torch.tensor([[2, 7, 3], [0, 6, 1], [6, 4, 1], [6, 5, 4]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case7(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data[0, 1, 0, 1] = 0
        volume_data[0, 1, 1, 0] = 0
        volume_data[0, 0, 1, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5000, 0.0000, 1.0000],
                [1.0000, 0.0000, 0.5000],
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [0.0000, 1.0000, 0.5000],
                [0.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )

        expected_faces = torch.tensor([[0, 1, 9], [4, 7, 8], [2, 3, 11], [5, 10, 6]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case8(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data[0, 0, 0, 1] = 0
        volume_data[0, 1, 0, 1] = 0
        volume_data[0, 0, 1, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [1.0000, 0.0000, 0.5000],
                [0.5000, 0.0000, 0.0000],
                [0.5000, 1.0000, 1.0000],
                [0.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )
        expected_faces = torch.tensor([[2, 3, 5], [4, 2, 5], [4, 5, 1], [4, 1, 0]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case9(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 1, 0, 0] = 0
        volume_data[0, 0, 0, 1] = 0
        volume_data[0, 1, 0, 1] = 0
        volume_data[0, 0, 1, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
                [0.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 0.0000],
            ]
        )
        expected_faces = torch.tensor([[0, 5, 4], [0, 4, 3], [0, 3, 1], [3, 4, 2]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case10(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data[0, 1, 1, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )

        expected_faces = torch.tensor([[4, 3, 2], [0, 1, 5]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case11(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data[0, 1, 0, 0] = 0
        volume_data[0, 1, 1, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [1.0000, 0.0000, 0.5000],
                [0.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )

        expected_faces = torch.tensor([[5, 1, 6], [5, 0, 1], [4, 3, 2]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case12(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 1, 0, 0] = 0
        volume_data[0, 0, 1, 0] = 0
        volume_data[0, 1, 1, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [1.0000, 0.0000, 0.5000],
                [0.5000, 0.0000, 0.0000],
                [0.5000, 1.0000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [0.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )

        expected_faces = torch.tensor([[6, 3, 2], [7, 0, 1], [5, 4, 8]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case13(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data[0, 0, 1, 0] = 0
        volume_data[0, 1, 0, 1] = 0
        volume_data[0, 1, 1, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5000, 0.0000, 1.0000],
                [1.0000, 0.0000, 0.5000],
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [0.0000, 1.0000, 0.5000],
            ]
        )

        expected_faces = torch.tensor([[3, 6, 2], [3, 7, 6], [1, 5, 0], [5, 4, 0]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case14(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data[0, 0, 0, 1] = 0
        volume_data[0, 1, 0, 1] = 0
        volume_data[0, 1, 1, 1] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [1.0000, 0.0000, 0.5000],
                [0.5000, 0.0000, 0.0000],
                [0.5000, 1.0000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [0.0000, 0.5000, 1.0000],
                [0.0000, 0.5000, 0.0000],
            ]
        )

        expected_faces = torch.tensor([[1, 0, 3], [1, 3, 4], [1, 4, 5], [2, 4, 3]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)


class TestMarchingCubes(TestCaseMixin, unittest.TestCase):
    def test_single_point(self):
        volume_data = torch.zeros(1, 3, 3, 3)  # (B, W, H, D)
        volume_data[0, 1, 1, 1] = 1
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.5, 1, 1],
                [1, 1, 0.5],
                [1, 0.5, 1],
                [1, 1, 1.5],
                [1, 1.5, 1],
                [1.5, 1, 1],
            ]
        )
        expected_faces = torch.tensor(
            [
                [2, 0, 1],
                [2, 3, 0],
                [0, 4, 1],
                [3, 4, 0],
                [5, 2, 1],
                [3, 2, 5],
                [5, 1, 4],
                [3, 5, 4],
            ]
        )

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 3)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)
        self.assertTrue(verts[0].ge(-1).all() and verts[0].le(1).all())

    def test_cube(self):
        volume_data = torch.zeros(1, 5, 5, 5)  # (B, W, H, D)
        volume_data[0, 1, 1, 1] = 1
        volume_data[0, 1, 1, 2] = 1
        volume_data[0, 2, 1, 1] = 1
        volume_data[0, 2, 1, 2] = 1
        volume_data[0, 1, 2, 1] = 1
        volume_data[0, 1, 2, 2] = 1
        volume_data[0, 2, 2, 1] = 1
        volume_data[0, 2, 2, 2] = 1
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, 0.9, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [0.9000, 1.0000, 1.0000],
                [1.0000, 1.0000, 0.9000],
                [1.0000, 0.9000, 1.0000],
                [0.9000, 1.0000, 2.0000],
                [1.0000, 0.9000, 2.0000],
                [1.0000, 1.0000, 2.1000],
                [0.9000, 2.0000, 1.0000],
                [1.0000, 2.0000, 0.9000],
                [0.9000, 2.0000, 2.0000],
                [1.0000, 2.0000, 2.1000],
                [1.0000, 2.1000, 1.0000],
                [1.0000, 2.1000, 2.0000],
                [2.0000, 1.0000, 0.9000],
                [2.0000, 0.9000, 1.0000],
                [2.0000, 0.9000, 2.0000],
                [2.0000, 1.0000, 2.1000],
                [2.0000, 2.0000, 0.9000],
                [2.0000, 2.0000, 2.1000],
                [2.0000, 2.1000, 1.0000],
                [2.0000, 2.1000, 2.0000],
                [2.1000, 1.0000, 1.0000],
                [2.1000, 1.0000, 2.0000],
                [2.1000, 2.0000, 1.0000],
                [2.1000, 2.0000, 2.0000],
            ]
        )

        expected_faces = torch.tensor(
            [
                [2, 0, 1],
                [2, 4, 3],
                [0, 2, 3],
                [4, 5, 3],
                [0, 6, 7],
                [1, 0, 7],
                [3, 8, 0],
                [8, 6, 0],
                [5, 9, 8],
                [3, 5, 8],
                [6, 10, 7],
                [11, 10, 6],
                [8, 11, 6],
                [9, 11, 8],
                [13, 2, 1],
                [12, 13, 1],
                [14, 4, 13],
                [13, 4, 2],
                [4, 14, 15],
                [5, 4, 15],
                [12, 1, 16],
                [1, 7, 16],
                [15, 17, 5],
                [5, 17, 9],
                [16, 7, 10],
                [18, 16, 10],
                [19, 18, 11],
                [18, 10, 11],
                [9, 17, 19],
                [11, 9, 19],
                [20, 13, 12],
                [20, 21, 14],
                [13, 20, 14],
                [15, 14, 21],
                [22, 20, 12],
                [16, 22, 12],
                [21, 20, 23],
                [23, 20, 22],
                [17, 15, 21],
                [23, 17, 21],
                [22, 16, 18],
                [23, 22, 18],
                [19, 23, 18],
                [17, 23, 19],
            ]
        )
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)
        verts, faces = marching_cubes_naive(volume_data, 0.9, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 5)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        # Check all values are in the range [-1, 1]
        self.assertTrue(verts[0].ge(-1).all() and verts[0].le(1).all())

    def test_cube_no_duplicate_verts(self):
        volume_data = torch.zeros(1, 5, 5, 5)  # (B, W, H, D)
        volume_data[0, 1, 1, 1] = 1
        volume_data[0, 1, 1, 2] = 1
        volume_data[0, 2, 1, 1] = 1
        volume_data[0, 2, 1, 2] = 1
        volume_data[0, 1, 2, 1] = 1
        volume_data[0, 1, 2, 2] = 1
        volume_data[0, 2, 2, 1] = 1
        volume_data[0, 2, 2, 2] = 1
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, 1, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 1.0],
                [1.0, 2.0, 2.0],
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 2.0],
                [2.0, 2.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )

        expected_faces = torch.tensor(
            [
                [1, 3, 0],
                [3, 2, 0],
                [5, 1, 4],
                [4, 1, 0],
                [4, 0, 6],
                [0, 2, 6],
                [5, 7, 1],
                [1, 7, 3],
                [7, 6, 3],
                [6, 2, 3],
                [5, 4, 7],
                [7, 4, 6],
            ]
        )

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, 1, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 5)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        # Check all values are in the range [-1, 1]
        self.assertTrue(verts[0].ge(-1).all() and verts[0].le(1).all())

    def test_sphere(self):
        # (B, W, H, D)
        volume = torch.Tensor(
            [
                [
                    [(x - 10) ** 2 + (y - 10) ** 2 + (z - 10) ** 2 for z in range(20)]
                    for y in range(20)
                ]
                for x in range(20)
            ]
        ).unsqueeze(0)
        volume = volume.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(
            volume, isolevel=64, return_local_coords=False
        )

        data_filename = "test_marching_cubes_data/sphere_level64.pickle"
        filename = os.path.join(DATA_DIR, data_filename)
        with open(filename, "rb") as file:
            verts_and_faces = pickle.load(file)
        expected_verts = verts_and_faces["verts"].squeeze()
        expected_faces = verts_and_faces["faces"].squeeze()

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(
            volume, isolevel=64, return_local_coords=True
        )

        expected_verts = convert_to_local(expected_verts, 20)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        # Check all values are in the range [-1, 1]
        self.assertTrue(verts[0].ge(-1).all() and verts[0].le(1).all())

    # Uses skimage.draw.ellipsoid
    def test_double_ellipsoid(self):
        if USE_SCIKIT:
            import numpy as np
            from skimage.draw import ellipsoid

            ellip_base = ellipsoid(6, 10, 16, levelset=True)
            ellip_double = np.concatenate(
                (ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0
            )
            volume = torch.Tensor(ellip_double).unsqueeze(0)
            volume = volume.permute(0, 3, 2, 1)  # (B, D, H, W)
            verts, faces = marching_cubes_naive(volume, isolevel=0.001)

            data_filename = "test_marching_cubes_data/double_ellipsoid.pickle"
            filename = os.path.join(DATA_DIR, data_filename)
            with open(filename, "rb") as file:
                verts_and_faces = pickle.load(file)
            expected_verts = verts_and_faces["verts"]
            expected_faces = verts_and_faces["faces"]

            self.assertClose(verts[0], expected_verts[0])
            self.assertClose(faces[0], expected_faces[0])

    def test_cube_surface_area(self):
        if USE_SCIKIT:
            from skimage.measure import marching_cubes_classic, mesh_surface_area

            volume_data = torch.zeros(1, 5, 5, 5)
            volume_data[0, 1, 1, 1] = 1
            volume_data[0, 1, 1, 2] = 1
            volume_data[0, 2, 1, 1] = 1
            volume_data[0, 2, 1, 2] = 1
            volume_data[0, 1, 2, 1] = 1
            volume_data[0, 1, 2, 2] = 1
            volume_data[0, 2, 2, 1] = 1
            volume_data[0, 2, 2, 2] = 1
            volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
            verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)
            verts_sci, faces_sci = marching_cubes_classic(volume_data[0])

            surf = mesh_surface_area(verts[0], faces[0])
            surf_sci = mesh_surface_area(verts_sci, faces_sci)

            self.assertClose(surf, surf_sci)

    def test_sphere_surface_area(self):
        if USE_SCIKIT:
            from skimage.measure import marching_cubes_classic, mesh_surface_area

            # (B, W, H, D)
            volume = torch.Tensor(
                [
                    [
                        [
                            (x - 10) ** 2 + (y - 10) ** 2 + (z - 10) ** 2
                            for z in range(20)
                        ]
                        for y in range(20)
                    ]
                    for x in range(20)
                ]
            ).unsqueeze(0)
            volume = volume.permute(0, 3, 2, 1)  # (B, D, H, W)
            verts, faces = marching_cubes_naive(volume, isolevel=64)
            verts_sci, faces_sci = marching_cubes_classic(volume[0], level=64)

            surf = mesh_surface_area(verts[0], faces[0])
            surf_sci = mesh_surface_area(verts_sci, faces_sci)

            self.assertClose(surf, surf_sci)

    def test_double_ellipsoid_surface_area(self):
        if USE_SCIKIT:
            import numpy as np
            from skimage.draw import ellipsoid
            from skimage.measure import marching_cubes_classic, mesh_surface_area

            ellip_base = ellipsoid(6, 10, 16, levelset=True)
            ellip_double = np.concatenate(
                (ellip_base[:-1, ...], ellip_base[2:, ...]), axis=0
            )
            volume = torch.Tensor(ellip_double).unsqueeze(0)
            volume = volume.permute(0, 3, 2, 1)  # (B, D, H, W)
            verts, faces = marching_cubes_naive(volume, isolevel=0)
            verts_sci, faces_sci = marching_cubes_classic(volume[0], level=0)

            surf = mesh_surface_area(verts[0], faces[0])
            surf_sci = mesh_surface_area(verts_sci, faces_sci)

            self.assertClose(surf, surf_sci)

    @staticmethod
    def marching_cubes_with_init(batch_size: int, V: int):
        device = torch.device("cuda:0")
        volume_data = torch.rand(
            (batch_size, V, V, V), dtype=torch.float32, device=device
        )
        torch.cuda.synchronize()

        def convert():
            marching_cubes_naive(volume_data, return_local_coords=False)
            torch.cuda.synchronize()

        return convert
