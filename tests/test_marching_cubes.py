# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import unittest

import torch
from pytorch3d.ops.marching_cubes import marching_cubes, marching_cubes_naive

from .common_testing import get_tests_dir, TestCaseMixin


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

        expected_verts = torch.tensor([[]])
        expected_faces = torch.tensor([[]], dtype=torch.int64)
        self.assertClose(verts, expected_verts)
        self.assertClose(faces, expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts, expected_verts)
        self.assertClose(faces, expected_faces)

    def test_case1(self):  # case 1
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0, 0, 0] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        expected_verts = torch.tensor(
            [
                [0.5, 0, 0],
                [0, 0.5, 0],
                [0, 0, 0.5],
            ]
        )
        expected_faces = torch.tensor([[0, 1, 2]])

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        expected_verts = convert_to_local(expected_verts, 2)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [0.0000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [1.0000, 0.5000, 0.0000],
            ]
        )
        expected_faces = torch.tensor([[0, 1, 2], [3, 1, 0]])
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [1.0000, 0.5000, 0.0000],
                [1.0000, 1.0000, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.5000],
            ]
        )
        expected_faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [0.0000, 0.0000, 0.5000],
                [1.0000, 0.5000, 0.0000],
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 1.0000],
            ]
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 3, 1], [3, 4, 1]])
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

    def test_case5(self):
        volume_data = torch.ones(1, 2, 2, 2)  # (B, W, H, D)
        volume_data[0, 0:2, 0, 0:2] = 0
        volume_data = volume_data.permute(0, 3, 2, 1)  # (B, D, H, W)
        verts, faces = marching_cubes_naive(volume_data, return_local_coords=False)

        expected_verts = torch.tensor(
            [
                [1.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 0.0000],
                [1.0000, 0.5000, 1.0000],
                [0.0000, 0.5000, 1.0000],
            ]
        )

        expected_faces = torch.tensor([[0, 1, 2], [2, 1, 3]])
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [0.5000, 1.0000, 0.0000],
                [0.0000, 1.0000, 0.5000],
                [0.0000, 0.5000, 0.0000],
                [1.0000, 0.5000, 0.0000],
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 1.0000],
                [0.0000, 0.0000, 0.5000],
            ]
        )
        expected_faces = torch.tensor([[0, 1, 2], [3, 4, 5], [3, 5, 6], [5, 4, 7]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [0.5000, 1.0000, 1.0000],
                [0.0000, 0.5000, 1.0000],
                [0.0000, 1.0000, 0.5000],
                [1.0000, 0.0000, 0.5000],
                [0.5000, 0.0000, 1.0000],
                [1.0000, 0.5000, 1.0000],
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 0.0000],
                [1.0000, 0.5000, 0.0000],
                [1.0000, 1.0000, 0.5000],
            ]
        )

        expected_faces = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [1.0000, 0.5000, 1.0000],
                [0.0000, 1.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
                [1.0000, 0.0000, 0.5000],
                [0.0000, 0.5000, 0.0000],
                [0.5000, 0.0000, 0.0000],
            ]
        )
        expected_faces = torch.tensor([[0, 1, 2], [3, 1, 0], [3, 4, 1], [3, 5, 4]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [0.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [1.0000, 0.5000, 0.0000],
                [0.5000, 1.0000, 1.0000],
            ]
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4], [5, 3, 2]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [0.0000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [1.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [0.5000, 1.0000, 1.0000],
            ]
        )

        expected_faces = torch.tensor([[0, 1, 2], [3, 4, 5]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [0.0000, 0.5000, 0.0000],
                [0.0000, 0.0000, 0.5000],
                [1.0000, 0.5000, 0.0000],
                [1.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [0.5000, 1.0000, 1.0000],
            ]
        )

        expected_faces = torch.tensor([[0, 1, 2], [0, 3, 1], [4, 5, 6]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [1.0000, 0.5000, 0.0000],
                [0.5000, 0.0000, 0.0000],
                [1.0000, 1.0000, 0.5000],
                [1.0000, 0.5000, 1.0000],
                [0.5000, 1.0000, 1.0000],
                [0.0000, 0.5000, 0.0000],
                [0.5000, 1.0000, 0.0000],
                [0.0000, 1.0000, 0.5000],
            ]
        )

        expected_faces = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [1.0000, 0.0000, 0.5000],
                [0.5000, 0.0000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
                [0.0000, 0.0000, 0.5000],
                [0.5000, 0.0000, 0.0000],
                [0.5000, 1.0000, 0.0000],
                [0.0000, 1.0000, 0.5000],
            ]
        )

        expected_faces = torch.tensor([[0, 1, 2], [2, 1, 3], [4, 5, 6], [4, 6, 7]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [0.5000, 0.0000, 0.0000],
                [0.0000, 0.5000, 0.0000],
                [0.0000, 0.5000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [1.0000, 0.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
            ]
        )

        expected_faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 4], [3, 2, 5]])

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 2)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
                [1.0000, 0.5000, 1.0000],
                [1.0000, 1.0000, 0.5000],
                [0.5000, 1.0000, 1.0000],
                [1.5000, 1.0000, 1.0000],
                [1.0000, 1.5000, 1.0000],
                [1.0000, 1.0000, 1.5000],
            ]
        )
        expected_faces = torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 3],
                [1, 4, 2],
                [1, 3, 4],
                [0, 2, 5],
                [3, 0, 5],
                [2, 4, 5],
                [3, 5, 4],
            ]
        )
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 3)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)
        self.assertTrue(verts[0].ge(-1).all() and verts[0].le(1).all())

        verts, faces = marching_cubes(volume_data, return_local_coords=True)
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
        expected_verts = torch.tensor(
            [
                [1.0000, 0.9000, 1.0000],
                [1.0000, 1.0000, 0.9000],
                [0.9000, 1.0000, 1.0000],
                [2.0000, 0.9000, 1.0000],
                [2.0000, 1.0000, 0.9000],
                [2.1000, 1.0000, 1.0000],
                [1.0000, 2.0000, 0.9000],
                [0.9000, 2.0000, 1.0000],
                [2.0000, 2.0000, 0.9000],
                [2.1000, 2.0000, 1.0000],
                [1.0000, 2.1000, 1.0000],
                [2.0000, 2.1000, 1.0000],
                [1.0000, 0.9000, 2.0000],
                [0.9000, 1.0000, 2.0000],
                [2.0000, 0.9000, 2.0000],
                [2.1000, 1.0000, 2.0000],
                [0.9000, 2.0000, 2.0000],
                [2.1000, 2.0000, 2.0000],
                [1.0000, 2.1000, 2.0000],
                [2.0000, 2.1000, 2.0000],
                [1.0000, 1.0000, 2.1000],
                [2.0000, 1.0000, 2.1000],
                [1.0000, 2.0000, 2.1000],
                [2.0000, 2.0000, 2.1000],
            ]
        )

        expected_faces = torch.tensor(
            [
                [0, 1, 2],
                [0, 3, 4],
                [1, 0, 4],
                [4, 3, 5],
                [1, 6, 7],
                [2, 1, 7],
                [4, 8, 1],
                [1, 8, 6],
                [8, 4, 5],
                [9, 8, 5],
                [6, 10, 7],
                [6, 8, 11],
                [10, 6, 11],
                [8, 9, 11],
                [12, 0, 2],
                [13, 12, 2],
                [3, 0, 14],
                [14, 0, 12],
                [15, 5, 3],
                [14, 15, 3],
                [2, 7, 13],
                [7, 16, 13],
                [5, 15, 9],
                [9, 15, 17],
                [10, 18, 16],
                [7, 10, 16],
                [11, 19, 10],
                [19, 18, 10],
                [9, 17, 19],
                [11, 9, 19],
                [12, 13, 20],
                [14, 12, 20],
                [21, 14, 20],
                [15, 14, 21],
                [13, 16, 22],
                [20, 13, 22],
                [21, 20, 23],
                [20, 22, 23],
                [17, 15, 21],
                [23, 17, 21],
                [16, 18, 22],
                [23, 22, 18],
                [19, 23, 18],
                [17, 23, 19],
            ]
        )
        verts, faces = marching_cubes_naive(volume_data, 0.9, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, 0.9, return_local_coords=False)
        verts2, faces2 = marching_cubes(volume_data, 0.9, return_local_coords=False)

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, 0.9, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 5)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        # Check all values are in the range [-1, 1]
        self.assertTrue(verts[0].ge(-1).all() and verts[0].le(1).all())

        verts, faces = marching_cubes(volume_data, 0.9, return_local_coords=True)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)
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
                [2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [2.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 2.0],
                [1.0, 1.0, 2.0],
                [1.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
                [1.0, 2.0, 2.0],
                [2.0, 1.0, 1.0],
                [2.0, 1.0, 2.0],
                [2.0, 2.0, 1.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 1.0],
                [2.0, 2.0, 2.0],
                [1.0, 2.0, 1.0],
                [1.0, 2.0, 2.0],
                [2.0, 1.0, 2.0],
                [1.0, 1.0, 2.0],
                [2.0, 2.0, 2.0],
                [1.0, 2.0, 2.0],
            ]
        )

        expected_faces = torch.tensor(
            [
                [0, 1, 2],
                [2, 1, 3],
                [4, 5, 6],
                [6, 5, 7],
                [8, 9, 10],
                [9, 11, 10],
                [12, 13, 14],
                [14, 13, 15],
                [16, 17, 18],
                [17, 19, 18],
                [20, 21, 22],
                [21, 23, 22],
            ]
        )
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume_data, 1, return_local_coords=False)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes_naive(volume_data, 1, return_local_coords=True)
        expected_verts = convert_to_local(expected_verts, 5)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)
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
        expected_verts = verts_and_faces["verts"]
        expected_faces = verts_and_faces["faces"]

        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)

        verts, faces = marching_cubes(volume, 64, return_local_coords=False)
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

        verts, faces = marching_cubes(volume, 64, return_local_coords=True)
        self.assertClose(verts[0], expected_verts)
        self.assertClose(faces[0], expected_faces)
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
            verts2, faces2 = marching_cubes(volume, isolevel=0.001)

            data_filename = "test_marching_cubes_data/double_ellipsoid.pickle"
            filename = os.path.join(DATA_DIR, data_filename)
            with open(filename, "rb") as file:
                verts_and_faces = pickle.load(file)
            expected_verts = verts_and_faces["verts"]
            expected_faces = verts_and_faces["faces"]

            self.assertClose(verts[0], expected_verts)
            self.assertClose(faces[0], expected_faces)
            self.assertClose(verts2[0], expected_verts)
            self.assertClose(faces2[0], expected_faces)

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
            verts_c, faces_c = marching_cubes(volume_data, return_local_coords=False)
            verts_sci, faces_sci = marching_cubes_classic(volume_data[0])

            surf = mesh_surface_area(verts[0], faces[0])
            surf_c = mesh_surface_area(verts_c[0], faces_c[0])
            surf_sci = mesh_surface_area(verts_sci, faces_sci)

            self.assertClose(surf, surf_sci)
            self.assertClose(surf, surf_c)

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
            verts_c, faces_c = marching_cubes(volume, isolevel=64)
            verts_sci, faces_sci = marching_cubes_classic(volume[0], level=64)

            surf = mesh_surface_area(verts[0], faces[0])
            surf_c = mesh_surface_area(verts_c[0], faces_c[0])
            surf_sci = mesh_surface_area(verts_sci, faces_sci)

            self.assertClose(surf, surf_sci)
            self.assertClose(surf, surf_c)

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
            verts_c, faces_c = marching_cubes(volume, isolevel=0)
            verts_sci, faces_sci = marching_cubes_classic(volume[0], level=0)

            surf = mesh_surface_area(verts[0], faces[0])
            surf_c = mesh_surface_area(verts_c[0], faces_c[0])
            surf_sci = mesh_surface_area(verts_sci, faces_sci)

            self.assertClose(surf, surf_sci)
            self.assertClose(surf, surf_c)

    def test_ball_example(self):
        N = 30
        axis_tensor = torch.arange(0, N)
        X, Y, Z = torch.meshgrid(axis_tensor, axis_tensor, axis_tensor, indexing="ij")
        u = (X - 15) ** 2 + (Y - 15) ** 2 + (Z - 15) ** 2 - 8**2
        u = u[None].float()
        verts, faces = marching_cubes_naive(u, 0, return_local_coords=False)
        verts2, faces2 = marching_cubes(u, 0, return_local_coords=False)
        self.assertClose(verts2[0], verts[0])
        self.assertClose(faces2[0], faces[0])
        verts3, faces3 = marching_cubes(u.cuda(), 0, return_local_coords=False)
        self.assertEqual(len(verts3), len(verts))
        self.assertEqual(len(faces3), len(faces))

    @staticmethod
    def marching_cubes_with_init(algo_type: str, batch_size: int, V: int, device: str):
        device = torch.device(device)
        volume_data = torch.rand(
            (batch_size, V, V, V), dtype=torch.float32, device=device
        )
        algo_table = {
            "naive": marching_cubes_naive,
            "extension": marching_cubes,
        }

        def convert():
            algo_table[algo_type](volume_data, return_local_coords=False)
            torch.cuda.synchronize()

        return convert
