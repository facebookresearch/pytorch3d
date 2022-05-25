# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.ops import cubify

from .common_testing import TestCaseMixin


class TestCubify(TestCaseMixin, unittest.TestCase):
    def test_allempty(self):
        N, V = 32, 14
        device = torch.device("cuda:0")
        voxels = torch.zeros((N, V, V, V), dtype=torch.float32, device=device)
        meshes = cubify(voxels, 0.5)
        self.assertTrue(meshes.isempty())

    def test_cubify(self):
        N, V = 4, 2
        device = torch.device("cuda:0")
        voxels = torch.zeros((N, V, V, V), dtype=torch.float32, device=device)

        # 1st example: (top left corner, znear) is on
        voxels[0, 0, 0, 0] = 1.0
        # 2nd example: all are on
        voxels[1] = 1.0
        # 3rd example: empty
        # 4th example
        voxels[3, :, :, 1] = 1.0
        voxels[3, 1, 1, 0] = 1.0

        # compute cubify
        meshes = cubify(voxels, 0.5)

        # 1st-check
        verts, faces = meshes.get_mesh_verts_faces(0)
        self.assertClose(faces.max().cpu(), torch.tensor(verts.size(0) - 1))
        self.assertClose(
            verts,
            torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
                device=device,
            ),
        )
        self.assertClose(
            faces,
            torch.tensor(
                [
                    [0, 1, 4],
                    [1, 5, 4],
                    [4, 5, 6],
                    [5, 7, 6],
                    [0, 4, 6],
                    [0, 6, 2],
                    [0, 3, 1],
                    [0, 2, 3],
                    [6, 7, 3],
                    [6, 3, 2],
                    [1, 7, 5],
                    [1, 3, 7],
                ],
                dtype=torch.int64,
                device=device,
            ),
        )
        # 2nd-check
        verts, faces = meshes.get_mesh_verts_faces(1)
        self.assertClose(faces.max().cpu(), torch.tensor(verts.size(0) - 1))
        self.assertClose(
            verts,
            torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [-1.0, -1.0, 3.0],
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, -1.0, 3.0],
                    [3.0, -1.0, -1.0],
                    [3.0, -1.0, 1.0],
                    [3.0, -1.0, 3.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, 1.0, 1.0],
                    [-1.0, 1.0, 3.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 3.0],
                    [3.0, 1.0, -1.0],
                    [3.0, 1.0, 1.0],
                    [3.0, 1.0, 3.0],
                    [-1.0, 3.0, -1.0],
                    [-1.0, 3.0, 1.0],
                    [-1.0, 3.0, 3.0],
                    [1.0, 3.0, -1.0],
                    [1.0, 3.0, 1.0],
                    [1.0, 3.0, 3.0],
                    [3.0, 3.0, -1.0],
                    [3.0, 3.0, 1.0],
                    [3.0, 3.0, 3.0],
                ],
                dtype=torch.float32,
                device=device,
            ),
        )
        self.assertClose(
            faces,
            torch.tensor(
                [
                    [0, 1, 9],
                    [1, 10, 9],
                    [0, 9, 12],
                    [0, 12, 3],
                    [0, 4, 1],
                    [0, 3, 4],
                    [1, 2, 10],
                    [2, 11, 10],
                    [1, 5, 2],
                    [1, 4, 5],
                    [2, 13, 11],
                    [2, 5, 13],
                    [3, 12, 14],
                    [3, 14, 6],
                    [3, 7, 4],
                    [3, 6, 7],
                    [14, 15, 7],
                    [14, 7, 6],
                    [4, 8, 5],
                    [4, 7, 8],
                    [15, 16, 8],
                    [15, 8, 7],
                    [5, 16, 13],
                    [5, 8, 16],
                    [9, 10, 17],
                    [10, 18, 17],
                    [17, 18, 20],
                    [18, 21, 20],
                    [9, 17, 20],
                    [9, 20, 12],
                    [10, 11, 18],
                    [11, 19, 18],
                    [18, 19, 21],
                    [19, 22, 21],
                    [11, 22, 19],
                    [11, 13, 22],
                    [20, 21, 23],
                    [21, 24, 23],
                    [12, 20, 23],
                    [12, 23, 14],
                    [23, 24, 15],
                    [23, 15, 14],
                    [21, 22, 24],
                    [22, 25, 24],
                    [24, 25, 16],
                    [24, 16, 15],
                    [13, 25, 22],
                    [13, 16, 25],
                ],
                dtype=torch.int64,
                device=device,
            ),
        )

        # 3rd-check
        verts, faces = meshes.get_mesh_verts_faces(2)
        self.assertTrue(verts.size(0) == 0)
        self.assertTrue(faces.size(0) == 0)

        # 4th-check
        verts, faces = meshes.get_mesh_verts_faces(3)
        self.assertClose(
            verts,
            torch.tensor(
                [
                    [1.0, -1.0, -1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, -1.0, 3.0],
                    [3.0, -1.0, -1.0],
                    [3.0, -1.0, 1.0],
                    [3.0, -1.0, 3.0],
                    [-1.0, 1.0, 1.0],
                    [-1.0, 1.0, 3.0],
                    [1.0, 1.0, -1.0],
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 3.0],
                    [3.0, 1.0, -1.0],
                    [3.0, 1.0, 1.0],
                    [3.0, 1.0, 3.0],
                    [-1.0, 3.0, 1.0],
                    [-1.0, 3.0, 3.0],
                    [1.0, 3.0, -1.0],
                    [1.0, 3.0, 1.0],
                    [1.0, 3.0, 3.0],
                    [3.0, 3.0, -1.0],
                    [3.0, 3.0, 1.0],
                    [3.0, 3.0, 3.0],
                ],
                dtype=torch.float32,
                device=device,
            ),
        )
        self.assertClose(
            faces,
            torch.tensor(
                [
                    [0, 1, 8],
                    [1, 9, 8],
                    [0, 8, 11],
                    [0, 11, 3],
                    [0, 4, 1],
                    [0, 3, 4],
                    [11, 12, 4],
                    [11, 4, 3],
                    [1, 2, 9],
                    [2, 10, 9],
                    [1, 5, 2],
                    [1, 4, 5],
                    [12, 13, 5],
                    [12, 5, 4],
                    [2, 13, 10],
                    [2, 5, 13],
                    [6, 7, 14],
                    [7, 15, 14],
                    [14, 15, 17],
                    [15, 18, 17],
                    [6, 14, 17],
                    [6, 17, 9],
                    [6, 10, 7],
                    [6, 9, 10],
                    [7, 18, 15],
                    [7, 10, 18],
                    [8, 9, 16],
                    [9, 17, 16],
                    [16, 17, 19],
                    [17, 20, 19],
                    [8, 16, 19],
                    [8, 19, 11],
                    [19, 20, 12],
                    [19, 12, 11],
                    [17, 18, 20],
                    [18, 21, 20],
                    [20, 21, 13],
                    [20, 13, 12],
                    [10, 21, 18],
                    [10, 13, 21],
                ],
                dtype=torch.int64,
                device=device,
            ),
        )

    def test_align(self):
        N, V = 1, 2
        device = torch.device("cuda:0")
        voxels = torch.ones((N, V, V, V), dtype=torch.float32, device=device)

        # topleft align
        mesh = cubify(voxels, 0.5)
        verts, faces = mesh.get_mesh_verts_faces(0)
        self.assertClose(verts.min(), torch.tensor(-1.0, device=device))
        self.assertClose(verts.max(), torch.tensor(3.0, device=device))

        # corner align
        mesh = cubify(voxels, 0.5, align="corner")
        verts, faces = mesh.get_mesh_verts_faces(0)
        self.assertClose(verts.min(), torch.tensor(-1.0, device=device))
        self.assertClose(verts.max(), torch.tensor(1.0, device=device))

        # center align
        mesh = cubify(voxels, 0.5, align="center")
        verts, faces = mesh.get_mesh_verts_faces(0)
        self.assertClose(verts.min(), torch.tensor(-2.0, device=device))
        self.assertClose(verts.max(), torch.tensor(2.0, device=device))

        # invalid align
        with self.assertRaisesRegex(ValueError, "Align mode must be one of"):
            cubify(voxels, 0.5, align="")

        # invalid align
        with self.assertRaisesRegex(ValueError, "Align mode must be one of"):
            cubify(voxels, 0.5, align="topright")

        # inside occupancy, similar to GH#185 use case
        N, V = 1, 4
        voxels = torch.zeros((N, V, V, V), dtype=torch.float32, device=device)
        voxels[0, : V // 2, : V // 2, : V // 2] = 1.0
        mesh = cubify(voxels, 0.5, align="corner")
        verts, faces = mesh.get_mesh_verts_faces(0)
        self.assertClose(verts.min(), torch.tensor(-1.0, device=device))
        self.assertClose(verts.max(), torch.tensor(0.0, device=device))

    @staticmethod
    def cubify_with_init(batch_size: int, V: int):
        device = torch.device("cuda:0")
        voxels = torch.rand((batch_size, V, V, V), dtype=torch.float32, device=device)
        torch.cuda.synchronize()

        def convert():
            cubify(voxels, 0.5)
            torch.cuda.synchronize()

        return convert
