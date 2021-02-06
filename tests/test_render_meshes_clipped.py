# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


"""
Checks for mesh rasterization in the case where the camera enters the
inside of the mesh and some mesh faces are partially
behind the image plane. These faces are clipped and then rasterized.
See pytorch3d/renderer/mesh/clip.py for more details about the
clipping process.
"""
import unittest

import torch
from common_testing import TestCaseMixin
from pytorch3d.renderer.mesh import ClipFrustum, clip_faces
from pytorch3d.structures.meshes import Meshes


class TestRenderMeshesClipping(TestCaseMixin, unittest.TestCase):
    @staticmethod
    def clip_faces(meshes):
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        face_verts = verts_packed[faces_packed]
        mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
        num_faces_per_mesh = meshes.num_faces_per_mesh()

        frustum = ClipFrustum(
            left=-1,
            right=1,
            top=-1,
            bottom=1,
            # In the unit tests for each case below the triangles are asummed
            #  to have already been projected onto the image plane.
            perspective_correct=False,
            z_clip_value=1e-2,
            cull=True,  # Cull to frustrum
        )

        clipped_faces = clip_faces(
            face_verts, mesh_to_face_first_idx, num_faces_per_mesh, frustum
        )
        return clipped_faces

    def test_case_1(self):
        """
        Case 1: Single triangle fully in front of the image plane (z=0)
        Triangle is not clipped or culled. The triangle is asummed to have
        already been projected onto the image plane so no perspective
        correction is needed.
        """
        device = "cuda:0"
        verts = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        clipped_faces = self.clip_faces(meshes)

        self.assertClose(clipped_faces.face_verts, verts[faces])
        self.assertEqual(clipped_faces.mesh_to_face_first_idx.item(), 0)
        self.assertEqual(clipped_faces.num_faces_per_mesh.item(), 1)
        self.assertIsNone(clipped_faces.faces_clipped_to_unclipped_idx)
        self.assertIsNone(clipped_faces.faces_clipped_to_conversion_idx)
        self.assertIsNone(clipped_faces.clipped_faces_neighbor_idx)
        self.assertIsNone(clipped_faces.barycentric_conversion)

    def test_case_2(self):
        """
        Case 2 triangles are fully behind the image plane (z=0) so are completely culled.
        Test with a single triangle behind the image plane.
        """

        device = "cuda:0"
        verts = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        clipped_faces = self.clip_faces(meshes)

        zero_t = torch.zeros(size=(1,), dtype=torch.int64, device=device)
        self.assertClose(
            clipped_faces.face_verts, torch.empty(device=device, size=(0, 3, 3))
        )
        self.assertClose(clipped_faces.mesh_to_face_first_idx, zero_t)
        self.assertClose(clipped_faces.num_faces_per_mesh, zero_t)
        self.assertClose(
            clipped_faces.faces_clipped_to_unclipped_idx,
            torch.empty(device=device, dtype=torch.int64, size=(0,)),
        )
        self.assertIsNone(clipped_faces.faces_clipped_to_conversion_idx)
        self.assertIsNone(clipped_faces.clipped_faces_neighbor_idx)
        self.assertIsNone(clipped_faces.barycentric_conversion)

    def test_case_3(self):
        """
        Case 3 triangles have exactly two vertices behind the clipping plane (z=0) so are
        clipped into a smaller triangle.

        Test with a single triangle parallel to the z axis which intersects with
        the image plane.
        """

        device = "cuda:0"
        verts = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        clipped_faces = self.clip_faces(meshes)

        zero_t = torch.zeros(size=(1,), dtype=torch.int64, device=device)
        clipped_face_verts = torch.tensor(
            [
                [
                    [0.4950, 0.0000, 0.0100],
                    [-0.4950, 0.0000, 0.0100],
                    [0.0000, 0.0000, 1.0000],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )

        # barycentric_conversion[i, :, k] stores the barycentric weights
        # in terms of the world coordinates of the original
        # (big) triangle for the kth vertex in the clipped (small) triangle.
        barycentric_conversion = torch.tensor(
            [
                [
                    [0.0000, 0.4950, 0.0000],
                    [0.5050, 0.5050, 1.0000],
                    [0.4950, 0.0000, 0.0000],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )

        self.assertClose(clipped_faces.face_verts, clipped_face_verts)
        self.assertEqual(clipped_faces.mesh_to_face_first_idx.item(), 0)
        self.assertEqual(clipped_faces.num_faces_per_mesh.item(), 1)
        self.assertClose(clipped_faces.faces_clipped_to_unclipped_idx, zero_t)
        self.assertClose(clipped_faces.faces_clipped_to_conversion_idx, zero_t)
        self.assertClose(
            clipped_faces.clipped_faces_neighbor_idx,
            zero_t - 1,  # default is -1
        )
        self.assertClose(clipped_faces.barycentric_conversion, barycentric_conversion)

    def test_case_4(self):
        """
        Case 4 triangles have exactly 1 vertex behind the clipping plane (z=0) so
        are clipped into a smaller quadrilateral and then divided into two triangles.

        Test with a single triangle parallel to the z axis which intersects with
        the image plane.
        """

        device = "cuda:0"
        verts = torch.tensor(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        clipped_faces = self.clip_faces(meshes)

        clipped_face_verts = torch.tensor(
            [
                # t1
                [
                    [-0.5050, 0.0000, 0.0100],
                    [-1.0000, 0.0000, 1.0000],
                    [0.5050, 0.0000, 0.0100],
                ],
                # t2
                [
                    [0.5050, 0.0000, 0.0100],
                    [-1.0000, 0.0000, 1.0000],
                    [1.0000, 0.0000, 1.0000],
                ],
            ],
            device=device,
            dtype=torch.float32,
        )

        barycentric_conversion = torch.tensor(
            [
                [
                    [0.4950, 0.0000, 0.4950],
                    [0.5050, 1.0000, 0.0000],
                    [0.0000, 0.0000, 0.5050],
                ],
                [
                    [0.4950, 0.0000, 0.0000],
                    [0.0000, 1.0000, 0.0000],
                    [0.5050, 0.0000, 1.0000],
                ],
            ],
            device=device,
            dtype=torch.float32,
        )

        self.assertClose(clipped_faces.face_verts, clipped_face_verts)
        self.assertEqual(clipped_faces.mesh_to_face_first_idx.item(), 0)
        self.assertEqual(
            clipped_faces.num_faces_per_mesh.item(), 2
        )  # now two faces instead of 1
        self.assertClose(
            clipped_faces.faces_clipped_to_unclipped_idx,
            torch.tensor([0, 0], device=device, dtype=torch.int64),
        )
        # Neighboring face for each of the sub triangles e.g. for t1, neighbor is t2,
        # and for t2, neighbor is t1
        self.assertClose(
            clipped_faces.clipped_faces_neighbor_idx,
            torch.tensor([1, 0], device=device, dtype=torch.int64),
        )
        # barycentric_conversion is of shape (F_clipped)
        self.assertEqual(clipped_faces.barycentric_conversion.shape[0], 2)
        self.assertClose(clipped_faces.barycentric_conversion, barycentric_conversion)
        # Index into barycentric_conversion for each clipped face.
        self.assertClose(
            clipped_faces.faces_clipped_to_conversion_idx,
            torch.tensor([0, 1], device=device, dtype=torch.int64),
        )

    def test_mixture_of_cases(self):
        """
        Test with two meshes composed of different cases to check all the
        indexing is correct.
        Case 4 faces are subdivided into two faces which are referred
        to as t1 and t2.
        """
        device = "cuda:0"
        # fmt: off
        verts = [
            torch.tensor(
                [
                    [-1.0,  0.0, -1.0],  # noqa: E241, E201
                    [ 0.0,  1.0, -1.0],  # noqa: E241, E201
                    [ 1.0,  0.0, -1.0],  # noqa: E241, E201
                    [ 0.0, -1.0, -1.0],  # noqa: E241, E201
                    [-1.0,  0.5,  0.5],  # noqa: E241, E201
                    [ 1.0,  1.0,  1.0],  # noqa: E241, E201
                    [ 0.0, -1.0,  1.0],  # noqa: E241, E201
                    [-1.0,  0.5, -0.5],  # noqa: E241, E201
                    [ 1.0,  1.0, -1.0],  # noqa: E241, E201
                    [-1.0,  0.0,  1.0],  # noqa: E241, E201
                    [ 0.0,  1.0,  1.0],  # noqa: E241, E201
                    [ 1.0,  0.0,  1.0],  # noqa: E241, E201
                ],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [
                    [ 0.0, -1.0, -1.0],  # noqa: E241, E201
                    [-1.0,  0.5,  0.5],  # noqa: E241, E201
                    [ 1.0,  1.0,  1.0],  # noqa: E241, E201
                ],
                dtype=torch.float32,
                device=device
            )
        ]
        faces = [
            torch.tensor(
                [
                    [0,  1,  2],  # noqa: E241, E201  Case 2 fully clipped
                    [3,  4,  5],  # noqa: E241, E201  Case 4 clipped and subdivided
                    [5,  4,  3],  # noqa: E241, E201  Repeat of Case 4
                    [6,  7,  8],  # noqa: E241, E201  Case 3 clipped
                    [9, 10, 11],  # noqa: E241, E201  Case 1 untouched
                ],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                [
                    [0,  1,  2],  # noqa: E241, E201  Case 4
                ],
                dtype=torch.int64,
                device=device,
            ),
        ]
        # fmt: on
        meshes = Meshes(verts=verts, faces=faces)

        # Clip meshes
        clipped_faces = self.clip_faces(meshes)

        # mesh 1: 4x faces (from Case 4) + 1 (from Case 3) + 1 (from Case 1)
        # mesh 2: 2x faces (from Case 4)
        self.assertEqual(clipped_faces.face_verts.shape[0], 6 + 2)

        # dummy idx type tensor to avoid having to initialize the dype/device each time
        idx = torch.empty(size=(1,), dtype=torch.int64, device=device)
        unclipped_idx = idx.new_tensor([1, 1, 2, 2, 3, 4, 5, 5])
        neighbors = idx.new_tensor([1, 0, 3, 2, -1, -1, 7, 6])
        first_idx = idx.new_tensor([0, 6])
        num_faces = idx.new_tensor([6, 2])

        self.assertClose(clipped_faces.clipped_faces_neighbor_idx, neighbors)
        self.assertClose(clipped_faces.faces_clipped_to_unclipped_idx, unclipped_idx)
        self.assertClose(clipped_faces.mesh_to_face_first_idx, first_idx)
        self.assertClose(clipped_faces.num_faces_per_mesh, num_faces)

        # faces_clipped_to_conversion_idx maps each output face to the
        # corresponding row of the barycentric_conversion matrix.
        # The barycentric_conversion matrix is composed by
        # finding the barycentric conversion weights for case 3 faces
        # case 4 (t1) faces and case 4 (t2) faces. These are then
        # concatenated. Therefore case 3 faces will be the first rows of
        # the barycentric_conversion matrix followed by t1 and then t2.
        # Case type of all faces: [4 (t1), 4 (t2), 4 (t1), 4 (t2), 3, 1, 4 (t1), 4 (t2)]
        # Based on this information we can calculate the indices into the
        # barycentric conversion matrix.
        bary_idx = idx.new_tensor([1, 4, 2, 5, 0, -1, 3, 6])
        self.assertClose(clipped_faces.faces_clipped_to_conversion_idx, bary_idx)
