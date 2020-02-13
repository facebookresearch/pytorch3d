#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest
import torch

from pytorch3d import _C
from pytorch3d.structures.meshes import Meshes

from common_testing import TestCaseMixin


class TestFaceAreasNormals(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    @staticmethod
    def init_meshes(
        num_meshes: int = 10,
        num_verts: int = 1000,
        num_faces: int = 3000,
        device: str = "cpu",
    ):
        device = torch.device(device)
        verts_list = []
        faces_list = []
        for _ in range(num_meshes):
            verts = torch.rand(
                (num_verts, 3), dtype=torch.float32, device=device
            )
            faces = torch.randint(
                num_verts, size=(num_faces, 3), dtype=torch.int64, device=device
            )
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts_list, faces_list)

        return meshes

    @staticmethod
    def face_areas_normals(verts, faces):
        """
        Pytorch implementation for face areas & normals.
        """
        vertices_faces = verts[faces]  # (F, 3, 3)
        # vector pointing from v0 to v1
        v01 = vertices_faces[:, 1] - vertices_faces[:, 0]
        # vector pointing from v0 to v2
        v02 = vertices_faces[:, 2] - vertices_faces[:, 0]
        normals = torch.cross(v01, v02, dim=1)  # (F, 3)
        face_areas = normals.norm(dim=-1) / 2
        face_normals = torch.nn.functional.normalize(
            normals, p=2, dim=1, eps=1e-6
        )
        return face_areas, face_normals

    def _test_face_areas_normals_helper(self, device):
        """
        Check the results from face_areas cuda/cpp and PyTorch implementation are
        the same.
        """
        meshes = self.init_meshes(10, 1000, 3000, device=device)
        verts = meshes.verts_packed()
        faces = meshes.faces_packed()

        areas_torch, normals_torch = self.face_areas_normals(verts, faces)
        areas, normals = _C.face_areas_normals(verts, faces)
        self.assertClose(areas_torch, areas, atol=1e-7)
        # normals get normalized by area thus sensitivity increases as areas
        # in our tests can be arbitrarily small. Thus we compare normals after
        # multiplying with areas
        unnormals = normals * areas.view(-1, 1)
        unnormals_torch = normals_torch * areas_torch.view(-1, 1)
        self.assertClose(unnormals_torch, unnormals, atol=1e-7)

    def test_face_areas_normals_cpu(self):
        self._test_face_areas_normals_helper("cpu")

    def test_face_areas_normals_cuda(self):
        self._test_face_areas_normals_helper("cuda:0")

    @staticmethod
    def face_areas_normals_with_init(
        num_meshes: int, num_verts: int, num_faces: int, cuda: bool = True
    ):
        device = "cuda:0" if cuda else "cpu"
        meshes = TestFaceAreasNormals.init_meshes(
            num_meshes, num_verts, num_faces, device
        )
        verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        torch.cuda.synchronize()

        def face_areas_normals():
            _C.face_areas_normals(verts, faces)
            torch.cuda.synchronize()

        return face_areas_normals

    @staticmethod
    def face_areas_normals_with_init_torch(
        num_meshes: int, num_verts: int, num_faces: int, cuda: bool = True
    ):
        device = "cuda:0" if cuda else "cpu"
        meshes = TestFaceAreasNormals.init_meshes(
            num_meshes, num_verts, num_faces, device
        )
        verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        torch.cuda.synchronize()

        def face_areas_normals():
            TestFaceAreasNormals.face_areas_normals(verts, faces)
            torch.cuda.synchronize()

        return face_areas_normals
