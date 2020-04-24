# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import torch
from common_testing import TestCaseMixin, get_random_cuda_device
from pytorch3d.ops import mesh_face_areas_normals
from pytorch3d.structures.meshes import Meshes


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
                (num_verts, 3), dtype=torch.float32, device=device, requires_grad=True
            )
            faces = torch.randint(
                num_verts, size=(num_faces, 3), dtype=torch.int64, device=device
            )
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts_list, faces_list)

        return meshes

    @staticmethod
    def face_areas_normals_python(verts, faces):
        """
        Pytorch implementation for face areas & normals.
        """
        # TODO(gkioxari) Change cast to floats once we add support for doubles.
        verts = verts.float()
        vertices_faces = verts[faces]  # (F, 3, 3)
        # vector pointing from v0 to v1
        v01 = vertices_faces[:, 1] - vertices_faces[:, 0]
        # vector pointing from v0 to v2
        v02 = vertices_faces[:, 2] - vertices_faces[:, 0]
        normals = torch.cross(v01, v02, dim=1)  # (F, 3)
        face_areas = normals.norm(dim=-1) / 2
        face_normals = torch.nn.functional.normalize(normals, p=2, dim=1, eps=1e-6)
        return face_areas, face_normals

    def _test_face_areas_normals_helper(self, device, dtype=torch.float32):
        """
        Check the results from face_areas cuda/cpp and PyTorch implementation are
        the same.
        """
        meshes = self.init_meshes(10, 200, 400, device=device)
        # make them leaf nodes
        verts = meshes.verts_packed().detach().clone().to(dtype)
        verts.requires_grad = True
        faces = meshes.faces_packed().detach().clone()

        # forward
        areas, normals = mesh_face_areas_normals(verts, faces)
        verts_torch = verts.detach().clone().to(dtype)
        verts_torch.requires_grad = True
        faces_torch = faces.detach().clone()
        (areas_torch, normals_torch) = TestFaceAreasNormals.face_areas_normals_python(
            verts_torch, faces_torch
        )
        self.assertClose(areas_torch, areas, atol=1e-7)
        # normals get normalized by area thus sensitivity increases as areas
        # in our tests can be arbitrarily small. Thus we compare normals after
        # multiplying with areas
        unnormals = normals * areas.view(-1, 1)
        unnormals_torch = normals_torch * areas_torch.view(-1, 1)
        self.assertClose(unnormals_torch, unnormals, atol=1e-6)

        # backward
        grad_areas = torch.rand(areas.shape, device=device, dtype=dtype)
        grad_normals = torch.rand(normals.shape, device=device, dtype=dtype)
        areas.backward((grad_areas, grad_normals))
        grad_verts = verts.grad
        areas_torch.backward((grad_areas, grad_normals))
        grad_verts_torch = verts_torch.grad
        self.assertClose(grad_verts_torch, grad_verts, atol=1e-6)

    def test_face_areas_normals_cpu(self):
        self._test_face_areas_normals_helper("cpu")

    def test_face_areas_normals_cuda(self):
        device = get_random_cuda_device()
        self._test_face_areas_normals_helper(device)

    def test_nonfloats_cpu(self):
        self._test_face_areas_normals_helper("cpu", dtype=torch.double)

    def test_nonfloats_cuda(self):
        device = get_random_cuda_device()
        self._test_face_areas_normals_helper(device, dtype=torch.double)

    @staticmethod
    def face_areas_normals_with_init(
        num_meshes: int, num_verts: int, num_faces: int, device: str = "cpu"
    ):
        meshes = TestFaceAreasNormals.init_meshes(
            num_meshes, num_verts, num_faces, device
        )
        verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        torch.cuda.synchronize()

        def face_areas_normals():
            mesh_face_areas_normals(verts, faces)
            torch.cuda.synchronize()

        return face_areas_normals

    @staticmethod
    def face_areas_normals_with_init_torch(
        num_meshes: int, num_verts: int, num_faces: int, device: str = "cpu"
    ):
        meshes = TestFaceAreasNormals.init_meshes(
            num_meshes, num_verts, num_faces, device
        )
        verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        torch.cuda.synchronize()

        def face_areas_normals():
            TestFaceAreasNormals.face_areas_normals_python(verts, faces)
            torch.cuda.synchronize()

        return face_areas_normals
