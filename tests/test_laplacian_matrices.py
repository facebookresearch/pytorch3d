# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.ops import cot_laplacian, laplacian, norm_laplacian
from pytorch3d.structures.meshes import Meshes

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestLaplacianMatrices(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def init_mesh(self) -> Meshes:
        V, F = 32, 64
        device = get_random_cuda_device()
        # random vertices
        verts = torch.rand((V, 3), dtype=torch.float32, device=device)
        # random valid faces (no self circles, e.g. (v0, v0, v1))
        faces = torch.stack([torch.randperm(V) for f in range(F)], dim=0)[:, :3]
        faces = faces.to(device=device)
        return Meshes(verts=[verts], faces=[faces])

    def test_laplacian(self):
        mesh = self.init_mesh()
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        V, E = verts.shape[0], edges.shape[0]

        L = laplacian(verts, edges)

        Lnaive = torch.zeros((V, V), dtype=torch.float32, device=verts.device)
        for e in range(E):
            e0, e1 = edges[e]
            Lnaive[e0, e1] = 1
            # symetric
            Lnaive[e1, e0] = 1

        deg = Lnaive.sum(1).view(-1, 1)
        deg[deg > 0] = 1.0 / deg[deg > 0]
        Lnaive = Lnaive * deg
        diag = torch.eye(V, dtype=torch.float32, device=mesh.device)
        Lnaive.masked_fill_(diag > 0, -1)

        self.assertClose(L.to_dense(), Lnaive)

    def test_cot_laplacian(self):
        mesh = self.init_mesh()
        verts = mesh.verts_packed()
        faces = mesh.faces_packed()
        V = verts.shape[0]

        eps = 1e-12

        L, inv_areas = cot_laplacian(verts, faces, eps=eps)

        Lnaive = torch.zeros((V, V), dtype=torch.float32, device=verts.device)
        inv_areas_naive = torch.zeros((V, 1), dtype=torch.float32, device=verts.device)

        for f in faces:
            v0 = verts[f[0], :]
            v1 = verts[f[1], :]
            v2 = verts[f[2], :]
            A = (v1 - v2).norm()
            B = (v0 - v2).norm()
            C = (v0 - v1).norm()
            s = 0.5 * (A + B + C)

            face_area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()
            inv_areas_naive[f[0]] += face_area
            inv_areas_naive[f[1]] += face_area
            inv_areas_naive[f[2]] += face_area

            A2, B2, C2 = A * A, B * B, C * C
            cota = (B2 + C2 - A2) / face_area / 4.0
            cotb = (A2 + C2 - B2) / face_area / 4.0
            cotc = (A2 + B2 - C2) / face_area / 4.0

            Lnaive[f[1], f[2]] += cota
            Lnaive[f[2], f[0]] += cotb
            Lnaive[f[0], f[1]] += cotc
            # symetric
            Lnaive[f[2], f[1]] += cota
            Lnaive[f[0], f[2]] += cotb
            Lnaive[f[1], f[0]] += cotc

        idx = inv_areas_naive > 0
        inv_areas_naive[idx] = 1.0 / inv_areas_naive[idx]

        self.assertClose(inv_areas, inv_areas_naive)
        self.assertClose(L.to_dense(), Lnaive)

    def test_norm_laplacian(self):
        mesh = self.init_mesh()
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
        V, E = verts.shape[0], edges.shape[0]

        eps = 1e-12

        L = norm_laplacian(verts, edges, eps=eps)

        Lnaive = torch.zeros((V, V), dtype=torch.float32, device=verts.device)
        for e in range(E):
            e0, e1 = edges[e]
            v0 = verts[e0]
            v1 = verts[e1]

            w01 = 1.0 / ((v0 - v1).norm() + eps)
            Lnaive[e0, e1] += w01
            Lnaive[e1, e0] += w01

        self.assertClose(L.to_dense(), Lnaive)
