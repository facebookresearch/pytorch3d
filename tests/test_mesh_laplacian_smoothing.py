# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from pytorch3d.loss.mesh_laplacian_smoothing import mesh_laplacian_smoothing
from pytorch3d.structures.meshes import Meshes


class TestLaplacianSmoothing(unittest.TestCase):
    @staticmethod
    def laplacian_smoothing_naive_uniform(meshes):
        """
        Naive implementation of laplacian smoothing with uniform weights.
        """
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
        V = verts_packed.shape[0]

        L = torch.zeros((V, V), dtype=torch.float32, device=meshes.device)

        # filling L with the face pairs should be the same as edge pairs
        for f in faces_packed:
            L[f[0], f[1]] = 1
            L[f[0], f[2]] = 1
            L[f[1], f[2]] = 1
            # symetric
            L[f[1], f[0]] = 1
            L[f[2], f[0]] = 1
            L[f[2], f[1]] = 1

        norm_w = L.sum(dim=1, keepdims=True)
        idx = norm_w > 0
        norm_w[idx] = 1.0 / norm_w[idx]

        loss = (L.mm(verts_packed) * norm_w - verts_packed).norm(dim=1)

        weights = torch.zeros(V, dtype=torch.float32, device=meshes.device)
        for v in range(V):
            weights[v] = meshes.num_verts_per_mesh()[
                meshes.verts_packed_to_mesh_idx()[v]
            ]
        weights = 1.0 / weights
        loss = loss * weights

        return loss.sum() / len(meshes)

    @staticmethod
    def laplacian_smoothing_naive_cot(meshes, method: str = "cot"):
        """
        Naive implementation of laplacian smoothing wit cotangent weights.
        """
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
        V = verts_packed.shape[0]

        L = torch.zeros((V, V), dtype=torch.float32, device=meshes.device)
        inv_areas = torch.zeros((V, 1), dtype=torch.float32, device=meshes.device)

        for f in faces_packed:
            v0 = verts_packed[f[0], :]
            v1 = verts_packed[f[1], :]
            v2 = verts_packed[f[2], :]
            A = (v1 - v2).norm()
            B = (v0 - v2).norm()
            C = (v0 - v1).norm()
            s = 0.5 * (A + B + C)

            face_area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()
            inv_areas[f[0]] += face_area
            inv_areas[f[1]] += face_area
            inv_areas[f[2]] += face_area

            A2, B2, C2 = A * A, B * B, C * C
            cota = (B2 + C2 - A2) / face_area / 4.0
            cotb = (A2 + C2 - B2) / face_area / 4.0
            cotc = (A2 + B2 - C2) / face_area / 4.0

            L[f[1], f[2]] += cota
            L[f[2], f[0]] += cotb
            L[f[0], f[1]] += cotc
            # symetric
            L[f[2], f[1]] += cota
            L[f[0], f[2]] += cotb
            L[f[1], f[0]] += cotc

        idx = inv_areas > 0
        inv_areas[idx] = 1.0 / inv_areas[idx]

        norm_w = L.sum(dim=1, keepdims=True)
        L_sum = norm_w.clone()
        idx = norm_w > 0
        norm_w[idx] = 1.0 / norm_w[idx]

        if method == "cotcurv":
            loss = (L.mm(verts_packed) - L_sum * verts_packed) * inv_areas * 0.25
            loss = loss.norm(dim=1)
        else:
            loss = L.mm(verts_packed) * norm_w - verts_packed
            loss = loss.norm(dim=1)

        weights = torch.zeros(V, dtype=torch.float32, device=meshes.device)
        for v in range(V):
            weights[v] = meshes.num_verts_per_mesh()[
                meshes.verts_packed_to_mesh_idx()[v]
            ]
        weights = 1.0 / weights
        loss = loss * weights

        return loss.sum() / len(meshes)

    @staticmethod
    def init_meshes(num_meshes: int = 10, num_verts: int = 1000, num_faces: int = 3000):
        device = torch.device("cuda:0")
        verts_list = []
        faces_list = []
        for _ in range(num_meshes):
            verts = (
                torch.rand((num_verts, 3), dtype=torch.float32, device=device) * 2.0
                - 1.0
            )  # verts in the space of [-1, 1]
            faces = torch.stack(
                [
                    torch.randperm(num_verts, device=device)[:3]
                    for _ in range(num_faces)
                ],
                dim=0,
            )
            # avoids duplicate vertices in a face
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts_list, faces_list)

        return meshes

    def test_laplacian_smoothing_uniform(self):
        """
        Test Laplacian Smoothing with uniform weights.
        """
        meshes = TestLaplacianSmoothing.init_meshes(10, 100, 300)

        # feats in list
        out = mesh_laplacian_smoothing(meshes, method="uniform")
        naive_out = TestLaplacianSmoothing.laplacian_smoothing_naive_uniform(meshes)

        self.assertTrue(torch.allclose(out, naive_out))

    def test_laplacian_smoothing_cot(self):
        """
        Test Laplacian Smoothing with cot weights.
        """
        meshes = TestLaplacianSmoothing.init_meshes(10, 100, 300)

        # feats in list
        out = mesh_laplacian_smoothing(meshes, method="cot")
        naive_out = TestLaplacianSmoothing.laplacian_smoothing_naive_cot(
            meshes, method="cot"
        )

        self.assertTrue(torch.allclose(out, naive_out))

    def test_laplacian_smoothing_cotcurv(self):
        """
        Test Laplacian Smoothing with cotcurv weights.
        """
        meshes = TestLaplacianSmoothing.init_meshes(10, 100, 300)

        # feats in list
        out = mesh_laplacian_smoothing(meshes, method="cotcurv")
        naive_out = TestLaplacianSmoothing.laplacian_smoothing_naive_cot(
            meshes, method="cotcurv"
        )

        self.assertTrue(torch.allclose(out, naive_out))

    @staticmethod
    def laplacian_smoothing_with_init(
        num_meshes: int, num_verts: int, num_faces: int, device: str = "cpu"
    ):
        device = torch.device(device)
        verts_list = []
        faces_list = []
        for _ in range(num_meshes):
            verts = torch.rand((num_verts, 3), dtype=torch.float32, device=device)
            faces = torch.randint(
                num_verts, size=(num_faces, 3), dtype=torch.int64, device=device
            )
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts_list, faces_list)
        torch.cuda.synchronize()

        def smooth():
            mesh_laplacian_smoothing(meshes, method="cotcurv")
            torch.cuda.synchronize()

        return smooth
