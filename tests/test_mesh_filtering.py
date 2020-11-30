# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import torch
from common_testing import TestCaseMixin, get_random_cuda_device
from pytorch3d.ops import taubin_smoothing
from pytorch3d.ops.mesh_filtering import norm_laplacian
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere


class TestTaubinSmoothing(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_taubin(self):
        N = 3
        device = get_random_cuda_device()

        mesh = ico_sphere(4, device).extend(N)
        ico_verts = mesh.verts_padded()
        ico_faces = mesh.faces_padded()

        rand_noise = torch.rand_like(ico_verts) * 0.2 - 0.1
        z_mask = (ico_verts[:, :, -1] > 0).view(N, -1, 1)
        rand_noise = rand_noise * z_mask
        verts = ico_verts + rand_noise
        mesh = Meshes(verts=verts, faces=ico_faces)

        smooth_mesh = taubin_smoothing(mesh, num_iter=50)
        smooth_verts = smooth_mesh.verts_padded()

        smooth_dist = (smooth_verts - ico_verts).norm(dim=-1).mean()
        dist = (verts - ico_verts).norm(dim=-1).mean()
        self.assertTrue(smooth_dist < dist)

    def test_norm_laplacian(self):
        V = 32
        F = 64
        device = get_random_cuda_device()
        # random vertices
        verts = torch.rand((V, 3), dtype=torch.float32, device=device)
        # random valid faces (no self circles, e.g. (v0, v0, v1))
        faces = torch.stack([torch.randperm(V) for f in range(F)], dim=0)[:, :3]
        faces = faces.to(device=device)
        mesh = Meshes(verts=[verts], faces=[faces])
        edges = mesh.edges_packed()

        eps = 1e-12

        L = norm_laplacian(verts, edges, eps=eps)

        Lnaive = torch.zeros((V, V), dtype=torch.float32, device=device)
        for f in range(F):
            f0, f1, f2 = faces[f]
            v0 = verts[f0]
            v1 = verts[f1]
            v2 = verts[f2]

            w12 = 1.0 / ((v1 - v2).norm() + eps)
            w02 = 1.0 / ((v0 - v2).norm() + eps)
            w01 = 1.0 / ((v0 - v1).norm() + eps)

            Lnaive[f0, f1] = w01
            Lnaive[f1, f0] = w01
            Lnaive[f0, f2] = w02
            Lnaive[f2, f0] = w02
            Lnaive[f1, f2] = w12
            Lnaive[f2, f1] = w12

        self.assertClose(L.to_dense(), Lnaive)
