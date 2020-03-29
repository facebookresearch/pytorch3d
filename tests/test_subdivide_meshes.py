# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import torch
from common_testing import TestCaseMixin
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils.ico_sphere import ico_sphere


class TestSubdivideMeshes(TestCaseMixin, unittest.TestCase):
    def test_simple_subdivide(self):
        # Create a mesh with one face and check the subdivided mesh has
        # 4 faces with the correct vertex coordinates.
        device = torch.device("cuda:0")
        verts = torch.tensor(
            [[0.5, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        mesh = Meshes(verts=[verts], faces=[faces])
        subdivide = SubdivideMeshes()
        new_mesh = subdivide(mesh)

        # Subdivided face:
        #
        #           v0
        #           /\
        #          /  \
        #         / f0 \
        #     v4 /______\ v3
        #       /\      /\
        #      /  \ f3 /  \
        #     / f2 \  / f1 \
        #    /______\/______\
        #  v2       v5       v1
        #
        gt_subdivide_verts = torch.tensor(
            [
                [0.5, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.75, 0.5, 0.0],
                [0.25, 0.5, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        gt_subdivide_faces = torch.tensor(
            [[0, 3, 4], [1, 5, 3], [2, 4, 5], [5, 4, 3]],
            dtype=torch.int64,
            device=device,
        )
        new_verts, new_faces = new_mesh.get_mesh_verts_faces(0)
        self.assertClose(new_verts, gt_subdivide_verts)
        self.assertClose(new_faces, gt_subdivide_faces)
        self.assertTrue(new_verts.requires_grad == verts.requires_grad)

    def test_heterogeneous_meshes(self):
        device = torch.device("cuda:0")
        verts1 = torch.tensor(
            [[0.5, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        faces1 = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)
        verts2 = torch.tensor(
            [[0.5, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.5, 1.0, 0.0]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        faces2 = torch.tensor([[0, 1, 2], [0, 3, 1]], dtype=torch.int64, device=device)
        faces3 = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=device)
        mesh = Meshes(verts=[verts1, verts2, verts2], faces=[faces1, faces2, faces3])
        subdivide = SubdivideMeshes()
        new_mesh = subdivide(mesh.clone())

        gt_subdivided_verts1 = torch.tensor(
            [
                [0.5, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.75, 0.5, 0.0],
                [0.25, 0.5, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        gt_subdivided_faces1 = torch.tensor(
            [[0, 3, 4], [1, 5, 3], [2, 4, 5], [5, 4, 3]],
            dtype=torch.int64,
            device=device,
        )
        # faces2:
        #
        #         v0 _______e2_______ v3
        #           /\              /
        #          /  \            /
        #         /    \          /
        #     e1 /      \ e0     / e4
        #       /        \      /
        #      /          \    /
        #     /            \  /
        #    /______________\/
        #  v2       e3      v1
        #
        # Subdivided faces2:
        #
        #         v0 _______v6_______ v3
        #           /\      /\      /
        #          /  \ f1 /  \ f3 /
        #         / f0 \  / f7 \  /
        #     v5 /______v4______\/v8
        #       /\      /\      /
        #      /  \ f6 /  \ f5 /
        #     / f4 \  / f2 \  /
        #    /______\/______\/
        #  v2       v7       v1
        #
        gt_subdivided_verts2 = torch.tensor(
            [
                [0.5, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.5, 1.0, 0.0],
                [0.75, 0.5, 0.0],
                [0.25, 0.5, 0.0],
                [1.0, 1.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.25, 0.5, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        gt_subdivided_faces2 = torch.tensor(
            [
                [0, 4, 5],
                [0, 6, 4],
                [1, 7, 4],
                [3, 8, 6],
                [2, 5, 7],
                [1, 4, 8],
                [7, 5, 4],
                [8, 4, 6],
            ],
            dtype=torch.int64,
            device=device,
        )
        gt_subdivided_verts3 = gt_subdivided_verts2.clone()
        gt_subdivided_verts3[-1, :] = torch.tensor(
            [0.75, 0.5, 0], dtype=torch.float32, device=device
        )
        gt_subdivided_faces3 = torch.tensor(
            [
                [0, 4, 5],
                [0, 5, 6],
                [1, 7, 4],
                [2, 8, 5],
                [2, 5, 7],
                [3, 6, 8],
                [7, 5, 4],
                [8, 6, 5],
            ],
            dtype=torch.int64,
            device=device,
        )
        new_mesh_verts1, new_mesh_faces1 = new_mesh.get_mesh_verts_faces(0)
        new_mesh_verts2, new_mesh_faces2 = new_mesh.get_mesh_verts_faces(1)
        new_mesh_verts3, new_mesh_faces3 = new_mesh.get_mesh_verts_faces(2)
        self.assertClose(new_mesh_verts1, gt_subdivided_verts1)
        self.assertClose(new_mesh_faces1, gt_subdivided_faces1)
        self.assertClose(new_mesh_verts2, gt_subdivided_verts2)
        self.assertClose(new_mesh_faces2, gt_subdivided_faces2)
        self.assertClose(new_mesh_verts3, gt_subdivided_verts3)
        self.assertClose(new_mesh_faces3, gt_subdivided_faces3)
        self.assertTrue(new_mesh_verts1.requires_grad == verts1.requires_grad)
        self.assertTrue(new_mesh_verts2.requires_grad == verts2.requires_grad)
        self.assertTrue(new_mesh_verts3.requires_grad == verts2.requires_grad)

    def test_subdivide_features(self):
        device = torch.device("cuda:0")
        mesh = ico_sphere(0, device)
        N = 10
        mesh = mesh.extend(N)
        edges = mesh.edges_packed()
        V = mesh.num_verts_per_mesh()[0]
        D = 256
        feats = torch.rand(
            (N * V, D), dtype=torch.float32, device=device, requires_grad=True
        )  # packed features
        app_feats = feats[edges].mean(1)
        subdivide = SubdivideMeshes()
        new_mesh, new_feats = subdivide(mesh, feats)
        gt_feats = torch.cat(
            (feats.view(N, V, D), app_feats.view(N, -1, D)), dim=1
        ).view(-1, D)
        self.assertClose(new_feats, gt_feats)
        self.assertTrue(new_feats.requires_grad == gt_feats.requires_grad)

    @staticmethod
    def subdivide_meshes_with_init(num_meshes: int = 10, same_topo: bool = False):
        device = torch.device("cuda:0")
        meshes = ico_sphere(0, device=device)
        if num_meshes > 1:
            meshes = meshes.extend(num_meshes)
        meshes_init = meshes.clone() if same_topo else None
        torch.cuda.synchronize()

        def subdivide_meshes():
            subdivide = SubdivideMeshes(meshes=meshes_init)
            subdivide(meshes=meshes.clone())
            torch.cuda.synchronize()

        return subdivide_meshes
