# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import torch
from pytorch3d.loss.mesh_normal_consistency import mesh_normal_consistency
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils.ico_sphere import ico_sphere


class TestMeshNormalConsistency(unittest.TestCase):
    @staticmethod
    def init_faces(num_verts: int = 1000):
        faces = []
        for f0 in range(num_verts):
            for f1 in range(f0 + 1, num_verts):
                f2 = torch.arange(f1 + 1, num_verts)
                n = f2.shape[0]
                if n == 0:
                    continue
                faces.append(
                    torch.stack(
                        [
                            torch.full((n,), f0, dtype=torch.int64),
                            torch.full((n,), f1, dtype=torch.int64),
                            f2,
                        ],
                        dim=1,
                    )
                )
        faces = torch.cat(faces, 0)
        return faces

    @staticmethod
    def init_meshes(num_meshes: int = 10, num_verts: int = 1000, num_faces: int = 3000):
        device = torch.device("cuda:0")
        valid_faces = TestMeshNormalConsistency.init_faces(num_verts).to(device)
        verts_list = []
        faces_list = []
        for _ in range(num_meshes):
            verts = (
                torch.rand((num_verts, 3), dtype=torch.float32, device=device) * 2.0
                - 1.0
            )  # verts in the space of [-1, 1]
            """
            faces = torch.stack(
                [
                    torch.randperm(num_verts, device=device)[:3]
                    for _ in range(num_faces)
                ],
                dim=0,
            )
            # avoids duplicate vertices in a face
            """
            idx = torch.randperm(valid_faces.shape[0], device=device)[
                : min(valid_faces.shape[0], num_faces)
            ]
            faces = valid_faces[idx]
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts_list, faces_list)
        return meshes

    @staticmethod
    def mesh_normal_consistency_naive(meshes):
        """
        Naive iterative implementation of mesh normal consistency.
        """
        N = len(meshes)
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        edges_packed = meshes.edges_packed()
        face_to_edge = meshes.faces_packed_to_edges_packed()
        edges_packed_to_mesh_idx = meshes.edges_packed_to_mesh_idx()

        E = edges_packed.shape[0]
        loss = []
        mesh_idx = []

        for e in range(E):
            face_idx = face_to_edge.eq(e).any(1).nonzero()  # indexed to faces
            v0 = verts_packed[edges_packed[e, 0]]
            v1 = verts_packed[edges_packed[e, 1]]
            normals = []
            for f in face_idx:
                v2 = -1
                for j in range(3):
                    if (
                        faces_packed[f, j] != edges_packed[e, 0]
                        and faces_packed[f, j] != edges_packed[e, 1]
                    ):
                        v2 = faces_packed[f, j]
                assert v2 > -1
                v2 = verts_packed[v2]
                normals.append((v1 - v0).view(-1).cross((v2 - v0).view(-1)))
            for i in range(len(normals) - 1):
                for j in range(1, len(normals)):
                    if i != j:
                        mesh_idx.append(edges_packed_to_mesh_idx[e])
                        loss.append(
                            (
                                1
                                - torch.cosine_similarity(
                                    normals[i].view(1, 3), -normals[j].view(1, 3)
                                )
                            )
                        )

        mesh_idx = torch.tensor(mesh_idx, device=meshes.device)
        num = mesh_idx.bincount(minlength=N)
        weights = 1.0 / num[mesh_idx].float()

        loss = torch.cat(loss) * weights
        return loss.sum() / N

    def test_mesh_normal_consistency_simple(self):
        r"""
        Mesh 1:
                        v3
                        /\
                       /  \
                   e4 / f1 \ e3
                     /      \
                 v2 /___e2___\ v1
                    \        /
                     \      /
                 e1   \ f0 / e0
                       \  /
                        \/
                        v0
        """
        device = torch.device("cuda:0")
        # mesh1 shown above
        verts1 = torch.rand((4, 3), dtype=torch.float32, device=device)
        faces1 = torch.tensor([[0, 1, 2], [2, 1, 3]], dtype=torch.int64, device=device)

        # mesh2 is a cuboid with 8 verts, 12 faces and 18 edges
        verts2 = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )
        faces2 = torch.tensor(
            [
                [0, 1, 2],
                [1, 3, 2],  # left face: 0, 1
                [2, 3, 6],
                [3, 7, 6],  # bottom face: 2, 3
                [0, 2, 6],
                [0, 6, 4],  # front face: 4, 5
                [0, 5, 1],
                [0, 4, 5],  # up face: 6, 7
                [6, 7, 5],
                [6, 5, 4],  # right face: 8, 9
                [1, 7, 3],
                [1, 5, 7],  # back face: 10, 11
            ],
            dtype=torch.int64,
            device=device,
        )

        # mesh3 is like mesh1 but with another face added to e2
        verts3 = torch.rand((5, 3), dtype=torch.float32, device=device)
        faces3 = torch.tensor(
            [[0, 1, 2], [2, 1, 3], [2, 1, 4]], dtype=torch.int64, device=device
        )

        meshes = Meshes(verts=[verts1, verts2, verts3], faces=[faces1, faces2, faces3])

        # mesh1: normal consistency computation
        n0 = (verts1[1] - verts1[2]).cross(verts1[3] - verts1[2])
        n1 = (verts1[1] - verts1[2]).cross(verts1[0] - verts1[2])
        loss1 = 1.0 - torch.cosine_similarity(n0.view(1, 3), -(n1.view(1, 3)))

        # mesh2: normal consistency computation
        # In the cube mesh, 6 edges are shared with coplanar faces (loss=0),
        # 12 edges are shared by perpendicular faces (loss=1)
        loss2 = 12.0 / 18

        # mesh3
        n0 = (verts3[1] - verts3[2]).cross(verts3[3] - verts3[2])
        n1 = (verts3[1] - verts3[2]).cross(verts3[0] - verts3[2])
        n2 = (verts3[1] - verts3[2]).cross(verts3[4] - verts3[2])
        loss3 = (
            3.0
            - torch.cosine_similarity(n0.view(1, 3), -(n1.view(1, 3)))
            - torch.cosine_similarity(n0.view(1, 3), -(n2.view(1, 3)))
            - torch.cosine_similarity(n1.view(1, 3), -(n2.view(1, 3)))
        )
        loss3 /= 3.0

        loss = (loss1 + loss2 + loss3) / 3.0

        out = mesh_normal_consistency(meshes)

        self.assertTrue(torch.allclose(out, loss))

    def test_mesh_normal_consistency(self):
        """
        Test Mesh Normal Consistency for random meshes.
        """
        meshes = TestMeshNormalConsistency.init_meshes(5, 100, 300)

        out1 = mesh_normal_consistency(meshes)
        out2 = TestMeshNormalConsistency.mesh_normal_consistency_naive(meshes)

        self.assertTrue(torch.allclose(out1, out2))

    @staticmethod
    def mesh_normal_consistency_with_ico(
        num_meshes: int, level: int = 3, device: str = "cpu"
    ):
        device = torch.device(device)
        mesh = ico_sphere(level, device)
        verts, faces = mesh.get_mesh_verts_faces(0)
        verts_list = [verts.clone() for _ in range(num_meshes)]
        faces_list = [faces.clone() for _ in range(num_meshes)]
        meshes = Meshes(verts_list, faces_list)
        torch.cuda.synchronize()

        def loss():
            mesh_normal_consistency(meshes)
            torch.cuda.synchronize()

        return loss
