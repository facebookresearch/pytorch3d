# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.loss import mesh_edge_loss
from pytorch3d.structures import Meshes

from .common_testing import TestCaseMixin
from .test_sample_points_from_meshes import init_meshes


class TestMeshEdgeLoss(TestCaseMixin, unittest.TestCase):
    def test_empty_meshes(self):
        device = torch.device("cuda:0")
        target_length = 0
        N = 10
        V = 32
        verts_list = []
        faces_list = []
        for _ in range(N):
            vn = torch.randint(3, high=V, size=(1,))[0].item()
            verts = torch.rand((vn, 3), dtype=torch.float32, device=device)
            faces = torch.tensor([], dtype=torch.int64, device=device)
            verts_list.append(verts)
            faces_list.append(faces)
        mesh = Meshes(verts=verts_list, faces=faces_list)
        loss = mesh_edge_loss(mesh, target_length=target_length)

        self.assertClose(loss, torch.tensor([0.0], dtype=torch.float32, device=device))
        self.assertTrue(loss.requires_grad)

    @staticmethod
    def mesh_edge_loss_naive(meshes, target_length: float = 0.0):
        """
        Naive iterative implementation of mesh loss calculation.
        """
        edges_packed = meshes.edges_packed()
        verts_packed = meshes.verts_packed()
        edge_to_mesh = meshes.edges_packed_to_mesh_idx()
        N = len(meshes)
        device = meshes.device
        valid = meshes.valid
        predlosses = torch.zeros((N,), dtype=torch.float32, device=device)

        for b in range(N):
            if valid[b] == 0:
                continue
            mesh_edges = edges_packed[edge_to_mesh == b]
            verts_edges = verts_packed[mesh_edges]
            num_edges = mesh_edges.size(0)
            for e in range(num_edges):
                v0, v1 = verts_edges[e, 0], verts_edges[e, 1]
                predlosses[b] += ((v0 - v1).norm(dim=0, p=2) - target_length) ** 2.0

            if num_edges > 0:
                predlosses[b] = predlosses[b] / num_edges

        return predlosses.mean()

    def test_mesh_edge_loss_output(self):
        """
        Check outputs of tensorized and iterative implementations are the same.
        """
        device = torch.device("cuda:0")
        target_length = 0.5
        num_meshes = 10
        num_verts = 32
        num_faces = 64

        verts_list = []
        faces_list = []
        valid = torch.randint(2, size=(num_meshes,))

        for n in range(num_meshes):
            if valid[n]:
                vn = torch.randint(3, high=num_verts, size=(1,))[0].item()
                fn = torch.randint(vn, high=num_faces, size=(1,))[0].item()
                verts = torch.rand((vn, 3), dtype=torch.float32, device=device)
                faces = torch.randint(
                    vn, size=(fn, 3), dtype=torch.int64, device=device
                )
            else:
                verts = torch.tensor([], dtype=torch.float32, device=device)
                faces = torch.tensor([], dtype=torch.int64, device=device)
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts=verts_list, faces=faces_list)
        loss = mesh_edge_loss(meshes, target_length=target_length)

        predloss = TestMeshEdgeLoss.mesh_edge_loss_naive(meshes, target_length)
        self.assertClose(loss, predloss)

    @staticmethod
    def mesh_edge_loss(num_meshes: int = 10, max_v: int = 100, max_f: int = 300):
        meshes = init_meshes(num_meshes, max_v, max_f, device="cuda:0")
        torch.cuda.synchronize()

        def compute_loss():
            mesh_edge_loss(meshes, target_length=0.0)
            torch.cuda.synchronize()

        return compute_loss
