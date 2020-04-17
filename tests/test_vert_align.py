# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import torch
import torch.nn.functional as F
from common_testing import TestCaseMixin
from pytorch3d.ops.vert_align import vert_align
from pytorch3d.structures.meshes import Meshes
from pytorch3d.structures.pointclouds import Pointclouds


class TestVertAlign(TestCaseMixin, unittest.TestCase):
    @staticmethod
    def vert_align_naive(
        feats, verts, return_packed: bool = False, align_corners: bool = True
    ):
        """
        Naive implementation of vert_align.
        """
        if torch.is_tensor(feats):
            feats = [feats]
        N = feats[0].shape[0]

        out_feats = []
        # sample every example in the batch separately
        for i in range(N):
            out_i_feats = []
            for feat in feats:
                feats_i = feat[i][None, :, :, :]  # (1, C, H, W)
                if torch.is_tensor(verts):
                    grid = verts[i][None, None, :, :2]  # (1, 1, V, 2)
                elif hasattr(verts, "verts_list"):
                    grid = verts.verts_list()[i][None, None, :, :2]  # (1, 1, V, 2)
                elif hasattr(verts, "points_list"):
                    grid = verts.points_list()[i][None, None, :, :2]  # (1, 1, V, 2)
                else:
                    raise ValueError("verts_or_meshes is invalid")
                feat_sampled_i = F.grid_sample(
                    feats_i,
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=align_corners,
                )  # (1, C, 1, V)
                feat_sampled_i = feat_sampled_i.squeeze(2).squeeze(0)  # (C, V)
                feat_sampled_i = feat_sampled_i.transpose(1, 0)  # (V, C)
                out_i_feats.append(feat_sampled_i)
            out_i_feats = torch.cat(out_i_feats, 1)  # (V, sum(C))
            out_feats.append(out_i_feats)

        if return_packed:
            out_feats = torch.cat(out_feats, 0)  # (sum(V), sum(C))
        else:
            out_feats = torch.stack(out_feats, 0)  # (N, V, sum(C))
        return out_feats

    @staticmethod
    def init_meshes(
        num_meshes: int = 10, num_verts: int = 1000, num_faces: int = 3000
    ) -> Meshes:
        device = torch.device("cuda:0")
        verts_list = []
        faces_list = []
        for _ in range(num_meshes):
            verts = (
                torch.rand((num_verts, 3), dtype=torch.float32, device=device) * 2.0
                - 1.0
            )  # verts in the space of [-1, 1]
            faces = torch.randint(
                num_verts, size=(num_faces, 3), dtype=torch.int64, device=device
            )
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts_list, faces_list)

        return meshes

    @staticmethod
    def init_pointclouds(num_clouds: int = 10, num_points: int = 1000) -> Pointclouds:
        device = torch.device("cuda:0")
        points_list = []
        for _ in range(num_clouds):
            points = (
                torch.rand((num_points, 3), dtype=torch.float32, device=device) * 2.0
                - 1.0
            )  # points in the space of [-1, 1]
            points_list.append(points)
        pointclouds = Pointclouds(points=points_list)

        return pointclouds

    @staticmethod
    def init_feats(batch_size: int = 10, num_channels: int = 256, device: str = "cuda"):
        H, W = [14, 28], [14, 28]
        feats = []
        for (h, w) in zip(H, W):
            feats.append(torch.rand((batch_size, num_channels, h, w), device=device))
        return feats

    def test_vert_align_with_meshes(self):
        """
        Test vert align vs naive implementation with meshes.
        """
        meshes = TestVertAlign.init_meshes(10, 1000, 3000)
        feats = TestVertAlign.init_feats(10, 256)

        # feats in list
        out = vert_align(feats, meshes, return_packed=True)
        naive_out = TestVertAlign.vert_align_naive(feats, meshes, return_packed=True)
        self.assertClose(out, naive_out)

        # feats as tensor
        out = vert_align(feats[0], meshes, return_packed=True)
        naive_out = TestVertAlign.vert_align_naive(feats[0], meshes, return_packed=True)
        self.assertClose(out, naive_out)

    def test_vert_align_with_pointclouds(self):
        """
        Test vert align vs naive implementation with meshes.
        """
        pointclouds = TestVertAlign.init_pointclouds(10, 1000)
        feats = TestVertAlign.init_feats(10, 256)

        # feats in list
        out = vert_align(feats, pointclouds, return_packed=True)
        naive_out = TestVertAlign.vert_align_naive(
            feats, pointclouds, return_packed=True
        )
        self.assertClose(out, naive_out)

        # feats as tensor
        out = vert_align(feats[0], pointclouds, return_packed=True)
        naive_out = TestVertAlign.vert_align_naive(
            feats[0], pointclouds, return_packed=True
        )
        self.assertClose(out, naive_out)

    def test_vert_align_with_verts(self):
        """
        Test vert align vs naive implementation with verts as tensor.
        """
        feats = TestVertAlign.init_feats(10, 256)
        verts = (
            torch.rand((10, 100, 3), dtype=torch.float32, device=feats[0].device) * 2.0
            - 1.0
        )

        # feats in list
        out = vert_align(feats, verts, return_packed=True)
        naive_out = TestVertAlign.vert_align_naive(feats, verts, return_packed=True)
        self.assertClose(out, naive_out)

        # feats as tensor
        out = vert_align(feats[0], verts, return_packed=True)
        naive_out = TestVertAlign.vert_align_naive(feats[0], verts, return_packed=True)
        self.assertClose(out, naive_out)

        out2 = vert_align(feats[0], verts, return_packed=True, align_corners=False)
        naive_out2 = TestVertAlign.vert_align_naive(
            feats[0], verts, return_packed=True, align_corners=False
        )
        self.assertFalse(torch.allclose(out, out2))
        self.assertTrue(torch.allclose(out2, naive_out2))

    @staticmethod
    def vert_align_with_init(
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
        feats = TestVertAlign.init_feats(num_meshes, device=device)
        torch.cuda.synchronize()

        def sample_features():
            vert_align(feats, meshes, return_packed=True)
            torch.cuda.synchronize()

        return sample_features
