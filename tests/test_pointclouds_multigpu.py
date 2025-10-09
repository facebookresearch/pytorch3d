# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from pytorch3d.structures.pointclouds import Pointclouds

from .common_testing import needs_multigpu, TestCaseMixin


class TestPointclouds(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def init_cloud(
        num_clouds: int = 3,
        max_points: int = 100,
        channels: int = 4,
        lists_to_tensors: bool = False,
        with_normals: bool = True,
        with_features: bool = True,
        min_points: int = 0,
        requires_grad: bool = False,
    ):
        """
        Function to generate a Pointclouds object of N meshes with
        random number of points.

        Args:
            num_clouds: Number of clouds to generate.
            channels: Number of features.
            max_points: Max number of points per cloud.
            lists_to_tensors: Determines whether the generated clouds should be
                              constructed from lists (=False) or
                              tensors (=True) of points/normals/features.
            with_normals: bool whether to include normals
            with_features: bool whether to include features
            min_points: Min number of points per cloud

        Returns:
            Pointclouds object.
        """
        device = torch.device("cuda:0")
        p = torch.randint(low=min_points, high=max_points, size=(num_clouds,))
        if lists_to_tensors:
            p.fill_(p[0])

        points_list = [
            torch.rand(
                (i, 3), device=device, dtype=torch.float32, requires_grad=requires_grad
            )
            for i in p
        ]
        normals_list, features_list = None, None
        if with_normals:
            normals_list = [
                torch.rand(
                    (i, 3),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=requires_grad,
                )
                for i in p
            ]
        if with_features:
            features_list = [
                torch.rand(
                    (i, channels),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=requires_grad,
                )
                for i in p
            ]

        if lists_to_tensors:
            points_list = torch.stack(points_list)
            if with_normals:
                normals_list = torch.stack(normals_list)
            if with_features:
                features_list = torch.stack(features_list)

        return Pointclouds(points_list, normals=normals_list, features=features_list)

    @needs_multigpu
    def test_to_list(self):
        cloud = self.init_cloud(5, 100, 10)
        device = torch.device("cuda:1")

        new_cloud = cloud.to(device)
        self.assertTrue(new_cloud.device == device)
        self.assertTrue(cloud.device == torch.device("cuda:0"))
        for attrib in [
            "points_padded",
            "points_packed",
            "normals_padded",
            "normals_packed",
            "features_padded",
            "features_packed",
            "num_points_per_cloud",
            "cloud_to_packed_first_idx",
            "padded_to_packed_idx",
        ]:
            self.assertClose(
                getattr(new_cloud, attrib)().cpu(), getattr(cloud, attrib)().cpu()
            )
        for i in range(len(cloud)):
            self.assertClose(
                cloud.points_list()[i].cpu(), new_cloud.points_list()[i].cpu()
            )
            self.assertClose(
                cloud.normals_list()[i].cpu(), new_cloud.normals_list()[i].cpu()
            )
            self.assertClose(
                cloud.features_list()[i].cpu(), new_cloud.features_list()[i].cpu()
            )
        self.assertTrue(all(cloud.valid.cpu() == new_cloud.valid.cpu()))
        self.assertTrue(cloud.equisized == new_cloud.equisized)
        self.assertTrue(cloud._N == new_cloud._N)
        self.assertTrue(cloud._P == new_cloud._P)
        self.assertTrue(cloud._C == new_cloud._C)

    @needs_multigpu
    def test_to_tensor(self):
        cloud = self.init_cloud(5, 100, 10, lists_to_tensors=True)
        device = torch.device("cuda:1")

        new_cloud = cloud.to(device)
        self.assertTrue(new_cloud.device == device)
        self.assertTrue(cloud.device == torch.device("cuda:0"))
        for attrib in [
            "points_padded",
            "points_packed",
            "normals_padded",
            "normals_packed",
            "features_padded",
            "features_packed",
            "num_points_per_cloud",
            "cloud_to_packed_first_idx",
            "padded_to_packed_idx",
        ]:
            self.assertClose(
                getattr(new_cloud, attrib)().cpu(), getattr(cloud, attrib)().cpu()
            )
        for i in range(len(cloud)):
            self.assertClose(
                cloud.points_list()[i].cpu(), new_cloud.points_list()[i].cpu()
            )
            self.assertClose(
                cloud.normals_list()[i].cpu(), new_cloud.normals_list()[i].cpu()
            )
            self.assertClose(
                cloud.features_list()[i].cpu(), new_cloud.features_list()[i].cpu()
            )
        self.assertTrue(all(cloud.valid.cpu() == new_cloud.valid.cpu()))
        self.assertTrue(cloud.equisized == new_cloud.equisized)
        self.assertTrue(cloud._N == new_cloud._N)
        self.assertTrue(cloud._P == new_cloud._P)
        self.assertTrue(cloud._C == new_cloud._C)
