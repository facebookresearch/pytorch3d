# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.implicitron.tools.point_cloud_utils import get_rgbd_point_cloud

from pytorch3d.renderer.cameras import PerspectiveCameras
from tests.common_testing import TestCaseMixin


class TestPointCloudUtils(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_unproject(self):
        H, W = 50, 100

        # Random RGBD image with depth 3
        # (depth 0 = at the camera)
        # and purple in the upper right corner

        image = torch.rand(4, H, W)
        depth = 3
        image[3] = depth
        image[1, H // 2 :, W // 2 :] *= 0.4

        # two ways to define the same camera:
        # at the origin facing the positive z axis
        ndc_camera = PerspectiveCameras(focal_length=1.0)
        screen_camera = PerspectiveCameras(
            focal_length=H // 2,
            in_ndc=False,
            image_size=((H, W),),
            principal_point=((W / 2, H / 2),),
        )

        for camera in (ndc_camera, screen_camera):
            # 1. z-depth
            cloud = get_rgbd_point_cloud(
                camera,
                image_rgb=image[:3][None],
                depth_map=image[3:][None],
                euclidean=False,
            )
            [points] = cloud.points_list()
            self.assertConstant(points[:, 2], depth)  # constant depth
            extremes = depth * torch.tensor([W / H - 1 / H, 1 - 1 / H])
            self.assertClose(points[:, :2].min(0).values, -extremes)
            self.assertClose(points[:, :2].max(0).values, extremes)

            # 2. euclidean
            cloud = get_rgbd_point_cloud(
                camera,
                image_rgb=image[:3][None],
                depth_map=image[3:][None],
                euclidean=True,
            )
            [points] = cloud.points_list()
            self.assertConstant(torch.norm(points, dim=1), depth, atol=1e-5)

            # 3. four channels
            get_rgbd_point_cloud(
                camera,
                image_rgb=image[None],
                depth_map=image[3:][None],
                euclidean=True,
            )
