# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.implicitron.tools.eval_video_trajectory import (
    generate_eval_video_cameras,
)
from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.transforms import axis_angle_to_matrix
from tests.common_testing import TestCaseMixin


class TestEvalCameras(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_circular(self):
        n_train_cameras = 10
        n_test_cameras = 100
        R, T = look_at_view_transform(azim=torch.rand(n_train_cameras) * 360)
        amplitude = 0.01
        R_jiggled = torch.bmm(
            R, axis_angle_to_matrix(torch.rand(n_train_cameras, 3) * amplitude)
        )
        cameras_train = PerspectiveCameras(R=R_jiggled, T=T)
        cameras_test = generate_eval_video_cameras(
            cameras_train, trajectory_type="circular_lsq_fit", trajectory_scale=1.0
        )

        positions_test = cameras_test.get_camera_center()
        center = positions_test.mean(0)
        self.assertClose(center, torch.zeros(3), atol=0.1)
        self.assertClose(
            (positions_test - center).norm(dim=[1]),
            torch.ones(n_test_cameras),
            atol=0.1,
        )
