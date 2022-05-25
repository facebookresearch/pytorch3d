# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import unittest

import numpy as np
import torch
from pytorch3d.ops import eyes
from pytorch3d.renderer.points.pulsar import Renderer as PulsarRenderer
from pytorch3d.transforms import so3_exp_map, so3_log_map
from pytorch3d.utils import (
    cameras_from_opencv_projection,
    opencv_from_cameras_projection,
    pulsar_from_opencv_projection,
)

from .common_testing import get_tests_dir, TestCaseMixin


DATA_DIR = get_tests_dir() / "data"


def cv2_project_points(pts, rvec, tvec, camera_matrix):
    """
    Reproduces the `cv2.projectPoints` function from OpenCV using PyTorch.
    """
    R = so3_exp_map(rvec)
    pts_proj_3d = (
        camera_matrix.bmm(R.bmm(pts.permute(0, 2, 1)) + tvec[:, :, None])
    ).permute(0, 2, 1)
    depth = pts_proj_3d[..., 2:]
    pts_proj_2d = pts_proj_3d[..., :2] / depth
    return pts_proj_2d


class TestCameraConversions(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    def test_cv2_project_points(self):
        """
        Tests that the local implementation of cv2_project_points gives the same
        restults OpenCV's `cv2.projectPoints`. The check is done against a set
        of precomputed results `cv_project_points_precomputed`.
        """
        with open(DATA_DIR / "cv_project_points_precomputed.json", "r") as f:
            cv_project_points_precomputed = json.load(f)

        for test_case in cv_project_points_precomputed:
            _pts_proj = cv2_project_points(
                **{
                    k: torch.tensor(test_case[k])[None]
                    for k in ("pts", "rvec", "tvec", "camera_matrix")
                }
            )
            pts_proj = torch.tensor(test_case["pts_proj"])[None]
            self.assertClose(_pts_proj, pts_proj, atol=1e-4)

    def test_opencv_conversion(self):
        """
        Tests that the cameras converted from opencv to pytorch3d convention
        return correct projections of random 3D points. The check is done
        against a set of results precomuted using `cv2.projectPoints` function.
        """
        device = torch.device("cuda:0")
        image_size = [[480, 640]] * 4
        R = [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        ]

        tvec = [
            [0.0, 0.0, 3.0],
            [0.3, -0.3, 3.0],
            [-0.15, 0.1, 4.0],
            [0.0, 0.0, 4.0],
        ]
        focal_length = [
            [100.0, 100.0],
            [115.0, 115.0],
            [105.0, 105.0],
            [120.0, 120.0],
        ]
        # These values are in y, x format, but they should be in x, y format.
        # The tests work like this because they only test for consistency,
        # but this format is misleading.
        principal_point = [
            [240, 320],
            [240.5, 320.3],
            [241, 318],
            [242, 322],
        ]

        principal_point, focal_length, R, tvec, image_size = [
            torch.tensor(x, device=device)
            for x in (principal_point, focal_length, R, tvec, image_size)
        ]
        camera_matrix = eyes(dim=3, N=4, device=device)
        camera_matrix[:, 0, 0], camera_matrix[:, 1, 1] = (
            focal_length[:, 0],
            focal_length[:, 1],
        )
        camera_matrix[:, :2, 2] = principal_point

        pts = torch.nn.functional.normalize(
            torch.randn(4, 1000, 3, device=device), dim=-1
        )

        # project the 3D points with the opencv projection function
        rvec = so3_log_map(R)
        pts_proj_opencv = cv2_project_points(pts, rvec, tvec, camera_matrix)

        # make the pytorch3d cameras
        cameras_opencv_to_pytorch3d = cameras_from_opencv_projection(
            R, tvec, camera_matrix, image_size
        )
        self.assertEqual(cameras_opencv_to_pytorch3d.device, device)

        # project the 3D points with converted cameras to screen space.
        pts_proj_pytorch3d_screen = cameras_opencv_to_pytorch3d.transform_points_screen(
            pts
        )[..., :2]

        # compare to the cached projected points
        self.assertClose(pts_proj_opencv, pts_proj_pytorch3d_screen, atol=1e-5)

        # Check the inverse.
        R_i, tvec_i, camera_matrix_i = opencv_from_cameras_projection(
            cameras_opencv_to_pytorch3d, image_size
        )
        self.assertClose(R, R_i)
        self.assertClose(tvec, tvec_i)
        self.assertClose(camera_matrix, camera_matrix_i)

    def test_pulsar_conversion(self):
        """
        Tests that the cameras converted from opencv to pulsar convention
        return correct projections of random 3D points. The check is done
        against a set of results precomputed using `cv2.projectPoints` function.
        """
        image_size = [[480, 640]]
        R = [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.1968, -0.6663, -0.7192],
                [0.7138, -0.4055, 0.5710],
                [-0.6721, -0.6258, 0.3959],
            ],
        ]
        tvec = [
            [10.0, 10.0, 3.0],
            [-0.0, -0.0, 20.0],
        ]
        focal_length = [
            [100.0, 100.0],
            [10.0, 10.0],
        ]
        principal_point = [
            [320, 240],
            [320, 240],
        ]

        principal_point, focal_length, R, tvec, image_size = [
            torch.FloatTensor(x)
            for x in (principal_point, focal_length, R, tvec, image_size)
        ]
        camera_matrix = eyes(dim=3, N=2)
        camera_matrix[:, 0, 0] = focal_length[:, 0]
        camera_matrix[:, 1, 1] = focal_length[:, 1]
        camera_matrix[:, :2, 2] = principal_point
        rvec = so3_log_map(R)
        pts = torch.tensor(
            [[[0.0, 0.0, 120.0]], [[0.0, 0.0, 120.0]]], dtype=torch.float32
        )
        radii = torch.tensor([[1e-5], [1e-5]], dtype=torch.float32)
        col = torch.zeros((2, 1, 1), dtype=torch.float32)

        # project the 3D points with the opencv projection function
        pts_proj_opencv = cv2_project_points(pts, rvec, tvec, camera_matrix)
        pulsar_cam = pulsar_from_opencv_projection(
            R, tvec, camera_matrix, image_size, znear=100.0
        )
        pulsar_rend = PulsarRenderer(
            640, 480, 1, right_handed_system=False, n_channels=1
        )
        rendered = torch.flip(
            pulsar_rend(
                pts,
                col,
                radii,
                pulsar_cam,
                1e-5,
                max_depth=150.0,
                min_depth=100.0,
            ),
            dims=(1,),
        )
        for batch_id in range(2):
            point_pos = torch.where(rendered[batch_id] == rendered[batch_id].min())
            point_pos = point_pos[1][0], point_pos[0][0]
            self.assertLess(
                torch.abs(point_pos[0] - pts_proj_opencv[batch_id, 0, 0]), 2
            )
            self.assertLess(
                torch.abs(point_pos[1] - pts_proj_opencv[batch_id, 0, 1]), 2
            )
