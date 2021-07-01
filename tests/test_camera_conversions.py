# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import json
import unittest

import numpy as np
import torch
from common_testing import TestCaseMixin, get_tests_dir
from pytorch3d.ops import eyes
from pytorch3d.transforms import so3_exp_map, so3_log_map
from pytorch3d.utils import (
    cameras_from_opencv_projection,
    opencv_from_cameras_projection,
)


DATA_DIR = get_tests_dir() / "data"


def _coords_opencv_screen_to_pytorch3d_ndc(xy_opencv, image_size):
    """
    Converts the OpenCV screen coordinates `xy_opencv` to PyTorch3D NDC coordinates.
    """
    xy_pytorch3d = -(2.0 * xy_opencv / image_size.flip(dims=(1,))[:, None] - 1.0)
    return xy_pytorch3d


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
        principal_point = [
            [240, 320],
            [240.5, 320.3],
            [241, 318],
            [242, 322],
        ]

        principal_point, focal_length, R, tvec, image_size = [
            torch.FloatTensor(x)
            for x in (principal_point, focal_length, R, tvec, image_size)
        ]
        camera_matrix = eyes(dim=3, N=4)
        camera_matrix[:, 0, 0], camera_matrix[:, 1, 1] = (
            focal_length[:, 0],
            focal_length[:, 1],
        )
        camera_matrix[:, :2, 2] = principal_point

        rvec = so3_log_map(R)

        pts = torch.nn.functional.normalize(torch.randn(4, 1000, 3), dim=-1)

        # project the 3D points with the opencv projection function
        pts_proj_opencv = cv2_project_points(pts, rvec, tvec, camera_matrix)

        # make the pytorch3d cameras
        cameras_opencv_to_pytorch3d = cameras_from_opencv_projection(
            rvec, tvec, camera_matrix, image_size
        )

        # project the 3D points with converted cameras
        pts_proj_pytorch3d = cameras_opencv_to_pytorch3d.transform_points(pts)[..., :2]

        # convert the opencv-projected points to pytorch3d screen coords
        pts_proj_opencv_in_pytorch3d_screen = _coords_opencv_screen_to_pytorch3d_ndc(
            pts_proj_opencv, image_size
        )

        # compare to the cached projected points
        self.assertClose(
            pts_proj_opencv_in_pytorch3d_screen, pts_proj_pytorch3d, atol=1e-5
        )

        # Check the inverse.
        rvec_i, tvec_i, camera_matrix_i = opencv_from_cameras_projection(
            cameras_opencv_to_pytorch3d, image_size
        )
        self.assertClose(rvec, rvec_i)
        self.assertClose(tvec, tvec_i)
        self.assertClose(camera_matrix, camera_matrix_i)
