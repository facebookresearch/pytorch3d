# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from pytorch3d.ops import corresponding_cameras_alignment
from pytorch3d.renderer.cameras import (
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    SfMOrthographicCameras,
    SfMPerspectiveCameras,
)
from pytorch3d.transforms.rotation_conversions import random_rotations
from pytorch3d.transforms.so3 import so3_exp_map, so3_relative_angle

from .common_testing import TestCaseMixin
from .test_cameras import init_random_cameras


class TestCamerasAlignment(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    def test_corresponding_cameras_alignment(self):
        """
        Checks the corresponding_cameras_alignment function.
        """
        device = torch.device("cuda:0")

        # try few different random setups
        for _ in range(3):
            for estimate_scale in (True, False):
                # init true alignment transform
                R_align_gt = random_rotations(1, device=device)[0]
                T_align_gt = torch.randn(3, dtype=torch.float32, device=device)

                # init true scale
                if estimate_scale:
                    s_align_gt = torch.randn(
                        1, dtype=torch.float32, device=device
                    ).exp()
                else:
                    s_align_gt = torch.tensor(1.0, dtype=torch.float32, device=device)

                for cam_type in (
                    SfMOrthographicCameras,
                    OpenGLPerspectiveCameras,
                    OpenGLOrthographicCameras,
                    SfMPerspectiveCameras,
                ):
                    # try well-determined and underdetermined cases
                    for batch_size in (10, 4, 3, 2, 1):
                        # get random cameras
                        cameras = init_random_cameras(
                            cam_type, batch_size, random_z=True
                        ).to(device)
                        # try all alignment modes
                        for mode in ("extrinsics", "centers"):
                            # try different noise levels
                            for add_noise in (0.0, 0.01, 1e-4):
                                self._corresponding_cameras_alignment_test_case(
                                    cameras,
                                    R_align_gt,
                                    T_align_gt,
                                    s_align_gt,
                                    estimate_scale,
                                    mode,
                                    add_noise,
                                )

    def _corresponding_cameras_alignment_test_case(
        self,
        cameras,
        R_align_gt,
        T_align_gt,
        s_align_gt,
        estimate_scale,
        mode,
        add_noise,
    ):
        batch_size = cameras.R.shape[0]

        # get target camera centers
        R_new = torch.bmm(R_align_gt[None].expand_as(cameras.R), cameras.R)
        T_new = (
            torch.bmm(T_align_gt[None, None].repeat(batch_size, 1, 1), cameras.R)[:, 0]
            + cameras.T
        ) * s_align_gt

        if add_noise != 0.0:
            R_new = torch.bmm(R_new, so3_exp_map(torch.randn_like(T_new) * add_noise))
            T_new += torch.randn_like(T_new) * add_noise

        # create new cameras from R_new and T_new
        cameras_tgt = cameras.clone()
        cameras_tgt.R = R_new
        cameras_tgt.T = T_new

        # align cameras and cameras_tgt
        cameras_aligned = corresponding_cameras_alignment(
            cameras, cameras_tgt, estimate_scale=estimate_scale, mode=mode
        )

        if batch_size <= 2 and mode == "centers":
            # underdetermined case - check only the center alignment error
            # since the rotation and translation are ambiguous here
            self.assertClose(
                cameras_aligned.get_camera_center(),
                cameras_tgt.get_camera_center(),
                atol=max(add_noise * 7.0, 1e-4),
            )

        else:

            def _rmse(a):
                return (torch.norm(a, dim=1, p=2) ** 2).mean().sqrt()

            if add_noise != 0.0:
                # in a noisy case check mean rotation/translation error for
                # extrinsic alignment and root mean center error for center alignment
                if mode == "centers":
                    self.assertNormsClose(
                        cameras_aligned.get_camera_center(),
                        cameras_tgt.get_camera_center(),
                        _rmse,
                        atol=max(add_noise * 10.0, 1e-4),
                    )
                elif mode == "extrinsics":
                    angle_err = so3_relative_angle(
                        cameras_aligned.R, cameras_tgt.R, cos_angle=True
                    ).mean()
                    self.assertClose(
                        angle_err, torch.ones_like(angle_err), atol=add_noise * 0.03
                    )
                    self.assertNormsClose(
                        cameras_aligned.T, cameras_tgt.T, _rmse, atol=add_noise * 7.0
                    )
                else:
                    raise ValueError(mode)

            else:
                # compare the rotations and translations of cameras
                self.assertClose(cameras_aligned.R, cameras_tgt.R, atol=3e-4)
                self.assertClose(cameras_aligned.T, cameras_tgt.T, atol=3e-4)
                # compare the centers
                self.assertClose(
                    cameras_aligned.get_camera_center(),
                    cameras_tgt.get_camera_center(),
                    atol=3e-4,
                )

    @staticmethod
    def corresponding_cameras_alignment(
        batch_size: int, estimate_scale: bool, mode: str, cam_type=SfMPerspectiveCameras
    ):
        device = torch.device("cuda:0")
        cameras_src, cameras_tgt = [
            init_random_cameras(cam_type, batch_size, random_z=True).to(device)
            for _ in range(2)
        ]

        torch.cuda.synchronize()

        def compute_corresponding_cameras_alignment():
            corresponding_cameras_alignment(
                cameras_src, cameras_tgt, estimate_scale=estimate_scale, mode=mode
            )
            torch.cuda.synchronize()

        return compute_corresponding_cameras_alignment
