# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from math import radians

import torch
from pytorch3d.renderer.camera_utils import camera_to_eye_at_up, rotate_on_spot
from pytorch3d.renderer.cameras import (
    get_world_to_view_transform,
    look_at_view_transform,
    PerspectiveCameras,
)
from pytorch3d.transforms import axis_angle_to_matrix
from torch.nn.functional import normalize

from .common_testing import TestCaseMixin


def _batched_dotprod(x: torch.Tensor, y: torch.Tensor):
    """
    Takes two tensors of shape (N,3) and returns their batched
    dot product along the last dimension as a tensor of shape
    (N,).
    """
    return torch.einsum("ij,ij->i", x, y)


class TestCameraUtils(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_invert_eye_at_up(self):
        # Generate random cameras and check we can reconstruct their eye, at,
        # and up vectors.
        N = 13
        eye = torch.rand(N, 3)
        at = torch.rand(N, 3)
        up = torch.rand(N, 3)

        R, T = look_at_view_transform(eye=eye, at=at, up=up)
        cameras = PerspectiveCameras(R=R, T=T)

        eye2, at2, up2 = camera_to_eye_at_up(cameras.get_world_to_view_transform())

        # The retrieved eye matches
        self.assertClose(eye, eye2, atol=1e-5)
        self.assertClose(cameras.get_camera_center(), eye)

        # at-eye as retrieved must be a vector in the same direction as
        # the original.
        self.assertClose(normalize(at - eye), normalize(at2 - eye2))

        # The up vector as retrieved should be rotated the same amount
        # around at-eye as the original. The component in the at-eye
        # direction is unimportant, as is the length.
        # So check that (up x (at-eye)) as retrieved is in the same
        # direction as its original value.
        up_check = torch.cross(up, at - eye, dim=-1)
        up_check2 = torch.cross(up2, at - eye, dim=-1)
        self.assertClose(normalize(up_check), normalize(up_check2))

        # Master check that we get the same camera if we reinitialise.
        R2, T2 = look_at_view_transform(eye=eye2, at=at2, up=up2)
        cameras2 = PerspectiveCameras(R=R2, T=T2)
        cam_trans = cameras.get_world_to_view_transform()
        cam_trans2 = cameras2.get_world_to_view_transform()

        self.assertClose(cam_trans.get_matrix(), cam_trans2.get_matrix(), atol=1e-5)

    def test_rotate_on_spot_yaw(self):
        N = 14
        eye = torch.rand(N, 3)
        at = torch.rand(N, 3)
        up = torch.rand(N, 3)

        R, T = look_at_view_transform(eye=eye, at=at, up=up)

        # Moving around the y axis looks left.
        angles = torch.FloatTensor([0, -radians(10), 0])
        rotation = axis_angle_to_matrix(angles)
        R_rot, T_rot = rotate_on_spot(R, T, rotation)

        eye_rot, at_rot, up_rot = camera_to_eye_at_up(
            get_world_to_view_transform(R=R_rot, T=T_rot)
        )
        self.assertClose(eye, eye_rot, atol=1e-5)

        # Make vectors pointing exactly left and up
        left = torch.cross(up, at - eye, dim=-1)
        left_rot = torch.cross(up_rot, at_rot - eye_rot, dim=-1)
        fully_up = torch.cross(at - eye, left, dim=-1)
        fully_up_rot = torch.cross(at_rot - eye_rot, left_rot, dim=-1)

        # The up direction is unchanged
        self.assertClose(normalize(fully_up), normalize(fully_up_rot), atol=1e-5)

        # The camera has moved left
        agree = _batched_dotprod(torch.cross(left, left_rot, dim=1), fully_up)
        self.assertGreater(agree.min(), 0)

        # Batch dimension for rotation
        R_rot2, T_rot2 = rotate_on_spot(R, T, rotation.expand(N, 3, 3))
        self.assertClose(R_rot, R_rot2)
        self.assertClose(T_rot, T_rot2)

        # No batch dimension for either
        R_rot3, T_rot3 = rotate_on_spot(R[0], T[0], rotation)
        self.assertClose(R_rot[:1], R_rot3)
        self.assertClose(T_rot[:1], T_rot3)

        # No batch dimension for R, T
        R_rot4, T_rot4 = rotate_on_spot(R[0], T[0], rotation.expand(N, 3, 3))
        self.assertClose(R_rot[:1].expand(N, 3, 3), R_rot4)
        self.assertClose(T_rot[:1].expand(N, 3), T_rot4)

    def test_rotate_on_spot_pitch(self):
        N = 14
        eye = torch.rand(N, 3)
        at = torch.rand(N, 3)
        up = torch.rand(N, 3)

        R, T = look_at_view_transform(eye=eye, at=at, up=up)

        # Moving around the x axis looks down.
        angles = torch.FloatTensor([-radians(10), 0, 0])
        rotation = axis_angle_to_matrix(angles)
        R_rot, T_rot = rotate_on_spot(R, T, rotation)
        eye_rot, at_rot, up_rot = camera_to_eye_at_up(
            get_world_to_view_transform(R=R_rot, T=T_rot)
        )
        self.assertClose(eye, eye_rot, atol=1e-5)

        # A vector pointing left is unchanged
        left = torch.cross(up, at - eye, dim=-1)
        left_rot = torch.cross(up_rot, at_rot - eye_rot, dim=-1)
        self.assertClose(normalize(left), normalize(left_rot), atol=1e-5)

        # The camera has moved down
        fully_up = torch.cross(at - eye, left, dim=-1)
        fully_up_rot = torch.cross(at_rot - eye_rot, left_rot, dim=-1)
        agree = _batched_dotprod(torch.cross(fully_up, fully_up_rot, dim=1), left)
        self.assertGreater(agree.min(), 0)

    def test_rotate_on_spot_roll(self):
        N = 14
        eye = torch.rand(N, 3)
        at = torch.rand(N, 3)
        up = torch.rand(N, 3)

        R, T = look_at_view_transform(eye=eye, at=at, up=up)

        # Moving around the z axis rotates the image.
        angles = torch.FloatTensor([0, 0, -radians(10)])
        rotation = axis_angle_to_matrix(angles)
        R_rot, T_rot = rotate_on_spot(R, T, rotation)
        eye_rot, at_rot, up_rot = camera_to_eye_at_up(
            get_world_to_view_transform(R=R_rot, T=T_rot)
        )
        self.assertClose(eye, eye_rot, atol=1e-5)
        self.assertClose(normalize(at - eye), normalize(at_rot - eye), atol=1e-5)

        # The camera has moved clockwise
        agree = _batched_dotprod(torch.cross(up, up_rot, dim=1), at - eye)
        self.assertGreater(agree.min(), 0)
