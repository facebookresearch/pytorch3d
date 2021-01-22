# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from common_testing import TestCaseMixin
from pytorch3d.renderer.camera_utils import camera_to_eye_at_up
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
from torch.nn.functional import normalize


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
