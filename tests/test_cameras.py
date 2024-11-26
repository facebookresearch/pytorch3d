# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# @licenselint-loose-mode

# Some of the code below is adapted from Soft Rasterizer (SoftRas)
#
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
# Copyright (c) 2019 Shichen Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import pickle
import unittest
from itertools import product

import numpy as np
import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import (
    camera_position_from_spherical_angles,
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    get_world_to_view_transform,
    look_at_rotation,
    look_at_view_transform,
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
    SfMOrthographicCameras,
    SfMPerspectiveCameras,
)
from pytorch3d.renderer.fisheyecameras import FishEyeCameras
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms.rotation_conversions import random_rotations
from pytorch3d.transforms.so3 import so3_exp_map

from .common_camera_utils import init_random_cameras

from .common_testing import TestCaseMixin


# Naive function adapted from SoftRasterizer for test purposes.
def perspective_project_naive(points, fov=60.0):
    """
    Compute perspective projection from a given viewing angle.
    Args:
        points: (N, V, 3) representing the padded points.
        viewing angle: degrees
    Returns:
        (N, V, 3) tensor of projected points preserving the view space z
        coordinate (no z renormalization)
    """
    device = points.device
    halfFov = torch.tensor((fov / 2) / 180 * np.pi, dtype=torch.float32, device=device)
    scale = torch.tan(halfFov[None])
    scale = scale[:, None]
    z = points[:, :, 2]
    x = points[:, :, 0] / z / scale
    y = points[:, :, 1] / z / scale
    points = torch.stack((x, y, z), dim=2)
    return points


def sfm_perspective_project_naive(points, fx=1.0, fy=1.0, p0x=0.0, p0y=0.0):
    """
    Compute perspective projection using focal length and principal point.

    Args:
        points: (N, V, 3) representing the padded points.
        fx: world units
        fy: world units
        p0x: pixels
        p0y: pixels
    Returns:
        (N, V, 3) tensor of projected points.
    """
    z = points[:, :, 2]
    x = (points[:, :, 0] * fx) / z + p0x
    y = (points[:, :, 1] * fy) / z + p0y
    points = torch.stack((x, y, 1.0 / z), dim=2)
    return points


# Naive function adapted from SoftRasterizer for test purposes.
def orthographic_project_naive(points, scale_xyz=(1.0, 1.0, 1.0)):
    """
    Compute orthographic projection from a given angle
    Args:
        points: (N, V, 3) representing the padded points.
        scaled: (N, 3) scaling factors for each of xyz directions
    Returns:
        (N, V, 3) tensor of projected points preserving the view space z
        coordinate (no z renormalization).
    """
    if not torch.is_tensor(scale_xyz):
        scale_xyz = torch.tensor(scale_xyz)
    scale_xyz = scale_xyz.view(-1, 3)
    z = points[:, :, 2]
    x = points[:, :, 0] * scale_xyz[:, 0]
    y = points[:, :, 1] * scale_xyz[:, 1]
    points = torch.stack((x, y, z), dim=2)
    return points


def ndc_to_screen_points_naive(points, imsize):
    """
    Transforms points from PyTorch3D's NDC space to screen space
    Args:
        points: (N, V, 3) representing padded points
        imsize: (N, 2) image size = (height, width)
    Returns:
        (N, V, 3) tensor of transformed points
    """
    height, width = imsize.unbind(1)
    width = width.view(-1, 1)
    half_width = width / 2.0
    height = height.view(-1, 1)
    half_height = height / 2.0

    scale = (
        half_width * (height > width).float() + half_height * (height <= width).float()
    )

    x, y, z = points.unbind(2)
    x = -scale * x + half_width
    y = -scale * y + half_height
    return torch.stack((x, y, z), dim=2)


class TestCameraHelpers(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    def test_look_at_view_transform_from_eye_point_tuple(self):
        dist = math.sqrt(2)
        elev = math.pi / 4
        azim = 0.0
        eye = ((0.0, 1.0, 1.0),)
        # using passed values for dist, elev, azim
        R, t = look_at_view_transform(dist, elev, azim, degrees=False)
        # using other values for dist, elev, azim - eye overrides
        R_eye, t_eye = look_at_view_transform(dist=3, elev=2, azim=1, eye=eye)
        # using only eye value

        R_eye_only, t_eye_only = look_at_view_transform(eye=eye)
        self.assertTrue(torch.allclose(R, R_eye, atol=2e-7))
        self.assertTrue(torch.allclose(t, t_eye, atol=2e-7))
        self.assertTrue(torch.allclose(R, R_eye_only, atol=2e-7))
        self.assertTrue(torch.allclose(t, t_eye_only, atol=2e-7))

    def test_look_at_view_transform_default_values(self):
        dist = 1.0
        elev = 0.0
        azim = 0.0
        # Using passed values for dist, elev, azim
        R, t = look_at_view_transform(dist, elev, azim)
        # Using default dist=1.0, elev=0.0, azim=0.0
        R_default, t_default = look_at_view_transform()
        # test default = passed = expected
        self.assertTrue(torch.allclose(R, R_default, atol=2e-7))
        self.assertTrue(torch.allclose(t, t_default, atol=2e-7))

    def test_look_at_view_transform_non_default_at_position(self):
        dist = 1.0
        elev = 0.0
        azim = 0.0
        at = ((1, 1, 1),)
        # Using passed values for dist, elev, azim, at
        R, t = look_at_view_transform(dist, elev, azim, at=at)
        # Using default dist=1.0, elev=0.0, azim=0.0
        R_default, t_default = look_at_view_transform()
        # test default = passed = expected
        # R must be the same, t must be translated by (1,-1,1) with respect to t_default
        t_trans = torch.tensor([1, -1, 1], dtype=torch.float32).view(1, 3)
        self.assertTrue(torch.allclose(R, R_default, atol=2e-7))
        self.assertTrue(torch.allclose(t, t_default + t_trans, atol=2e-7))

    def test_camera_position_from_angles_python_scalar(self):
        dist = 2.7
        elev = 90.0
        azim = 0.0
        expected_position = torch.tensor([0.0, 2.7, 0.0], dtype=torch.float32).view(
            1, 3
        )
        position = camera_position_from_spherical_angles(dist, elev, azim)
        self.assertClose(position, expected_position, atol=2e-7)

    def test_camera_position_from_angles_python_scalar_radians(self):
        dist = 2.7
        elev = math.pi / 2
        azim = 0.0
        expected_position = torch.tensor([0.0, 2.7, 0.0], dtype=torch.float32)
        expected_position = expected_position.view(1, 3)
        position = camera_position_from_spherical_angles(
            dist, elev, azim, degrees=False
        )
        self.assertClose(position, expected_position, atol=2e-7)

    def test_camera_position_from_angles_torch_scalars(self):
        dist = torch.tensor(2.7)
        elev = torch.tensor(0.0)
        azim = torch.tensor(90.0)
        expected_position = torch.tensor([2.7, 0.0, 0.0], dtype=torch.float32).view(
            1, 3
        )
        position = camera_position_from_spherical_angles(dist, elev, azim)
        self.assertClose(position, expected_position, atol=2e-7)

    def test_camera_position_from_angles_mixed_scalars(self):
        dist = 2.7
        elev = torch.tensor(0.0)
        azim = 90.0
        expected_position = torch.tensor([2.7, 0.0, 0.0], dtype=torch.float32).view(
            1, 3
        )
        position = camera_position_from_spherical_angles(dist, elev, azim)
        self.assertClose(position, expected_position, atol=2e-7)

    def test_camera_position_from_angles_torch_scalar_grads(self):
        dist = torch.tensor(2.7, requires_grad=True)
        elev = torch.tensor(45.0, requires_grad=True)
        azim = torch.tensor(45.0)
        position = camera_position_from_spherical_angles(dist, elev, azim)
        position.sum().backward()
        self.assertTrue(hasattr(elev, "grad"))
        self.assertTrue(hasattr(dist, "grad"))
        elev_grad = elev.grad.clone()
        dist_grad = dist.grad.clone()
        elev = math.pi / 180.0 * elev.detach()
        azim = math.pi / 180.0 * azim
        grad_dist = (
            torch.cos(elev) * torch.sin(azim)
            + torch.sin(elev)
            + torch.cos(elev) * torch.cos(azim)
        )
        grad_elev = (
            -(torch.sin(elev)) * torch.sin(azim)
            + torch.cos(elev)
            - torch.sin(elev) * torch.cos(azim)
        )
        grad_elev = dist * (math.pi / 180.0) * grad_elev
        self.assertClose(elev_grad, grad_elev)
        self.assertClose(dist_grad, grad_dist)

    def test_camera_position_from_angles_vectors(self):
        dist = torch.tensor([2.0, 2.0])
        elev = torch.tensor([0.0, 90.0])
        azim = torch.tensor([90.0, 0.0])
        expected_position = torch.tensor(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32
        )
        position = camera_position_from_spherical_angles(dist, elev, azim)
        self.assertClose(position, expected_position, atol=2e-7)

    def test_camera_position_from_angles_vectors_broadcast(self):
        dist = torch.tensor([2.0, 3.0, 5.0])
        elev = torch.tensor([0.0])
        azim = torch.tensor([90.0])
        expected_position = torch.tensor(
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float32
        )
        position = camera_position_from_spherical_angles(dist, elev, azim)
        self.assertClose(position, expected_position, atol=3e-7)

    def test_camera_position_from_angles_vectors_mixed_broadcast(self):
        dist = torch.tensor([2.0, 3.0, 5.0])
        elev = 0.0
        azim = torch.tensor(90.0)
        expected_position = torch.tensor(
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=torch.float32
        )
        position = camera_position_from_spherical_angles(dist, elev, azim)
        self.assertClose(position, expected_position, atol=3e-7)

    def test_camera_position_from_angles_vectors_mixed_broadcast_grads(self):
        dist = torch.tensor([2.0, 3.0, 5.0], requires_grad=True)
        elev = torch.tensor(45.0, requires_grad=True)
        azim = 45.0
        position = camera_position_from_spherical_angles(dist, elev, azim)
        position.sum().backward()
        self.assertTrue(hasattr(elev, "grad"))
        self.assertTrue(hasattr(dist, "grad"))
        elev_grad = elev.grad.clone()
        dist_grad = dist.grad.clone()
        azim = torch.tensor(azim)
        elev = math.pi / 180.0 * elev.detach()
        azim = math.pi / 180.0 * azim
        grad_dist = (
            torch.cos(elev) * torch.sin(azim)
            + torch.sin(elev)
            + torch.cos(elev) * torch.cos(azim)
        )
        grad_elev = (
            -(torch.sin(elev)) * torch.sin(azim)
            + torch.cos(elev)
            - torch.sin(elev) * torch.cos(azim)
        )
        grad_elev = (dist * (math.pi / 180.0) * grad_elev).sum()
        self.assertClose(elev_grad, grad_elev)
        self.assertClose(dist_grad, torch.full([3], grad_dist))

    def test_camera_position_from_angles_vectors_bad_broadcast(self):
        # Batch dim for broadcast must be N or 1
        dist = torch.tensor([2.0, 3.0, 5.0])
        elev = torch.tensor([0.0, 90.0])
        azim = torch.tensor([90.0])
        with self.assertRaises(ValueError):
            camera_position_from_spherical_angles(dist, elev, azim)

    def test_look_at_rotation_python_list(self):
        camera_position = [[0.0, 0.0, -1.0]]  # camera pointing along negative z
        rot_mat = look_at_rotation(camera_position)
        self.assertClose(rot_mat, torch.eye(3)[None], atol=2e-7)

    def test_look_at_rotation_input_fail(self):
        camera_position = [-1.0]  # expected to have xyz positions
        with self.assertRaises(ValueError):
            look_at_rotation(camera_position)

    def test_look_at_rotation_list_broadcast(self):
        # fmt: off
        camera_positions = [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]
        rot_mats_expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ],
                [
                    [-1.0, 0.0,  0.0],  # noqa: E241, E201
                    [ 0.0, 1.0,  0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, -1.0]   # noqa: E241, E201
                ],
            ],
            dtype=torch.float32
        )
        # fmt: on
        rot_mats = look_at_rotation(camera_positions)
        self.assertClose(rot_mats, rot_mats_expected, atol=2e-7)

    def test_look_at_rotation_tensor_broadcast(self):
        # fmt: off
        camera_positions = torch.tensor([
            [0.0, 0.0, -1.0],
            [0.0, 0.0,  1.0]   # noqa: E241, E201
        ], dtype=torch.float32)
        rot_mats_expected = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ],
                [
                    [-1.0, 0.0,  0.0],  # noqa: E241, E201
                    [ 0.0, 1.0,  0.0],  # noqa: E241, E201
                    [ 0.0, 0.0, -1.0]   # noqa: E241, E201
                ],
            ],
            dtype=torch.float32
        )
        # fmt: on
        rot_mats = look_at_rotation(camera_positions)
        self.assertClose(rot_mats, rot_mats_expected, atol=2e-7)

    def test_look_at_rotation_tensor_grad(self):
        camera_position = torch.tensor([[0.0, 0.0, -1.0]], requires_grad=True)
        rot_mat = look_at_rotation(camera_position)
        rot_mat.sum().backward()
        self.assertTrue(hasattr(camera_position, "grad"))
        self.assertClose(
            camera_position.grad, torch.zeros_like(camera_position), atol=2e-7
        )

    def test_view_transform(self):
        T = torch.tensor([0.0, 0.0, -1.0], requires_grad=True).view(1, -1)
        R = look_at_rotation(T)
        RT = get_world_to_view_transform(R=R, T=T)
        self.assertTrue(isinstance(RT, Transform3d))

    def test_look_at_view_transform_corner_case(self):
        dist = 2.7
        elev = 90
        azim = 90
        expected_position = torch.tensor([0.0, 2.7, 0.0], dtype=torch.float32).view(
            1, 3
        )
        position = camera_position_from_spherical_angles(dist, elev, azim)
        self.assertClose(position, expected_position, atol=2e-7)
        R, _ = look_at_view_transform(eye=position)
        x_axis = R[:, :, 0]
        expected_x_axis = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32).view(1, 3)
        self.assertClose(x_axis, expected_x_axis, atol=5e-3)


class TestCamerasCommon(TestCaseMixin, unittest.TestCase):
    def test_K(self, batch_size=10):
        T = torch.randn(batch_size, 3)
        R = random_rotations(batch_size)
        K = torch.randn(batch_size, 4, 4)
        for cam_type in (
            FoVOrthographicCameras,
            FoVPerspectiveCameras,
            OrthographicCameras,
            PerspectiveCameras,
        ):
            cam = cam_type(R=R, T=T, K=K)
            cam.get_projection_transform()
            # Just checking that we don't crash or anything

    def test_view_transform_class_method(self):
        T = torch.tensor([0.0, 0.0, -1.0], requires_grad=True).view(1, -1)
        R = look_at_rotation(T)
        RT = get_world_to_view_transform(R=R, T=T)
        for cam_type in (
            OpenGLPerspectiveCameras,
            OpenGLOrthographicCameras,
            SfMOrthographicCameras,
            SfMPerspectiveCameras,
            FoVOrthographicCameras,
            FoVPerspectiveCameras,
            OrthographicCameras,
            PerspectiveCameras,
        ):
            cam = cam_type(R=R, T=T)
            RT_class = cam.get_world_to_view_transform()
            self.assertTrue(torch.allclose(RT.get_matrix(), RT_class.get_matrix()))

        self.assertTrue(isinstance(RT, Transform3d))

    def test_get_camera_center(self, batch_size=10):
        T = torch.randn(batch_size, 3)
        R = random_rotations(batch_size)
        for cam_type in (
            OpenGLPerspectiveCameras,
            OpenGLOrthographicCameras,
            SfMOrthographicCameras,
            SfMPerspectiveCameras,
            FoVOrthographicCameras,
            FoVPerspectiveCameras,
            OrthographicCameras,
            PerspectiveCameras,
        ):
            cam = cam_type(R=R, T=T)
            C = cam.get_camera_center()
            C_ = -torch.bmm(R, T[:, :, None])[:, :, 0]
            self.assertTrue(torch.allclose(C, C_, atol=1e-05))

    @staticmethod
    def init_equiv_cameras_ndc_screen(cam_type: CamerasBase, batch_size: int):
        T = torch.randn(batch_size, 3) * 0.03
        T[:, 2] = 4
        R = so3_exp_map(torch.randn(batch_size, 3) * 3.0)
        screen_cam_params = {"R": R, "T": T}
        ndc_cam_params = {"R": R, "T": T}
        if cam_type in (OrthographicCameras, PerspectiveCameras):
            fcl = torch.rand((batch_size, 2)) * 3.0 + 0.1
            prc = torch.randn((batch_size, 2)) * 0.2
            # (height, width)
            image_size = torch.randint(low=2, high=64, size=(batch_size, 2))
            # scale
            scale = (image_size.min(dim=1, keepdim=True).values) / 2.0

            ndc_cam_params["focal_length"] = fcl
            ndc_cam_params["principal_point"] = prc
            ndc_cam_params["image_size"] = image_size

            screen_cam_params["image_size"] = image_size
            screen_cam_params["focal_length"] = fcl * scale
            screen_cam_params["principal_point"] = (
                (image_size[:, [1, 0]]) / 2.0 - prc * scale
            )
            screen_cam_params["in_ndc"] = False
        else:
            raise ValueError(str(cam_type))
        return cam_type(**ndc_cam_params), cam_type(**screen_cam_params)

    def test_unproject_points(self, batch_size=50, num_points=100):
        """
        Checks that an unprojection of a randomly projected point cloud
        stays the same.
        """

        for cam_type in (
            SfMOrthographicCameras,
            OpenGLPerspectiveCameras,
            OpenGLOrthographicCameras,
            SfMPerspectiveCameras,
            FoVOrthographicCameras,
            FoVPerspectiveCameras,
            OrthographicCameras,
            PerspectiveCameras,
        ):
            # init the cameras
            cameras = init_random_cameras(cam_type, batch_size)
            # xyz - the ground truth point cloud
            xyz = torch.randn(batch_size, num_points, 3) * 0.3
            # xyz in camera coordinates
            xyz_cam = cameras.get_world_to_view_transform().transform_points(xyz)
            # depth = z-component of xyz_cam
            depth = xyz_cam[:, :, 2:]
            # project xyz
            xyz_proj = cameras.transform_points(xyz)
            xy, cam_depth = xyz_proj.split(2, dim=2)
            # input to the unprojection function
            xy_depth = torch.cat((xy, depth), dim=2)

            for to_world in (False, True):
                if to_world:
                    matching_xyz = xyz
                else:
                    matching_xyz = xyz_cam

                # if we have FoV (= OpenGL) cameras
                # test for scaled_depth_input=True/False
                if cam_type in (
                    OpenGLPerspectiveCameras,
                    OpenGLOrthographicCameras,
                    FoVPerspectiveCameras,
                    FoVOrthographicCameras,
                ):
                    for scaled_depth_input in (True, False):
                        if scaled_depth_input:
                            xy_depth_ = xyz_proj
                        else:
                            xy_depth_ = xy_depth
                        xyz_unproj = cameras.unproject_points(
                            xy_depth_,
                            world_coordinates=to_world,
                            scaled_depth_input=scaled_depth_input,
                        )
                        self.assertTrue(
                            torch.allclose(xyz_unproj, matching_xyz, atol=1e-4)
                        )
                else:
                    xyz_unproj = cameras.unproject_points(
                        xy_depth, world_coordinates=to_world
                    )
                    self.assertTrue(torch.allclose(xyz_unproj, matching_xyz, atol=1e-4))

    @staticmethod
    def unproject_points(
        cam_type, batch_size=50, num_points=100, device: Device = "cpu"
    ):
        """
        Checks that an unprojection of a randomly projected point cloud
        stays the same.
        """
        if device == "cuda":
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        str2cls = {  # noqa
            "OpenGLOrthographicCameras": OpenGLOrthographicCameras,
            "OpenGLPerspectiveCameras": OpenGLPerspectiveCameras,
            "SfMOrthographicCameras": SfMOrthographicCameras,
            "SfMPerspectiveCameras": SfMPerspectiveCameras,
            "FoVOrthographicCameras": FoVOrthographicCameras,
            "FoVPerspectiveCameras": FoVPerspectiveCameras,
            "OrthographicCameras": OrthographicCameras,
            "PerspectiveCameras": PerspectiveCameras,
            "FishEyeCameras": FishEyeCameras,
        }

        def run_cameras():
            # init the cameras
            cameras = init_random_cameras(str2cls[cam_type], batch_size, device=device)
            # xyz - the ground truth point cloud
            xyz = torch.randn(num_points, 3) * 0.3
            xyz = cameras.unproject_points(xyz, scaled_depth_input=True)

        return run_cameras

    def test_project_points_screen(self, batch_size=50, num_points=100):
        """
        Checks that an unprojection of a randomly projected point cloud
        stays the same.
        """

        for cam_type in (
            OpenGLOrthographicCameras,
            OpenGLPerspectiveCameras,
            SfMOrthographicCameras,
            SfMPerspectiveCameras,
            FoVOrthographicCameras,
            FoVPerspectiveCameras,
            OrthographicCameras,
            PerspectiveCameras,
        ):
            # init the cameras
            cameras = init_random_cameras(cam_type, batch_size)
            # xyz - the ground truth point cloud
            xy = torch.randn(batch_size, num_points, 2) * 2.0 - 1.0
            z = torch.randn(batch_size, num_points, 1) * 3.0 + 1.0
            xyz = torch.cat((xy, z), dim=2)
            # image size
            image_size = torch.randint(low=32, high=64, size=(batch_size, 2))
            # project points
            xyz_project_ndc = cameras.transform_points_ndc(xyz)
            xyz_project_screen = cameras.transform_points_screen(
                xyz, image_size=image_size
            )
            # naive
            xyz_project_screen_naive = ndc_to_screen_points_naive(
                xyz_project_ndc, image_size
            )
            # we set atol to 1e-4, remember that screen points are in [0, W]x[0, H] space
            self.assertClose(xyz_project_screen, xyz_project_screen_naive, atol=1e-4)

    @staticmethod
    def transform_points(
        cam_type, batch_size=50, num_points=100, device: Device = "cpu"
    ):
        """
        Checks that an unprojection of a randomly projected point cloud
        stays the same.
        """

        if device == "cuda":
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        str2cls = {  # noqa
            "OpenGLOrthographicCameras": OpenGLOrthographicCameras,
            "OpenGLPerspectiveCameras": OpenGLPerspectiveCameras,
            "SfMOrthographicCameras": SfMOrthographicCameras,
            "SfMPerspectiveCameras": SfMPerspectiveCameras,
            "FoVOrthographicCameras": FoVOrthographicCameras,
            "FoVPerspectiveCameras": FoVPerspectiveCameras,
            "OrthographicCameras": OrthographicCameras,
            "PerspectiveCameras": PerspectiveCameras,
            "FishEyeCameras": FishEyeCameras,
        }

        def run_cameras():
            # init the cameras
            cameras = init_random_cameras(str2cls[cam_type], batch_size, device=device)
            # xyz - the ground truth point cloud
            xy = torch.randn(num_points, 2) * 2.0 - 1.0
            z = torch.randn(num_points, 1) * 3.0 + 1.0
            xyz = torch.cat((xy, z), dim=-1)
            xy = cameras.transform_points(xyz)

        return run_cameras

    def test_equiv_project_points(self, batch_size=50, num_points=100):
        """
        Checks that NDC and screen cameras project points to ndc correctly.
        Applies only to OrthographicCameras and PerspectiveCameras.
        """
        for cam_type in (OrthographicCameras, PerspectiveCameras):
            # init the cameras
            (
                ndc_cameras,
                screen_cameras,
            ) = TestCamerasCommon.init_equiv_cameras_ndc_screen(cam_type, batch_size)
            # xyz - the ground truth point cloud in Py3D space
            xy = torch.randn(batch_size, num_points, 2) * 0.3
            z = torch.rand(batch_size, num_points, 1) + 3.0 + 0.1
            xyz = torch.cat((xy, z), dim=2)
            # project points
            xyz_ndc = ndc_cameras.transform_points_ndc(xyz)
            xyz_screen = screen_cameras.transform_points_ndc(xyz)
            # check correctness
            self.assertClose(xyz_ndc, xyz_screen, atol=1e-5)

    def test_clone(self, batch_size: int = 10):
        """
        Checks the clone function of the cameras.
        """
        for cam_type in (
            SfMOrthographicCameras,
            OpenGLPerspectiveCameras,
            OpenGLOrthographicCameras,
            SfMPerspectiveCameras,
            FoVOrthographicCameras,
            FoVPerspectiveCameras,
            OrthographicCameras,
            PerspectiveCameras,
        ):
            cameras = init_random_cameras(cam_type, batch_size)
            cameras = cameras.to(torch.device("cpu"))
            cameras_clone = cameras.clone()

            for var in cameras.__dict__.keys():
                val = getattr(cameras, var)
                val_clone = getattr(cameras_clone, var)
                if torch.is_tensor(val):
                    self.assertClose(val, val_clone)
                    self.assertSeparate(val, val_clone)
                else:
                    self.assertTrue(val == val_clone)

    def test_join_cameras_as_batch_errors(self):
        cam0 = PerspectiveCameras(device="cuda:0")
        cam1 = OrthographicCameras(device="cuda:0")

        # Cameras not of the same type
        with self.assertRaisesRegex(ValueError, "same type"):
            join_cameras_as_batch([cam0, cam1])

        cam2 = OrthographicCameras(device="cpu")
        # Cameras not on the same device
        with self.assertRaisesRegex(ValueError, "same device"):
            join_cameras_as_batch([cam1, cam2])

        cam3 = OrthographicCameras(in_ndc=False, device="cuda:0")
        # Different coordinate systems -- all should be in ndc or in screen
        with self.assertRaisesRegex(
            ValueError, "Attribute _in_ndc is not constant across inputs"
        ):
            join_cameras_as_batch([cam1, cam3])

    def join_cameras_as_batch_fov(self, camera_cls):
        R0 = torch.randn((6, 3, 3))
        R1 = torch.randn((3, 3, 3))
        cam0 = camera_cls(znear=10.0, zfar=100.0, R=R0, device="cuda:0")
        cam1 = camera_cls(znear=10.0, zfar=200.0, R=R1, device="cuda:0")

        cam_batch = join_cameras_as_batch([cam0, cam1])

        self.assertEqual(cam_batch._N, cam0._N + cam1._N)
        self.assertEqual(cam_batch.device, cam0.device)
        self.assertClose(cam_batch.R, torch.cat((R0, R1), dim=0).to(device="cuda:0"))

    def join_cameras_as_batch(self, camera_cls):
        R0 = torch.randn((6, 3, 3))
        R1 = torch.randn((3, 3, 3))
        p0 = torch.randn((6, 2, 1))
        p1 = torch.randn((3, 2, 1))
        f0 = 5.0
        f1 = torch.randn(3, 2)
        f2 = torch.randn(3, 1)
        cam0 = camera_cls(
            R=R0,
            focal_length=f0,
            principal_point=p0,
        )
        cam1 = camera_cls(
            R=R1,
            focal_length=f0,
            principal_point=p1,
        )
        cam2 = camera_cls(
            R=R1,
            focal_length=f1,
            principal_point=p1,
        )
        cam3 = camera_cls(
            R=R1,
            focal_length=f2,
            principal_point=p1,
        )
        cam_batch = join_cameras_as_batch([cam0, cam1])

        self.assertEqual(cam_batch._N, cam0._N + cam1._N)
        self.assertEqual(cam_batch.device, cam0.device)
        self.assertClose(cam_batch.R, torch.cat((R0, R1), dim=0))
        self.assertClose(cam_batch.principal_point, torch.cat((p0, p1), dim=0))
        self.assertEqual(cam_batch._in_ndc, cam0._in_ndc)

        # Test one broadcasted value and one fixed value
        # Focal length as (N,) in one camera and (N, 2) in the other
        cam_batch = join_cameras_as_batch([cam0, cam2])
        self.assertEqual(cam_batch._N, cam0._N + cam2._N)
        self.assertClose(cam_batch.R, torch.cat((R0, R1), dim=0))
        self.assertClose(
            cam_batch.focal_length,
            torch.cat([torch.tensor([[f0, f0]]).expand(6, -1), f1], dim=0),
        )

        # Focal length as (N, 1) in one camera and (N, 2) in the other
        cam_batch = join_cameras_as_batch([cam2, cam3])
        self.assertClose(
            cam_batch.focal_length,
            torch.cat([f1, f2.expand(-1, 2)], dim=0),
        )

    def test_join_batch_perspective(self):
        self.join_cameras_as_batch_fov(FoVPerspectiveCameras)
        self.join_cameras_as_batch(PerspectiveCameras)

    def test_join_batch_orthographic(self):
        self.join_cameras_as_batch_fov(FoVOrthographicCameras)
        self.join_cameras_as_batch(OrthographicCameras)

    def test_iterable(self):
        for camera_type in [PerspectiveCameras, OrthographicCameras]:
            a_list = list(camera_type())
            self.assertEqual(len(a_list), 1)


############################################################
#                FoVPerspective Camera                     #
############################################################


class TestFoVPerspectiveProjection(TestCaseMixin, unittest.TestCase):
    def test_perspective(self):
        far = 10.0
        near = 1.0
        cameras = FoVPerspectiveCameras(znear=near, zfar=far, fov=60.0)
        P = cameras.get_projection_transform()
        # vertices are at the far clipping plane so z gets mapped to 1.
        vertices = torch.tensor([1, 2, far], dtype=torch.float32)
        projected_verts = torch.tensor(
            [np.sqrt(3) / far, 2 * np.sqrt(3) / far, 1.0], dtype=torch.float32
        )
        vertices = vertices[None, None, :]
        v1 = P.transform_points(vertices)
        v2 = perspective_project_naive(vertices, fov=60.0)
        self.assertClose(v1[..., :2], v2[..., :2])
        self.assertClose(far * v1[..., 2], v2[..., 2])
        self.assertClose(v1.squeeze(), projected_verts)

        # vertices are at the near clipping plane so z gets mapped to 0.0.
        vertices[..., 2] = near
        projected_verts = torch.tensor(
            [np.sqrt(3) / near, 2 * np.sqrt(3) / near, 0.0], dtype=torch.float32
        )
        v1 = P.transform_points(vertices)
        v2 = perspective_project_naive(vertices, fov=60.0)
        self.assertClose(v1[..., :2], v2[..., :2])
        self.assertClose(v1.squeeze(), projected_verts)

    def test_perspective_kwargs(self):
        cameras = FoVPerspectiveCameras(znear=5.0, zfar=100.0, fov=0.0)
        # Override defaults by passing in values to get_projection_transform
        far = 10.0
        P = cameras.get_projection_transform(znear=1.0, zfar=far, fov=60.0)
        vertices = torch.tensor([1, 2, far], dtype=torch.float32)
        projected_verts = torch.tensor(
            [np.sqrt(3) / far, 2 * np.sqrt(3) / far, 1.0], dtype=torch.float32
        )
        vertices = vertices[None, None, :]
        v1 = P.transform_points(vertices)
        self.assertClose(v1.squeeze(), projected_verts)

    def test_perspective_mixed_inputs_broadcast(self):
        far = torch.tensor([10.0, 20.0], dtype=torch.float32)
        near = 1.0
        fov = torch.tensor(60.0)
        cameras = FoVPerspectiveCameras(znear=near, zfar=far, fov=fov)
        P = cameras.get_projection_transform()
        vertices = torch.tensor([1, 2, 10], dtype=torch.float32)
        z1 = 1.0  # vertices at far clipping plane so z = 1.0
        z2 = (20.0 / (20.0 - 1.0) * 10.0 + -20.0 / (20.0 - 1.0)) / 10.0
        projected_verts = torch.tensor(
            [
                [np.sqrt(3) / 10.0, 2 * np.sqrt(3) / 10.0, z1],
                [np.sqrt(3) / 10.0, 2 * np.sqrt(3) / 10.0, z2],
            ],
            dtype=torch.float32,
        )
        vertices = vertices[None, None, :]
        v1 = P.transform_points(vertices)
        v2 = perspective_project_naive(vertices, fov=60.0)
        self.assertClose(v1[..., :2], torch.cat([v2, v2])[..., :2])
        self.assertClose(v1.squeeze(), projected_verts)

    def test_perspective_mixed_inputs_grad(self):
        far = torch.tensor([10.0])
        near = 1.0
        fov = torch.tensor(60.0, requires_grad=True)
        cameras = FoVPerspectiveCameras(znear=near, zfar=far, fov=fov)
        P = cameras.get_projection_transform()
        vertices = torch.tensor([1, 2, 10], dtype=torch.float32)
        vertices_batch = vertices[None, None, :]
        v1 = P.transform_points(vertices_batch).squeeze()
        v1.sum().backward()
        self.assertTrue(hasattr(fov, "grad"))
        fov_grad = fov.grad.clone()
        half_fov_rad = (math.pi / 180.0) * fov.detach() / 2.0
        grad_cotan = -(1.0 / (torch.sin(half_fov_rad) ** 2.0) * 1 / 2.0)
        grad_fov = (math.pi / 180.0) * grad_cotan
        grad_fov = (vertices[0] + vertices[1]) * grad_fov / 10.0
        self.assertClose(fov_grad, grad_fov)

    def test_camera_class_init(self):
        device = torch.device("cuda:0")
        cam = FoVPerspectiveCameras(znear=10.0, zfar=(100.0, 200.0))

        # Check broadcasting
        self.assertTrue(cam.znear.shape == (2,))
        self.assertTrue(cam.zfar.shape == (2,))

        # Test to
        new_cam = cam.to(device=device)
        self.assertTrue(new_cam.device == device)

    def test_getitem(self):
        N_CAMERAS = 6
        R_matrix = torch.randn((N_CAMERAS, 3, 3))
        cam = FoVPerspectiveCameras(znear=10.0, zfar=100.0, R=R_matrix)

        # Check get item returns an instance of the same class
        # with all the same keys
        c0 = cam[0]
        self.assertTrue(isinstance(c0, FoVPerspectiveCameras))
        self.assertEqual(cam.__dict__.keys(), c0.__dict__.keys())

        # Check all fields correct in get item with int index
        self.assertEqual(len(c0), 1)
        self.assertClose(c0.zfar, torch.tensor([100.0]))
        self.assertClose(c0.znear, torch.tensor([10.0]))
        self.assertClose(c0.R, R_matrix[0:1, ...])
        self.assertEqual(c0.device, torch.device("cpu"))

        # Check list(int) index
        c012 = cam[[0, 1, 2]]
        self.assertEqual(len(c012), 3)
        self.assertClose(c012.zfar, torch.tensor([100.0] * 3))
        self.assertClose(c012.znear, torch.tensor([10.0] * 3))
        self.assertClose(c012.R, R_matrix[0:3, ...])

        # Check torch.LongTensor index
        SLICE = [1, 3, 5]
        index = torch.tensor(SLICE, dtype=torch.int64)
        c135 = cam[index]
        self.assertEqual(len(c135), 3)
        self.assertClose(c135.zfar, torch.tensor([100.0] * 3))
        self.assertClose(c135.znear, torch.tensor([10.0] * 3))
        self.assertClose(c135.R, R_matrix[SLICE, ...])

        # Check torch.BoolTensor index
        bool_slice = [i in SLICE for i in range(N_CAMERAS)]
        index = torch.tensor(bool_slice, dtype=torch.bool)
        c135 = cam[index]
        self.assertEqual(len(c135), 3)
        self.assertClose(c135.zfar, torch.tensor([100.0] * 3))
        self.assertClose(c135.znear, torch.tensor([10.0] * 3))
        self.assertClose(c135.R, R_matrix[SLICE, ...])

        # Check errors with get item
        with self.assertRaisesRegex(IndexError, "out of bounds"):
            cam[N_CAMERAS]

        index = torch.tensor([1, 0, 1], dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "does not match cameras"):
            cam[index]

        with self.assertRaisesRegex(ValueError, "Invalid index type"):
            cam[slice(0, 1)]

        with self.assertRaisesRegex(ValueError, "Invalid index type"):
            cam[[True, False]]

        index = torch.tensor(SLICE, dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "Invalid index type"):
            cam[index]

    def test_get_full_transform(self):
        cam = FoVPerspectiveCameras()
        T = torch.tensor([0.0, 0.0, 1.0]).view(1, -1)
        R = look_at_rotation(T)
        P = cam.get_full_projection_transform(R=R, T=T)
        self.assertTrue(isinstance(P, Transform3d))
        self.assertClose(cam.R, R)
        self.assertClose(cam.T, T)

    def test_transform_points(self):
        # Check transform_points methods works with default settings for
        # RT and P
        far = 10.0
        cam = FoVPerspectiveCameras(znear=1.0, zfar=far, fov=60.0)
        points = torch.tensor([1, 2, far], dtype=torch.float32)
        points = points.view(1, 1, 3).expand(5, 10, -1)
        projected_points = torch.tensor(
            [np.sqrt(3) / far, 2 * np.sqrt(3) / far, 1.0], dtype=torch.float32
        )
        projected_points = projected_points.view(1, 1, 3).expand(5, 10, -1)
        new_points = cam.transform_points(points)
        self.assertClose(new_points, projected_points)

    def test_perspective_type(self):
        cam = FoVPerspectiveCameras(znear=1.0, zfar=10.0, fov=60.0)
        self.assertTrue(cam.is_perspective())
        self.assertEqual(cam.get_znear(), 1.0)


############################################################
#                FoVOrthographic Camera                    #
############################################################


class TestFoVOrthographicProjection(TestCaseMixin, unittest.TestCase):
    def test_orthographic(self):
        far = 10.0
        near = 1.0
        cameras = FoVOrthographicCameras(znear=near, zfar=far)
        P = cameras.get_projection_transform()

        vertices = torch.tensor([1, 2, far], dtype=torch.float32)
        projected_verts = torch.tensor([1, 2, 1], dtype=torch.float32)
        vertices = vertices[None, None, :]
        v1 = P.transform_points(vertices)
        v2 = orthographic_project_naive(vertices)
        self.assertClose(v1[..., :2], v2[..., :2])
        self.assertClose(v1.squeeze(), projected_verts)

        vertices[..., 2] = near
        projected_verts[2] = 0.0
        v1 = P.transform_points(vertices)
        v2 = orthographic_project_naive(vertices)
        self.assertClose(v1[..., :2], v2[..., :2])
        self.assertClose(v1.squeeze(), projected_verts)

    def test_orthographic_scaled(self):
        vertices = torch.tensor([1, 2, 0.5], dtype=torch.float32)
        vertices = vertices[None, None, :]
        scale = torch.tensor([[2.0, 0.5, 20]])
        # applying the scale puts the z coordinate at the far clipping plane
        # so the z is mapped to 1.0
        projected_verts = torch.tensor([2, 1, 1], dtype=torch.float32)
        cameras = FoVOrthographicCameras(znear=1.0, zfar=10.0, scale_xyz=scale)
        P = cameras.get_projection_transform()
        v1 = P.transform_points(vertices)
        v2 = orthographic_project_naive(vertices, scale)
        self.assertClose(v1[..., :2], v2[..., :2])
        self.assertClose(v1, projected_verts[None, None])

    def test_orthographic_kwargs(self):
        cameras = FoVOrthographicCameras(znear=5.0, zfar=100.0)
        far = 10.0
        P = cameras.get_projection_transform(znear=1.0, zfar=far)
        vertices = torch.tensor([1, 2, far], dtype=torch.float32)
        projected_verts = torch.tensor([1, 2, 1], dtype=torch.float32)
        vertices = vertices[None, None, :]
        v1 = P.transform_points(vertices)
        self.assertClose(v1.squeeze(), projected_verts)

    def test_orthographic_mixed_inputs_broadcast(self):
        far = torch.tensor([10.0, 20.0])
        near = 1.0
        cameras = FoVOrthographicCameras(znear=near, zfar=far)
        P = cameras.get_projection_transform()
        vertices = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float32)
        z2 = 1.0 / (20.0 - 1.0) * 10.0 + -1.0 / (20.0 - 1.0)
        projected_verts = torch.tensor(
            [[1.0, 2.0, 1.0], [1.0, 2.0, z2]], dtype=torch.float32
        )
        vertices = vertices[None, None, :]
        v1 = P.transform_points(vertices)
        v2 = orthographic_project_naive(vertices)
        self.assertClose(v1[..., :2], torch.cat([v2, v2])[..., :2])
        self.assertClose(v1.squeeze(), projected_verts)

    def test_orthographic_mixed_inputs_grad(self):
        far = torch.tensor([10.0])
        near = 1.0
        scale = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
        cameras = FoVOrthographicCameras(znear=near, zfar=far, scale_xyz=scale)
        P = cameras.get_projection_transform()
        vertices = torch.tensor([1.0, 2.0, 10.0], dtype=torch.float32)
        vertices_batch = vertices[None, None, :]
        v1 = P.transform_points(vertices_batch)
        v1.sum().backward()
        self.assertTrue(hasattr(scale, "grad"))
        scale_grad = scale.grad.clone()
        grad_scale = torch.tensor(
            [
                [
                    vertices[0] * P._matrix[:, 0, 0],
                    vertices[1] * P._matrix[:, 1, 1],
                    vertices[2] * P._matrix[:, 2, 2],
                ]
            ]
        )
        self.assertClose(scale_grad, grad_scale)

    def test_perspective_type(self):
        cam = FoVOrthographicCameras(znear=1.0, zfar=10.0)
        self.assertFalse(cam.is_perspective())
        self.assertEqual(cam.get_znear(), 1.0)

    def test_getitem(self):
        R_matrix = torch.randn((6, 3, 3))
        scale = torch.tensor([[1.0, 1.0, 1.0]], requires_grad=True)
        cam = FoVOrthographicCameras(
            znear=10.0, zfar=100.0, R=R_matrix, scale_xyz=scale
        )

        # Check get item returns an instance of the same class
        # with all the same keys
        c0 = cam[0]
        self.assertTrue(isinstance(c0, FoVOrthographicCameras))
        self.assertEqual(cam.__dict__.keys(), c0.__dict__.keys())

        # Check torch.LongTensor index
        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        c135 = cam[index]
        self.assertEqual(len(c135), 3)
        self.assertClose(c135.zfar, torch.tensor([100.0] * 3))
        self.assertClose(c135.znear, torch.tensor([10.0] * 3))
        self.assertClose(c135.min_x, torch.tensor([-1.0] * 3))
        self.assertClose(c135.max_x, torch.tensor([1.0] * 3))
        self.assertClose(c135.R, R_matrix[[1, 3, 5], ...])
        self.assertClose(c135.scale_xyz, scale.expand(3, -1))


############################################################
#                Orthographic Camera                       #
############################################################


class TestOrthographicProjection(TestCaseMixin, unittest.TestCase):
    def test_orthographic(self):
        cameras = OrthographicCameras()
        P = cameras.get_projection_transform()

        vertices = torch.randn([3, 4, 3], dtype=torch.float32)
        projected_verts = vertices.clone()
        v1 = P.transform_points(vertices)
        v2 = orthographic_project_naive(vertices)

        self.assertClose(v1[..., :2], v2[..., :2])
        self.assertClose(v1, projected_verts)

    def test_orthographic_scaled(self):
        focal_length_x = 10.0
        focal_length_y = 15.0

        cameras = OrthographicCameras(focal_length=((focal_length_x, focal_length_y),))
        P = cameras.get_projection_transform()

        vertices = torch.randn([3, 4, 3], dtype=torch.float32)
        projected_verts = vertices.clone()
        projected_verts[:, :, 0] *= focal_length_x
        projected_verts[:, :, 1] *= focal_length_y
        v1 = P.transform_points(vertices)
        v2 = orthographic_project_naive(
            vertices, scale_xyz=(focal_length_x, focal_length_y, 1.0)
        )
        v3 = cameras.transform_points(vertices)
        self.assertClose(v1[..., :2], v2[..., :2])
        self.assertClose(v3[..., :2], v2[..., :2])
        self.assertClose(v1, projected_verts)

    def test_orthographic_kwargs(self):
        cameras = OrthographicCameras(focal_length=5.0, principal_point=((2.5, 2.5),))
        P = cameras.get_projection_transform(
            focal_length=2.0, principal_point=((2.5, 3.5),)
        )
        vertices = torch.randn([3, 4, 3], dtype=torch.float32)
        projected_verts = vertices.clone()
        projected_verts[:, :, :2] *= 2.0
        projected_verts[:, :, 0] += 2.5
        projected_verts[:, :, 1] += 3.5
        v1 = P.transform_points(vertices)
        self.assertClose(v1, projected_verts)

    def test_perspective_type(self):
        cam = OrthographicCameras(focal_length=5.0, principal_point=((2.5, 2.5),))
        self.assertFalse(cam.is_perspective())
        self.assertIsNone(cam.get_znear())

    def test_getitem(self):
        R_matrix = torch.randn((6, 3, 3))
        principal_point = torch.randn((6, 2, 1))
        focal_length = 5.0
        cam = OrthographicCameras(
            R=R_matrix,
            focal_length=focal_length,
            principal_point=principal_point,
        )

        # Check get item returns an instance of the same class
        # with all the same keys
        c0 = cam[0]
        self.assertTrue(isinstance(c0, OrthographicCameras))
        self.assertEqual(cam.__dict__.keys(), c0.__dict__.keys())

        # Check torch.LongTensor index
        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        c135 = cam[index]
        self.assertEqual(len(c135), 3)
        self.assertClose(c135.focal_length, torch.tensor([[5.0, 5.0]] * 3))
        self.assertClose(c135.R, R_matrix[[1, 3, 5], ...])
        self.assertClose(c135.principal_point, principal_point[[1, 3, 5], ...])


############################################################
#                Perspective Camera                        #
############################################################


class TestPerspectiveProjection(TestCaseMixin, unittest.TestCase):
    def test_perspective(self):
        cameras = PerspectiveCameras()
        P = cameras.get_projection_transform()

        vertices = torch.randn([3, 4, 3], dtype=torch.float32)
        v1 = P.transform_points(vertices)
        v2 = sfm_perspective_project_naive(vertices)
        self.assertClose(v1, v2)

    def test_perspective_scaled(self):
        focal_length_x = 10.0
        focal_length_y = 15.0
        p0x = 15.0
        p0y = 30.0

        cameras = PerspectiveCameras(
            focal_length=((focal_length_x, focal_length_y),),
            principal_point=((p0x, p0y),),
        )
        P = cameras.get_projection_transform()

        vertices = torch.randn([3, 4, 3], dtype=torch.float32)
        v1 = P.transform_points(vertices)
        v2 = sfm_perspective_project_naive(
            vertices, fx=focal_length_x, fy=focal_length_y, p0x=p0x, p0y=p0y
        )
        v3 = cameras.transform_points(vertices)
        self.assertClose(v1, v2)
        self.assertClose(v3[..., :2], v2[..., :2])

    def test_perspective_kwargs(self):
        cameras = PerspectiveCameras(focal_length=5.0, principal_point=((2.5, 2.5),))
        P = cameras.get_projection_transform(
            focal_length=2.0, principal_point=((2.5, 3.5),)
        )
        vertices = torch.randn([3, 4, 3], dtype=torch.float32)
        v1 = P.transform_points(vertices)
        v2 = sfm_perspective_project_naive(vertices, fx=2.0, fy=2.0, p0x=2.5, p0y=3.5)
        self.assertClose(v1, v2, atol=1e-6)

    def test_perspective_type(self):
        cam = PerspectiveCameras(focal_length=5.0, principal_point=((2.5, 2.5),))
        self.assertTrue(cam.is_perspective())
        self.assertIsNone(cam.get_znear())

    def test_getitem(self):
        R_matrix = torch.randn((6, 3, 3))
        principal_point = torch.randn((6, 2, 1))
        focal_length = 5.0
        cam = PerspectiveCameras(
            R=R_matrix,
            focal_length=focal_length,
            principal_point=principal_point,
        )

        # Check get item returns an instance of the same class
        # with all the same keys
        c0 = cam[0]
        self.assertTrue(isinstance(c0, PerspectiveCameras))
        self.assertEqual(cam.__dict__.keys(), c0.__dict__.keys())

        # Check torch.LongTensor index
        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        c135 = cam[index]
        self.assertEqual(len(c135), 3)
        self.assertClose(c135.focal_length, torch.tensor([[5.0, 5.0]] * 3))
        self.assertClose(c135.R, R_matrix[[1, 3, 5], ...])
        self.assertClose(c135.principal_point, principal_point[[1, 3, 5], ...])

        # Check in_ndc is handled correctly
        self.assertEqual(cam._in_ndc, c0._in_ndc)

    def test_clone_picklable(self):
        camera = PerspectiveCameras()
        pickle.dumps(camera)
        pickle.dumps(camera.clone())


############################################################
#                FishEye Camera                        #
############################################################


class TestFishEyeProjection(TestCaseMixin, unittest.TestCase):
    def setUpSimpleCase(self) -> None:
        super().setUp()
        focal = torch.tensor([[240]], dtype=torch.float32)
        principal_point = torch.tensor([[320, 240]])
        p_3d = torch.tensor(
            [
                [2.0, 3.0, 1.0],
                [3.0, 2.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return focal, principal_point, p_3d

    def setUpAriaCase(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        focal = torch.tensor([[608.9255557152]], dtype=torch.float32)
        principal_point = torch.tensor(
            [[712.0114821205, 706.8666571177]], dtype=torch.float32
        )
        radial_params = torch.tensor(
            [
                [
                    0.3877090026,
                    -0.315613384,
                    -0.3434984955,
                    1.8565874201,
                    -2.1799372221,
                    0.7713834763,
                ],
            ],
            dtype=torch.float32,
        )
        tangential_params = torch.tensor(
            [[-0.0002747019, 0.0005228974]], dtype=torch.float32
        )
        thin_prism_params = torch.tensor(
            [
                [0.000134884, -0.000084822, -0.0009420014, -0.0001276838],
            ],
            dtype=torch.float32,
        )
        return (
            focal,
            principal_point,
            radial_params,
            tangential_params,
            thin_prism_params,
        )

    def setUpBatchCameras(self, combination: None) -> None:
        super().setUp()
        focal, principal_point, p_3d = self.setUpSimpleCase()
        radial_params = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        tangential_params = torch.tensor([[0, 0]], dtype=torch.float32)
        thin_prism_params = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)
        (
            focal1,
            principal_point1,
            radial_params1,
            tangential_params1,
            thin_prism_params1,
        ) = self.setUpAriaCase()
        focal = torch.cat([focal, focal1], dim=0)
        principal_point = torch.cat([principal_point, principal_point1], dim=0)
        radial_params = torch.cat([radial_params, radial_params1], dim=0)
        tangential_params = torch.cat([tangential_params, tangential_params1], dim=0)
        thin_prism_params = torch.cat([thin_prism_params, thin_prism_params1], dim=0)
        if combination is None:
            combination = [True, True, True]
        cameras = FishEyeCameras(
            use_radial=combination[0],
            use_tangential=combination[1],
            use_thin_prism=combination[2],
            focal_length=focal,
            principal_point=principal_point,
            radial_params=radial_params,
            tangential_params=tangential_params,
            thin_prism_params=thin_prism_params,
        )

        return cameras

    def test_distortion_params_set_to_zeors(self):
        # test case 1: all distortion params are 0. Note that
        # setting radial_params to zeros is not equivalent to
        # disabling radial distortions, set use_radial=False does
        focal, principal_point, p_3d = self.setUpSimpleCase()
        cameras = FishEyeCameras(
            focal_length=focal,
            principal_point=principal_point,
        )
        uv_case1 = cameras.transform_points(p_3d)
        self.assertClose(
            uv_case1,
            torch.tensor(
                [[493.0993, 499.6489, 1.0], [579.6489, 413.0993, 1.0]],
            ),
        )
        # test case 2: equivalent of test case 1 by
        # disabling use_tangential and use_thin_prism
        cameras = FishEyeCameras(
            focal_length=focal,
            principal_point=principal_point,
            use_tangential=False,
            use_thin_prism=False,
        )
        uv_case2 = cameras.transform_points(p_3d)
        self.assertClose(uv_case2, uv_case1)

    def test_fisheye_against_perspective_cameras(self):
        # test case: check equivalence with PerspectiveCameras
        # by disabling all distortions
        focal, principal_point, p_3d = self.setUpSimpleCase()
        cameras = PerspectiveCameras(
            focal_length=focal,
            principal_point=principal_point,
        )
        P = cameras.get_projection_transform()
        uv_perspective = P.transform_points(p_3d)

        # disable all distortions
        cameras = FishEyeCameras(
            focal_length=focal,
            principal_point=principal_point,
            use_radial=False,
            use_tangential=False,
            use_thin_prism=False,
        )
        uv = cameras.transform_points(p_3d)
        self.assertClose(uv, uv_perspective)

    def test_project_shape_broadcasts(self):
        focal, principal_point, p_3d = self.setUpSimpleCase()
        torch.set_printoptions(precision=6)
        combinations = product([0, 1], repeat=3)
        for combination in combinations:
            cameras = FishEyeCameras(
                use_radial=combination[0],
                use_tangential=combination[1],
                use_thin_prism=combination[2],
                focal_length=focal,
                principal_point=principal_point,
            )
            # test case 1:
            # 1 transform with points of shape (P, 3) -> (P, 3)
            # 1 transform with points of shape (1, P, 3) -> (1, P, 3)
            # 1 transform with points of shape (M, P, 3) -> (M, P, 3)
            points = p_3d.repeat(1, 1, 1)
            cameras = FishEyeCameras(
                focal_length=focal,
                principal_point=principal_point,
                use_radial=False,
                use_tangential=False,
                use_thin_prism=False,
            )
            uv = cameras.transform_points(p_3d)
            uv_point_batch = cameras.transform_points(points)
            self.assertClose(uv_point_batch, uv.repeat(1, 1, 1))

        points = p_3d.repeat(3, 1, 1)
        uv_point_batch = cameras.transform_points(points)
        self.assertClose(uv_point_batch, uv.repeat(3, 1, 1))

        # test case 2
        # test with N transforms and points of shape (P, 3) -> (N, P, 3)
        # test with N transforms and points of shape (1, P, 3) -> (N, P, 3)
        torch.set_printoptions(sci_mode=False)
        p_3d = torch.tensor(
            [
                [2.0, 3.0, 1.0],
                [3.0, 2.0, 1.0],
            ]
        )
        expected_res = torch.tensor(
            [
                [
                    [
                        [800.000000, 960.000000, 1.000000],
                        [1040.000000, 720.000000, 1.000000],
                    ],
                    [
                        [1929.862549, 2533.643311, 1.000000],
                        [2538.788086, 1924.717773, 1.000000],
                    ],
                ],
                [
                    [
                        [800.000000, 960.000000, 1.000000],
                        [1040.000000, 720.000000, 1.000000],
                    ],
                    [
                        [1927.272095, 2524.220459, 1.000000],
                        [2536.197754, 1915.295166, 1.000000],
                    ],
                ],
                [
                    [
                        [800.000000, 960.000000, 1.000000],
                        [1040.000000, 720.000000, 1.000000],
                    ],
                    [
                        [1930.050293, 2538.434814, 1.000000],
                        [2537.956543, 1927.569092, 1.000000],
                    ],
                ],
                [
                    [
                        [800.000000, 960.000000, 1.000000],
                        [1040.000000, 720.000000, 1.000000],
                    ],
                    [
                        [1927.459839, 2529.011963, 1.000000],
                        [2535.366211, 1918.146484, 1.000000],
                    ],
                ],
                [
                    [
                        [493.099304, 499.648926, 1.000000],
                        [579.648926, 413.099304, 1.000000],
                    ],
                    [
                        [1662.673950, 2132.860352, 1.000000],
                        [2138.005127, 1657.529053, 1.000000],
                    ],
                ],
                [
                    [
                        [493.099304, 499.648926, 1.000000],
                        [579.648926, 413.099304, 1.000000],
                    ],
                    [
                        [1660.083496, 2123.437744, 1.000000],
                        [2135.414795, 1648.106445, 1.000000],
                    ],
                ],
                [
                    [
                        [493.099304, 499.648926, 1.000000],
                        [579.648926, 413.099304, 1.000000],
                    ],
                    [
                        [1662.861816, 2137.651855, 1.000000],
                        [2137.173828, 1660.380371, 1.000000],
                    ],
                ],
                [
                    [
                        [493.099304, 499.648926, 1.000000],
                        [579.648926, 413.099304, 1.000000],
                    ],
                    [
                        [1660.271240, 2128.229248, 1.000000],
                        [2134.583496, 1650.957764, 1.000000],
                    ],
                ],
            ]
        )
        combinations = product([0, 1], repeat=3)
        for i, combination in enumerate(combinations):
            cameras = self.setUpBatchCameras(combination)
            uv_point_batch = cameras.transform_points(p_3d)
            self.assertClose(uv_point_batch, expected_res[i])

            uv_point_batch = cameras.transform_points(p_3d.repeat(1, 1, 1))
            self.assertClose(uv_point_batch, expected_res[i].repeat(1, 1, 1))

    def test_cuda(self):
        """
        Test cuda device
        """
        focal, principal_point, p_3d = self.setUpSimpleCase()
        cameras_cuda = FishEyeCameras(
            focal_length=focal,
            principal_point=principal_point,
            device="cuda:0",
        )
        uv = cameras_cuda.transform_points(p_3d)
        expected_res = torch.tensor(
            [[493.0993, 499.6489, 1.0], [579.6489, 413.0993, 1.0]],
        )
        self.assertClose(uv, expected_res.to("cuda:0"))

        rep_3d = cameras_cuda.unproject_points(uv)
        self.assertClose(rep_3d, p_3d.to("cuda:0"))

    def test_unproject_shape_broadcasts(self):
        # test case 1:
        # 1 transform with points of (P, 3) -> (P, 3)
        # 1 transform with points of (M, P, 3) -> (M, P, 3)
        (
            focal,
            principal_point,
            radial_params,
            tangential_params,
            thin_prism_params,
        ) = self.setUpAriaCase()
        xy_depth = torch.tensor(
            [
                [2134.5814033, 1650.95653328, 1.0],
                [1074.25442904, 1159.52461285, 1.0],
            ]
        )
        cameras = FishEyeCameras(
            focal_length=focal,
            principal_point=principal_point,
            radial_params=radial_params,
            tangential_params=tangential_params,
            thin_prism_params=thin_prism_params,
        )
        rep_3d = cameras.unproject_points(xy_depth)
        expected_res = torch.tensor(
            [
                [[2.999442, 1.990583, 1.000000], [0.666728, 0.833142, 1.000000]],
                [[2.997338, 2.005411, 1.000000], [0.666859, 0.834456, 1.000000]],
                [[3.002090, 1.985229, 1.000000], [0.666537, 0.832025, 1.000000]],
                [[2.999999, 2.000000, 1.000000], [0.666667, 0.833333, 1.000000]],
                [[2.999442, 1.990583, 1.000000], [0.666728, 0.833142, 1.000000]],
                [[2.997338, 2.005411, 1.000000], [0.666859, 0.834456, 1.000000]],
                [[3.002090, 1.985229, 1.000000], [0.666537, 0.832025, 1.000000]],
                [[2.999999, 2.000000, 1.000000], [0.666667, 0.833333, 1.000000]],
            ]
        )
        torch.set_printoptions(precision=6)
        combinations = product([0, 1], repeat=3)
        for i, combination in enumerate(combinations):
            cameras = FishEyeCameras(
                use_radial=combination[0],
                use_tangential=combination[1],
                use_thin_prism=combination[2],
                focal_length=focal,
                principal_point=principal_point,
                radial_params=radial_params,
                tangential_params=tangential_params,
                thin_prism_params=thin_prism_params,
            )
            rep_3d = cameras.unproject_points(xy_depth)
            self.assertClose(rep_3d, expected_res[i])
            rep_3d = cameras.unproject_points(xy_depth.repeat(3, 1, 1))
            self.assertClose(rep_3d, expected_res[i].repeat(3, 1, 1))

            # test case 2:
            # N transforms with points of (P, 3) -> (N, P, 3)
            # N transforms with points of (1, P, 3) -> (N, P, 3)
            cameras = FishEyeCameras(
                use_radial=combination[0],
                use_tangential=combination[1],
                use_thin_prism=combination[2],
                focal_length=focal.repeat(2, 1),
                principal_point=principal_point.repeat(2, 1),
                radial_params=radial_params.repeat(2, 1),
                tangential_params=tangential_params.repeat(2, 1),
                thin_prism_params=thin_prism_params.repeat(2, 1),
            )
            rep_3d = cameras.unproject_points(xy_depth)
            self.assertClose(rep_3d, expected_res[i].repeat(2, 1, 1))

    def test_unhandled_shape(self):
        """
        Test error handling when shape of transforms
        and points are not expected.
        """
        cameras = self.setUpBatchCameras(None)
        points = torch.rand(3, 3, 1)
        with self.assertRaises(ValueError):
            cameras.transform_points(points)

    def test_getitem(self):
        # Check get item returns an instance of the same class
        # with all the same keys
        cam = self.setUpBatchCameras(None)
        c0 = cam[0]
        self.assertTrue(isinstance(c0, FishEyeCameras))
        self.assertEqual(cam.__dict__.keys(), c0.__dict__.keys())
