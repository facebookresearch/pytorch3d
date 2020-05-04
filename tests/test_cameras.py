# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


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
import unittest

import numpy as np
import torch
from common_testing import TestCaseMixin
from pytorch3d.renderer.cameras import (
    CamerasBase,
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    SfMOrthographicCameras,
    SfMPerspectiveCameras,
    camera_position_from_spherical_angles,
    get_world_to_view_transform,
    look_at_rotation,
    look_at_view_transform,
)
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms.so3 import so3_exponential_map


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


class TestCameraHelpers(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

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


class TestCamerasCommon(TestCaseMixin, unittest.TestCase):
    def test_view_transform_class_method(self):
        T = torch.tensor([0.0, 0.0, -1.0], requires_grad=True).view(1, -1)
        R = look_at_rotation(T)
        RT = get_world_to_view_transform(R=R, T=T)
        for cam_type in (
            OpenGLPerspectiveCameras,
            OpenGLOrthographicCameras,
            SfMOrthographicCameras,
            SfMPerspectiveCameras,
        ):
            cam = cam_type(R=R, T=T)
            RT_class = cam.get_world_to_view_transform()
            self.assertTrue(torch.allclose(RT.get_matrix(), RT_class.get_matrix()))

        self.assertTrue(isinstance(RT, Transform3d))

    def test_get_camera_center(self, batch_size=10):
        T = torch.randn(batch_size, 3)
        R = so3_exponential_map(torch.randn(batch_size, 3) * 3.0)
        for cam_type in (
            OpenGLPerspectiveCameras,
            OpenGLOrthographicCameras,
            SfMOrthographicCameras,
            SfMPerspectiveCameras,
        ):
            cam = cam_type(R=R, T=T)
            C = cam.get_camera_center()
            C_ = -torch.bmm(R, T[:, :, None])[:, :, 0]
            self.assertTrue(torch.allclose(C, C_, atol=1e-05))

    @staticmethod
    def init_random_cameras(cam_type: CamerasBase, batch_size: int):
        cam_params = {}
        T = torch.randn(batch_size, 3) * 0.03
        T[:, 2] = 4
        R = so3_exponential_map(torch.randn(batch_size, 3) * 3.0)
        cam_params = {"R": R, "T": T}
        if cam_type in (OpenGLPerspectiveCameras, OpenGLOrthographicCameras):
            cam_params["znear"] = torch.rand(batch_size) * 10 + 0.1
            cam_params["zfar"] = torch.rand(batch_size) * 4 + 1 + cam_params["znear"]
            if cam_type == OpenGLPerspectiveCameras:
                cam_params["fov"] = torch.rand(batch_size) * 60 + 30
                cam_params["aspect_ratio"] = torch.rand(batch_size) * 0.5 + 0.5
            else:
                cam_params["top"] = torch.rand(batch_size) * 0.2 + 0.9
                cam_params["bottom"] = -(torch.rand(batch_size)) * 0.2 - 0.9
                cam_params["left"] = -(torch.rand(batch_size)) * 0.2 - 0.9
                cam_params["right"] = torch.rand(batch_size) * 0.2 + 0.9
        elif cam_type in (SfMOrthographicCameras, SfMPerspectiveCameras):
            cam_params["focal_length"] = torch.rand(batch_size) * 10 + 0.1
            cam_params["principal_point"] = torch.randn((batch_size, 2))
        else:
            raise ValueError(str(cam_type))
        return cam_type(**cam_params)

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
        ):
            # init the cameras
            cameras = TestCamerasCommon.init_random_cameras(cam_type, batch_size)
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

                # if we have OpenGL cameras
                # test for scaled_depth_input=True/False
                if cam_type in (OpenGLPerspectiveCameras, OpenGLOrthographicCameras):
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

    def test_clone(self, batch_size: int = 10):
        """
        Checks the clone function of the cameras.
        """
        for cam_type in (
            SfMOrthographicCameras,
            OpenGLPerspectiveCameras,
            OpenGLOrthographicCameras,
            SfMPerspectiveCameras,
        ):
            cameras = TestCamerasCommon.init_random_cameras(cam_type, batch_size)
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


class TestPerspectiveProjection(TestCaseMixin, unittest.TestCase):
    def test_perspective(self):
        far = 10.0
        near = 1.0
        cameras = OpenGLPerspectiveCameras(znear=near, zfar=far, fov=60.0)
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
        cameras = OpenGLPerspectiveCameras(znear=5.0, zfar=100.0, fov=0.0)
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
        cameras = OpenGLPerspectiveCameras(znear=near, zfar=far, fov=fov)
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
        cameras = OpenGLPerspectiveCameras(znear=near, zfar=far, fov=fov)
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
        cam = OpenGLPerspectiveCameras(znear=10.0, zfar=(100.0, 200.0))

        # Check broadcasting
        self.assertTrue(cam.znear.shape == (2,))
        self.assertTrue(cam.zfar.shape == (2,))

        # update znear element 1
        cam[1].znear = 20.0
        self.assertTrue(cam.znear[1] == 20.0)

        # Get item and get value
        c0 = cam[0]
        self.assertTrue(c0.zfar == 100.0)

        # Test to
        new_cam = cam.to(device=device)
        self.assertTrue(new_cam.device == device)

    def test_get_full_transform(self):
        cam = OpenGLPerspectiveCameras()
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
        cam = OpenGLPerspectiveCameras(znear=1.0, zfar=far, fov=60.0)
        points = torch.tensor([1, 2, far], dtype=torch.float32)
        points = points.view(1, 1, 3).expand(5, 10, -1)
        projected_points = torch.tensor(
            [np.sqrt(3) / far, 2 * np.sqrt(3) / far, 1.0], dtype=torch.float32
        )
        projected_points = projected_points.view(1, 1, 3).expand(5, 10, -1)
        new_points = cam.transform_points(points)
        self.assertClose(new_points, projected_points)


class TestOpenGLOrthographicProjection(TestCaseMixin, unittest.TestCase):
    def test_orthographic(self):
        far = 10.0
        near = 1.0
        cameras = OpenGLOrthographicCameras(znear=near, zfar=far)
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
        cameras = OpenGLOrthographicCameras(znear=1.0, zfar=10.0, scale_xyz=scale)
        P = cameras.get_projection_transform()
        v1 = P.transform_points(vertices)
        v2 = orthographic_project_naive(vertices, scale)
        self.assertClose(v1[..., :2], v2[..., :2])
        self.assertClose(v1, projected_verts[None, None])

    def test_orthographic_kwargs(self):
        cameras = OpenGLOrthographicCameras(znear=5.0, zfar=100.0)
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
        cameras = OpenGLOrthographicCameras(znear=near, zfar=far)
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
        cameras = OpenGLOrthographicCameras(znear=near, zfar=far, scale_xyz=scale)
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


class TestSfMOrthographicProjection(TestCaseMixin, unittest.TestCase):
    def test_orthographic(self):
        cameras = SfMOrthographicCameras()
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

        cameras = SfMOrthographicCameras(
            focal_length=((focal_length_x, focal_length_y),)
        )
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
        cameras = SfMOrthographicCameras(
            focal_length=5.0, principal_point=((2.5, 2.5),)
        )
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


class TestSfMPerspectiveProjection(TestCaseMixin, unittest.TestCase):
    def test_perspective(self):
        cameras = SfMPerspectiveCameras()
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

        cameras = SfMPerspectiveCameras(
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
        cameras = SfMPerspectiveCameras(focal_length=5.0, principal_point=((2.5, 2.5),))
        P = cameras.get_projection_transform(
            focal_length=2.0, principal_point=((2.5, 3.5),)
        )
        vertices = torch.randn([3, 4, 3], dtype=torch.float32)
        v1 = P.transform_points(vertices)
        v2 = sfm_perspective_project_naive(vertices, fx=2.0, fy=2.0, p0x=2.5, p0y=3.5)
        self.assertClose(v1, v2, atol=1e-6)
