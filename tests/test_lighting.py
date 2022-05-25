# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from pytorch3d.renderer.lighting import AmbientLights, DirectionalLights, PointLights
from pytorch3d.transforms import RotateAxisAngle

from .common_testing import TestCaseMixin


class TestLights(TestCaseMixin, unittest.TestCase):
    def test_init_lights(self):
        """
        Initialize Lights class with the default values.
        """
        device = torch.device("cuda:0")
        light = DirectionalLights(device=device)
        keys = ["ambient_color", "diffuse_color", "specular_color", "direction"]
        for k in keys:
            prop = getattr(light, k)
            self.assertTrue(torch.is_tensor(prop))
            self.assertTrue(prop.device == device)
            self.assertTrue(prop.shape == (1, 3))

        light = PointLights(device=device)
        keys = ["ambient_color", "diffuse_color", "specular_color", "location"]
        for k in keys:
            prop = getattr(light, k)
            self.assertTrue(torch.is_tensor(prop))
            self.assertTrue(prop.device == device)
            self.assertTrue(prop.shape == (1, 3))

    def test_lights_clone_to(self):
        device = torch.device("cuda:0")
        cpu = torch.device("cpu")
        light = DirectionalLights()
        new_light = light.clone().to(device)
        keys = ["ambient_color", "diffuse_color", "specular_color", "direction"]
        for k in keys:
            prop = getattr(light, k)
            new_prop = getattr(new_light, k)
            self.assertTrue(prop.device == cpu)
            self.assertTrue(new_prop.device == device)
            self.assertSeparate(new_prop, prop)

        light = PointLights()
        new_light = light.clone().to(device)
        keys = ["ambient_color", "diffuse_color", "specular_color", "location"]
        for k in keys:
            prop = getattr(light, k)
            new_prop = getattr(new_light, k)
            self.assertTrue(prop.device == cpu)
            self.assertTrue(new_prop.device == device)
            self.assertSeparate(new_prop, prop)

    def test_lights_accessor(self):
        d_light = DirectionalLights(ambient_color=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        p_light = PointLights(ambient_color=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
        for light in [d_light, p_light]:
            # Update element
            color = (0.5, 0.5, 0.5)
            light[1].ambient_color = color
            self.assertClose(light.ambient_color[1], torch.tensor(color))
            # Get item and get value
            l0 = light[0]
            self.assertClose(l0.ambient_color, torch.tensor((0.0, 0.0, 0.0)))

    def test_initialize_lights_broadcast(self):
        light = DirectionalLights(
            ambient_color=torch.randn(10, 3),
            diffuse_color=torch.randn(1, 3),
            specular_color=torch.randn(1, 3),
        )
        keys = ["ambient_color", "diffuse_color", "specular_color", "direction"]
        for k in keys:
            prop = getattr(light, k)
            self.assertTrue(prop.shape == (10, 3))

        light = PointLights(
            ambient_color=torch.randn(10, 3),
            diffuse_color=torch.randn(1, 3),
            specular_color=torch.randn(1, 3),
        )
        keys = ["ambient_color", "diffuse_color", "specular_color", "location"]
        for k in keys:
            prop = getattr(light, k)
            self.assertTrue(prop.shape == (10, 3))

    def test_initialize_lights_broadcast_fail(self):
        """
        Batch dims have to be the same or 1.
        """
        with self.assertRaises(ValueError):
            DirectionalLights(
                ambient_color=torch.randn(10, 3), diffuse_color=torch.randn(15, 3)
            )

        with self.assertRaises(ValueError):
            PointLights(
                ambient_color=torch.randn(10, 3), diffuse_color=torch.randn(15, 3)
            )

    def test_initialize_lights_dimensions_fail(self):
        """
        Color should have shape (N, 3) or (1, 3)
        """
        with self.assertRaises(ValueError):
            DirectionalLights(ambient_color=torch.randn(10, 4))

        with self.assertRaises(ValueError):
            DirectionalLights(direction=torch.randn(10, 4))

        with self.assertRaises(ValueError):
            PointLights(ambient_color=torch.randn(10, 4))

        with self.assertRaises(ValueError):
            PointLights(location=torch.randn(10, 4))

    def test_initialize_ambient(self):
        N = 13
        color = 0.8 * torch.ones((N, 3))
        lights = AmbientLights(ambient_color=color)
        self.assertEqual(len(lights), N)
        self.assertClose(lights.ambient_color, color)

        lights = AmbientLights(ambient_color=color[:1])
        self.assertEqual(len(lights), 1)
        self.assertClose(lights.ambient_color, color[:1])


class TestDiffuseLighting(TestCaseMixin, unittest.TestCase):
    def test_diffuse_directional_lights(self):
        """
        Test with a single point where:
        1) the normal and light direction are 45 degrees apart.
        2) the normal and light direction are 90 degrees apart. The output
           should be zero for this case
        """
        color = torch.tensor([1, 1, 1], dtype=torch.float32)
        direction = torch.tensor(
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32
        )
        normals = torch.tensor([0, 0, 1], dtype=torch.float32)
        normals = normals[None, None, :]
        expected_output = torch.tensor(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32
        )
        expected_output = expected_output.view(1, 1, 3).repeat(3, 1, 1)
        light = DirectionalLights(diffuse_color=color, direction=direction)
        output_light = light.diffuse(normals=normals)
        self.assertClose(output_light, expected_output)

        # Change light direction to be 90 degrees apart from normal direction.
        direction = torch.tensor([0, 1, 0], dtype=torch.float32)
        light.direction = direction
        expected_output = torch.zeros_like(expected_output)
        output_light = light.diffuse(normals=normals)
        self.assertClose(output_light, expected_output)

    def test_diffuse_point_lights(self):
        """
        Test with a single point at the origin. Test two cases:
        1) the point light is at (1, 0, 1) hence the light direction is 45
           degrees apart from the normal direction
        1) the point light is at (0, 1, 0) hence the light direction is 90
           degrees apart from the normal direction. The output
           should be zero for this case
        """
        color = torch.tensor([1, 1, 1], dtype=torch.float32)
        location = torch.tensor(
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32
        )
        points = torch.tensor([0, 0, 0], dtype=torch.float32)
        normals = torch.tensor([0, 0, 1], dtype=torch.float32)
        expected_output = torch.tensor(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32
        )
        expected_output = expected_output.view(-1, 1, 3)
        light = PointLights(diffuse_color=color[None, :], location=location[None, :])
        output_light = light.diffuse(
            points=points[None, None, :], normals=normals[None, None, :]
        )
        self.assertClose(output_light, expected_output)

        # Change light direction to be 90 degrees apart from normal direction.
        location = torch.tensor([0, 1, 0], dtype=torch.float32)
        expected_output = torch.zeros_like(expected_output)
        light = PointLights(diffuse_color=color[None, :], location=location[None, :])
        output_light = light.diffuse(
            points=points[None, None, :], normals=normals[None, None, :]
        )
        self.assertClose(output_light, expected_output)

    def test_diffuse_batched(self):
        """
        Test with a batch where each batch element has one point
        where the normal and light direction are 45 degrees apart.
        """
        batch_size = 10
        color = torch.tensor([1, 1, 1], dtype=torch.float32)
        direction = torch.tensor(
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32
        )
        normals = torch.tensor([0, 0, 1], dtype=torch.float32)
        expected_out = torch.tensor(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32
        )

        # Reshape
        direction = direction.view(-1, 3).expand(batch_size, -1)
        normals = normals.view(-1, 1, 3).expand(batch_size, -1, -1)
        color = color.view(-1, 3).expand(batch_size, -1)
        expected_out = expected_out.view(-1, 1, 3).expand(batch_size, 1, 3)

        lights = DirectionalLights(diffuse_color=color, direction=direction)
        output_light = lights.diffuse(normals=normals)
        self.assertClose(output_light, expected_out)

    def test_diffuse_batched_broadcast_inputs(self):
        """
        Test with a batch where each batch element has one point
        where the normal and light direction are 45 degrees apart.
        The color and direction are the same for each batch element.
        """
        batch_size = 10
        color = torch.tensor([1, 1, 1], dtype=torch.float32)
        direction = torch.tensor(
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32
        )
        normals = torch.tensor([0, 0, 1], dtype=torch.float32)
        expected_out = torch.tensor(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32
        )

        # Reshape
        normals = normals.view(-1, 1, 3).expand(batch_size, -1, -1)
        expected_out = expected_out.view(-1, 1, 3).expand(batch_size, 1, 3)

        # Don't expand the direction or color. Broadcasting should happen
        # in the diffuse function.
        direction = direction.view(1, 3)
        color = color.view(1, 3)

        lights = DirectionalLights(diffuse_color=color, direction=direction)
        output_light = lights.diffuse(normals=normals)
        self.assertClose(output_light, expected_out)

    def test_diffuse_batched_arbitrary_input_dims(self):
        """
        Test with a batch of inputs where shape of the input is mimicking the
        shape in a shading function i.e. an interpolated normal per pixel for
        top K faces per pixel.
        """
        N, H, W, K = 16, 256, 256, 100
        device = torch.device("cuda:0")
        color = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
        direction = torch.tensor(
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=torch.float32, device=device
        )
        normals = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        normals = normals.view(1, 1, 1, 1, 3).expand(N, H, W, K, -1)
        direction = direction.view(1, 3)
        color = color.view(1, 3)
        expected_output = torch.tensor(
            [1 / np.sqrt(2), 1 / np.sqrt(2), 1 / np.sqrt(2)],
            dtype=torch.float32,
            device=device,
        )
        expected_output = expected_output.view(1, 1, 1, 1, 3)
        expected_output = expected_output.expand(N, H, W, K, -1)

        lights = DirectionalLights(diffuse_color=color, direction=direction)
        output_light = lights.diffuse(normals=normals)
        self.assertClose(output_light, expected_output)

    def test_diffuse_batched_packed(self):
        """
        Test with a batch of 2 meshes each of which has faces on a single plane.
        The normal and light direction are 45 degrees apart for the first mesh
        and 90 degrees apart for the second mesh.

        The points and normals are in the packed format i.e. no batch dimension.
        """
        verts_packed = torch.rand((10, 3))  # points aren't used
        faces_per_mesh = [6, 4]
        mesh_to_vert_idx = [0] * faces_per_mesh[0] + [1] * faces_per_mesh[1]
        mesh_to_vert_idx = torch.tensor(mesh_to_vert_idx, dtype=torch.int64)
        color = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float32)
        direction = torch.tensor(
            [
                [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                [0, 1, 0],  # 90 degrees to normal so zero diffuse light
            ],
            dtype=torch.float32,
        )
        normals = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.float32)
        expected_output = torch.zeros_like(verts_packed, dtype=torch.float32)
        expected_output[:6, :] += 1 / np.sqrt(2)
        expected_output[6:, :] = 0.0
        lights = DirectionalLights(
            diffuse_color=color[mesh_to_vert_idx, :],
            direction=direction[mesh_to_vert_idx, :],
        )
        output_light = lights.diffuse(normals=normals[mesh_to_vert_idx, :])
        self.assertClose(output_light, expected_output)


class TestSpecularLighting(TestCaseMixin, unittest.TestCase):
    def test_specular_directional_lights(self):
        """
        Specular highlights depend on the camera position as well as the light
        position/direction.
        Test with a single point where:
        1) the normal and light direction are -45 degrees apart and the normal
           and camera position are +45 degrees apart. The reflected light ray
           will be perfectly aligned with the camera so the output is 1.0.
        2) the normal and light direction are -45 degrees apart and the
           camera position is behind the point. The output should be zero for
           this case.
        """
        color = torch.tensor([1, 0, 1], dtype=torch.float32)
        direction = torch.tensor(
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        camera_position = torch.tensor(
            [+1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        points = torch.tensor([0, 0, 0], dtype=torch.float32)
        normals = torch.tensor([0, 1, 0], dtype=torch.float32)
        expected_output = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        expected_output = expected_output.view(1, 1, 3).repeat(3, 1, 1)
        lights = DirectionalLights(specular_color=color, direction=direction)
        output_light = lights.specular(
            points=points[None, None, :],
            normals=normals[None, None, :],
            camera_position=camera_position[None, :],
            shininess=torch.tensor(10),
        )
        self.assertClose(output_light, expected_output)

        # Change camera position to be behind the point.
        camera_position = torch.tensor(
            [+1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=torch.float32
        )
        expected_output = torch.zeros_like(expected_output)
        output_light = lights.specular(
            points=points[None, None, :],
            normals=normals[None, None, :],
            camera_position=camera_position[None, :],
            shininess=torch.tensor(10),
        )
        self.assertClose(output_light, expected_output)

    def test_specular_point_lights(self):
        """
        Replace directional lights with point lights and check the output
        is the same.

        Test an additional case where the angle between the light reflection
        direction and the view direction is 30 degrees.
        """
        color = torch.tensor([1, 0, 1], dtype=torch.float32)
        location = torch.tensor([-1, 1, 0], dtype=torch.float32)
        camera_position = torch.tensor(
            [+1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        points = torch.tensor([0, 0, 0], dtype=torch.float32)
        normals = torch.tensor([0, 1, 0], dtype=torch.float32)
        expected_output = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
        expected_output = expected_output.view(-1, 1, 3)
        lights = PointLights(specular_color=color[None, :], location=location[None, :])
        output_light = lights.specular(
            points=points[None, None, :],
            normals=normals[None, None, :],
            camera_position=camera_position[None, :],
            shininess=torch.tensor(10),
        )
        self.assertClose(output_light, expected_output)

        # Change camera position to be behind the point
        camera_position = torch.tensor(
            [+1 / np.sqrt(2), -1 / np.sqrt(2), 0], dtype=torch.float32
        )
        expected_output = torch.zeros_like(expected_output)
        output_light = lights.specular(
            points=points[None, None, :],
            normals=normals[None, None, :],
            camera_position=camera_position[None, :],
            shininess=torch.tensor(10),
        )
        self.assertClose(output_light, expected_output)

        # Change camera direction to be 30 degrees from the reflection direction
        camera_position = torch.tensor(
            [+1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        rotate_30 = RotateAxisAngle(-30, axis="z")
        camera_position = rotate_30.transform_points(camera_position[None, :])
        expected_output = torch.tensor(
            [np.cos(30.0 * np.pi / 180), 0.0, np.cos(30.0 * np.pi / 180)],
            dtype=torch.float32,
        )
        expected_output = expected_output.view(-1, 1, 3)
        output_light = lights.specular(
            points=points[None, None, :],
            normals=normals[None, None, :],
            camera_position=camera_position[None, :],
            shininess=torch.tensor(10),
        )
        self.assertClose(output_light, expected_output**10)

    def test_specular_batched(self):
        batch_size = 10
        color = torch.tensor([1, 0, 1], dtype=torch.float32)
        direction = torch.tensor(
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        camera_position = torch.tensor(
            [+1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        points = torch.tensor([0, 0, 0], dtype=torch.float32)
        normals = torch.tensor([0, 1, 0], dtype=torch.float32)
        expected_out = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

        # Reshape
        direction = direction.view(1, 3).expand(batch_size, -1)
        camera_position = camera_position.view(1, 3).expand(batch_size, -1)
        normals = normals.view(1, 1, 3).expand(batch_size, -1, -1)
        points = points.view(1, 1, 3).expand(batch_size, -1, -1)
        color = color.view(1, 3).expand(batch_size, -1)
        expected_out = expected_out.view(1, 1, 3).expand(batch_size, 1, 3)

        lights = DirectionalLights(specular_color=color, direction=direction)
        output_light = lights.specular(
            points=points,
            normals=normals,
            camera_position=camera_position,
            shininess=torch.tensor(10),
        )
        self.assertClose(output_light, expected_out)

    def test_specular_batched_broadcast_inputs(self):
        batch_size = 10
        color = torch.tensor([1, 0, 1], dtype=torch.float32)
        direction = torch.tensor(
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        camera_position = torch.tensor(
            [+1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        points = torch.tensor([0, 0, 0], dtype=torch.float32)
        normals = torch.tensor([0, 1, 0], dtype=torch.float32)
        expected_out = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

        # Reshape
        normals = normals.view(1, 1, 3).expand(batch_size, -1, -1)
        points = points.view(1, 1, 3).expand(batch_size, -1, -1)
        expected_out = expected_out.view(1, 1, 3).expand(batch_size, 1, 3)

        # Don't expand the direction, color or camera_position.
        # These should be broadcasted in the specular function
        direction = direction.view(1, 3)
        camera_position = camera_position.view(1, 3)
        color = color.view(1, 3)

        lights = DirectionalLights(specular_color=color, direction=direction)
        output_light = lights.specular(
            points=points,
            normals=normals,
            camera_position=camera_position,
            shininess=torch.tensor(10),
        )
        self.assertClose(output_light, expected_out)

    def test_specular_batched_arbitrary_input_dims(self):
        """
        Test with a batch of inputs where shape of the input is mimicking the
        shape expected after rasterization i.e. a normal per pixel for
        top K faces per pixel.
        """
        device = torch.device("cuda:0")
        N, H, W, K = 8, 128, 128, 100
        color = torch.tensor([1, 0, 1], dtype=torch.float32, device=device)
        direction = torch.tensor(
            [-1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        camera_position = torch.tensor(
            [+1 / np.sqrt(2), 1 / np.sqrt(2), 0], dtype=torch.float32
        )
        points = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        normals = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
        points = points.view(1, 1, 1, 1, 3).expand(N, H, W, K, 3)
        normals = normals.view(1, 1, 1, 1, 3).expand(N, H, W, K, 3)

        direction = direction.view(1, 3)
        color = color.view(1, 3)
        camera_position = camera_position.view(1, 3)

        expected_output = torch.tensor(
            [1.0, 0.0, 1.0], dtype=torch.float32, device=device
        )
        expected_output = expected_output.view(-1, 1, 1, 1, 3)
        expected_output = expected_output.expand(N, H, W, K, -1)

        lights = DirectionalLights(specular_color=color, direction=direction)
        output_light = lights.specular(
            points=points,
            normals=normals,
            camera_position=camera_position,
            shininess=10.0,
        )
        self.assertClose(output_light, expected_output)

    def test_specular_batched_packed(self):
        """
        Test with a batch of 2 meshes each of which has faces on a single plane.
        The points and normals are in the packed format i.e. no batch dimension.
        """
        faces_per_mesh = [6, 4]
        mesh_to_vert_idx = [0] * faces_per_mesh[0] + [1] * faces_per_mesh[1]
        mesh_to_vert_idx = torch.tensor(mesh_to_vert_idx, dtype=torch.int64)
        color = torch.tensor([[1, 1, 1], [1, 0, 1]], dtype=torch.float32)
        direction = torch.tensor(
            [[-1 / np.sqrt(2), 1 / np.sqrt(2), 0], [-1, 1, 0]], dtype=torch.float32
        )
        camera_position = torch.tensor(
            [
                [+1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                [+1 / np.sqrt(2), -1 / np.sqrt(2), 0],
            ],
            dtype=torch.float32,
        )
        points = torch.tensor([[0, 0, 0]], dtype=torch.float32)
        normals = torch.tensor([[0, 1, 0], [0, 1, 0]], dtype=torch.float32)
        expected_output = torch.zeros((10, 3), dtype=torch.float32)
        expected_output[:6, :] += 1.0

        lights = DirectionalLights(
            specular_color=color[mesh_to_vert_idx, :],
            direction=direction[mesh_to_vert_idx, :],
        )
        output_light = lights.specular(
            points=points.view(-1, 3).expand(10, -1),
            normals=normals.view(-1, 3)[mesh_to_vert_idx, :],
            camera_position=camera_position[mesh_to_vert_idx, :],
            shininess=10.0,
        )
        self.assertClose(output_light, expected_output)
