# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    NDCMultinomialRaysampler,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PulsarPointsRenderer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes, Pointclouds

from .common_testing import TestCaseMixin


"""
PyTorch3D renderers operate in an align_corners=False manner.
This file demonstrates the pixel-perfect calculation by very simple
examples.
"""


class _CommonData:
    """
    Contains data for all these tests.

    - Firstly, a non-square at the origin specified in ndc space and
    screen space. Principal point is in the center of the image.
    Focal length is 1.0 in world space.
    This camera has the identity as its world to view transformation, so
    it is facing down the positive z axis with y being up and x being left.
    A point on the z=1.0 focal plane has its x,y world coordinate equal to
    its NDC.

    - Secondly, batched together with that, is a camera with the same
    focal length facing in the same direction but located so that it faces
    the corner of the corner pixel of the first image, with its principal
    point located at its corner, so that it maps the z=1 plane to NDC just
    like the first.

    - a single point self.point in world space which is located on a plane 1.0
    in front from the camera which is located exactly in the center
    of a known pixel (self.x, self.y), specifically with negative x and slightly
    positive y, so it is in the top right quadrant of the image.

    - A second batch of cameras defined in screen space which exactly match the
    first ones.

    So that this data can be copied for making demos, it is easiest to leave
    it as a freestanding class.
    """

    def __init__(self):
        self.H, self.W = 249, 125
        self.image_size = (self.H, self.W)
        self.camera_ndc = PerspectiveCameras(
            focal_length=1.0,
            image_size=(self.image_size,),
            in_ndc=True,
            T=torch.tensor([[0.0, 0.0, 0.0], [-1.0, self.H / self.W, 0.0]]),
            principal_point=((-0.0, -0.0), (1.0, -self.H / self.W)),
        )
        # Note how principal point is  specifiied
        self.camera_screen = PerspectiveCameras(
            focal_length=self.W / 2.0,
            principal_point=((self.W / 2.0, self.H / 2.0), (0.0, self.H)),
            image_size=(self.image_size,),
            T=torch.tensor([[0.0, 0.0, 0.0], [-1.0, self.H / self.W, 0.0]]),
            in_ndc=False,
        )

        # 81 is more than half of 125, 113 is a bit less than half of 249
        self.x, self.y = 81, 113
        self.point = [-0.304, 0.176, 1]
        # The point is in the center of pixel (81, 113)
        # where pixel (0,0) is the top left.
        # 81 is 38/2 pixels over the midpoint (125-1)/2=62
        # and 38/125=0.304
        # 113 is 22/2 pixels under the midpoint (249-1)/2=124
        # and 22/125=0.176


class TestPixels(TestCaseMixin, unittest.TestCase):
    def test_mesh(self):
        data = _CommonData()
        # Three points on the plane at unit 1 from the camera in
        # world space, whose mean is the known point.
        verts = torch.tensor(
            [[-0.288, 0.192, 1], [-0.32, 0.192, 1], [-0.304, 0.144, 1]]
        )
        self.assertClose(verts.mean(0), torch.tensor(data.point))
        faces = torch.LongTensor([[0, 1, 2]])
        # A mesh of one triangular face whose centroid is the known point
        # duplicated so it can be rendered from two cameras.
        meshes = Meshes(verts=[verts], faces=[faces]).extend(2)
        faces_per_pixel = 2
        for camera in (data.camera_ndc, data.camera_screen):
            rasterizer = MeshRasterizer(
                cameras=camera,
                raster_settings=RasterizationSettings(
                    image_size=data.image_size, faces_per_pixel=faces_per_pixel
                ),
            )
            barycentric_coords_found = rasterizer(meshes).bary_coords
            self.assertTupleEqual(
                barycentric_coords_found.shape,
                (2,) + data.image_size + (faces_per_pixel, 3),
            )
            # We see that the barycentric coordinates at the expected
            # pixel are (1/3, 1/3, 1/3), indicating that this pixel
            # hits the centroid of the triangle.
            self.assertClose(
                barycentric_coords_found[:, data.y, data.x, 0],
                torch.full((2, 3), 1 / 3.0),
                atol=1e-5,
            )

    def test_pointcloud(self):
        data = _CommonData()
        clouds = Pointclouds(points=torch.tensor([[data.point]])).extend(2)
        colorful_cloud = Pointclouds(
            points=torch.tensor([[data.point]]), features=torch.ones(1, 1, 3)
        ).extend(2)
        points_per_pixel = 2
        # for camera in [data.camera_screen]:
        for camera in (data.camera_ndc, data.camera_screen):
            rasterizer = PointsRasterizer(
                cameras=camera,
                raster_settings=PointsRasterizationSettings(
                    image_size=data.image_size,
                    radius=0.0001,
                    points_per_pixel=points_per_pixel,
                ),
            )
            # when rasterizing we expect only one pixel to be occupied
            rasterizer_output = rasterizer(clouds).idx
            self.assertTupleEqual(
                rasterizer_output.shape, (2,) + data.image_size + (points_per_pixel,)
            )
            found = torch.nonzero(rasterizer_output != -1)
            self.assertTupleEqual(found.shape, (2, 4))
            self.assertListEqual(found[0].tolist(), [0, data.y, data.x, 0])
            self.assertListEqual(found[1].tolist(), [1, data.y, data.x, 0])

            if camera.in_ndc():
                # Pulsar not currently working in screen space.
                pulsar_renderer = PulsarPointsRenderer(rasterizer=rasterizer)
                pulsar_output = pulsar_renderer(
                    colorful_cloud, gamma=(0.1, 0.1), znear=(0.1, 0.1), zfar=(70, 70)
                )
                self.assertTupleEqual(
                    pulsar_output.shape, (2,) + data.image_size + (3,)
                )
                # Look for points rendered in the red channel only, expecting our one.
                # Check the first batch element only.
                # TODO: Something is odd with the second.
                found = torch.nonzero(pulsar_output[0, :, :, 0])
                self.assertTupleEqual(found.shape, (1, 2))
                self.assertListEqual(found[0].tolist(), [data.y, data.x])
                # Should be:
                # found = torch.nonzero(pulsar_output[:, :, :, 0])
                # self.assertTupleEqual(found.shape, (2, 3))
                # self.assertListEqual(found[0].tolist(), [0, data.y, data.x])
                # self.assertListEqual(found[1].tolist(), [1, data.y, data.x])

    def test_raysampler(self):
        data = _CommonData()
        gridsampler = NDCMultinomialRaysampler(
            image_width=data.W,
            image_height=data.H,
            n_pts_per_ray=2,
            min_depth=1.0,
            max_depth=2.0,
        )
        for camera in (data.camera_ndc, data.camera_screen):
            bundle = gridsampler(camera)
            self.assertTupleEqual(bundle.xys.shape, (2,) + data.image_size + (2,))
            self.assertTupleEqual(
                bundle.directions.shape, (2,) + data.image_size + (3,)
            )
            self.assertClose(
                bundle.xys[:, data.y, data.x],
                torch.tensor(data.point[:2]).expand(2, -1),
            )
            # We check only the first batch element.
            # Second element varies because of camera location.
            self.assertClose(
                bundle.directions[0, data.y, data.x],
                torch.tensor(data.point),
            )

    def test_camera(self):
        data = _CommonData()
        # Our point, plus the image center, and a corner of the image.
        # Located at the focal-length distance away
        points = torch.tensor([data.point, [0, 0, 1], [1, data.H / data.W, 1]])
        for cameras in (data.camera_ndc, data.camera_screen):
            ndc_points = cameras.transform_points_ndc(points)
            screen_points = cameras.transform_points_screen(points)
            screen_points_without_xyflip = cameras.transform_points_screen(
                points, with_xyflip=False
            )
            camera_points = cameras.transform_points(points)
            for batch_idx in range(2):
                # NDC space agrees with the original
                self.assertClose(ndc_points[batch_idx], points, atol=1e-5)
                # First point in screen space is the center of our expected pixel
                self.assertClose(
                    screen_points[batch_idx][0],
                    torch.tensor([data.x + 0.5, data.y + 0.5, 1.0]),
                    atol=1e-5,
                )
                # Screen coords without xyflip should have x, y that negate the non-
                # flipped values, and unchanged z.
                self.assertClose(
                    screen_points_without_xyflip[batch_idx][0],
                    torch.tensor([-(data.x + 0.5), -(data.y + 0.5), 1.0]),
                    atol=1e-5,
                )
                # Second point in screen space is the center of the screen
                self.assertClose(
                    screen_points[batch_idx][1],
                    torch.tensor([data.W / 2.0, data.H / 2.0, 1.0]),
                    atol=1e-5,
                )
                # Third point in screen space is the corner of the screen
                # (corner of corner pixels)
                self.assertClose(
                    screen_points[batch_idx][2],
                    torch.tensor([0.0, 0.0, 1.0]),
                    atol=1e-5,
                )

                if cameras.in_ndc():
                    self.assertClose(camera_points[batch_idx], ndc_points[batch_idx])
                else:
                    # transform_points does something strange for screen cameras
                    if batch_idx == 0:
                        wanted = torch.stack(
                            [
                                data.W - screen_points[batch_idx, :, 0],
                                data.H - screen_points[batch_idx, :, 1],
                                torch.ones(3),
                            ],
                            dim=1,
                        )
                    else:
                        wanted = torch.stack(
                            [
                                -screen_points[batch_idx, :, 0],
                                2 * data.H - screen_points[batch_idx, :, 1],
                                torch.ones(3),
                            ],
                            dim=1,
                        )
                    self.assertClose(camera_points[batch_idx], wanted)
