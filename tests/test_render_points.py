# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


"""
Sanity checks for output images from the pointcloud renderer.
"""
import unittest
import warnings
from os import path
from pathlib import Path

import numpy as np
import torch
from common_testing import TestCaseMixin, load_rgb_image
from PIL import Image
from pytorch3d.renderer.cameras import (
    OpenGLOrthographicCameras,
    OpenGLPerspectiveCameras,
    look_at_view_transform,
)
from pytorch3d.renderer.points import (
    AlphaCompositor,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.utils.ico_sphere import ico_sphere


# If DEBUG=True, save out images generated in the tests for debugging.
# All saved images have prefix DEBUG_
DEBUG = False
DATA_DIR = Path(__file__).resolve().parent / "data"


class TestRenderPoints(TestCaseMixin, unittest.TestCase):
    def test_simple_sphere(self):
        device = torch.device("cuda:0")
        sphere_mesh = ico_sphere(1, device)
        verts_padded = sphere_mesh.verts_padded()
        # Shift vertices to check coordinate frames are correct.
        verts_padded[..., 1] += 0.2
        verts_padded[..., 0] += 0.2
        pointclouds = Pointclouds(
            points=verts_padded, features=torch.ones_like(verts_padded)
        )
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = PointsRasterizationSettings(
            image_size=256, radius=5e-2, points_per_pixel=1
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        compositor = NormWeightedCompositor()
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

        # Load reference image
        filename = "simple_pointcloud_sphere.png"
        image_ref = load_rgb_image("test_%s" % filename, DATA_DIR)

        for bin_size in [0, None]:
            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size
            images = renderer(pointclouds)
            rgb = images[0, ..., :3].squeeze().cpu()
            if DEBUG:
                filename = "DEBUG_%s" % filename
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / filename
                )
            self.assertClose(rgb, image_ref)

    def test_pointcloud_with_features(self):
        device = torch.device("cuda:0")
        file_dir = Path(__file__).resolve().parent.parent / "docs/tutorials/data"
        pointcloud_filename = file_dir / "PittsburghBridge/pointcloud.npz"

        # Note, this file is too large to check in to the repo.
        # Download the file to run the test locally.
        if not path.exists(pointcloud_filename):
            url = "https://dl.fbaipublicfiles.com/pytorch3d/data/PittsburghBridge/pointcloud.npz"
            msg = (
                "pointcloud.npz not found, download from %s, save it at the path %s, and rerun"
                % (url, pointcloud_filename)
            )
            warnings.warn(msg)
            return True

        # Load point cloud
        pointcloud = np.load(pointcloud_filename)
        verts = torch.Tensor(pointcloud["verts"]).to(device)
        rgb_feats = torch.Tensor(pointcloud["rgb"]).to(device)

        verts.requires_grad = True
        rgb_feats.requires_grad = True
        point_cloud = Pointclouds(points=[verts], features=[rgb_feats])

        R, T = look_at_view_transform(20, 10, 0)
        cameras = OpenGLOrthographicCameras(device=device, R=R, T=T, znear=0.01)

        raster_settings = PointsRasterizationSettings(
            # Set image_size so it is not a multiple of 16 (min bin_size)
            # in order to confirm that there are no errors in coarse rasterization.
            image_size=500,
            radius=0.003,
            points_per_pixel=10,
        )

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(
                cameras=cameras, raster_settings=raster_settings
            ),
            compositor=AlphaCompositor(),
        )

        images = renderer(point_cloud)

        # Load reference image
        filename = "bridge_pointcloud.png"
        image_ref = load_rgb_image("test_%s" % filename, DATA_DIR)

        for bin_size in [0, None]:
            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size
            images = renderer(point_cloud)
            rgb = images[0, ..., :3].squeeze().cpu()
            if DEBUG:
                filename = "DEBUG_%s" % filename
                Image.fromarray((rgb.detach().numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / filename
                )
            self.assertClose(rgb, image_ref, atol=0.015)

        # Check grad exists.
        grad_images = torch.randn_like(images)
        images.backward(grad_images)
        self.assertIsNotNone(verts.grad)
        self.assertIsNotNone(rgb_feats.grad)

    def test_simple_sphere_batched(self):
        device = torch.device("cuda:0")
        sphere_mesh = ico_sphere(1, device)
        verts_padded = sphere_mesh.verts_padded()
        verts_padded[..., 1] += 0.2
        verts_padded[..., 0] += 0.2
        pointclouds = Pointclouds(
            points=verts_padded, features=torch.ones_like(verts_padded)
        )
        batch_size = 20
        pointclouds = pointclouds.extend(batch_size)
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = PointsRasterizationSettings(
            image_size=256, radius=5e-2, points_per_pixel=1
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        compositor = NormWeightedCompositor()
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

        # Load reference image
        filename = "simple_pointcloud_sphere.png"
        image_ref = load_rgb_image("test_%s" % filename, DATA_DIR)

        images = renderer(pointclouds)
        for i in range(batch_size):
            rgb = images[i, ..., :3].squeeze().cpu()
            if i == 0 and DEBUG:
                filename = "DEBUG_%s" % filename
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / filename
                )
            self.assertClose(rgb, image_ref)
