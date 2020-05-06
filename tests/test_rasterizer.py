# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer.cameras import OpenGLPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.points.rasterizer import (
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.utils.ico_sphere import ico_sphere


DATA_DIR = Path(__file__).resolve().parent / "data"
DEBUG = False  # Set DEBUG to true to save outputs from the tests.


def convert_image_to_binary_mask(filename):
    with Image.open(filename) as raw_image:
        image = torch.from_numpy(np.array(raw_image))
    min = image.min()
    max = image.max()
    image_norm = (image - min) / (max - min)
    image_norm[image_norm > 0] == 1.0
    image_norm = image_norm.to(torch.int64)
    return image_norm


class TestMeshRasterizer(unittest.TestCase):
    def test_simple_sphere(self):
        device = torch.device("cuda:0")
        ref_filename = "test_rasterized_sphere.png"
        image_ref_filename = DATA_DIR / ref_filename

        # Rescale image_ref to the 0 - 1 range and convert to a binary mask.
        image_ref = convert_image_to_binary_mask(image_ref_filename)

        # Init mesh
        sphere_mesh = ico_sphere(5, device)

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        # Init rasterizer
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        ####################################
        # 1. Test rasterizing a single mesh
        ####################################

        fragments = rasterizer(sphere_mesh)
        image = fragments.pix_to_face[0, ..., 0].squeeze().cpu()
        # Convert pix_to_face to a binary mask
        image[image >= 0] = 1.0
        image[image < 0] = 0.0

        if DEBUG:
            Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / "DEBUG_test_rasterized_sphere.png"
            )

        self.assertTrue(torch.allclose(image, image_ref))

        ##################################
        #  2. Test with a batch of meshes
        ##################################

        batch_size = 10
        sphere_meshes = sphere_mesh.extend(batch_size)
        fragments = rasterizer(sphere_meshes)
        for i in range(batch_size):
            image = fragments.pix_to_face[i, ..., 0].squeeze().cpu()
            image[image >= 0] = 1.0
            image[image < 0] = 0.0
            self.assertTrue(torch.allclose(image, image_ref))

        ####################################################
        #  3. Test that passing kwargs to rasterizer works.
        ####################################################

        #  Change the view transform to zoom in.
        R, T = look_at_view_transform(2.0, 0, 0, device=device)
        fragments = rasterizer(sphere_mesh, R=R, T=T)
        image = fragments.pix_to_face[0, ..., 0].squeeze().cpu()
        image[image >= 0] = 1.0
        image[image < 0] = 0.0

        ref_filename = "test_rasterized_sphere_zoom.png"
        image_ref_filename = DATA_DIR / ref_filename
        image_ref = convert_image_to_binary_mask(image_ref_filename)

        if DEBUG:
            Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / "DEBUG_test_rasterized_sphere_zoom.png"
            )
        self.assertTrue(torch.allclose(image, image_ref))

        #################################
        #  4. Test init without cameras.
        ##################################

        # Create a new empty rasterizer:
        rasterizer = MeshRasterizer()

        # Check that omitting the cameras in both initialization
        # and the forward pass throws an error:
        with self.assertRaisesRegex(ValueError, "Cameras must be specified"):
            rasterizer(sphere_mesh)

        # Now pass in the cameras as a kwarg
        fragments = rasterizer(
            sphere_mesh, cameras=cameras, raster_settings=raster_settings
        )
        image = fragments.pix_to_face[0, ..., 0].squeeze().cpu()
        # Convert pix_to_face to a binary mask
        image[image >= 0] = 1.0
        image[image < 0] = 0.0

        if DEBUG:
            Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / "DEBUG_test_rasterized_sphere.png"
            )

        self.assertTrue(torch.allclose(image, image_ref))


class TestPointRasterizer(unittest.TestCase):
    def test_simple_sphere(self):
        device = torch.device("cuda:0")

        # Load reference image
        ref_filename = "test_simple_pointcloud_sphere.png"
        image_ref_filename = DATA_DIR / ref_filename

        # Rescale image_ref to the 0 - 1 range and convert to a binary mask.
        image_ref = convert_image_to_binary_mask(image_ref_filename).to(torch.int32)

        sphere_mesh = ico_sphere(1, device)
        verts_padded = sphere_mesh.verts_padded()
        verts_padded[..., 1] += 0.2
        verts_padded[..., 0] += 0.2
        pointclouds = Pointclouds(points=verts_padded)
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = PointsRasterizationSettings(
            image_size=256, radius=5e-2, points_per_pixel=1
        )

        #################################
        #  1. Test init without cameras.
        ##################################

        # Initialize without passing in the cameras
        rasterizer = PointsRasterizer()

        # Check that omitting the cameras in both initialization
        # and the forward pass throws an error:
        with self.assertRaisesRegex(ValueError, "Cameras must be specified"):
            rasterizer(pointclouds)

        ##########################################
        # 2. Test rasterizing a single pointcloud
        ##########################################

        fragments = rasterizer(
            pointclouds, cameras=cameras, raster_settings=raster_settings
        )

        # Convert idx to a binary mask
        image = fragments.idx[0, ..., 0].squeeze().cpu()
        image[image >= 0] = 1.0
        image[image < 0] = 0.0

        if DEBUG:
            Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / "DEBUG_test_rasterized_sphere_points.png"
            )

        self.assertTrue(torch.allclose(image, image_ref[..., 0]))

        ########################################
        #  3. Test with a batch of pointclouds
        ########################################

        batch_size = 10
        pointclouds = pointclouds.extend(batch_size)
        fragments = rasterizer(
            pointclouds, cameras=cameras, raster_settings=raster_settings
        )
        for i in range(batch_size):
            image = fragments.idx[i, ..., 0].squeeze().cpu()
            image[image >= 0] = 1.0
            image[image < 0] = 0.0
            self.assertTrue(torch.allclose(image, image_ref[..., 0]))
