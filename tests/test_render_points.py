# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Sanity checks for output images from the pointcloud renderer.
"""

import unittest
import warnings
from os import path

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer.cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
    OrthographicCameras,
    PerspectiveCameras,
)
from pytorch3d.renderer.compositing import alpha_composite, norm_weighted_sum
from pytorch3d.renderer.fisheyecameras import FishEyeCameras
from pytorch3d.renderer.points import (
    AlphaCompositor,
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    PulsarPointsRenderer,
)
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.utils.ico_sphere import ico_sphere

from .common_testing import (
    get_pytorch3d_dir,
    get_tests_dir,
    load_rgb_image,
    TestCaseMixin,
)


# If DEBUG=True, save out images generated in the tests for debugging.
# All saved images have prefix DEBUG_
DEBUG = False
DATA_DIR = get_tests_dir() / "data"


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
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
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

    def test_simple_sphere_fisheye(self):
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
        cameras = FishEyeCameras(
            device=device,
            R=R,
            T=T,
            use_radial=False,
            use_tangential=False,
            use_thin_prism=False,
            world_coordinates=True,
        )
        raster_settings = PointsRasterizationSettings(
            image_size=256, radius=5e-2, points_per_pixel=1
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        compositor = NormWeightedCompositor()
        renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)

        # Load reference image
        filename = "render_fisheye_sphere_points.png"
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

    def test_simple_sphere_pulsar(self):
        for device in [torch.device("cpu"), torch.device("cuda")]:
            sphere_mesh = ico_sphere(1, device)
            verts_padded = sphere_mesh.verts_padded()
            # Shift vertices to check coordinate frames are correct.
            verts_padded[..., 1] += 0.2
            verts_padded[..., 0] += 0.2
            pointclouds = Pointclouds(
                points=verts_padded, features=torch.ones_like(verts_padded)
            )
            for azimuth in [0.0, 90.0]:
                R, T = look_at_view_transform(2.7, 0.0, azimuth)
                for camera_name, cameras in [
                    ("fovperspective", FoVPerspectiveCameras(device=device, R=R, T=T)),
                    (
                        "fovorthographic",
                        FoVOrthographicCameras(device=device, R=R, T=T),
                    ),
                    ("perspective", PerspectiveCameras(device=device, R=R, T=T)),
                    ("orthographic", OrthographicCameras(device=device, R=R, T=T)),
                ]:
                    raster_settings = PointsRasterizationSettings(
                        image_size=256, radius=5e-2, points_per_pixel=1
                    )
                    rasterizer = PointsRasterizer(
                        cameras=cameras, raster_settings=raster_settings
                    )
                    renderer = PulsarPointsRenderer(rasterizer=rasterizer).to(device)
                    # Load reference image
                    filename = (
                        "pulsar_simple_pointcloud_sphere_"
                        f"azimuth{azimuth}_{camera_name}.png"
                    )
                    image_ref = load_rgb_image("test_%s" % filename, DATA_DIR)
                    images = renderer(
                        pointclouds, gamma=(1e-3,), znear=(1.0,), zfar=(100.0,)
                    )
                    rgb = images[0, ..., :3].squeeze().cpu()
                    if DEBUG:
                        filename = "DEBUG_%s" % filename
                        Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                            DATA_DIR / filename
                        )
                    self.assertClose(rgb, image_ref, rtol=7e-3, atol=5e-3)

    def test_unified_inputs_pulsar(self):
        # Test data on different devices.
        for device in [torch.device("cpu"), torch.device("cuda")]:
            sphere_mesh = ico_sphere(1, device)
            verts_padded = sphere_mesh.verts_padded()
            pointclouds = Pointclouds(
                points=verts_padded, features=torch.ones_like(verts_padded)
            )
            R, T = look_at_view_transform(2.7, 0.0, 0.0)
            # Test the different camera types.
            for _, cameras in [
                ("fovperspective", FoVPerspectiveCameras(device=device, R=R, T=T)),
                (
                    "fovorthographic",
                    FoVOrthographicCameras(device=device, R=R, T=T),
                ),
                ("perspective", PerspectiveCameras(device=device, R=R, T=T)),
                ("orthographic", OrthographicCameras(device=device, R=R, T=T)),
            ]:
                # Test different ways for image size specification.
                for image_size in (256, (256, 256)):
                    raster_settings = PointsRasterizationSettings(
                        image_size=image_size, radius=5e-2, points_per_pixel=1
                    )
                    rasterizer = PointsRasterizer(
                        cameras=cameras, raster_settings=raster_settings
                    )
                    # Test that the compositor can be provided. It's value is ignored
                    # so use a dummy.
                    _ = PulsarPointsRenderer(rasterizer=rasterizer, compositor=1).to(
                        device
                    )
                    # Constructor without compositor.
                    _ = PulsarPointsRenderer(rasterizer=rasterizer).to(device)
                    # Constructor with n_channels.
                    _ = PulsarPointsRenderer(rasterizer=rasterizer, n_channels=3).to(
                        device
                    )
                    # Constructor with max_num_spheres.
                    renderer = PulsarPointsRenderer(
                        rasterizer=rasterizer, max_num_spheres=1000
                    ).to(device)
                    # Test the forward function.
                    if isinstance(cameras, (PerspectiveCameras, OrthographicCameras)):
                        # znear and zfar is required in this case.
                        self.assertRaises(
                            ValueError,
                            lambda renderer=renderer,
                            pointclouds=pointclouds: renderer.forward(
                                point_clouds=pointclouds, gamma=(1e-4,)
                            ),
                        )
                        renderer.forward(
                            point_clouds=pointclouds,
                            gamma=(1e-4,),
                            znear=(1.0,),
                            zfar=(2.0,),
                        )
                        # znear and zfar must be batched.
                        self.assertRaises(
                            TypeError,
                            lambda renderer=renderer,
                            pointclouds=pointclouds: renderer.forward(
                                point_clouds=pointclouds,
                                gamma=(1e-4,),
                                znear=1.0,
                                zfar=(2.0,),
                            ),
                        )
                        self.assertRaises(
                            TypeError,
                            lambda renderer=renderer,
                            pointclouds=pointclouds: renderer.forward(
                                point_clouds=pointclouds,
                                gamma=(1e-4,),
                                znear=(1.0,),
                                zfar=2.0,
                            ),
                        )
                    else:
                        # gamma must be batched.
                        self.assertRaises(
                            TypeError,
                            lambda renderer=renderer,
                            pointclouds=pointclouds: renderer.forward(
                                point_clouds=pointclouds, gamma=1e-4
                            ),
                        )
                        renderer.forward(point_clouds=pointclouds, gamma=(1e-4,))
                        # rasterizer width and height change.
                        renderer.rasterizer.raster_settings.image_size = 0
                        self.assertRaises(
                            ValueError,
                            lambda renderer=renderer,
                            pointclouds=pointclouds: renderer.forward(
                                point_clouds=pointclouds, gamma=(1e-4,)
                            ),
                        )

    def test_pointcloud_with_features(self):
        device = torch.device("cuda:0")
        file_dir = get_pytorch3d_dir() / "docs/tutorials/data"
        pointcloud_filename = file_dir / "PittsburghBridge/pointcloud.npz"

        # Note, this file is too large to check in to the repo.
        # Download the file to run the test locally.
        if not path.exists(pointcloud_filename):
            url = (
                "https://dl.fbaipublicfiles.com/pytorch3d/data/"
                "PittsburghBridge/pointcloud.npz"
            )
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
        cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

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
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
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

    def test_compositor_background_color_rgba(self):
        N, H, W, K, C, P = 1, 15, 15, 20, 4, 225
        ptclds = torch.randn((C, P))
        alphas = torch.rand((N, K, H, W))
        pix_idxs = torch.randint(-1, 20, (N, K, H, W))  # 20 < P, large amount of -1
        background_color = [0.5, 0, 1]

        compositor_funcs = [
            (NormWeightedCompositor, norm_weighted_sum),
            (AlphaCompositor, alpha_composite),
        ]

        for compositor_class, composite_func in compositor_funcs:
            compositor = compositor_class(background_color)

            # run the forward method to generate masked images
            masked_images = compositor.forward(pix_idxs, alphas, ptclds)

            # generate unmasked images for testing purposes
            images = composite_func(pix_idxs, alphas, ptclds)

            is_foreground = pix_idxs[:, 0] >= 0

            # make sure foreground values are unchanged
            self.assertClose(
                torch.masked_select(masked_images, is_foreground[:, None]),
                torch.masked_select(images, is_foreground[:, None]),
            )

            is_background = ~is_foreground[..., None].expand(-1, -1, -1, C)

            # permute masked_images to correctly get rgb values
            masked_images = masked_images.permute(0, 2, 3, 1)
            for i in range(3):
                channel_color = background_color[i]

                # check if background colors are properly changed
                self.assertTrue(
                    masked_images[is_background]
                    .view(-1, C)[..., i]
                    .eq(channel_color)
                    .all()
                )

            # check background color alpha values
            self.assertTrue(
                masked_images[is_background].view(-1, C)[..., 3].eq(1).all()
            )

    def test_compositor_background_color_rgb(self):
        N, H, W, K, C, P = 1, 15, 15, 20, 3, 225
        ptclds = torch.randn((C, P))
        alphas = torch.rand((N, K, H, W))
        pix_idxs = torch.randint(-1, 20, (N, K, H, W))  # 20 < P, large amount of -1
        background_color = [0.5, 0, 1]

        compositor_funcs = [
            (NormWeightedCompositor, norm_weighted_sum),
            (AlphaCompositor, alpha_composite),
        ]

        for compositor_class, composite_func in compositor_funcs:
            compositor = compositor_class(background_color)

            # run the forward method to generate masked images
            masked_images = compositor.forward(pix_idxs, alphas, ptclds)

            # generate unmasked images for testing purposes
            images = composite_func(pix_idxs, alphas, ptclds)

            is_foreground = pix_idxs[:, 0] >= 0

            # make sure foreground values are unchanged
            self.assertClose(
                torch.masked_select(masked_images, is_foreground[:, None]),
                torch.masked_select(images, is_foreground[:, None]),
            )

            is_background = ~is_foreground[..., None].expand(-1, -1, -1, C)

            # permute masked_images to correctly get rgb values
            masked_images = masked_images.permute(0, 2, 3, 1)
            for i in range(3):
                channel_color = background_color[i]

                # check if background colors are properly changed
                self.assertTrue(
                    masked_images[is_background]
                    .view(-1, C)[..., i]
                    .eq(channel_color)
                    .all()
                )
