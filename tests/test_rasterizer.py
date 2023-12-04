# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    OrthographicCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    RasterizationSettings,
)
from pytorch3d.renderer.fisheyecameras import FishEyeCameras
from pytorch3d.renderer.opengl.rasterizer_opengl import (
    _check_cameras,
    _check_raster_settings,
    _convert_meshes_to_gl_ndc,
    _parse_and_verify_image_size,
    MeshRasterizerOpenGL,
)
from pytorch3d.structures import Pointclouds
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils.ico_sphere import ico_sphere

from .common_testing import get_tests_dir, TestCaseMixin


DATA_DIR = get_tests_dir() / "data"
DEBUG = False  # Set DEBUG to true to save outputs from the tests.


def convert_image_to_binary_mask(filename):
    with Image.open(filename) as raw_image:
        image = torch.from_numpy(np.array(raw_image))
    mx = image.max()
    image_norm = (image == mx).to(torch.int64)
    return image_norm


class TestMeshRasterizer(unittest.TestCase):
    def test_simple_sphere(self):
        self._simple_sphere(MeshRasterizer)

    def test_simple_sphere_fisheye(self):
        self._simple_sphere_fisheye_against_perspective(MeshRasterizer)

    def test_simple_sphere_opengl(self):
        self._simple_sphere(MeshRasterizerOpenGL)

    def _simple_sphere(self, rasterizer_type):
        device = torch.device("cuda:0")
        ref_filename = f"test_rasterized_sphere_{rasterizer_type.__name__}.png"
        image_ref_filename = DATA_DIR / ref_filename

        # Rescale image_ref to the 0 - 1 range and convert to a binary mask.
        image_ref = convert_image_to_binary_mask(image_ref_filename)

        # Init mesh
        sphere_mesh = ico_sphere(5, device)

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        # Init rasterizer
        rasterizer = rasterizer_type(cameras=cameras, raster_settings=raster_settings)

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
                DATA_DIR
                / f"DEBUG_test_rasterized_sphere_{rasterizer_type.__name__}.png"
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

        #  Change the view transform to zoom out.
        R, T = look_at_view_transform(20.0, 0, 0, device=device)
        fragments = rasterizer(sphere_mesh, R=R, T=T)
        image = fragments.pix_to_face[0, ..., 0].squeeze().cpu()
        image[image >= 0] = 1.0
        image[image < 0] = 0.0

        ref_filename = f"test_rasterized_sphere_zoom_{rasterizer_type.__name__}.png"
        image_ref_filename = DATA_DIR / ref_filename
        image_ref = convert_image_to_binary_mask(image_ref_filename)

        if DEBUG:
            Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR
                / f"DEBUG_test_rasterized_sphere_zoom_{rasterizer_type.__name__}.png"
            )
        self.assertTrue(torch.allclose(image, image_ref))

        #################################
        #  4. Test init without cameras.
        ##################################

        # Create a new empty rasterizer:
        rasterizer = rasterizer_type(raster_settings=raster_settings)

        # Check that omitting the cameras in both initialization
        # and the forward pass throws an error:
        with self.assertRaisesRegex(ValueError, "Cameras must be specified"):
            rasterizer(sphere_mesh)

        # Now pass in the cameras as a kwarg
        fragments = rasterizer(sphere_mesh, cameras=cameras)
        image = fragments.pix_to_face[0, ..., 0].squeeze().cpu()
        # Convert pix_to_face to a binary mask
        image[image >= 0] = 1.0
        image[image < 0] = 0.0

        if DEBUG:
            Image.fromarray((image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR
                / f"DEBUG_test_rasterized_sphere_{rasterizer_type.__name__}.png"
            )

        self.assertTrue(torch.allclose(image, image_ref))

    def _simple_sphere_fisheye_against_perspective(self, rasterizer_type):
        device = torch.device("cuda:0")

        # Init mesh
        sphere_mesh = ico_sphere(5, device)

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)

        # Init Fisheye camera params
        focal = torch.tensor([[1.7321]], dtype=torch.float32)
        principal_point = torch.tensor([[0.0101, -0.0101]])
        perspective_cameras = PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal,
            principal_point=principal_point,
            device="cuda:0",
        )
        fisheye_cameras = FishEyeCameras(
            device=device,
            R=R,
            T=T,
            focal_length=focal,
            principal_point=principal_point,
            world_coordinates=True,
            use_radial=False,
            use_tangential=False,
            use_thin_prism=False,
        )
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        # Init rasterizer
        perspective_rasterizer = rasterizer_type(
            cameras=perspective_cameras, raster_settings=raster_settings
        )
        fisheye_rasterizer = rasterizer_type(
            cameras=fisheye_cameras, raster_settings=raster_settings
        )

        ####################################################################################
        # Test rasterizing a single mesh comparing fisheye camera against perspective camera
        ####################################################################################

        perspective_fragments = perspective_rasterizer(sphere_mesh)
        perspective_image = perspective_fragments.pix_to_face[0, ..., 0].squeeze().cpu()
        # Convert pix_to_face to a binary mask
        perspective_image[perspective_image >= 0] = 1.0
        perspective_image[perspective_image < 0] = 0.0

        if DEBUG:
            Image.fromarray((perspective_image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR
                / f"DEBUG_test_perspective_rasterized_sphere_{rasterizer_type.__name__}.png"
            )

        fisheye_fragments = fisheye_rasterizer(sphere_mesh)
        fisheye_image = fisheye_fragments.pix_to_face[0, ..., 0].squeeze().cpu()
        # Convert pix_to_face to a binary mask
        fisheye_image[fisheye_image >= 0] = 1.0
        fisheye_image[fisheye_image < 0] = 0.0

        if DEBUG:
            Image.fromarray((fisheye_image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR
                / f"DEBUG_test_fisheye_rasterized_sphere_{rasterizer_type.__name__}.png"
            )

        self.assertTrue(torch.allclose(fisheye_image, perspective_image))

        ##################################
        #  2. Test with a batch of meshes
        ##################################

        batch_size = 10
        sphere_meshes = sphere_mesh.extend(batch_size)
        fragments = fisheye_rasterizer(sphere_meshes)
        for i in range(batch_size):
            image = fragments.pix_to_face[i, ..., 0].squeeze().cpu()
            image[image >= 0] = 1.0
            image[image < 0] = 0.0
            self.assertTrue(torch.allclose(image, perspective_image))

    def test_simple_to(self):
        # Check that to() works without a cameras object.
        device = torch.device("cuda:0")
        rasterizer = MeshRasterizer()
        rasterizer.to(device)

        rasterizer = MeshRasterizerOpenGL()
        rasterizer.to(device)

    def test_compare_rasterizers(self):
        device = torch.device("cuda:0")

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
            perspective_correct=True,
        )
        from pytorch3d.io import load_obj
        from pytorch3d.renderer import TexturesAtlas

        from .common_testing import get_pytorch3d_dir

        TUTORIAL_DATA_DIR = get_pytorch3d_dir() / "docs/tutorials/data"
        obj_filename = TUTORIAL_DATA_DIR / "cow_mesh/cow.obj"

        # Load mesh and texture as a per face texture atlas.
        verts, faces, aux = load_obj(
            obj_filename,
            device=device,
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=8,
            texture_wrap=None,
        )
        atlas = aux.texture_atlas
        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesAtlas(atlas=[atlas]),
        )

        # Rasterize using both rasterizers and compare results.
        rasterizer = MeshRasterizerOpenGL(
            cameras=cameras, raster_settings=raster_settings
        )
        fragments_opengl = rasterizer(mesh)

        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        fragments = rasterizer(mesh)

        # Ensure that 99.9% of bary_coords is at most 0.001 different.
        self.assertLess(
            torch.quantile(
                (fragments.bary_coords - fragments_opengl.bary_coords).abs(), 0.999
            ),
            0.001,
        )

        # Ensure that 99.9% of zbuf vals is at most 0.001 different.
        self.assertLess(
            torch.quantile((fragments.zbuf - fragments_opengl.zbuf).abs(), 0.999), 0.001
        )

        # Ensure that 99.99% of pix_to_face is identical.
        self.assertEqual(
            torch.quantile(
                (fragments.pix_to_face != fragments_opengl.pix_to_face).float(), 0.9999
            ),
            0,
        )


class TestMeshRasterizerOpenGLUtils(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        verts = torch.tensor(
            [[-1, 1, 0], [1, 1, 0], [1, -1, 0]], dtype=torch.float32
        ).cuda()
        faces = torch.tensor([[0, 1, 2]]).cuda()
        self.meshes_world = Meshes(verts=[verts], faces=[faces])

    # Test various utils specific to the OpenGL rasterizer. Full "integration tests"
    # live in test_render_meshes and test_render_multigpu.
    def test_check_cameras(self):
        _check_cameras(FoVPerspectiveCameras())
        _check_cameras(FoVPerspectiveCameras())
        with self.assertRaisesRegex(ValueError, "Cameras must be specified"):
            _check_cameras(None)
        with self.assertRaisesRegex(ValueError, "MeshRasterizerOpenGL only works with"):
            _check_cameras(PerspectiveCameras())
        with self.assertRaisesRegex(ValueError, "MeshRasterizerOpenGL only works with"):
            _check_cameras(OrthographicCameras())

        MeshRasterizerOpenGL(FoVPerspectiveCameras().cuda())(self.meshes_world)
        MeshRasterizerOpenGL(FoVOrthographicCameras().cuda())(self.meshes_world)
        MeshRasterizerOpenGL()(
            self.meshes_world, cameras=FoVPerspectiveCameras().cuda()
        )

        with self.assertRaisesRegex(ValueError, "MeshRasterizerOpenGL only works with"):
            MeshRasterizerOpenGL(PerspectiveCameras().cuda())(self.meshes_world)
        with self.assertRaisesRegex(ValueError, "MeshRasterizerOpenGL only works with"):
            MeshRasterizerOpenGL(OrthographicCameras().cuda())(self.meshes_world)
        with self.assertRaisesRegex(ValueError, "Cameras must be specified"):
            MeshRasterizerOpenGL()(self.meshes_world)

    def test_check_raster_settings(self):
        raster_settings = RasterizationSettings()
        raster_settings.faces_per_pixel = 100
        with self.assertWarnsRegex(UserWarning, ".* one face per pixel"):
            _check_raster_settings(raster_settings)

        with self.assertWarnsRegex(UserWarning, ".* one face per pixel"):
            MeshRasterizerOpenGL(raster_settings=raster_settings)(
                self.meshes_world, cameras=FoVPerspectiveCameras().cuda()
            )

    def test_convert_meshes_to_gl_ndc_square_img(self):
        R, T = look_at_view_transform(1, 90, 180)
        cameras = FoVOrthographicCameras(R=R, T=T).cuda()

        meshes_gl_ndc = _convert_meshes_to_gl_ndc(
            self.meshes_world, (100, 100), cameras
        )

        # After look_at_view_transform rotating 180 deg around z-axis, we recover
        # the original coordinates. After additionally elevating the view by 90
        # deg, we "zero out" the y-coordinate. Finally, we negate the x and y axes
        # to adhere to OpenGL conventions (which go against the PyTorch3D convention).
        self.assertClose(
            meshes_gl_ndc.verts_list()[0],
            torch.tensor(
                [[-1, 0, 0], [1, 0, 0], [1, 0, 2]], dtype=torch.float32
            ).cuda(),
            atol=1e-5,
        )

    def test_parse_and_verify_image_size(self):
        img_size = _parse_and_verify_image_size(512)
        self.assertEqual(img_size, (512, 512))

        img_size = _parse_and_verify_image_size((2047, 10))
        self.assertEqual(img_size, (2047, 10))

        img_size = _parse_and_verify_image_size((10, 2047))
        self.assertEqual(img_size, (10, 2047))

        with self.assertRaisesRegex(ValueError, "Max rasterization size is"):
            _parse_and_verify_image_size((2049, 512))

        with self.assertRaisesRegex(ValueError, "Max rasterization size is"):
            _parse_and_verify_image_size((512, 2049))

        with self.assertRaisesRegex(ValueError, "Max rasterization size is"):
            _parse_and_verify_image_size((2049, 2049))

        rasterizer = MeshRasterizerOpenGL(FoVPerspectiveCameras().cuda())
        raster_settings = RasterizationSettings()

        raster_settings.image_size = 512
        fragments = rasterizer(self.meshes_world, raster_settings=raster_settings)
        self.assertEqual(fragments.pix_to_face.shape, torch.Size([1, 512, 512, 1]))

        raster_settings.image_size = (2047, 10)
        fragments = rasterizer(self.meshes_world, raster_settings=raster_settings)
        self.assertEqual(fragments.pix_to_face.shape, torch.Size([1, 2047, 10, 1]))

        raster_settings.image_size = (10, 2047)
        fragments = rasterizer(self.meshes_world, raster_settings=raster_settings)
        self.assertEqual(fragments.pix_to_face.shape, torch.Size([1, 10, 2047, 1]))

        raster_settings.image_size = (2049, 512)
        with self.assertRaisesRegex(ValueError, "Max rasterization size is"):
            rasterizer(self.meshes_world, raster_settings=raster_settings)

        raster_settings.image_size = (512, 2049)
        with self.assertRaisesRegex(ValueError, "Max rasterization size is"):
            rasterizer(self.meshes_world, raster_settings=raster_settings)

        raster_settings.image_size = (2049, 2049)
        with self.assertRaisesRegex(ValueError, "Max rasterization size is"):
            rasterizer(self.meshes_world, raster_settings=raster_settings)


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
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
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

    def test_simple_sphere_fisheye_against_perspective(self):
        device = torch.device("cuda:0")

        # Rescale image_ref to the 0 - 1 range and convert to a binary mask.
        sphere_mesh = ico_sphere(1, device)
        verts_padded = sphere_mesh.verts_padded()
        verts_padded[..., 1] += 0.2
        verts_padded[..., 0] += 0.2
        pointclouds = Pointclouds(points=verts_padded)
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        perspective_cameras = PerspectiveCameras(
            R=R,
            T=T,
            device=device,
        )
        fisheye_cameras = FishEyeCameras(
            device=device,
            R=R,
            T=T,
            world_coordinates=True,
            use_radial=False,
            use_tangential=False,
            use_thin_prism=False,
        )
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

        ########################################################################################
        # 2. Test rasterizing a single pointcloud with fisheye camera agasint perspective camera
        ########################################################################################

        perspective_fragments = rasterizer(
            pointclouds, cameras=perspective_cameras, raster_settings=raster_settings
        )
        fisheye_fragments = rasterizer(
            pointclouds, cameras=fisheye_cameras, raster_settings=raster_settings
        )

        # Convert idx to a binary mask
        perspective_image = perspective_fragments.idx[0, ..., 0].squeeze().cpu()
        perspective_image[perspective_image >= 0] = 1.0
        perspective_image[perspective_image < 0] = 0.0

        fisheye_image = fisheye_fragments.idx[0, ..., 0].squeeze().cpu()
        fisheye_image[fisheye_image >= 0] = 1.0
        fisheye_image[fisheye_image < 0] = 0.0

        if DEBUG:
            Image.fromarray((perspective_image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / "DEBUG_test_rasterized_perspective_sphere_points.png"
            )
            Image.fromarray((fisheye_image.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / "DEBUG_test_rasterized_fisheye_sphere_points.png"
            )

        self.assertTrue(torch.allclose(fisheye_image, perspective_image))

    def test_simple_to(self):
        # Check that to() works without a cameras object.
        device = torch.device("cuda:0")
        rasterizer = PointsRasterizer()
        rasterizer.to(device)
