# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


"""
Sanity checks for output images from the renderer.
"""
import unittest
from pathlib import Path

import numpy as np
import torch
from common_testing import TestCaseMixin, load_rgb_image
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.renderer.cameras import OpenGLPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh import TexturesAtlas, TexturesUV, TexturesVertex
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.shader import (
    BlendParams,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftSilhouetteShader,
    TexturedSoftPhongShader,
)
from pytorch3d.structures.meshes import Meshes, join_mesh
from pytorch3d.utils.ico_sphere import ico_sphere


# If DEBUG=True, save out images generated in the tests for debugging.
# All saved images have prefix DEBUG_
DEBUG = False
DATA_DIR = Path(__file__).resolve().parent / "data"


class TestRenderMeshes(TestCaseMixin, unittest.TestCase):
    def test_simple_sphere(self, elevated_camera=False):
        """
        Test output of phong and gouraud shading matches a reference image using
        the default values for the light sources.

        Args:
            elevated_camera: Defines whether the camera observing the scene should
                           have an elevation of 45 degrees.
        """
        device = torch.device("cuda:0")

        # Init mesh
        sphere_mesh = ico_sphere(5, device)
        verts_padded = sphere_mesh.verts_padded()
        faces_padded = sphere_mesh.faces_padded()
        feats = torch.ones_like(verts_padded, device=device)
        textures = TexturesVertex(verts_features=feats)
        sphere_mesh = Meshes(verts=verts_padded, faces=faces_padded, textures=textures)

        # Init rasterizer settings
        if elevated_camera:
            # Elevated and rotated camera
            R, T = look_at_view_transform(dist=2.7, elev=45.0, azim=45.0)
            postfix = "_elevated_camera"
            # If y axis is up, the spot of light should
            # be on the bottom left of the sphere.
        else:
            # No elevation or azimuth rotation
            R, T = look_at_view_transform(2.7, 0.0, 0.0)
            postfix = ""
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]

        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))

        # Test several shaders
        shaders = {
            "phong": HardPhongShader,
            "gouraud": HardGouraudShader,
            "flat": HardFlatShader,
        }
        for (name, shader_init) in shaders.items():
            shader = shader_init(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            images = renderer(sphere_mesh)
            filename = "simple_sphere_light_%s%s.png" % (name, postfix)
            image_ref = load_rgb_image("test_%s" % filename, DATA_DIR)
            rgb = images[0, ..., :3].squeeze().cpu()

            if DEBUG:
                filename = "DEBUG_%s" % filename
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / filename
                )
            self.assertClose(rgb, image_ref, atol=0.05)

        ########################################################
        # Move the light to the +z axis in world space so it is
        # behind the sphere. Note that +Z is in, +Y up,
        # +X left for both world and camera space.
        ########################################################
        lights.location[..., 2] = -2.0
        phong_shader = HardPhongShader(
            lights=lights,
            cameras=cameras,
            materials=materials,
            blend_params=blend_params,
        )
        phong_renderer = MeshRenderer(rasterizer=rasterizer, shader=phong_shader)
        images = phong_renderer(sphere_mesh, lights=lights)
        rgb = images[0, ..., :3].squeeze().cpu()
        if DEBUG:
            filename = "DEBUG_simple_sphere_dark%s.png" % postfix
            Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / filename
            )

        # Load reference image
        image_ref_phong_dark = load_rgb_image(
            "test_simple_sphere_dark%s.png" % postfix, DATA_DIR
        )
        self.assertClose(rgb, image_ref_phong_dark, atol=0.05)

    def test_simple_sphere_elevated_camera(self):
        """
        Test output of phong and gouraud shading matches a reference image using
        the default values for the light sources.

        The rendering is performed with a camera that has non-zero elevation.
        """
        self.test_simple_sphere(elevated_camera=True)

    def test_simple_sphere_batched(self):
        """
        Test a mesh with vertex textures can be extended to form a batch, and
        is rendered correctly with Phong, Gouraud and Flat Shaders.
        """
        batch_size = 5
        device = torch.device("cuda:0")

        # Init mesh with vertex textures.
        sphere_meshes = ico_sphere(5, device).extend(batch_size)
        verts_padded = sphere_meshes.verts_padded()
        faces_padded = sphere_meshes.faces_padded()
        feats = torch.ones_like(verts_padded, device=device)
        textures = TexturesVertex(verts_features=feats)
        sphere_meshes = Meshes(
            verts=verts_padded, faces=faces_padded, textures=textures
        )

        # Init rasterizer settings
        dist = torch.tensor([2.7]).repeat(batch_size).to(device)
        elev = torch.zeros_like(dist)
        azim = torch.zeros_like(dist)
        R, T = look_at_view_transform(dist, elev, azim)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]
        blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))

        # Init renderer
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shaders = {
            "phong": HardPhongShader,
            "gouraud": HardGouraudShader,
            "flat": HardFlatShader,
        }
        for (name, shader_init) in shaders.items():
            shader = shader_init(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            images = renderer(sphere_meshes)
            image_ref = load_rgb_image(
                "test_simple_sphere_light_%s.png" % name, DATA_DIR
            )
            for i in range(batch_size):
                rgb = images[i, ..., :3].squeeze().cpu()
                if i == 0 and DEBUG:
                    filename = "DEBUG_simple_sphere_batched_%s.png" % name
                    Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                        DATA_DIR / filename
                    )
                self.assertClose(rgb, image_ref, atol=0.05)

    def test_silhouette_with_grad(self):
        """
        Test silhouette blending. Also check that gradient calculation works.
        """
        device = torch.device("cuda:0")
        ref_filename = "test_silhouette.png"
        image_ref_filename = DATA_DIR / ref_filename
        sphere_mesh = ico_sphere(5, device)
        verts, faces = sphere_mesh.get_mesh_verts_faces(0)
        sphere_mesh = Meshes(verts=[verts], faces=[faces])

        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=80,
            clip_barycentric_coords=True,
        )

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        # Init renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftSilhouetteShader(blend_params=blend_params),
        )
        images = renderer(sphere_mesh)
        alpha = images[0, ..., 3].squeeze().cpu()
        if DEBUG:
            Image.fromarray((alpha.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / "DEBUG_silhouette.png"
            )

        with Image.open(image_ref_filename) as raw_image_ref:
            image_ref = torch.from_numpy(np.array(raw_image_ref))

        image_ref = image_ref.to(dtype=torch.float32) / 255.0
        self.assertClose(alpha, image_ref, atol=0.055)

        # Check grad exist
        verts.requires_grad = True
        sphere_mesh = Meshes(verts=[verts], faces=[faces])
        images = renderer(sphere_mesh)
        images[0, ...].sum().backward()
        self.assertIsNotNone(verts.grad)

    def test_texture_map(self):
        """
        Test a mesh with a texture map is loaded and rendered correctly.
        The pupils in the eyes of the cow should always be looking to the left.
        """
        device = torch.device("cuda:0")
        obj_dir = Path(__file__).resolve().parent.parent / "docs/tutorials/data"
        obj_filename = obj_dir / "cow_mesh/cow.obj"

        # Load mesh + texture
        verts, faces, aux = load_obj(
            obj_filename, device=device, load_textures=True, texture_wrap=None
        )
        tex_map = list(aux.texture_images.values())[0]
        tex_map = tex_map[None, ...].to(faces.textures_idx.device)
        textures = TexturesUV(
            maps=tex_map, faces_uvs=[faces.textures_idx], verts_uvs=[aux.verts_uvs]
        )
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx], textures=textures)

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)

        # Place light behind the cow in world space. The front of
        # the cow is facing the -z direction.
        lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

        blend_params = BlendParams(
            sigma=1e-1,
            gamma=1e-4,
            background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
        )
        # Init renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=TexturedSoftPhongShader(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            ),
        )

        # Load reference image
        image_ref = load_rgb_image("test_texture_map_back.png", DATA_DIR)

        for bin_size in [0, None]:
            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size
            images = renderer(mesh)
            rgb = images[0, ..., :3].squeeze().cpu()

            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / "DEBUG_texture_map_back.png"
                )

            # NOTE some pixels can be flaky and will not lead to
            # `cond1` being true. Add `cond2` and check `cond1 or cond2`
            cond1 = torch.allclose(rgb, image_ref, atol=0.05)
            cond2 = ((rgb - image_ref).abs() > 0.05).sum() < 5
            self.assertTrue(cond1 or cond2)

        # Check grad exists
        [verts] = mesh.verts_list()
        verts.requires_grad = True
        mesh2 = Meshes(verts=[verts], faces=mesh.faces_list(), textures=mesh.textures)
        images = renderer(mesh2)
        images[0, ...].sum().backward()
        self.assertIsNotNone(verts.grad)

        ##########################################
        # Check rendering of the front of the cow
        ##########################################

        R, T = look_at_view_transform(2.7, 0, 180)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        # Move light to the front of the cow in world space
        lights.location = torch.tensor([0.0, 0.0, -2.0], device=device)[None]

        # Load reference image
        image_ref = load_rgb_image("test_texture_map_front.png", DATA_DIR)

        for bin_size in [0, None]:
            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size

            images = renderer(mesh, cameras=cameras, lights=lights)
            rgb = images[0, ..., :3].squeeze().cpu()

            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / "DEBUG_texture_map_front.png"
                )

            # NOTE some pixels can be flaky and will not lead to
            # `cond1` being true. Add `cond2` and check `cond1 or cond2`
            cond1 = torch.allclose(rgb, image_ref, atol=0.05)
            cond2 = ((rgb - image_ref).abs() > 0.05).sum() < 5
            self.assertTrue(cond1 or cond2)

        #################################
        # Add blurring to rasterization
        #################################
        R, T = look_at_view_transform(2.7, 0, 180)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        blend_params = BlendParams(sigma=5e-4, gamma=1e-4)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
            faces_per_pixel=100,
            clip_barycentric_coords=True,
        )

        # Load reference image
        image_ref = load_rgb_image("test_blurry_textured_rendering.png", DATA_DIR)

        for bin_size in [0, None]:
            # Check both naive and coarse to fine produce the same output.
            renderer.rasterizer.raster_settings.bin_size = bin_size

            images = renderer(
                mesh.clone(),
                cameras=cameras,
                raster_settings=raster_settings,
                blend_params=blend_params,
            )
            rgb = images[0, ..., :3].squeeze().cpu()

            if DEBUG:
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / "DEBUG_blurry_textured_rendering.png"
                )

            self.assertClose(rgb, image_ref, atol=0.05)

    def test_joined_spheres(self):
        """
        Test a list of Meshes can be joined as a single mesh and
        the single mesh is rendered correctly with Phong, Gouraud
        and Flat Shaders.
        """
        device = torch.device("cuda:0")

        # Init mesh with vertex textures.
        # Initialize a list containing two ico spheres of different sizes.
        sphere_list = [ico_sphere(3, device), ico_sphere(4, device)]
        # [(42 verts, 80 faces), (162 verts, 320 faces)]
        # The scale the vertices need to be set at to resize the spheres
        scales = [0.25, 1]
        # The distance the spheres ought to be offset horizontally to prevent overlap.
        offsets = [1.2, -0.3]
        # Initialize a list containing the adjusted sphere meshes.
        sphere_mesh_list = []
        for i in range(len(sphere_list)):
            verts = sphere_list[i].verts_padded() * scales[i]
            verts[0, :, 0] += offsets[i]
            sphere_mesh_list.append(
                Meshes(verts=verts, faces=sphere_list[i].faces_padded())
            )
        joined_sphere_mesh = join_mesh(sphere_mesh_list)
        joined_sphere_mesh.textures = TexturesVertex(
            verts_features=torch.ones_like(joined_sphere_mesh.verts_padded())
        )

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0.0, 0.0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, +2.0], device=device)[None]
        blend_params = BlendParams(1e-4, 1e-4, (0, 0, 0))

        # Init renderer
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
        shaders = {
            "phong": HardPhongShader,
            "gouraud": HardGouraudShader,
            "flat": HardFlatShader,
        }
        for (name, shader_init) in shaders.items():
            shader = shader_init(
                lights=lights,
                cameras=cameras,
                materials=materials,
                blend_params=blend_params,
            )
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            image = renderer(joined_sphere_mesh)
            rgb = image[..., :3].squeeze().cpu()
            if DEBUG:
                file_name = "DEBUG_joined_spheres_%s.png" % name
                Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / file_name
                )
            image_ref = load_rgb_image("test_joined_spheres_%s.png" % name, DATA_DIR)
            self.assertClose(rgb, image_ref, atol=0.05)

    def test_texture_map_atlas(self):
        """
        Test a mesh with a texture map as a per face atlas is loaded and rendered correctly.
        """
        device = torch.device("cuda:0")
        obj_dir = Path(__file__).resolve().parent.parent / "docs/tutorials/data"
        obj_filename = obj_dir / "cow_mesh/cow.obj"

        # Load mesh and texture as a per face texture atlas.
        verts, faces, aux = load_obj(
            obj_filename,
            device=device,
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=8,
            texture_wrap=None,
        )
        mesh = Meshes(
            verts=[verts],
            faces=[faces.verts_idx],
            textures=TexturesAtlas(atlas=[aux.texture_atlas]),
        )

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True
        )

        # Init shader settings
        materials = Materials(device=device, specular_color=((0, 0, 0),), shininess=0.0)
        lights = PointLights(device=device)

        # Place light behind the cow in world space. The front of
        # the cow is facing the -z direction.
        lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

        # The HardPhongShader can be used directly with atlas textures.
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=HardPhongShader(lights=lights, cameras=cameras, materials=materials),
        )

        images = renderer(mesh)
        rgb = images[0, ..., :3].squeeze().cpu()

        # Load reference image
        image_ref = load_rgb_image("test_texture_atlas_8x8_back.png", DATA_DIR)

        if DEBUG:
            Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / "DEBUG_texture_atlas_8x8_back.png"
            )

        self.assertClose(rgb, image_ref, atol=0.05)
