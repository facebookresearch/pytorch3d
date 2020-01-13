#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


"""
Sanity checks for output images from the renderer.
"""
import numpy as np
import unittest
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from skimage.io import imread

from pytorch3d.io import load_obj
from pytorch3d.renderer.cameras import (
    OpenGLPerspectiveCameras,
    look_at_view_transform,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.rasterizer import (
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.shader import (
    BlendParams,
    GouradShader,
    PhongShader,
    SilhouetteShader,
    TexturedPhongShader,
)
from pytorch3d.renderer.mesh.texturing import Textures
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils.ico_sphere import ico_sphere

# Save out images generated in the tests for debugging
# All saved images have prefix DEBUG_
DEBUG = False
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_rgb_image(filename, data_dir=DATA_DIR):
    filepath = data_dir / filename
    image = torch.from_numpy(imread(filepath) / 255.0)
    image = image.to(dtype=torch.float32)
    return image[..., :3]


class TestRenderingMeshes(unittest.TestCase):
    def test_simple_sphere(self):
        """
        Test output of phong and gourad shading matches a reference image using
        the default values for the light sources.
        """
        device = torch.device("cuda:0")

        # Init mesh
        sphere_mesh = ico_sphere(5, device)
        verts_padded = sphere_mesh.verts_padded()
        faces_padded = sphere_mesh.faces_padded()
        textures = Textures(verts_rgb=torch.ones_like(verts_padded))
        sphere_mesh = Meshes(
            verts=verts_padded, faces=faces_padded, textures=textures
        )

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 0, 0)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, -2.0], device=device)[None]

        # Init renderer
        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        )
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=PhongShader(
                lights=lights, cameras=cameras, materials=materials
            ),
        )
        images = renderer(sphere_mesh)
        rgb = images[0, ..., :3].squeeze().cpu()
        if DEBUG:
            plt.imsave("DEBUG_simple_sphere_light.png", rgb.numpy())

        # Load reference image
        image_ref_phong = load_rgb_image("test_simple_sphere_illuminated.png")
        self.assertTrue(torch.allclose(rgb, image_ref_phong, atol=0.05))

        ###################################
        # Move the light behind the object
        ###################################
        # Check the image is dark
        lights.location[..., 2] = +2.0
        image_ref_phong_dark = load_rgb_image("test_simple_sphere_dark.png")
        images = renderer(sphere_mesh, lights=lights)
        rgb = images[0, ..., :3].squeeze().cpu()
        if DEBUG:
            plt.imsave("DEBUG_simple_sphere_dark.png", rgb.numpy())
        self.assertTrue(torch.allclose(rgb, image_ref_phong_dark, atol=0.05))

        ######################################
        # Change the shader to a GouradShader
        ######################################
        lights.location = torch.tensor([0.0, 0.0, -2.0], device=device)[None]
        renderer = MeshRenderer(
            rasterizer=rasterizer,
            shader=GouradShader(
                lights=lights, cameras=cameras, materials=materials
            ),
        )
        image_ref_gourad = load_rgb_image("test_simple_sphere_light_gourad.png")
        images = renderer(sphere_mesh)
        rgb = images[0, ..., :3].squeeze().cpu()
        if DEBUG:
            plt.imsave("DEBUG_simple_sphere_light_gourad.png", rgb.numpy())
        self.assertTrue(torch.allclose(rgb, image_ref_gourad, atol=0.005))
        self.assertFalse(torch.allclose(rgb, image_ref_phong, atol=0.005))

    def test_simple_sphere_batched(self):
        """
        Test output of phong shading matches a reference image using
        the default values for the light sources.
        """
        batch_size = 5
        device = torch.device("cuda:0")

        # Init mesh
        sphere_meshes = ico_sphere(5, device).extend(batch_size)
        verts_padded = sphere_meshes.verts_padded()
        faces_padded = sphere_meshes.faces_padded()
        textures = Textures(verts_rgb=torch.ones_like(verts_padded))
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
            image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, -2.0], device=device)[None]

        # Init renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings
            ),
            shader=PhongShader(
                lights=lights, cameras=cameras, materials=materials
            ),
        )
        images = renderer(sphere_meshes)

        # Load ref image
        image_ref = load_rgb_image("test_simple_sphere_illuminated.png")

        for i in range(batch_size):
            rgb = images[i, ..., :3].squeeze().cpu()
            if DEBUG:
                plt.imsave("DEBUG_simple_sphere_%d.png" % i, rgb.numpy())
            self.assertTrue(torch.allclose(rgb, image_ref, atol=0.05))

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
            bin_size=0,
        )

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 10, 20)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

        # Init renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings
            ),
            shader=SilhouetteShader(blend_params=blend_params),
        )
        images = renderer(sphere_mesh)
        alpha = images[0, ..., 3].squeeze().cpu()
        if DEBUG:
            plt.imsave("DEBUG_silhouette_grad.png", alpha.numpy())

        # Rescale image_ref to the 0-1 range
        image_ref = torch.from_numpy(imread(image_ref_filename, as_grey=True))
        min = image_ref.min()
        max = image_ref.max()
        image_ref_norm = (image_ref - min) / (max - min)
        image_ref_norm = image_ref_norm.to(dtype=torch.float32)
        self.assertTrue(torch.allclose(alpha, image_ref_norm, atol=0.055))

        # Check grad exist
        verts.requires_grad = True
        sphere_mesh = Meshes(verts=[verts], faces=[faces])
        images = renderer(sphere_mesh)
        images[0, ...].sum().backward()
        self.assertIsNotNone(verts.grad)

    def test_texture_map(self):
        """
        Test a mesh with a texture map is loaded and rendered correctly
        """
        device = torch.device("cuda:0")
        obj_filename = DATA_DIR / "textured_obj/spot_triangulated.obj"

        # Load mesh + texture
        verts, faces, aux = load_obj(obj_filename)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)
        texture_uvs = aux.verts_uvs
        materials = aux.material_colors
        tex_maps = aux.texture_images

        # tex_maps is a dictionary of material names as keys and texture images
        # as values. Only need the images for this example.
        textures = Textures(
            maps=list(tex_maps.values()),
            faces_uvs=faces.textures_idx.to(torch.int64).to(device)[None, :],
            verts_uvs=texture_uvs.to(torch.float32).to(device)[None, :],
        )
        mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

        # Init rasterizer settings
        R, T = look_at_view_transform(2.7, 10, 20)
        cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        # Init shader settings
        materials = Materials(device=device)
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, -2.0], device=device)[None]
        raster_settings = RasterizationSettings(
            image_size=512, blur_radius=0.0, faces_per_pixel=1, bin_size=0
        )

        # Init renderer
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings
            ),
            shader=TexturedPhongShader(
                lights=lights, cameras=cameras, materials=materials
            ),
        )
        images = renderer(mesh)
        rgb = images[0, ..., :3].squeeze().cpu()

        # Load reference image
        image_ref = load_rgb_image("test_texture_map.png")

        if DEBUG:
            plt.imsave("DEBUG_texture_map.png", rgb.numpy())

        self.assertTrue(torch.allclose(rgb, image_ref, atol=0.05))

        # Check grad exists
        verts = verts.clone()
        verts.requires_grad = True
        mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)
        images = renderer(mesh)
        images[0, ...].sum().backward()
        self.assertIsNotNone(verts.grad)
