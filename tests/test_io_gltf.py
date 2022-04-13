# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from math import radians

import numpy as np
import torch
from common_testing import get_pytorch3d_dir, get_tests_dir, TestCaseMixin
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
    rotate_on_spot,
)
from pytorch3d.renderer.mesh import (
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.vis.texture_vis import texturesuv_image_PIL


DATA_DIR = get_tests_dir() / "data"
TUTORIAL_DATA_DIR = get_pytorch3d_dir() / "docs/tutorials/data"
DEBUG = False


def _load(path, **kwargs) -> Meshes:
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    return io.load_mesh(path, **kwargs)


def _render(
    mesh: Meshes,
    name: str,
    dist: float = 3.0,
    elev: float = 10.0,
    azim: float = 0,
    image_size: int = 256,
    pan=None,
    RT=None,
    use_ambient=False,
):
    device = mesh.device
    if RT is not None:
        R, T = RT
    else:
        R, T = look_at_view_transform(dist, elev, azim)
        if pan is not None:
            R, T = rotate_on_spot(R, T, pan)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1
    )

    # Init shader settings
    if use_ambient:
        lights = AmbientLights(device=device)
    else:
        lights = PointLights(device=device)
        lights.location = torch.tensor([0.0, 0.0, 2.0], device=device)[None]

    blend_params = BlendParams(
        sigma=1e-1,
        gamma=1e-4,
        background_color=torch.tensor([1.0, 1.0, 1.0], device=device),
    )
    # Init renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device, lights=lights, cameras=cameras, blend_params=blend_params
        ),
    )

    output = renderer(mesh)

    image = (output[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)

    if DEBUG:
        Image.fromarray(image).save(DATA_DIR / f"glb_{name}_.png")

    return image


class TestMeshGltfIO(TestCaseMixin, unittest.TestCase):
    def test_load_apartment(self):
        """
        This is the example habitat example scene from inside
        http://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip

        The scene is "already lit", i.e. the textures reflect the lighting
        already, so we want to render them with full ambient light.
        """
        self.skipTest("Data not available")

        glb = DATA_DIR / "apartment_1.glb"
        self.assertTrue(glb.is_file())
        device = torch.device("cuda:0")
        mesh = _load(glb, device=device)

        if DEBUG:
            texturesuv_image_PIL(mesh.textures).save(DATA_DIR / "out_apartment.png")

        for i in range(19):
            # random locations in the apartment
            eye = ((np.random.uniform(-6, 0.5), np.random.uniform(-8, 2), 0),)
            at = ((np.random.uniform(-6, 0.5), np.random.uniform(-8, 2), 0),)
            up = ((0, 0, -1),)
            RT = look_at_view_transform(eye=eye, at=at, up=up)
            _render(mesh, f"apartment_eau{i}", RT=RT, use_ambient=True)

        for i in range(12):
            # panning around the inner room from one location
            pan = axis_angle_to_matrix(torch.FloatTensor([0, radians(30 * i), 0]))
            _render(mesh, f"apartment{i}", 1.0, -90, pan, use_ambient=True)

    def test_load_cow(self):
        """
        Load the cow as converted to a single mesh in a glb file.
        """
        glb = DATA_DIR / "cow.glb"
        self.assertTrue(glb.is_file())
        device = torch.device("cuda:0")
        mesh = _load(glb, device=device)
        self.assertEqual(mesh.device, device)

        self.assertEqual(mesh.faces_packed().shape, (5856, 3))
        self.assertEqual(mesh.verts_packed().shape, (3225, 3))
        mesh_obj = _load(TUTORIAL_DATA_DIR / "cow_mesh/cow.obj")
        self.assertClose(
            mesh_obj.get_bounding_boxes().cpu(), mesh_obj.get_bounding_boxes()
        )

        self.assertClose(
            mesh.textures.verts_uvs_padded().cpu(), mesh_obj.textures.verts_uvs_padded()
        )

        self.assertClose(
            mesh.textures.faces_uvs_padded().cpu(), mesh_obj.textures.faces_uvs_padded()
        )

        self.assertClose(
            mesh.textures.maps_padded().cpu(), mesh_obj.textures.maps_padded()
        )

        if DEBUG:
            texturesuv_image_PIL(mesh.textures).save(DATA_DIR / "out_cow.png")

            image = _render(mesh, "cow", azim=4)
            with Image.open(DATA_DIR / "glb_cow.png") as f:
                expected = np.array(f)

            self.assertClose(image, expected)

    def test_load_cow_no_texture(self):
        """
        Load the cow as converted to a single mesh in a glb file.
        """
        glb = DATA_DIR / "cow.glb"
        self.assertTrue(glb.is_file())
        device = torch.device("cuda:0")
        mesh = _load(glb, device=device, include_textures=False)
        self.assertEqual(len(mesh), 1)
        self.assertIsNone(mesh.textures)

        self.assertEqual(mesh.faces_packed().shape, (5856, 3))
        self.assertEqual(mesh.verts_packed().shape, (3225, 3))
        mesh_obj = _load(TUTORIAL_DATA_DIR / "cow_mesh/cow.obj")
        self.assertClose(
            mesh_obj.get_bounding_boxes().cpu(), mesh_obj.get_bounding_boxes()
        )

        mesh.textures = TexturesVertex(0.5 * torch.ones_like(mesh.verts_padded()))

        image = _render(mesh, "cow_gray")

        with Image.open(DATA_DIR / "glb_cow_gray.png") as f:
            expected = np.array(f)

        self.assertClose(image, expected)
