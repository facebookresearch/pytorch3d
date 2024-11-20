# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import unittest
from math import radians

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import IO
from pytorch3d.io.experimental_gltf_io import _read_header, MeshGlbFormat
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
from pytorch3d.utils import ico_sphere
from pytorch3d.vis.texture_vis import texturesuv_image_PIL

from .common_testing import get_pytorch3d_dir, get_tests_dir, TestCaseMixin


DATA_DIR = get_tests_dir() / "data"
TUTORIAL_DATA_DIR = get_pytorch3d_dir() / "docs/tutorials/data"
DEBUG = False


def _load(path, **kwargs) -> Meshes:
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    return io.load_mesh(path, **kwargs)


def _write(mesh, path, **kwargs) -> None:
    io = IO()
    io.register_meshes_format(MeshGlbFormat())
    io.save_mesh(mesh, path, **kwargs)

    with open(path, "rb") as f:
        _, stored_length = _read_header(f)  # pyre-ignore
    assert stored_length == os.path.getsize(path)


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
        self.assertClose(mesh.get_bounding_boxes().cpu(), mesh_obj.get_bounding_boxes())

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

    def test_save_cow(self):
        """
        Save the cow mesh to a glb file
        """
        # load cow mesh from a glb file
        glb = DATA_DIR / "cow.glb"
        self.assertTrue(glb.is_file())
        device = torch.device("cuda:0")
        mesh = _load(glb, device=device)

        # save the mesh to a glb file
        glb_reload = DATA_DIR / "cow_write.glb"
        _write(mesh, glb_reload)

        # load again
        self.assertTrue(glb_reload.is_file())
        device = torch.device("cuda:0")
        mesh_reload = _load(glb_reload, device=device)
        glb_reload.unlink()

        # assertions
        self.assertEqual(mesh_reload.faces_packed().shape, (5856, 3))
        self.assertEqual(mesh_reload.verts_packed().shape, (3225, 3))
        self.assertClose(
            mesh_reload.get_bounding_boxes().cpu(), mesh.get_bounding_boxes().cpu()
        )

        self.assertClose(
            mesh_reload.textures.verts_uvs_padded().cpu(),
            mesh.textures.verts_uvs_padded().cpu(),
        )

        self.assertClose(
            mesh_reload.textures.faces_uvs_padded().cpu(),
            mesh.textures.faces_uvs_padded().cpu(),
        )

        self.assertClose(
            mesh_reload.textures.maps_padded().cpu(), mesh.textures.maps_padded().cpu()
        )

    def test_save_ico_sphere(self):
        """
        save the ico_sphere mesh in a glb file
        """
        ico_sphere_mesh = ico_sphere(level=3)
        glb = DATA_DIR / "ico_sphere.glb"
        _write(ico_sphere_mesh, glb)

        # reload the ico_sphere
        device = torch.device("cuda:0")
        mesh_reload = _load(glb, device=device, include_textures=False)
        glb.unlink()

        self.assertClose(
            ico_sphere_mesh.verts_padded().cpu(),
            mesh_reload.verts_padded().cpu(),
        )

        self.assertClose(
            ico_sphere_mesh.faces_padded().cpu(),
            mesh_reload.faces_padded().cpu(),
        )

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
        self.assertClose(mesh.get_bounding_boxes().cpu(), mesh_obj.get_bounding_boxes())

        mesh.textures = TexturesVertex(0.5 * torch.ones_like(mesh.verts_padded()))

        image = _render(mesh, "cow_gray")

        with Image.open(DATA_DIR / "glb_cow_gray.png") as f:
            expected = np.array(f)

        self.assertClose(image, expected)

    def test_load_save_load_cow_texturesvertex(self):
        """
        Load the cow as converted to a single mesh in a glb file and then save it to a glb file.
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
        self.assertClose(mesh.get_bounding_boxes().cpu(), mesh_obj.get_bounding_boxes())

        mesh.textures = TexturesVertex(0.5 * torch.ones_like(mesh.verts_padded()))

        image = _render(mesh, "cow_gray")

        with Image.open(DATA_DIR / "glb_cow_gray.png") as f:
            expected = np.array(f)

        self.assertClose(image, expected)

        # save the mesh to a glb file
        glb = DATA_DIR / "cow_write_texturesvertex.glb"
        _write(mesh, glb)

        # reload the mesh glb file saved in TexturesVertex format
        self.assertTrue(glb.is_file())
        mesh_dash = _load(glb, device=device)
        glb.unlink()
        self.assertEqual(len(mesh_dash), 1)

        self.assertEqual(mesh_dash.faces_packed().shape, (5856, 3))
        self.assertEqual(mesh_dash.verts_packed().shape, (3225, 3))
        self.assertEqual(mesh_dash.textures.verts_features_list()[0].shape, (3225, 3))

        # check the re-rendered image with expected
        image_dash = _render(mesh, "cow_gray_texturesvertex")
        self.assertClose(image_dash, expected)

    def test_save_toy(self):
        """
        Construct a simple mesh and save it to a glb file in TexturesVertex mode.
        """

        example = {}
        example["POSITION"] = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                    [-1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0],
                ]
            ]
        )
        example["indices"] = torch.tensor(
            [
                [
                    [1, 4, 2],
                    [4, 3, 2],
                    [3, 7, 2],
                    [7, 6, 2],
                    [3, 4, 7],
                    [4, 8, 7],
                    [8, 5, 7],
                    [5, 6, 7],
                    [5, 2, 6],
                    [5, 1, 2],
                    [1, 5, 4],
                    [5, 8, 4],
                ]
            ]
        )
        example["indices"] -= 1
        example["COLOR_0"] = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ]
        )
        # example['prop'] = {'material':
        #                       {'pbrMetallicRoughness':
        #                           {'baseColorFactor':
        #                                torch.tensor([[0.7, 0.7, 1, 0.5]]),
        #                            'metallicFactor': torch.tensor([1]),
        #                            'roughnessFactor': torch.tensor([0.1])},
        #                    'alphaMode': 'BLEND',
        #                    'doubleSided': True}}

        texture = TexturesVertex(example["COLOR_0"])
        mesh = Meshes(
            verts=example["POSITION"], faces=example["indices"], textures=texture
        )

        glb = DATA_DIR / "example_write_texturesvertex.glb"
        _write(mesh, glb)
        glb.unlink()
