# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import warnings
from collections import Counter
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import torch
from iopath.common.file_io import PathManager
from pytorch3d.io import (
    IO,
    load_obj,
    load_objs_as_meshes,
    save_obj,
    subset_obj,
)
from pytorch3d.io.obj_io import _Faces, _Aux
from pytorch3d.utils.obj_utils import parse_obj_to_mesh_by_texture
from pytorch3d.io.mtl_io import (
    _bilinear_interpolation_grid_sample,
    _bilinear_interpolation_vectorized,
    _parse_mtl,
)
from pytorch3d.renderer import (
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    PointLights,
    MeshRenderer,
    SoftPhongShader,
    MeshRasterizer,
)
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.structures import join_meshes_as_batch, join_meshes_as_scene, Meshes
from pytorch3d.utils import torus
from pytorch3d.ops.estimate_pointcloud_normals import estimate_pointcloud_normals
from pytorch3d.ops import sample_points_from_meshes, sample_points_from_obj
from .common_testing import (
    get_pytorch3d_dir,
    get_tests_dir,
    load_rgb_image,
    TestCaseMixin,
)
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = get_tests_dir() / "data"
TUTORIAL_DATA_DIR = get_pytorch3d_dir() / "docs/tutorials/data"


class TestMeshObjIO(TestCaseMixin, unittest.TestCase):
    def test_load_obj_simple(self):
        obj_file = "\n".join(
            [
                "# this is a comment",  # Comments should be ignored.
                "v 0.1 0.2 0.3",
                "v 0.2 0.3 0.4",
                "v 0.3 0.4 0.5",
                "v  0.4 0.5 0.6",  # some obj files have multiple spaces after v
                "f 1 2 3",
                "f 1 2 4 3 1",  # Polygons should be split into triangles
            ]
        )
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            verts, faces, aux = load_obj(Path(f.name))
            normals = aux.normals
            textures = aux.verts_uvs
            materials = aux.material_colors
            tex_maps = aux.texture_images

            expected_verts = torch.tensor(
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
                dtype=torch.float32,
            )
            expected_faces = torch.tensor(
                [
                    [0, 1, 2],  # First face
                    [0, 1, 3],  # Second face (polygon)
                    [0, 3, 2],  # Second face (polygon)
                    [0, 2, 0],  # Second face (polygon)
                ],
                dtype=torch.int64,
            )
            self.assertTrue(torch.all(verts == expected_verts))
            self.assertTrue(torch.all(faces.verts_idx == expected_faces))
            padded_vals = -(torch.ones_like(faces.verts_idx))
            self.assertTrue(torch.all(faces.normals_idx == padded_vals))
            self.assertTrue(torch.all(faces.textures_idx == padded_vals))
            self.assertTrue(
                torch.all(faces.materials_idx == -(torch.ones(len(expected_faces))))
            )
            self.assertTrue(normals is None)
            self.assertTrue(textures is None)
            self.assertTrue(materials is None)
            self.assertTrue(tex_maps is None)

    def test_load_obj_complex(self):
        obj_file = "\n".join(
            [
                "# this is a comment",  # Comments should be ignored.
                "v 0.1 0.2 0.3",
                "v 0.2 0.3 0.4",
                "v 0.3 0.4 0.5",
                "v 0.4 0.5 0.6",
                "vn 0.000000 0.000000 -1.000000",
                "vn -1.000000 -0.000000 -0.000000",
                "vn -0.000000 -0.000000 1.000000",  # Normals should not be ignored.
                "v 0.5 0.6 0.7",
                "vt 0.749279 0.501284 0.0",  # Some files add 0.0 - ignore this.
                "vt 0.999110 0.501077",
                "vt 0.999455 0.750380",
                "f 1 2 3",
                "f 1 2 4 3 5",  # Polygons should be split into triangles
                "f 2/1/2 3/1/2 4/2/2",  # Texture/normals are loaded correctly.
                "f -1 -2 1",  # Negative indexing counts from the end.
            ]
        )

        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            verts, faces, aux = load_obj(Path(f.name))
            normals = aux.normals
            textures = aux.verts_uvs
            materials = aux.material_colors
            tex_maps = aux.texture_images

            expected_verts = torch.tensor(
                [
                    [0.1, 0.2, 0.3],
                    [0.2, 0.3, 0.4],
                    [0.3, 0.4, 0.5],
                    [0.4, 0.5, 0.6],
                    [0.5, 0.6, 0.7],
                ],
                dtype=torch.float32,
            )
            expected_faces = torch.tensor(
                [
                    [0, 1, 2],  # First face
                    [0, 1, 3],  # Second face (polygon)
                    [0, 3, 2],  # Second face (polygon)
                    [0, 2, 4],  # Second face (polygon)
                    [1, 2, 3],  # Third face (normals / texture)
                    [4, 3, 0],  # Fourth face (negative indices)
                ],
                dtype=torch.int64,
            )
            expected_normals = torch.tensor(
                [
                    [0.000000, 0.000000, -1.000000],
                    [-1.000000, -0.000000, -0.000000],
                    [-0.000000, -0.000000, 1.000000],
                ],
                dtype=torch.float32,
            )
            expected_textures = torch.tensor(
                [[0.749279, 0.501284], [0.999110, 0.501077], [0.999455, 0.750380]],
                dtype=torch.float32,
            )
            expected_faces_normals_idx = -(
                torch.ones_like(expected_faces, dtype=torch.int64)
            )
            expected_faces_normals_idx[4, :] = torch.tensor(
                [1, 1, 1], dtype=torch.int64
            )
            expected_faces_textures_idx = -(
                torch.ones_like(expected_faces, dtype=torch.int64)
            )
            expected_faces_textures_idx[4, :] = torch.tensor(
                [0, 0, 1], dtype=torch.int64
            )

            self.assertTrue(torch.all(verts == expected_verts))
            self.assertTrue(torch.all(faces.verts_idx == expected_faces))
            self.assertClose(normals, expected_normals)
            self.assertClose(textures, expected_textures)
            self.assertClose(faces.normals_idx, expected_faces_normals_idx)
            self.assertClose(faces.textures_idx, expected_faces_textures_idx)
            self.assertTrue(materials is None)
            self.assertTrue(tex_maps is None)

    def test_load_obj_complex_pluggable(self):
        """
        This won't work on Windows due to the behavior of NamedTemporaryFile
        """
        obj_file = "\n".join(
            [
                "# this is a comment",  # Comments should be ignored.
                "v 0.1 0.2 0.3",
                "v 0.2 0.3 0.4",
                "v 0.3 0.4 0.5",
                "v 0.4 0.5 0.6",
                "vn 0.000000 0.000000 -1.000000",
                "vn -1.000000 -0.000000 -0.000000",
                "vn -0.000000 -0.000000 1.000000",  # Normals should not be ignored.
                "v 0.5 0.6 0.7",
                "vt 0.749279 0.501284 0.0",  # Some files add 0.0 - ignore this.
                "vt 0.999110 0.501077",
                "vt 0.999455 0.750380",
                "f 1 2 3",
                "f 1 2 4 3 5",  # Polygons should be split into triangles
                "f 2/1/2 3/1/2 4/2/2",  # Texture/normals are loaded correctly.
                "f -1 -2 1",  # Negative indexing counts from the end.
            ]
        )
        io = IO()
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()
            mesh = io.load_mesh(f.name)
            mesh_from_path = io.load_mesh(Path(f.name))

        with NamedTemporaryFile(mode="w", suffix=".ply") as f:
            f.write(obj_file)
            f.flush()
            with self.assertRaisesRegex(ValueError, "Invalid file header."):
                io.load_mesh(f.name)

        expected_verts = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
                [0.5, 0.6, 0.7],
            ],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor(
            [
                [0, 1, 2],  # First face
                [0, 1, 3],  # Second face (polygon)
                [0, 3, 2],  # Second face (polygon)
                [0, 2, 4],  # Second face (polygon)
                [1, 2, 3],  # Third face (normals / texture)
                [4, 3, 0],  # Fourth face (negative indices)
            ],
            dtype=torch.int64,
        )
        self.assertClose(mesh.verts_padded(), expected_verts[None])
        self.assertClose(mesh.faces_padded(), expected_faces[None])
        self.assertClose(mesh_from_path.verts_padded(), expected_verts[None])
        self.assertClose(mesh_from_path.faces_padded(), expected_faces[None])
        self.assertIsNone(mesh.textures)

    def test_load_obj_normals_only(self):
        obj_file = "\n".join(
            [
                "v 0.1 0.2 0.3",
                "v 0.2 0.3 0.4",
                "v 0.3 0.4 0.5",
                "v 0.4 0.5 0.6",
                "vn 0.000000 0.000000 -1.000000",
                "vn -1.000000 -0.000000 -0.000000",
                "f 2//1 3//1 4//2",
            ]
        )

        expected_faces_normals_idx = torch.tensor([[0, 0, 1]], dtype=torch.int64)
        expected_normals = torch.tensor(
            [[0.000000, 0.000000, -1.000000], [-1.000000, -0.000000, -0.000000]],
            dtype=torch.float32,
        )
        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )

        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            verts, faces, aux = load_obj(Path(f.name))
            normals = aux.normals
            textures = aux.verts_uvs
            materials = aux.material_colors
            tex_maps = aux.texture_images
            self.assertClose(faces.normals_idx, expected_faces_normals_idx)
            self.assertClose(normals, expected_normals)
            self.assertClose(verts, expected_verts)
            # Textures idx padded  with -1.
            self.assertClose(faces.textures_idx, torch.ones_like(faces.verts_idx) * -1)
            self.assertTrue(textures is None)
            self.assertTrue(materials is None)
            self.assertTrue(tex_maps is None)

    def test_load_obj_textures_only(self):
        obj_file = "\n".join(
            [
                "v 0.1 0.2 0.3",
                "v 0.2 0.3 0.4",
                "v 0.3 0.4 0.5",
                "v 0.4 0.5 0.6",
                "vt 0.999110 0.501077",
                "vt 0.999455 0.750380",
                "f 2/1 3/1 4/2",
            ]
        )

        expected_faces_textures_idx = torch.tensor([[0, 0, 1]], dtype=torch.int64)
        expected_textures = torch.tensor(
            [[0.999110, 0.501077], [0.999455, 0.750380]], dtype=torch.float32
        )
        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )

        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            verts, faces, aux = load_obj(Path(f.name))
            normals = aux.normals
            textures = aux.verts_uvs
            materials = aux.material_colors
            tex_maps = aux.texture_images

            self.assertClose(faces.textures_idx, expected_faces_textures_idx)
            self.assertClose(expected_textures, textures)
            self.assertClose(expected_verts, verts)
            self.assertTrue(
                torch.all(faces.normals_idx == -(torch.ones_like(faces.textures_idx)))
            )
            self.assertTrue(normals is None)
            self.assertTrue(materials is None)
            self.assertTrue(tex_maps is None)

    def test_load_obj_error_textures(self):
        obj_file = "\n".join(["vt 0.1"])
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertRaises(ValueError) as err:
                load_obj(Path(f.name))
            self.assertTrue("does not have 2 values" in str(err.exception))

    def test_load_obj_error_normals(self):
        obj_file = "\n".join(["vn 0.1"])
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertRaises(ValueError) as err:
                load_obj(Path(f.name))
            self.assertTrue("does not have 3 values" in str(err.exception))

    def test_load_obj_error_vertices(self):
        obj_file = "\n".join(["v 1"])
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertRaises(ValueError) as err:
                load_obj(Path(f.name))
            self.assertTrue("does not have 3 values" in str(err.exception))

    def test_load_obj_error_inconsistent_triplets(self):
        obj_file = "\n".join(["f 2//1 3/1 4/1/2"])
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertRaises(ValueError) as err:
                load_obj(Path(f.name))
            self.assertTrue("Vertex properties are inconsistent" in str(err.exception))

    def test_load_obj_error_too_many_vertex_properties(self):
        obj_file = "\n".join(["f 2/1/1/3"])
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertRaises(ValueError) as err:
                load_obj(Path(f.name))
            self.assertTrue(
                "Face vertices can only have 3 properties" in str(err.exception)
            )

    def test_load_obj_error_invalid_vertex_indices(self):
        obj_file = "\n".join(
            ["v 0.1 0.2 0.3", "v 0.1 0.2 0.3", "v 0.1 0.2 0.3", "f -2 5 1"]
        )
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertWarnsRegex(UserWarning, "Faces have invalid indices"):
                load_obj(Path(f.name))

    def test_load_obj_error_invalid_normal_indices(self):
        obj_file = "\n".join(
            [
                "v 0.1 0.2 0.3",
                "v 0.1 0.2 0.3",
                "v 0.1 0.2 0.3",
                "vn 0.1 0.2 0.3",
                "vn 0.1 0.2 0.3",
                "vn 0.1 0.2 0.3",
                "f -2/2 2/4 1/1",
            ]
        )
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertWarnsRegex(UserWarning, "Faces have invalid indices"):
                load_obj(Path(f.name))

    def test_load_obj_error_invalid_texture_indices(self):
        obj_file = "\n".join(
            [
                "v 0.1 0.2 0.3",
                "v 0.1 0.2 0.3",
                "v 0.1 0.2 0.3",
                "vt 0.1 0.2",
                "vt 0.1 0.2",
                "vt 0.1 0.2",
                "f -2//2 2//6 1//1",
            ]
        )
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertWarnsRegex(UserWarning, "Faces have invalid indices"):
                load_obj(Path(f.name))

    def test_save_obj_invalid_shapes(self):
        # Invalid vertices shape
        with self.assertRaises(ValueError) as error:
            verts = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])  # (V, 4)
            faces = torch.LongTensor([[0, 1, 2]])
            with NamedTemporaryFile(mode="w", suffix=".obj") as f:
                save_obj(Path(f.name), verts, faces)
        expected_message = (
            "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        )
        self.assertTrue(expected_message, error.exception)

        # Invalid faces shape
        with self.assertRaises(ValueError) as error:
            verts = torch.FloatTensor([[0.1, 0.2, 0.3]])
            faces = torch.LongTensor([[0, 1, 2, 3]])  # (F, 4)
            with NamedTemporaryFile(mode="w", suffix=".obj") as f:
                save_obj(Path(f.name), verts, faces)
        expected_message = (
            "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        )
        self.assertTrue(expected_message, error.exception)

    def test_save_obj_invalid_indices(self):
        message_regex = "Faces have invalid indices"
        verts = torch.FloatTensor([[0.1, 0.2, 0.3]])
        faces = torch.LongTensor([[0, 1, 2]])
        with self.assertWarnsRegex(UserWarning, message_regex):
            with NamedTemporaryFile(mode="w", suffix=".obj") as f:
                save_obj(Path(f.name), verts, faces)

        faces = torch.LongTensor([[-1, 0, 1]])
        with self.assertWarnsRegex(UserWarning, message_regex):
            with NamedTemporaryFile(mode="w", suffix=".obj") as f:
                save_obj(Path(f.name), verts, faces)

    def _test_save_load(self, verts, faces):
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            file_path = Path(f.name)
            save_obj(file_path, verts, faces)
            f.flush()

            expected_verts, expected_faces = verts, faces
            if not len(expected_verts):  # Always compare with a (V, 3) tensor
                expected_verts = torch.zeros(size=(0, 3), dtype=torch.float32)
            if not len(expected_faces):  # Always compare with an (F, 3) tensor
                expected_faces = torch.zeros(size=(0, 3), dtype=torch.int64)
            actual_verts, actual_faces, _ = load_obj(file_path)
            self.assertClose(expected_verts, actual_verts)
            self.assertClose(expected_faces, actual_faces.verts_idx)

    def test_empty_save_load_obj(self):
        # Vertices + empty faces
        verts = torch.FloatTensor([[0.1, 0.2, 0.3]])
        faces = torch.LongTensor([])
        self._test_save_load(verts, faces)

        faces = torch.zeros(size=(0, 3), dtype=torch.int64)
        self._test_save_load(verts, faces)

        # Faces + empty vertices
        message_regex = "Faces have invalid indices"
        verts = torch.FloatTensor([])
        faces = torch.LongTensor([[0, 1, 2]])
        with self.assertWarnsRegex(UserWarning, message_regex):
            self._test_save_load(verts, faces)

        verts = torch.zeros(size=(0, 3), dtype=torch.float32)
        with self.assertWarnsRegex(UserWarning, message_regex):
            self._test_save_load(verts, faces)

        # Empty vertices + empty faces
        message_regex = "Empty 'verts' and 'faces' arguments provided"
        verts0 = torch.FloatTensor([])
        faces0 = torch.LongTensor([])
        with self.assertWarnsRegex(UserWarning, message_regex):
            self._test_save_load(verts0, faces0)

        faces3 = torch.zeros(size=(0, 3), dtype=torch.int64)
        with self.assertWarnsRegex(UserWarning, message_regex):
            self._test_save_load(verts0, faces3)

        verts3 = torch.zeros(size=(0, 3), dtype=torch.float32)
        with self.assertWarnsRegex(UserWarning, message_regex):
            self._test_save_load(verts3, faces0)

        with self.assertWarnsRegex(UserWarning, message_regex):
            self._test_save_load(verts3, faces3)

    def test_save_obj(self):
        verts = torch.tensor(
            [[0.01, 0.2, 0.301], [0.2, 0.03, 0.408], [0.3, 0.4, 0.05], [0.6, 0.7, 0.8]],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 2], [3, 2, 1], [3, 1, 0]], dtype=torch.int64
        )
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            save_obj(Path(f.name), verts, faces, decimal_places=2)

            expected_file = "\n".join(
                [
                    "v 0.01 0.20 0.30",
                    "v 0.20 0.03 0.41",
                    "v 0.30 0.40 0.05",
                    "v 0.60 0.70 0.80",
                    "f 1 3 2",
                    "f 1 2 3",
                    "f 4 3 2",
                    "f 4 2 1",
                ]
            )
            self.assertEqual(Path(f.name).read_text(), expected_file)

    def test_load_mtl(self):
        obj_filename = "cow_mesh/cow.obj"
        filename = os.path.join(TUTORIAL_DATA_DIR, obj_filename)
        verts, faces, aux = load_obj(filename)
        materials = aux.material_colors
        tex_maps = aux.texture_images

        dtype = torch.float32
        expected_materials = {
            "material_1": {
                "ambient_color": torch.tensor([1.0, 1.0, 1.0], dtype=dtype),
                "diffuse_color": torch.tensor([1.0, 1.0, 1.0], dtype=dtype),
                "specular_color": torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
                "shininess": torch.tensor([10.0], dtype=dtype),
            }
        }
        # Texture atlas is not created as `create_texture_atlas=True` was
        # not set in the load_obj args
        self.assertTrue(aux.texture_atlas is None)
        # Check that there is an image with material name material_1.
        self.assertTrue(tuple(tex_maps.keys()) == ("material_1",))
        self.assertTrue(torch.is_tensor(tuple(tex_maps.values())[0]))
        self.assertTrue(
            torch.all(faces.materials_idx == torch.zeros(len(faces.verts_idx)))
        )

        # Check all keys and values in dictionary are the same.
        for n1, n2 in zip(materials.keys(), expected_materials.keys()):
            self.assertTrue(n1 == n2)
            for k1, k2 in zip(materials[n1].keys(), expected_materials[n2].keys()):
                self.assertTrue(
                    torch.allclose(materials[n1][k1], expected_materials[n2][k2])
                )

    def test_load_mtl_with_spaces_in_resource_filename(self):
        """
        Check that the texture image for materials in mtl files
        is loaded correctly even if there is a space in the file name
        e.g. material 1.png
        """
        mtl_file = "\n".join(
            [
                "newmtl material_1",
                "map_Kd material 1.png",
                "Ka 1.000 1.000 1.000",  # white
                "Kd 1.000 1.000 1.000",  # white
                "Ks 0.000 0.000 0.000",  # black
                "Ns 10.0",
            ]
        )
        with NamedTemporaryFile(mode="w", suffix=".mtl") as f:
            f.write(mtl_file)
            f.flush()

            material_properties, texture_files = _parse_mtl(
                Path(f.name), path_manager=PathManager(), device="cpu"
            )

            dtype = torch.float32
            expected_materials = {
                "material_1": {
                    "ambient_color": torch.tensor([1.0, 1.0, 1.0], dtype=dtype),
                    "diffuse_color": torch.tensor([1.0, 1.0, 1.0], dtype=dtype),
                    "specular_color": torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
                    "shininess": torch.tensor([10.0], dtype=dtype),
                }
            }
            # Check that there is a material with name material_1
            self.assertTrue(tuple(texture_files.keys()) == ("material_1",))
            # Check that there is an image with name material 1.png
            self.assertTrue(texture_files["material_1"] == "material 1.png")

            # Check all keys and values in dictionary are the same.
            for n1, n2 in zip(material_properties.keys(), expected_materials.keys()):
                self.assertTrue(n1 == n2)
                for k1, k2 in zip(
                    material_properties[n1].keys(), expected_materials[n2].keys()
                ):
                    self.assertTrue(
                        torch.allclose(
                            material_properties[n1][k1], expected_materials[n2][k2]
                        )
                    )

    def test_load_mtl_texture_atlas_compare_softras(self):
        # Load saved texture atlas created with SoftRas.
        device = torch.device("cuda:0")
        obj_filename = TUTORIAL_DATA_DIR / "cow_mesh/cow.obj"
        expected_atlas_fname = DATA_DIR / "cow_texture_atlas_softras.pt"

        # Note, the reference texture atlas generated using SoftRas load_obj function
        # is too large to check in to the repo. Download the file to run the test locally.
        if not os.path.exists(expected_atlas_fname):
            url = (
                "https://dl.fbaipublicfiles.com/pytorch3d/data/"
                "tests/cow_texture_atlas_softras.pt"
            )
            msg = (
                "cow_texture_atlas_softras.pt not found, download from %s, "
                "save it at the path %s, and rerun" % (url, expected_atlas_fname)
            )
            warnings.warn(msg)
            return True

        expected_atlas = torch.load(expected_atlas_fname)
        _, _, aux = load_obj(
            obj_filename,
            load_textures=True,
            device=device,
            create_texture_atlas=True,
            texture_atlas_size=15,
            texture_wrap="repeat",
        )

        self.assertClose(expected_atlas, aux.texture_atlas, atol=5e-5)

    def test_load_mtl_noload(self):
        obj_filename = "cow_mesh/cow.obj"
        filename = os.path.join(TUTORIAL_DATA_DIR, obj_filename)
        verts, faces, aux = load_obj(filename, load_textures=False)

        self.assertTrue(aux.material_colors is None)
        self.assertTrue(aux.texture_images is None)

    def test_load_no_usemtl(self):
        obj_filename = "missing_usemtl/cow.obj"
        # obj_filename has no "usemtl material_1" line
        filename = os.path.join(DATA_DIR, obj_filename)
        # TexturesUV type
        mesh = IO().load_mesh(filename)
        self.assertIsNotNone(mesh.textures)

        verts, faces, aux = load_obj(filename)
        self.assertTrue("material_1" in aux.material_colors)
        self.assertTrue("material_1" in aux.texture_images)

    def test_load_mtl_fail(self):
        # Faces have a material
        obj_file = "\n".join(
            [
                "v 0.1 0.2 0.3",
                "v 0.2 0.3 0.4",
                "v 0.3 0.4 0.5",
                "v 0.4 0.5 0.6",
                "usemtl material_1",
                "f 1 2 3",
                "f 1 2 4",
            ]
        )

        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()

            with self.assertWarnsRegex(UserWarning, "No mtl file provided"):
                verts, faces, aux = load_obj(Path(f.name))

            expected_verts = torch.tensor(
                [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
                dtype=torch.float32,
            )
            expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
            self.assertTrue(torch.allclose(verts, expected_verts))
            self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))
            self.assertTrue(aux.material_colors is None)
            self.assertTrue(aux.texture_images is None)
            self.assertTrue(aux.normals is None)
            self.assertTrue(aux.verts_uvs is None)

    def test_load_obj_mtl_no_image(self):
        obj_filename = "obj_mtl_no_image/model.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
        R = 8
        verts, faces, aux = load_obj(
            filename,
            load_textures=True,
            create_texture_atlas=True,
            texture_atlas_size=R,
            texture_wrap=None,
        )

        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))

        # Check that the material diffuse color has been assigned to all the
        # values in the texture atlas.
        expected_atlas = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float32)
        expected_atlas = expected_atlas[None, None, None, :].expand(2, R, R, -1)
        self.assertTrue(torch.allclose(aux.texture_atlas, expected_atlas))
        self.assertEqual(len(aux.material_colors.keys()), 1)
        self.assertEqual(list(aux.material_colors.keys()), ["material_1"])

    def test_load_obj_missing_texture(self):
        obj_filename = "missing_files_obj/model.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
        with self.assertWarnsRegex(UserWarning, "Texture file does not exist"):
            verts, faces, aux = load_obj(filename)

        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))

    def test_load_obj_missing_texture_noload(self):
        obj_filename = "missing_files_obj/model.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
        verts, faces, aux = load_obj(filename, load_textures=False)

        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))
        self.assertTrue(aux.material_colors is None)
        self.assertTrue(aux.texture_images is None)

    def test_load_obj_missing_mtl(self):
        obj_filename = "missing_files_obj/model2.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
        with self.assertWarnsRegex(UserWarning, "Mtl file does not exist"):
            verts, faces, aux = load_obj(filename)

        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))

    def test_load_obj_missing_mtl_noload(self):
        obj_filename = "missing_files_obj/model2.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
        verts, faces, aux = load_obj(filename, load_textures=False)

        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))
        self.assertTrue(aux.material_colors is None)
        self.assertTrue(aux.texture_images is None)

    def test_join_meshes_as_batch(self):
        """
        Test that join_meshes_as_batch and load_objs_as_meshes are consistent
        with single meshes.
        """

        def check_triple(mesh, mesh3):
            """
            Verify that mesh3 is three copies of mesh.
            """

            def check_item(x, y):
                self.assertEqual(x is None, y is None)
                if x is not None:
                    self.assertClose(torch.cat([x, x, x]), y)

            check_item(mesh.verts_padded(), mesh3.verts_padded())
            check_item(mesh.faces_padded(), mesh3.faces_padded())

            if mesh.textures is not None:
                if isinstance(mesh.textures, TexturesUV):
                    check_item(
                        mesh.textures.faces_uvs_padded(),
                        mesh3.textures.faces_uvs_padded(),
                    )
                    check_item(
                        mesh.textures.verts_uvs_padded(),
                        mesh3.textures.verts_uvs_padded(),
                    )
                    check_item(
                        mesh.textures.maps_padded(), mesh3.textures.maps_padded()
                    )
                elif isinstance(mesh.textures, TexturesVertex):
                    check_item(
                        mesh.textures.verts_features_padded(),
                        mesh3.textures.verts_features_padded(),
                    )
                elif isinstance(mesh.textures, TexturesAtlas):
                    check_item(
                        mesh.textures.atlas_padded(), mesh3.textures.atlas_padded()
                    )

        obj_filename = TUTORIAL_DATA_DIR / "cow_mesh/cow.obj"

        mesh = load_objs_as_meshes([obj_filename])
        mesh3 = load_objs_as_meshes([obj_filename, obj_filename, obj_filename])
        check_triple(mesh, mesh3)
        self.assertTupleEqual(mesh.textures.maps_padded().shape, (1, 1024, 1024, 3))

        # Try mismatched texture map sizes, which needs a call to interpolate()
        mesh2048 = mesh.clone()
        maps = mesh.textures.maps_padded()
        mesh2048.textures._maps_padded = torch.cat([maps, maps], dim=1)
        join_meshes_as_batch([mesh.to("cuda:0"), mesh2048.to("cuda:0")])

        mesh_notex = load_objs_as_meshes([obj_filename], load_textures=False)
        mesh3_notex = load_objs_as_meshes(
            [obj_filename, obj_filename, obj_filename], load_textures=False
        )
        check_triple(mesh_notex, mesh3_notex)
        self.assertIsNone(mesh_notex.textures)

        # meshes with vertex texture, join into a batch.
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        vert_tex = torch.ones_like(verts)
        rgb_tex = TexturesVertex(verts_features=[vert_tex])
        mesh_rgb = Meshes(verts=[verts], faces=[faces], textures=rgb_tex)
        mesh_rgb3 = join_meshes_as_batch([mesh_rgb, mesh_rgb, mesh_rgb])
        check_triple(mesh_rgb, mesh_rgb3)
        nums_rgb = mesh_rgb.textures._num_verts_per_mesh
        nums_rgb3 = mesh_rgb3.textures._num_verts_per_mesh
        self.assertEqual(type(nums_rgb), list)
        self.assertEqual(type(nums_rgb3), list)
        self.assertListEqual(nums_rgb * 3, nums_rgb3)

        # meshes with texture atlas, join into a batch.
        device = "cuda:0"
        atlas = torch.rand((2, 4, 4, 3), dtype=torch.float32, device=device)
        atlas_tex = TexturesAtlas(atlas=[atlas])
        mesh_atlas = Meshes(verts=[verts], faces=[faces], textures=atlas_tex)
        mesh_atlas3 = join_meshes_as_batch([mesh_atlas, mesh_atlas, mesh_atlas])
        check_triple(mesh_atlas, mesh_atlas3)

        # Test load multiple meshes with textures into a batch.
        teapot_obj = TUTORIAL_DATA_DIR / "teapot.obj"
        mesh_teapot = load_objs_as_meshes([teapot_obj])
        teapot_verts, teapot_faces = mesh_teapot.get_mesh_verts_faces(0)
        mix_mesh = load_objs_as_meshes([obj_filename, teapot_obj], load_textures=False)
        self.assertEqual(len(mix_mesh), 2)
        self.assertClose(mix_mesh.verts_list()[0], mesh.verts_list()[0])
        self.assertClose(mix_mesh.faces_list()[0], mesh.faces_list()[0])
        self.assertClose(mix_mesh.verts_list()[1], teapot_verts)
        self.assertClose(mix_mesh.faces_list()[1], teapot_faces)

        cow3_tea = join_meshes_as_batch([mesh3, mesh_teapot], include_textures=False)
        self.assertEqual(len(cow3_tea), 4)
        check_triple(mesh_notex, cow3_tea[:3])
        self.assertClose(cow3_tea.verts_list()[3], mesh_teapot.verts_list()[0])
        self.assertClose(cow3_tea.faces_list()[3], mesh_teapot.faces_list()[0])

        # Check error raised if all meshes in the batch don't have the same texture type
        with self.assertRaisesRegex(ValueError, "same type of texture"):
            join_meshes_as_batch([mesh_atlas, mesh_rgb, mesh_atlas])

    def test_save_obj_with_normal(self):
        verts = torch.tensor(
            [[0.01, 0.2, 0.301], [0.2, 0.03, 0.408], [0.3, 0.4, 0.05], [0.6, 0.7, 0.8]],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 2], [3, 2, 1], [3, 1, 0]], dtype=torch.int64
        )
        normals = torch.tensor(
            [
                [0.02, 0.5, 0.73],
                [0.3, 0.03, 0.361],
                [0.32, 0.12, 0.47],
                [0.36, 0.17, 0.9],
                [0.40, 0.7, 0.19],
                [1.0, 0.00, 0.000],
                [0.00, 1.00, 0.00],
                [0.00, 0.00, 1.0],
            ],
            dtype=torch.float32,
        )
        faces_normals_idx = torch.tensor(
            [[0, 1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 0]], dtype=torch.int64
        )

        with TemporaryDirectory() as temp_dir:
            obj_file = os.path.join(temp_dir, "mesh.obj")
            save_obj(
                obj_file,
                verts,
                faces,
                decimal_places=2,
                normals=normals,
                faces_normals_idx=faces_normals_idx,
            )

            expected_obj_file = "\n".join(
                [
                    "v 0.01 0.20 0.30",
                    "v 0.20 0.03 0.41",
                    "v 0.30 0.40 0.05",
                    "v 0.60 0.70 0.80",
                    "vn 0.02 0.50 0.73",
                    "vn 0.30 0.03 0.36",
                    "vn 0.32 0.12 0.47",
                    "vn 0.36 0.17 0.90",
                    "vn 0.40 0.70 0.19",
                    "vn 1.00 0.00 0.00",
                    "vn 0.00 1.00 0.00",
                    "vn 0.00 0.00 1.00",
                    "f 1//1 3//2 2//3",
                    "f 1//3 2//4 3//5",
                    "f 4//5 3//6 2//7",
                    "f 4//7 2//8 1//1",
                ]
            )

            # Check the obj file is saved correctly
            with open(obj_file, "r") as actual_file:
                self.assertEqual(actual_file.read(), expected_obj_file)

    def test_save_obj_with_texture(self):
        verts = torch.tensor(
            [[0.01, 0.2, 0.301], [0.2, 0.03, 0.408], [0.3, 0.4, 0.05], [0.6, 0.7, 0.8]],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 2], [3, 2, 1], [3, 1, 0]], dtype=torch.int64
        )
        verts_uvs = torch.tensor(
            [[0.02, 0.5], [0.3, 0.03], [0.32, 0.12], [0.36, 0.17]],
            dtype=torch.float32,
        )
        faces_uvs = faces
        texture_map = torch.randint(size=(2, 2, 3), high=255) / 255.0

        with TemporaryDirectory() as temp_dir:
            obj_file = os.path.join(temp_dir, "mesh.obj")
            save_obj(
                obj_file,
                verts,
                faces,
                decimal_places=2,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs,
                texture_map=texture_map,
            )

            expected_obj_file = "\n".join(
                [
                    "",
                    "mtllib mesh.mtl",
                    "usemtl mesh",
                    "",
                    "v 0.01 0.20 0.30",
                    "v 0.20 0.03 0.41",
                    "v 0.30 0.40 0.05",
                    "v 0.60 0.70 0.80",
                    "vt 0.02 0.50",
                    "vt 0.30 0.03",
                    "vt 0.32 0.12",
                    "vt 0.36 0.17",
                    "f 1/1 3/3 2/2",
                    "f 1/1 2/2 3/3",
                    "f 4/4 3/3 2/2",
                    "f 4/4 2/2 1/1",
                ]
            )
            expected_mtl_file = "\n".join(["newmtl mesh", "map_Kd mesh.png", ""])

            # Check there are only 3 files in the temp dir
            tempfiles = ["mesh.obj", "mesh.png", "mesh.mtl"]
            tempfiles_dir = os.listdir(temp_dir)
            self.assertEqual(Counter(tempfiles), Counter(tempfiles_dir))

            # Check the obj file is saved correctly
            with open(obj_file, "r") as actual_file:
                self.assertEqual(actual_file.read(), expected_obj_file)

            # Check the mtl file is saved correctly
            mtl_file_name = os.path.join(temp_dir, "mesh.mtl")
            with open(mtl_file_name, "r") as mtl_file:
                self.assertEqual(mtl_file.read(), expected_mtl_file)

            # Check the texture image file is saved correctly
            texture_image = load_rgb_image("mesh.png", temp_dir)
            self.assertClose(texture_image, texture_map)

    def test_save_obj_with_normal_and_texture(self):
        verts = torch.tensor(
            [[0.01, 0.2, 0.301], [0.2, 0.03, 0.408], [0.3, 0.4, 0.05], [0.6, 0.7, 0.8]],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 2], [3, 2, 1], [3, 1, 0]], dtype=torch.int64
        )
        normals = torch.tensor(
            [
                [0.02, 0.5, 0.73],
                [0.3, 0.03, 0.361],
                [0.32, 0.12, 0.47],
                [0.36, 0.17, 0.9],
            ],
            dtype=torch.float32,
        )
        faces_normals_idx = faces
        verts_uvs = torch.tensor(
            [[0.02, 0.5], [0.3, 0.03], [0.32, 0.12], [0.36, 0.17]],
            dtype=torch.float32,
        )
        faces_uvs = faces
        texture_map = torch.randint(size=(2, 2, 3), high=255) / 255.0

        with TemporaryDirectory() as temp_dir:
            obj_file = os.path.join(temp_dir, "mesh.obj")
            save_obj(
                obj_file,
                verts,
                faces,
                decimal_places=2,
                normals=normals,
                faces_normals_idx=faces_normals_idx,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs,
                texture_map=texture_map,
            )

            expected_obj_file = "\n".join(
                [
                    "",
                    "mtllib mesh.mtl",
                    "usemtl mesh",
                    "",
                    "v 0.01 0.20 0.30",
                    "v 0.20 0.03 0.41",
                    "v 0.30 0.40 0.05",
                    "v 0.60 0.70 0.80",
                    "vn 0.02 0.50 0.73",
                    "vn 0.30 0.03 0.36",
                    "vn 0.32 0.12 0.47",
                    "vn 0.36 0.17 0.90",
                    "vt 0.02 0.50",
                    "vt 0.30 0.03",
                    "vt 0.32 0.12",
                    "vt 0.36 0.17",
                    "f 1/1/1 3/3/3 2/2/2",
                    "f 1/1/1 2/2/2 3/3/3",
                    "f 4/4/4 3/3/3 2/2/2",
                    "f 4/4/4 2/2/2 1/1/1",
                ]
            )
            expected_mtl_file = "\n".join(["newmtl mesh", "map_Kd mesh.png", ""])

            # Check there are only 3 files in the temp dir
            tempfiles = ["mesh.obj", "mesh.png", "mesh.mtl"]
            tempfiles_dir = os.listdir(temp_dir)
            self.assertEqual(Counter(tempfiles), Counter(tempfiles_dir))

            # Check the obj file is saved correctly
            with open(obj_file, "r") as actual_file:
                self.assertEqual(actual_file.read(), expected_obj_file)

            # Check the mtl file is saved correctly
            mtl_file_name = os.path.join(temp_dir, "mesh.mtl")
            with open(mtl_file_name, "r") as mtl_file:
                self.assertEqual(mtl_file.read(), expected_mtl_file)

            # Check the texture image file is saved correctly
            texture_image = load_rgb_image("mesh.png", temp_dir)
            self.assertClose(texture_image, texture_map)

    def test_save_obj_with_texture_errors(self):
        verts = torch.tensor(
            [[0.01, 0.2, 0.301], [0.2, 0.03, 0.408], [0.3, 0.4, 0.05], [0.6, 0.7, 0.8]],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 2], [3, 2, 1], [3, 1, 0]], dtype=torch.int64
        )
        verts_uvs = torch.tensor(
            [[0.02, 0.5], [0.3, 0.03], [0.32, 0.12], [0.36, 0.17]],
            dtype=torch.float32,
        )
        faces_uvs = faces
        texture_map = torch.randint(size=(2, 2, 3), high=255)

        expected_obj_file = "\n".join(
            [
                "v 0.01 0.20 0.30",
                "v 0.20 0.03 0.41",
                "v 0.30 0.40 0.05",
                "v 0.60 0.70 0.80",
                "f 1 3 2",
                "f 1 2 3",
                "f 4 3 2",
                "f 4 2 1",
            ]
        )
        with TemporaryDirectory() as temp_dir:
            obj_file = os.path.join(temp_dir, "mesh.obj")

            # If only one of verts_uvs/faces_uvs/texture_map is provided
            # then textures are not saved
            for arg in [
                {"verts_uvs": verts_uvs},
                {"faces_uvs": faces_uvs},
                {"texture_map": texture_map},
            ]:
                save_obj(
                    obj_file,
                    verts,
                    faces,
                    decimal_places=2,
                    **arg,
                )

                # Check there is only 1 file in the temp dir
                tempfiles = ["mesh.obj"]
                tempfiles_dir = os.listdir(temp_dir)
                self.assertEqual(tempfiles, tempfiles_dir)

                # Check the obj file is saved correctly
                with open(obj_file, "r") as actual_file:
                    self.assertEqual(actual_file.read(), expected_obj_file)

        obj_file = StringIO()
        with self.assertRaises(ValueError):
            save_obj(
                obj_file,
                verts,
                faces,
                decimal_places=2,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs[..., 2],  # Incorrect shape
                texture_map=texture_map,
            )

        with self.assertRaises(ValueError):
            save_obj(
                obj_file,
                verts,
                faces,
                decimal_places=2,
                verts_uvs=verts_uvs[..., 0],  # Incorrect shape
                faces_uvs=faces_uvs,
                texture_map=texture_map,
            )

        with self.assertRaises(ValueError):
            save_obj(
                obj_file,
                verts,
                faces,
                decimal_places=2,
                verts_uvs=verts_uvs,
                faces_uvs=faces_uvs,
                texture_map=texture_map[..., 1],  # Incorrect shape
            )

    def test_save_obj_with_texture_IO(self):
        verts = torch.tensor(
            [[0.01, 0.2, 0.301], [0.2, 0.03, 0.408], [0.3, 0.4, 0.05], [0.6, 0.7, 0.8]],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 2], [3, 2, 1], [3, 1, 0]], dtype=torch.int64
        )
        verts_uvs = torch.tensor(
            [[0.02, 0.5], [0.3, 0.03], [0.32, 0.12], [0.36, 0.17]],
            dtype=torch.float32,
        )
        faces_uvs = faces
        texture_map = torch.randint(size=(2, 2, 3), high=255) / 255.0

        with TemporaryDirectory() as temp_dir:
            obj_file = os.path.join(temp_dir, "mesh.obj")
            textures_uv = TexturesUV([texture_map], [faces_uvs], [verts_uvs])
            test_mesh = Meshes(verts=[verts], faces=[faces], textures=textures_uv)

            IO().save_mesh(data=test_mesh, path=obj_file, decimal_places=2)

            expected_obj_file = "\n".join(
                [
                    "",
                    "mtllib mesh.mtl",
                    "usemtl mesh",
                    "",
                    "v 0.01 0.20 0.30",
                    "v 0.20 0.03 0.41",
                    "v 0.30 0.40 0.05",
                    "v 0.60 0.70 0.80",
                    "vt 0.02 0.50",
                    "vt 0.30 0.03",
                    "vt 0.32 0.12",
                    "vt 0.36 0.17",
                    "f 1/1 3/3 2/2",
                    "f 1/1 2/2 3/3",
                    "f 4/4 3/3 2/2",
                    "f 4/4 2/2 1/1",
                ]
            )
            expected_mtl_file = "\n".join(["newmtl mesh", "map_Kd mesh.png", ""])

            # Check there are only 3 files in the temp dir
            tempfiles = ["mesh.obj", "mesh.png", "mesh.mtl"]
            tempfiles_dir = os.listdir(temp_dir)
            self.assertEqual(Counter(tempfiles), Counter(tempfiles_dir))

            # Check the obj file is saved correctly
            with open(obj_file, "r") as actual_file:
                self.assertEqual(actual_file.read(), expected_obj_file)

            # Check the mtl file is saved correctly
            mtl_file_name = os.path.join(temp_dir, "mesh.mtl")
            with open(mtl_file_name, "r") as mtl_file:
                self.assertEqual(mtl_file.read(), expected_mtl_file)

            # Check the texture image file is saved correctly
            texture_image = load_rgb_image("mesh.png", temp_dir)
            self.assertClose(texture_image, texture_map)

    def test_multitexture_obj_IO(self):
        """checking IO with multiple txtures.
        Coverage for the following functions and their helper functions:
            - obj_io.load_obj
            - obj_io.subset_obj
            - obj_io.save_obj
            - utils.obj_utils.parse_obj_to_mesh_by_texture
        """
        obj_filename = "cow_mesh/cow.obj"
        filename = os.path.join(TUTORIAL_DATA_DIR, obj_filename)
        # load the cow mesh into as its individual elements
        verts, faces, aux = load_obj(
            filename, load_textures=True, create_texture_atlas=True, texture_wrap=None
        )
        # generate face normals based on verts to test subsetting operations
        normals = estimate_pointcloud_normals(obj[0][None]).squeeze()

        # create a new obj tuple from its individual components with normals
        _faces = _Faces(
            verts_idx=faces.verts_idx,
            normals_idx=faces.verts_idx,  # include face normals
            textures_idx=faces.textures_idx,
            materials_idx=faces.materials_idx,
        )
        _texture_images = dict(
            material_1=aux.texture_images["material_1"],
        )
        _material_colors = dict(
            material_1=aux.material_colors["material_1"],
        )
        _aux = _Aux(
            normals=normals,
            verts_uvs=aux.verts_uvs,
            material_colors=_material_colors,
            texture_images=_texture_images,
            texture_atlas=aux.texture_atlas,
        )
        # create a new obj object
        obj = (verts, _faces, _aux)

        # test internal function for parsing objs to mesh list
        with self.assertRaises(ValueError) as err:
            parse_obj_to_mesh_by_texture(
                verts=verts,
                faces=faces.verts_idx,
                verts_uvs=aux.verts_uvs,
                faces_uvs=faces.textures_idx,
                texture_images=list(aux.texture_images.items()),
                device=verts.device,
                materials_idx=faces.materials_idx,
                texture_atlas=aux.texture_atlas,
                normals=normals,
                faces_normals_idx=faces.verts_idx
            )
        self.assertTrue("texture_images must be a dictionary" in str(err.exception))

        list_of_meshes = parse_obj_to_mesh_by_texture(
            verts=verts,
            faces=faces.verts_idx,
            verts_uvs=aux.verts_uvs,
            faces_uvs=faces.textures_idx,
            texture_images=aux.texture_images,
            device=verts.device,
            materials_idx=faces.materials_idx,
            texture_atlas=aux.texture_atlas,
            normals=normals,
            faces_normals_idx=faces.verts_idx
        )

        mesh = join_meshes_as_scene(list_of_meshes)

        self.assertTrue(mesh.verts_packed().shape[0] == verts.shape[0])
        self.assertTrue(mesh.faces_packed().shape[0] == faces.verts_idx.shape[0])

        # check error conditions
        with self.assertRaises(ValueError) as err:
            subset_obj(obj=obj, faces_to_subset=torch.tensor([]), device=verts.device)
        self.assertTrue("faces_to_subset is empty." in str(err.exception))

        with self.assertRaises(ValueError) as err:
            subset_obj(obj=obj, faces_to_subset=torch.tensor([-1]), device=verts.device)
        self.assertTrue(
            "faces_to_subset contains invalid indices." in str(err.exception)
        )

        with self.assertRaises(ValueError) as err:
            subset_obj(
                obj=obj,
                faces_to_subset=torch.tensor([0, 1, obj[1].verts_idx.shape[0]]),
                device=verts.device,
            )
        self.assertTrue(
            "faces_to_subset contains invalid indices." in str(err.exception)
        )

        with self.assertRaises(ValueError) as err:
            subset_obj(
                obj=obj, faces_to_subset=np.array([0, 1, 2, 3]), device=verts.device
            )
        self.assertTrue("faces_to_subset must be a torch.Tensor" in str(err.exception))

        message = "Face indices are repeated in faces_to_subset."
        with self.assertWarnsRegex(UserWarning, message):
            subset_obj(
                obj=obj,
                faces_to_subset=torch.tensor([0, 0, 0, 1, 1, 1, 2]).to(verts.device),
                device=verts.device,
            )

        with self.assertRaises(ValueError) as err:
            subset_obj(
                obj=(obj[0],),
                faces_to_subset=np.array([0, 1, 2, 3]),
                device=verts.device,
            )
        self.assertTrue("obj must be 3-tuple" in str(err.exception))

        with self.assertRaises(ValueError) as err:
            subset_obj(
                obj=(obj[0], torch.tensor([0, 1]), obj[2]),
                faces_to_subset=np.array([0, 1, 2, 3]),
                device=verts.device,
            )
        self.assertTrue(
            "obj[1] must be a _Faces NamedTuple object that defines obj faces"
            in str(err.exception)
        )

        with self.assertRaises(ValueError) as err:
            subset_obj(
                obj=(obj[0], obj[1], torch.tensor([0, 1])),
                faces_to_subset=np.array([0, 1, 2, 3]),
                device=verts.device,
            )
        self.assertTrue(
            "obj[2] must be an _Aux NamedTuple object that defines obj properties"
            in str(err.exception)
        )

        with self.assertRaises(ValueError) as err:
            subset_obj(
                obj=obj,
                faces_to_subset=torch.tensor([0, 1, 2, 3]).to(torch.device("cuda:0")),
                device=torch.device("cpu"),
            )
        self.assertTrue(
            "obj and faces_to_subset are not on the same device" in str(err.exception)
        )

        # set up renderer according to tutorial
        R, T = look_at_view_transform(2.7, 0, 180)
        cameras = FoVPerspectiveCameras(device=verts.device, R=R, T=T)

        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=verts.device, location=[[0.0, 0.0, -3.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=verts.device, cameras=cameras, lights=lights),
        )

        # simulate aribitrary obj segmentation by quadrant
        quadrants = [1, 2, 3, 4]
        # use a copy of materails_idx to apply new materials for testing
        quadrants_materials = _faces.materials_idx.clone()

        # for each qudrant, output an obj of only the current faces
        for quadrant_ix, quadrant in enumerate(quadrants):
            # select verts by their quadrant in 3d space
            verts_idx_to_select = self._split_verts_by_quadrant(verts, quadrant)
            verts_idx = faces.verts_idx.numpy()
            # mask provides which faces contain at least one vert that belong the current quadrant
            mask = np.in1d(verts_idx, verts_idx_to_select).reshape(verts_idx.shape)
            # select the faces, by index, that belong to a given quadrant as faces_to_subset
            faces_to_subset = np.where(np.any(mask, axis=1))[0]
            # apply zero-index quadrant refernces to materials_idx as a psuedo label
            quadrants_materials[faces_to_subset] = quadrant_ix
            # apply faces_to_subset to the obj
            obj_subset = subset_obj(
                obj=obj,
                faces_to_subset=torch.from_numpy(faces_to_subset),
                device=verts.device,
            )
            # split the subset obj into its elements for reference
            _verts, _faces, _aux = obj_subset[0], obj_subset[1], obj_subset[2]
            # generate an array of index to reference faces
            _test_index = torch.arange(_faces.verts_idx.shape[0])
            # check normals are split property according to origin data
            self.assertClose(
                obj[2].normals[obj[1].normals_idx[faces_to_subset]],
                obj_subset[2].normals[_faces.normals_idx[_test_index]],
            )
            # check texture atlas are split property according to origin data
            self.assertClose(
                obj[2].texture_atlas[faces_to_subset],
                obj_subset[2].texture_atlas[_test_index],
            )
            # check verts_uvs are split properly according to origin data
            self.assertClose(
                obj[2].verts_uvs[torch.unique(obj[1].textures_idx[faces_to_subset])],
                obj_subset[2].verts_uvs,
            )
            # check that output faces are equal to expected input size
            self.assertEqual(faces_to_subset.shape[0], obj_subset[1].verts_idx.shape[0])
            # assert the obj_subset is valid given a warning message ("Faces have invalid indices") about invalid dims in obj_io
            self.assertFalse(obj_subset[1].verts_idx.max() >= obj_subset[0].shape[0])
            self.assertFalse(obj_subset[1].verts_idx.min() < 0)
            # check that material images are the same as the origin data, if they are part of the subset
            for material_key, material_value in obj[2].material_colors.items():
                if material_key in obj_subset[2].material_colors.keys():
                    np.testing.assert_array_equal(
                        material_value, obj_subset[2].material_colors[material_key]
                    )
            # configure variable settings for writing mtl names
            image_name_kwargs = dict(
                material_name_as_file_name=True if quadrant == 1 else False,
                reuse_material_files=True if quadrant in [2, 3] else False,
            )
            if quadrant == 4:
                image_name_kwargs = None

            obj_basename = "test_multitexture_obj_IO_Q"
            obj_name = f"{obj_basename}{quadrant}.obj"
            obj_file = os.path.join(DATA_DIR, obj_name)

            # save an obj to disk having multiple textures
            save_obj(
                f=obj_file,
                verts=obj_subset[0],
                faces=obj_subset[1].verts_idx,
                verts_uvs=obj_subset[2].verts_uvs,
                faces_uvs=obj_subset[1].textures_idx,
                texture_images=obj_subset[2].texture_images,
                materials_idx=obj_subset[1].materials_idx,
                image_name_kwargs=image_name_kwargs,
                normals=obj_subset[2].normals,
                faces_normals_idx=obj_subset[1].normals_idx
            )

            # test IO on subset obj as meshes object
            mesh = load_objs_as_meshes(files=[obj_file], device=_verts.device)
            # check that the output image array is equal to the input
            # self.assertEqual(
            #     obj[2].texture_images["material_1"].shape[1],
            #     mesh.textures._maps_list[0].shape[1],
            # )
            # check the total expected length of faces against the input
            self.assertEqual(
                faces_to_subset.shape[0], mesh.textures.faces_uvs_list()[0].shape[0]
            )

            images = renderer(mesh)
            plt.figure(figsize=(10, 10))
            plt.imshow(images[0, ..., :3].cpu().numpy())
            plt.axis("off")
            plt.tight_layout()
            image_name = f"{obj_basename}{quadrant}_render.png"
            plt.savefig(os.path.join(DATA_DIR, image_name))
            plt.clf()

            mtl_name = f"{obj_basename}{quadrant}.mtl"
            mtl_file = os.path.join(DATA_DIR, mtl_name)

            with open(mtl_file, "r") as mtl_f:
                mtl_lines = [line.strip().split(" ")[1] for line in mtl_f]
                if quadrant in [1, 2, 3]:
                    self.assertTrue("material_1" in mtl_lines)
                if quadrant == 4:
                    self.assertTrue(f"{obj_basename}4.png" in mtl_lines)

        # test IO for a single output with multiple textures
        # create a new obj tuple
        _faces = _Faces(
            verts_idx=faces.verts_idx,
            normals_idx=faces.verts_idx,  # include face normals
            textures_idx=faces.textures_idx,
            materials_idx=quadrants_materials,  # use new material assignments per face
        )
        # change up colors per material aribitrarily
        _texture_images = dict(
            material_1=aux.texture_images["material_1"],
            material_2=aux.texture_images["material_1"] + 0.4,
            material_3=aux.texture_images["material_1"] + 0.6,
            material_4=aux.texture_images["material_1"] + 0.8,
        )
        _material_colors = dict(
            material_1=aux.material_colors["material_1"],
            material_2=aux.material_colors["material_1"],
            material_3=aux.material_colors["material_1"],
            material_4=aux.material_colors["material_1"],
        )
        _aux = _Aux(
            normals=normals,
            verts_uvs=aux.verts_uvs,
            material_colors=_material_colors,
            texture_images=_texture_images,
            texture_atlas=aux.texture_atlas,
        )

        # create a new obj object of the input mesh but with four textures
        obj_basename = "test_multitexture_obj_IO_quad_cow"
        obj_quad = (verts, _faces, _aux)

        obj_name = f"{obj_basename}.obj"
        obj_file = os.path.join(DATA_DIR, obj_name)

        save_obj(
            f=obj_file,
            verts=obj_quad[0],
            faces=obj_quad[1].verts_idx,
            verts_uvs=obj_quad[2].verts_uvs,
            faces_uvs=obj_quad[1].textures_idx,
            texture_images=obj_quad[2].texture_images,
            materials_idx=obj_quad[1].materials_idx,
        )

        _obj_name = f"{obj_basename}_im_100"
        _obj_file = os.path.join(DATA_DIR, f"{_obj_name}.obj")
        # check expected warnings and error condtions for save_obj
        with self.assertRaises(ValueError) as err:
            save_obj(
                f=_obj_file,
                verts=obj_quad[0],
                faces=obj_quad[1].verts_idx,
                verts_uvs=obj_quad[2].verts_uvs,
                faces_uvs=obj_quad[1].textures_idx,
                texture_images=obj_quad[2].texture_images,
                materials_idx=obj_quad[1].materials_idx,
                image_format='tiff'
        )
        self.assertTrue(
            "'image_format' must be either 'png' or 'jpeg'"
            in str(err.exception)
        )

        with self.assertRaises(ValueError) as err:
            save_obj(
                f=_obj_file,
                verts=obj_quad[0],
                faces=obj_quad[1].verts_idx,
                verts_uvs=obj_quad[2].verts_uvs,
                faces_uvs=obj_quad[1].textures_idx,
                texture_images=obj_quad[2].texture_images,
                texture_map=torch.rand((256, 256, 3)),
                materials_idx=obj_quad[1].materials_idx,
        )
        self.assertTrue(
            "texture_map is not None and texture_images is not None; only one can be provided"
            in str(err.exception)
        )

        message = "'image_quality is recommended to be set between 0 and 95 according to PIL documentation"
        with self.assertWarnsRegex(UserWarning, message):
            save_obj(
                f=_obj_file,
                verts=obj_quad[0],
                faces=obj_quad[1].verts_idx,
                verts_uvs=obj_quad[2].verts_uvs,
                faces_uvs=obj_quad[1].textures_idx,
                texture_images=obj_quad[2].texture_images,
                materials_idx=obj_quad[1].materials_idx,
                image_quality=100
        )

        # test subset_obj functions in reading back multitexture obj
        obj_multi = load_obj(obj_file, load_textures=True, create_texture_atlas=True)
        # assert the obj_multi is valid given a warning message ("Faces have invalid indices") about invalid dims in obj_io
        self.assertFalse(obj_multi[1].verts_idx.max() >= obj_multi[0].shape[0])
        self.assertFalse(obj_multi[1].verts_idx.min() < 0)

        # reading back multi texture obj as meshes
        mesh = load_objs_as_meshes(files=[obj_file], device=verts.device)

        images = renderer(mesh)
        plt.figure(figsize=(10, 10))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off")
        plt.tight_layout()
        image_name = f"{obj_basename}_render.png"
        plt.savefig(os.path.join(DATA_DIR, image_name))
        plt.clf()

        mtl_name = f"{obj_basename}.mtl"
        mtl_file = os.path.join(DATA_DIR, mtl_name)

        # check that the multitexture subsetter functions enable reading back multiple textures
        image_size = 0
        with open(mtl_file, "r") as mtl_f:
            mtl_lines = [line.strip().split(" ")[1] for line in mtl_f]
            for quadrant_ix, quadrant in enumerate(quadrants):
                # check the contents of mtl
                self.assertTrue(f"material_{quadrant}" in mtl_lines)
                self.assertTrue(f"{obj_basename}_{quadrant_ix}.png" in mtl_lines)
                # check the dimensions of the associated image array equal the sum of the concatenated texture inputs
                image_size += (
                    obj_quad[2].texture_images[f"material_{quadrant}"].shape[1]
                )
            self.assertEqual(image_size, mesh.textures._maps_list[0].shape[1])

        # plot the visulization of verts to texture from the resulting mesh
        plt.figure(figsize=(7, 7))
        texturesuv_image_matplotlib(mesh.textures, subsample=None)
        plt.axis("off")
        plt.tight_layout()
        image_name = f"test_multitexture_obj_IO_quad_cow_uv.png"
        plt.savefig(os.path.join(DATA_DIR, image_name), bbox_inches="tight")
        plt.clf()

    def test_high_precision_obj_IO(self):
        """checking obj IO with high_precision.
        """
        torch.manual_seed(1)
        obj_filename = "cow_mesh/cow.obj"
        filename = os.path.join(TUTORIAL_DATA_DIR, obj_filename)
        # load the cow mesh into as its individual elements
        obj = load_obj(
            filename,
            load_textures=True,
            create_texture_atlas=True,
            texture_wrap=None,
            high_precision=False
        )
        self.assertTrue(obj[0].dtype == torch.tensor([0], dtype=torch.float32).dtype)
        
        obj = load_obj(
            filename,
            load_textures=True,
            create_texture_atlas=True,
            texture_wrap=None,
            high_precision=True
        )
        self.assertTrue(obj[0].dtype == torch.tensor([0], dtype=torch.float64).dtype)

        mesh = load_objs_as_meshes(
            files=[filename],
            load_textures=True,
            create_texture_atlas=False,
            texture_wrap=None,
            high_precision=True
        )
        self.assertTrue(mesh.verts_packed().dtype == torch.tensor([0], dtype=torch.float64).dtype)

        points, _, _ = sample_points_from_meshes(
            meshes=mesh,
            num_samples=100,
            return_normals=True,
            return_textures=True
        )

        self.assertTrue(points.dtype == torch.tensor([0], dtype=torch.float64).dtype)
        
        points, _, _, _ = sample_points_from_obj(
            verts=obj[0],
            faces=obj[1].verts_idx,
            verts_uvs=obj[2].verts_uvs,
            faces_uvs=obj[1].textures_idx,
            texture_images=obj[2].texture_images,
            materials_idx=obj[1].materials_idx,
            texture_atlas=obj[2].texture_atlas,
            num_samples=100,
            sample_all_faces=True,
            return_mappers=True, 
            return_textures=True, 
            return_normals=True
        )

        self.assertTrue(points.dtype == torch.tensor([0], dtype=torch.float64).dtype)

        # simulate verts from real world high precision geometries
        obj_file = "\n".join(
                    [
                        "v 400000.62547364098 5000000.3990083504 100000.13215637207031",
                        "v 400000.84158577782 5000000.1198769379 100000.57968044281006",
                        "v 400000.72239937645 5000000.8835740853 100000.39289093017578",
                        "v 400000.99848046165 5000000.1990514565 100000.06639862060547",
                        "vn 0.000000 0.000000 -1.000000",
                        "vn -1.000000 -0.000000 -0.000000",
                        "f 2//1 3//1 4//2",
                    ]
                )

        # expected verts in float 32
        expected_verts_fp32 = torch.tensor(
            [
                [400000.6250, 5000000.5000, 100000.1322],
                [400000.8438, 5000000.0000, 100000.5797],
                [400000.7188, 5000001.0000, 100000.3929],
                [400001.0000, 5000000.0000, 100000.0664]],
            dtype=torch.float32,
        )

        # expected unique y vert values in float 32 (rounded from text input)
        # rounded verts produce rounded points which may not be desired
        expected_y_points_fp32 = [-0.5000,  0.0000,  0.5000,  1.0000]

        # expected unique y vert values in float 64 (identicle to text input)
        expected_verts_fp64 = torch.tensor(
            [
                [400000.62547364098, 5000000.3990083504, 100000.13215637207031],
                [400000.84158577782, 5000000.1198769379, 100000.57968044281006],
                [400000.72239937645, 5000000.8835740853, 100000.39289093017578],
                [400000.99848046165, 5000000.1990514565, 100000.06639862060547]],
            dtype=torch.float64,
        )

        # simulate obj IO on real world geometries
        with NamedTemporaryFile(mode="w", suffix=".obj") as f:
            f.write(obj_file)
            f.flush()
            obj = load_obj(Path(f.name))
            # in normal IO, returned verts are rounded
            self.assertTrue(torch.all(obj[0] == expected_verts_fp32))

            points, _, _, _ = sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx,
                verts_uvs=obj[2].verts_uvs,
                faces_uvs=obj[1].textures_idx,
                texture_images=obj[2].texture_images,
                materials_idx=obj[1].materials_idx,
                texture_atlas=obj[2].texture_atlas,
                num_samples=100,
                sample_all_faces=True,
                return_mappers=True, 
                return_textures=True, 
                return_normals=True
            )
            
            u_diff = torch.unique(points[..., 1]) - 5000000
            # subtract left of decimal and check that values are rounded
            self.assertTrue(torch.all(sum(u_diff==i for i in expected_y_points_fp32).bool()))

            obj = load_obj(Path(f.name), high_precision=True)
            # check that verts match the text input at float 64
            self.assertTrue(torch.all(obj[0] == expected_verts_fp64))

            points, _, _, _ = sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx,
                verts_uvs=obj[2].verts_uvs,
                faces_uvs=obj[1].textures_idx,
                texture_images=obj[2].texture_images,
                materials_idx=obj[1].materials_idx,
                texture_atlas=obj[2].texture_atlas,
                num_samples=100,
                sample_all_faces=True,
                return_mappers=True, 
                return_textures=True, 
                return_normals=True
            )
            u_diff = torch.unique(points[..., 1]) - 5000000
            # check that unique y val verts in high precision do not contain rounded
            self.assertFalse(torch.all(sum(u_diff==i for i in expected_y_points_fp32).bool()))

    @staticmethod
    def _split_verts_by_quadrant(verts: np.ndarray, quadrant: int = 1):
        """A utilty function for splitting 3D coordinate array (verts) into quadrants
        Reference: https://stackoverflow.com/questions/69398528/how-to-determine-quadrant-given-x-and-y-coordinate-columns-in-pandas-dataframe
        Args:
            verts: An Nx3 array intended to represent vertices in a mesh.
            quadrant: An integer (range from 1 to 4) of the desired quadrant.
        Returns:
            np.ndarray: A Nx array representing the indices to select that represent the given quadrant
        """
        if isinstance(verts, torch.Tensor):
            verts = verts.cpu().numpy()

        deg = np.round(180 * np.arctan2(verts[..., 1], verts[..., 0]) / np.pi).astype(
            int
        )
        quadrants = 1 + ((deg + 360) % 360) // 90
        return np.flatnonzero([quadrants == quadrant])

    @staticmethod
    def _bm_save_obj(verts: torch.Tensor, faces: torch.Tensor, decimal_places: int):
        return lambda: save_obj(StringIO(), verts, faces, decimal_places)

    @staticmethod
    def _bm_load_obj(verts: torch.Tensor, faces: torch.Tensor, decimal_places: int):
        f = StringIO()
        save_obj(f, verts, faces, decimal_places)
        s = f.getvalue()
        # Recreate stream so it's unaffected by how it was created.
        return lambda: load_obj(StringIO(s))

    @staticmethod
    def bm_save_simple_obj_with_init(V: int, F: int):
        verts = torch.tensor(V * [[0.11, 0.22, 0.33]]).view(-1, 3)
        faces = torch.tensor(F * [[1, 2, 3]]).view(-1, 3)
        return TestMeshObjIO._bm_save_obj(verts, faces, decimal_places=2)

    @staticmethod
    def bm_load_simple_obj_with_init(V: int, F: int):
        verts = torch.tensor(V * [[0.1, 0.2, 0.3]]).view(-1, 3)
        faces = torch.tensor(F * [[1, 2, 3]]).view(-1, 3)
        return TestMeshObjIO._bm_load_obj(verts, faces, decimal_places=2)

    @staticmethod
    def bm_save_complex_obj(N: int):
        meshes = torus(r=0.25, R=1.0, sides=N, rings=2 * N)
        [verts], [faces] = meshes.verts_list(), meshes.faces_list()
        return TestMeshObjIO._bm_save_obj(verts, faces, decimal_places=5)

    @staticmethod
    def bm_load_complex_obj(N: int):
        meshes = torus(r=0.25, R=1.0, sides=N, rings=2 * N)
        [verts], [faces] = meshes.verts_list(), meshes.faces_list()
        return TestMeshObjIO._bm_load_obj(verts, faces, decimal_places=5)

    @staticmethod
    def bm_load_texture_atlas(R: int):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        data_dir = "/data/users/nikhilar/fbsource/fbcode/vision/fair/pytorch3d/docs/"
        obj_filename = os.path.join(data_dir, "tutorials/data/cow_mesh/cow.obj")
        torch.cuda.synchronize()

        def load():
            load_obj(
                obj_filename,
                load_textures=True,
                device=device,
                create_texture_atlas=True,
                texture_atlas_size=R,
            )
            torch.cuda.synchronize()

        return load

    @staticmethod
    def bm_bilinear_sampling_vectorized(S: int, F: int, R: int):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        image = torch.rand((S, S, 3))
        grid = torch.rand((F, R, R, 2))
        torch.cuda.synchronize()

        def load():
            _bilinear_interpolation_vectorized(image, grid)
            torch.cuda.synchronize()

        return load

    @staticmethod
    def bm_bilinear_sampling_grid_sample(S: int, F: int, R: int):
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        image = torch.rand((S, S, 3))
        grid = torch.rand((F, R, R, 2))
        torch.cuda.synchronize()

        def load():
            _bilinear_interpolation_grid_sample(image, grid)
            torch.cuda.synchronize()

        return load
