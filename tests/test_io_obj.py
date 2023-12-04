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
from pytorch3d.io import IO, load_obj, load_objs_as_meshes, save_obj
from pytorch3d.io.mtl_io import (
    _bilinear_interpolation_grid_sample,
    _bilinear_interpolation_vectorized,
    _parse_mtl,
)
from pytorch3d.renderer import TexturesAtlas, TexturesUV, TexturesVertex
from pytorch3d.structures import join_meshes_as_batch, Meshes
from pytorch3d.utils import torus

from .common_testing import (
    get_pytorch3d_dir,
    get_tests_dir,
    load_rgb_image,
    TestCaseMixin,
)


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
        verts = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])  # (V, 4)
        faces = torch.LongTensor([[0, 1, 2]])
        with self.assertRaises(ValueError) as error:
            with NamedTemporaryFile(mode="w", suffix=".obj") as f:
                save_obj(Path(f.name), verts, faces)
        expected_message = (
            "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        )
        self.assertTrue(expected_message, error.exception)

        # Invalid faces shape
        verts = torch.FloatTensor([[0.1, 0.2, 0.3]])
        faces = torch.LongTensor([[0, 1, 2, 3]])  # (F, 4)
        with self.assertRaises(ValueError) as error:
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
