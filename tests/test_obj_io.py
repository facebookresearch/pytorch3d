#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import unittest
from io import StringIO
from pathlib import Path
import torch

from pytorch3d.io import load_obj, save_obj


class TestMeshObjIO(unittest.TestCase):
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
        obj_file = StringIO(obj_file)
        verts, faces, aux = load_obj(obj_file)
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
            ],
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
        self.assertTrue(faces.normals_idx == [])
        self.assertTrue(faces.textures_idx == [])
        self.assertTrue(
            torch.all(faces.materials_idx == -torch.ones(len(expected_faces)))
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
        obj_file = StringIO(obj_file)
        verts, faces, aux = load_obj(obj_file)
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
        expected_faces_normals_idx = torch.tensor(
            [[1, 1, 1]], dtype=torch.int64
        )
        expected_faces_textures_idx = torch.tensor(
            [[0, 0, 1]], dtype=torch.int64
        )

        self.assertTrue(torch.all(verts == expected_verts))
        self.assertTrue(torch.all(faces.verts_idx == expected_faces))
        self.assertTrue(torch.allclose(normals, expected_normals))
        self.assertTrue(torch.allclose(textures, expected_textures))
        self.assertTrue(
            torch.allclose(faces.normals_idx, expected_faces_normals_idx)
        )
        self.assertTrue(
            torch.allclose(faces.textures_idx, expected_faces_textures_idx)
        )
        self.assertTrue(materials is None)
        self.assertTrue(tex_maps is None)

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
        obj_file = StringIO(obj_file)
        expected_faces_normals_idx = torch.tensor(
            [[0, 0, 1]], dtype=torch.int64
        )
        expected_normals = torch.tensor(
            [
                [0.000000, 0.000000, -1.000000],
                [-1.000000, -0.000000, -0.000000],
            ],
            dtype=torch.float32,
        )
        expected_verts = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        verts, faces, aux = load_obj(obj_file)
        normals = aux.normals
        textures = aux.verts_uvs
        materials = aux.material_colors
        tex_maps = aux.texture_images
        self.assertTrue(
            torch.allclose(faces.normals_idx, expected_faces_normals_idx)
        )
        self.assertTrue(torch.allclose(normals, expected_normals))
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(faces.textures_idx == [])
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
        obj_file = StringIO(obj_file)
        expected_faces_textures_idx = torch.tensor(
            [[0, 0, 1]], dtype=torch.int64
        )
        expected_textures = torch.tensor(
            [[0.999110, 0.501077], [0.999455, 0.750380]], dtype=torch.float32
        )
        expected_verts = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        verts, faces, aux = load_obj(obj_file)
        normals = aux.normals
        textures = aux.verts_uvs
        materials = aux.material_colors
        tex_maps = aux.texture_images

        self.assertTrue(
            torch.allclose(faces.textures_idx, expected_faces_textures_idx)
        )
        self.assertTrue(torch.allclose(expected_textures, textures))
        self.assertTrue(torch.allclose(expected_verts, verts))
        self.assertTrue(faces.normals_idx == [])
        self.assertTrue(normals is None)
        self.assertTrue(materials is None)
        self.assertTrue(tex_maps is None)

    def test_load_obj_error_textures(self):
        obj_file = "\n".join(["vt 0.1"])
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue("does not have 2 values" in str(err.exception))

    def test_load_obj_error_normals(self):
        obj_file = "\n".join(["vn 0.1"])
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue("does not have 3 values" in str(err.exception))

    def test_load_obj_error_vertices(self):
        obj_file = "\n".join(["v 1"])
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue("does not have 3 values" in str(err.exception))

    def test_load_obj_error_inconsistent_triplets(self):
        obj_file = "\n".join(["f 2//1 3/1 4/1/2"])
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue(
            "Vertex properties are inconsistent" in str(err.exception)
        )

    def test_load_obj_error_too_many_vertex_properties(self):
        obj_file = "\n".join(["f 2/1/1/3"])
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue(
            "Face vertices can ony have 3 properties" in str(err.exception)
        )

    def test_load_obj_error_invalid_vertex_indices(self):
        obj_file = "\n".join(
            ["v 0.1 0.2 0.3", "v 0.1 0.2 0.3", "v 0.1 0.2 0.3", "f -2 5 1"]
        )
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue("Faces have invalid indices." in str(err.exception))

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
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue("Faces have invalid indices." in str(err.exception))

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
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue("Faces have invalid indices." in str(err.exception))

    def test_save_obj(self):
        verts = torch.tensor(
            [
                [0.01, 0.2, 0.301],
                [0.2, 0.03, 0.408],
                [0.3, 0.4, 0.05],
                [0.6, 0.7, 0.8],
            ],
            dtype=torch.float32,
        )
        faces = torch.tensor(
            [[0, 2, 1], [0, 1, 2], [3, 2, 1], [3, 1, 0]], dtype=torch.int64
        )
        obj_file = StringIO()
        save_obj(obj_file, verts, faces, decimal_places=2)
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
        actual_file = obj_file.getvalue()
        self.assertEqual(actual_file, expected_file)

    def test_load_mtl(self):
        DATA_DIR = (
            Path(__file__).resolve().parent.parent / "docs/tutorials/data"
        )
        obj_filename = "cow_mesh/cow.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
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
        # Check that there is an image with material name material_1.
        self.assertTrue(tuple(tex_maps.keys()) == ("material_1",))
        self.assertTrue(torch.is_tensor(tuple(tex_maps.values())[0]))
        self.assertTrue(
            torch.all(faces.materials_idx == torch.zeros(len(faces.verts_idx)))
        )

        # Check all keys and values in dictionary are the same.
        for n1, n2 in zip(materials.keys(), expected_materials.keys()):
            self.assertTrue(n1 == n2)
            for k1, k2 in zip(
                materials[n1].keys(), expected_materials[n2].keys()
            ):
                self.assertTrue(
                    torch.allclose(
                        materials[n1][k1], expected_materials[n2][k2]
                    )
                )

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
        obj_file = StringIO(obj_file)
        with self.assertWarnsRegex(Warning, "No mtl file provided"):
            verts, faces, aux = load_obj(obj_file)

        expected_verts = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))
        self.assertTrue(aux.material_colors is None)
        self.assertTrue(aux.texture_images is None)
        self.assertTrue(aux.normals is None)
        self.assertTrue(aux.verts_uvs is None)

    def test_load_obj_missing_texture(self):
        DATA_DIR = Path(__file__).resolve().parent / "data"
        obj_filename = "missing_files_obj/model.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
        with self.assertWarnsRegex(Warning, "Texture file does not exist"):
            verts, faces, aux = load_obj(filename)

        expected_verts = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))

    def test_load_obj_missing_mtl(self):
        DATA_DIR = Path(__file__).resolve().parent / "data"
        obj_filename = "missing_files_obj/model2.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
        with self.assertWarnsRegex(Warning, "Mtl file does not exist"):
            verts, faces, aux = load_obj(filename)

        expected_verts = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        expected_faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
        self.assertTrue(torch.allclose(verts, expected_verts))
        self.assertTrue(torch.allclose(faces.verts_idx, expected_faces))

    @staticmethod
    def save_obj_with_init(V: int, F: int):
        verts_list = torch.tensor(V * [[0.11, 0.22, 0.33]]).view(-1, 3)
        faces_list = torch.tensor(F * [[1, 2, 3]]).view(-1, 3)
        obj_file = StringIO()

        def save_mesh():
            save_obj(obj_file, verts_list, faces_list, decimal_places=2)

        return save_mesh

    @staticmethod
    def load_obj_with_init(V: int, F: int):
        obj = "\n".join(["v 0.1 0.2 0.3"] * V + ["f 1 2 3"] * F)

        def load_mesh():
            obj_file = StringIO(obj)
            verts, faces, aux = load_obj(obj_file)

        return load_mesh
