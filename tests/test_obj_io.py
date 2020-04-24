# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import os
import unittest
import warnings
from io import StringIO
from pathlib import Path

import torch
from common_testing import TestCaseMixin
from pytorch3d.io import load_obj, load_objs_as_meshes, save_obj
from pytorch3d.io.mtl_io import (
    _bilinear_interpolation_grid_sample,
    _bilinear_interpolation_vectorized,
)
from pytorch3d.structures import Meshes, Textures, join_meshes_as_batch
from pytorch3d.utils import torus


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
        obj_file = StringIO(obj_file)
        verts, faces, aux = load_obj(obj_file)
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
        padded_vals = -torch.ones_like(faces.verts_idx)
        self.assertTrue(torch.all(faces.normals_idx == padded_vals))
        self.assertTrue(torch.all(faces.textures_idx == padded_vals))
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
        expected_faces_normals_idx = -torch.ones_like(expected_faces, dtype=torch.int64)
        expected_faces_normals_idx[4, :] = torch.tensor([1, 1, 1], dtype=torch.int64)
        expected_faces_textures_idx = -torch.ones_like(
            expected_faces, dtype=torch.int64
        )
        expected_faces_textures_idx[4, :] = torch.tensor([0, 0, 1], dtype=torch.int64)

        self.assertTrue(torch.all(verts == expected_verts))
        self.assertTrue(torch.all(faces.verts_idx == expected_faces))
        self.assertClose(normals, expected_normals)
        self.assertClose(textures, expected_textures)
        self.assertClose(faces.normals_idx, expected_faces_normals_idx)
        self.assertClose(faces.textures_idx, expected_faces_textures_idx)
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
        expected_faces_normals_idx = torch.tensor([[0, 0, 1]], dtype=torch.int64)
        expected_normals = torch.tensor(
            [[0.000000, 0.000000, -1.000000], [-1.000000, -0.000000, -0.000000]],
            dtype=torch.float32,
        )
        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        verts, faces, aux = load_obj(obj_file)
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
        obj_file = StringIO(obj_file)
        expected_faces_textures_idx = torch.tensor([[0, 0, 1]], dtype=torch.int64)
        expected_textures = torch.tensor(
            [[0.999110, 0.501077], [0.999455, 0.750380]], dtype=torch.float32
        )
        expected_verts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        verts, faces, aux = load_obj(obj_file)
        normals = aux.normals
        textures = aux.verts_uvs
        materials = aux.material_colors
        tex_maps = aux.texture_images

        self.assertClose(faces.textures_idx, expected_faces_textures_idx)
        self.assertClose(expected_textures, textures)
        self.assertClose(expected_verts, verts)
        self.assertTrue(
            torch.all(faces.normals_idx == -torch.ones_like(faces.textures_idx))
        )
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
        self.assertTrue("Vertex properties are inconsistent" in str(err.exception))

    def test_load_obj_error_too_many_vertex_properties(self):
        obj_file = "\n".join(["f 2/1/1/3"])
        obj_file = StringIO(obj_file)

        with self.assertRaises(ValueError) as err:
            load_obj(obj_file)
        self.assertTrue("Face vertices can ony have 3 properties" in str(err.exception))

    def test_load_obj_error_invalid_vertex_indices(self):
        obj_file = "\n".join(
            ["v 0.1 0.2 0.3", "v 0.1 0.2 0.3", "v 0.1 0.2 0.3", "f -2 5 1"]
        )
        obj_file = StringIO(obj_file)

        with self.assertWarnsRegex(UserWarning, "Faces have invalid indices"):
            load_obj(obj_file)

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

        with self.assertWarnsRegex(UserWarning, "Faces have invalid indices"):
            load_obj(obj_file)

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

        with self.assertWarnsRegex(UserWarning, "Faces have invalid indices"):
            load_obj(obj_file)

    def test_save_obj_invalid_shapes(self):
        # Invalid vertices shape
        with self.assertRaises(ValueError) as error:
            verts = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])  # (V, 4)
            faces = torch.LongTensor([[0, 1, 2]])
            save_obj(StringIO(), verts, faces)
        expected_message = (
            "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        )
        self.assertTrue(expected_message, error.exception)

        # Invalid faces shape
        with self.assertRaises(ValueError) as error:
            verts = torch.FloatTensor([[0.1, 0.2, 0.3]])
            faces = torch.LongTensor([[0, 1, 2, 3]])  # (F, 4)
            save_obj(StringIO(), verts, faces)
        expected_message = (
            "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        )
        self.assertTrue(expected_message, error.exception)

    def test_save_obj_invalid_indices(self):
        message_regex = "Faces have invalid indices"
        verts = torch.FloatTensor([[0.1, 0.2, 0.3]])
        faces = torch.LongTensor([[0, 1, 2]])
        with self.assertWarnsRegex(UserWarning, message_regex):
            save_obj(StringIO(), verts, faces)

        faces = torch.LongTensor([[-1, 0, 1]])
        with self.assertWarnsRegex(UserWarning, message_regex):
            save_obj(StringIO(), verts, faces)

    def _test_save_load(self, verts, faces):
        f = StringIO()
        save_obj(f, verts, faces)
        f.seek(0)
        expected_verts, expected_faces = verts, faces
        if not len(expected_verts):  # Always compare with a (V, 3) tensor
            expected_verts = torch.zeros(size=(0, 3), dtype=torch.float32)
        if not len(expected_faces):  # Always compare with an (F, 3) tensor
            expected_faces = torch.zeros(size=(0, 3), dtype=torch.int64)
        actual_verts, actual_faces, _ = load_obj(f)
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
        DATA_DIR = Path(__file__).resolve().parent.parent / "docs/tutorials/data"
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

    def test_load_mtl_texture_atlas_compare_softras(self):
        # Load saved texture atlas created with SoftRas.
        device = torch.device("cuda:0")
        DATA_DIR = Path(__file__).resolve().parent.parent
        obj_filename = DATA_DIR / "docs/tutorials/data/cow_mesh/cow.obj"
        expected_atlas_fname = DATA_DIR / "tests/data/cow_texture_atlas_softras.pt"

        # Note, the reference texture atlas generated using SoftRas load_obj function
        # is too large to check in to the repo. Download the file to run the test locally.
        if not os.path.exists(expected_atlas_fname):
            url = "https://dl.fbaipublicfiles.com/pytorch3d/data/tests/cow_texture_atlas_softras.pt"
            msg = (
                "cow_texture_atlas_softras.pt not found, download from %s, save it at the path %s, and rerun"
                % (url, expected_atlas_fname)
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
        DATA_DIR = Path(__file__).resolve().parent.parent / "docs/tutorials/data"
        obj_filename = "cow_mesh/cow.obj"
        filename = os.path.join(DATA_DIR, obj_filename)
        verts, faces, aux = load_obj(filename, load_textures=False)

        self.assertTrue(aux.material_colors is None)
        self.assertTrue(aux.texture_images is None)

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
        with self.assertWarnsRegex(UserWarning, "No mtl file provided"):
            verts, faces, aux = load_obj(obj_file)

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

    def test_load_obj_missing_texture(self):
        DATA_DIR = Path(__file__).resolve().parent / "data"
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
        DATA_DIR = Path(__file__).resolve().parent / "data"
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
        DATA_DIR = Path(__file__).resolve().parent / "data"
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
        DATA_DIR = Path(__file__).resolve().parent / "data"
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
                check_item(mesh.textures.maps_padded(), mesh3.textures.maps_padded())
                check_item(
                    mesh.textures.faces_uvs_padded(), mesh3.textures.faces_uvs_padded()
                )
                check_item(
                    mesh.textures.verts_uvs_padded(), mesh3.textures.verts_uvs_padded()
                )
                check_item(
                    mesh.textures.verts_rgb_padded(), mesh3.textures.verts_rgb_padded()
                )

        DATA_DIR = Path(__file__).resolve().parent.parent / "docs/tutorials/data"
        obj_filename = DATA_DIR / "cow_mesh/cow.obj"

        mesh = load_objs_as_meshes([obj_filename])
        mesh3 = load_objs_as_meshes([obj_filename, obj_filename, obj_filename])
        check_triple(mesh, mesh3)
        self.assertTupleEqual(mesh.textures.maps_padded().shape, (1, 1024, 1024, 3))

        mesh_notex = load_objs_as_meshes([obj_filename], load_textures=False)
        mesh3_notex = load_objs_as_meshes(
            [obj_filename, obj_filename, obj_filename], load_textures=False
        )
        check_triple(mesh_notex, mesh3_notex)
        self.assertIsNone(mesh_notex.textures)

        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        vert_tex = torch.tensor(
            [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32
        )
        tex = Textures(verts_rgb=vert_tex[None, :])
        mesh_rgb = Meshes(verts=[verts], faces=[faces], textures=tex)
        mesh_rgb3 = join_meshes_as_batch([mesh_rgb, mesh_rgb, mesh_rgb])
        check_triple(mesh_rgb, mesh_rgb3)

        teapot_obj = DATA_DIR / "teapot.obj"
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
        DATA_DIR = "/data/users/nikhilar/fbsource/fbcode/vision/fair/pytorch3d/docs/"
        obj_filename = os.path.join(DATA_DIR, "tutorials/data/cow_mesh/cow.obj")
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
