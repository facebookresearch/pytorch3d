# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import struct
import unittest
from io import BytesIO, StringIO
from tempfile import NamedTemporaryFile, TemporaryFile

import numpy as np
import pytorch3d.io.ply_io
import torch
from iopath.common.file_io import PathManager
from pytorch3d.io import IO
from pytorch3d.io.ply_io import load_ply, save_ply
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import torus

from .common_testing import get_tests_dir, TestCaseMixin


global_path_manager = PathManager()
DATA_DIR = get_tests_dir() / "data"


def _load_ply_raw(stream):
    return pytorch3d.io.ply_io._load_ply_raw(stream, global_path_manager)


CUBE_PLY_LINES = [
    "ply",
    "format ascii 1.0",
    "comment made by Greg Turk",
    "comment this file is a cube",
    "element vertex 8",
    "property float x",
    "property float y",
    "property float z",
    "element face 6",
    "property list uchar int vertex_index",
    "end_header",
    "0 0 0",
    "0 0 1",
    "0 1 1",
    "0 1 0",
    "1 0 0",
    "1 0 1",
    "1 1 1",
    "1 1 0",
    "4 0 1 2 3",
    "4 7 6 5 4",
    "4 0 4 5 1",
    "4 1 5 6 2",
    "4 2 6 7 3",
    "4 3 7 4 0",
]

CUBE_VERTS = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 1],
    [1, 1, 0],
]
CUBE_FACES = [
    [0, 1, 2],
    [7, 6, 5],
    [0, 4, 5],
    [1, 5, 6],
    [2, 6, 7],
    [3, 7, 4],
    [0, 2, 3],
    [7, 5, 4],
    [0, 5, 1],
    [1, 6, 2],
    [2, 7, 3],
    [3, 4, 0],
]


class TestMeshPlyIO(TestCaseMixin, unittest.TestCase):
    def test_raw_load_simple_ascii(self):
        ply_file = "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "comment made by Greg Turk",
                "comment this file is a cube",
                "element vertex 8",
                "property float x",
                "property float y",
                "property float z",
                "element face 6",
                "property list uchar int vertex_index",
                "element irregular_list 3",
                "property list uchar int vertex_index",
                "end_header",
                "0 0 0",
                "0 0 1",
                "0 1 1",
                "0 1 0",
                "1 0 0",
                "1 0 1",
                "1 1 1",
                "1 1 0",
                "4 0 1 2 3",
                "4 7 6 5 4",
                "4 0 4 5 1",
                "4 1 5 6 2",
                "4 2 6 7 3",
                "4 3 7 4 0",  # end of faces
                "4 0 1 2 3",
                "4 7 6 5 4",
                "3 4 5 1",
            ]
        )
        for line_ending in [None, "\n", "\r\n"]:
            if line_ending is None:
                stream = StringIO(ply_file)
            else:
                byte_file = ply_file.encode("ascii")
                if line_ending == "\r\n":
                    byte_file = byte_file.replace(b"\n", b"\r\n")
                stream = BytesIO(byte_file)
            header, data = _load_ply_raw(stream)
            self.assertTrue(header.ascii)
            self.assertEqual(len(data), 3)
            self.assertTupleEqual(data["face"].shape, (6, 4))
            self.assertClose([0, 1, 2, 3], data["face"][0])
            self.assertClose([3, 7, 4, 0], data["face"][5])
            [vertex0] = data["vertex"]
            self.assertTupleEqual(vertex0.shape, (8, 3))
            irregular = data["irregular_list"]
            self.assertEqual(len(irregular), 3)
            self.assertEqual(type(irregular), list)
            [x] = irregular[0]
            self.assertClose(x, [0, 1, 2, 3])
            [x] = irregular[1]
            self.assertClose(x, [7, 6, 5, 4])
            [x] = irregular[2]
            self.assertClose(x, [4, 5, 1])

    def test_load_simple_ascii(self):
        ply_file = "\n".join(CUBE_PLY_LINES)
        for line_ending in [None, "\n", "\r\n"]:
            if line_ending is None:
                stream = StringIO(ply_file)
            else:
                byte_file = ply_file.encode("ascii")
                if line_ending == "\r\n":
                    byte_file = byte_file.replace(b"\n", b"\r\n")
                stream = BytesIO(byte_file)
            verts, faces = load_ply(stream)
            self.assertEqual(verts.shape, (8, 3))
            self.assertEqual(faces.shape, (12, 3))
            self.assertClose(verts, torch.FloatTensor(CUBE_VERTS))
            self.assertClose(faces, torch.LongTensor(CUBE_FACES))

    def test_pluggable_load_cube(self):
        """
        This won't work on Windows due to NamedTemporaryFile being reopened.
        Use the testpath package instead?
        """
        ply_file = "\n".join(CUBE_PLY_LINES)
        io = IO()
        with NamedTemporaryFile(mode="w", suffix=".ply") as f:
            f.write(ply_file)
            f.flush()
            mesh = io.load_mesh(f.name)
        self.assertClose(mesh.verts_padded(), torch.FloatTensor(CUBE_VERTS)[None])
        self.assertClose(mesh.faces_padded(), torch.LongTensor(CUBE_FACES)[None])

        device = torch.device("cuda:0")

        with NamedTemporaryFile(mode="w", suffix=".ply") as f2:
            io.save_mesh(mesh, f2.name)
            f2.flush()
            mesh2 = io.load_mesh(f2.name, device=device)
        self.assertEqual(mesh2.verts_padded().device, device)
        self.assertClose(mesh2.verts_padded().cpu(), mesh.verts_padded())
        self.assertClose(mesh2.faces_padded().cpu(), mesh.faces_padded())

        with NamedTemporaryFile(mode="w") as f3:
            with self.assertRaisesRegex(
                ValueError, "No mesh interpreter found to write to"
            ):
                io.save_mesh(mesh, f3.name)
            with self.assertRaisesRegex(
                ValueError, "No mesh interpreter found to read "
            ):
                io.load_mesh(f3.name)

    def test_heterogenous_verts_per_face(self):
        # The cube but where one face is pentagon not square.
        text = CUBE_PLY_LINES.copy()
        text[-1] = "5 3 7 4 0 1"
        stream = StringIO("\n".join(text))
        verts, faces = load_ply(stream)
        self.assertEqual(verts.shape, (8, 3))
        self.assertEqual(faces.shape, (13, 3))

    def test_save_too_many_colors(self):
        verts = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32
        )
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        vert_colors = torch.rand((4, 7))
        texture_with_seven_colors = TexturesVertex(verts_features=[vert_colors])

        mesh = Meshes(
            verts=[verts],
            faces=[faces],
            textures=texture_with_seven_colors,
        )

        io = IO()
        msg = "Texture will not be saved as it has 7 colors, not 3."
        with NamedTemporaryFile(mode="w", suffix=".ply") as f:
            with self.assertWarnsRegex(UserWarning, msg):
                io.save_mesh(mesh.cuda(), f.name)

    def test_save_load_meshes(self):
        verts = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32
        )
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        normals = torch.tensor(
            [[0, 1, 0], [1, 0, 0], [1, 4, 1], [1, 0, 0]], dtype=torch.float32
        )
        vert_colors = torch.rand_like(verts)
        texture = TexturesVertex(verts_features=[vert_colors])

        for do_textures, do_normals in itertools.product([True, False], [True, False]):
            mesh = Meshes(
                verts=[verts],
                faces=[faces],
                textures=texture if do_textures else None,
                verts_normals=[normals] if do_normals else None,
            )
            device = torch.device("cuda:0")

            io = IO()
            with NamedTemporaryFile(mode="w", suffix=".ply") as f:
                io.save_mesh(mesh.cuda(), f.name)
                f.flush()
                mesh2 = io.load_mesh(f.name, device=device)
            self.assertEqual(mesh2.device, device)
            mesh2 = mesh2.cpu()
            self.assertClose(mesh2.verts_padded(), mesh.verts_padded())
            self.assertClose(mesh2.faces_padded(), mesh.faces_padded())
            if do_normals:
                self.assertTrue(mesh.has_verts_normals())
                self.assertTrue(mesh2.has_verts_normals())
                self.assertClose(
                    mesh2.verts_normals_padded(), mesh.verts_normals_padded()
                )
            else:
                self.assertFalse(mesh.has_verts_normals())
                self.assertFalse(mesh2.has_verts_normals())
                self.assertFalse(torch.allclose(mesh2.verts_normals_padded(), normals))
            if do_textures:
                self.assertIsInstance(mesh2.textures, TexturesVertex)
                self.assertClose(mesh2.textures.verts_features_list()[0], vert_colors)
            else:
                self.assertIsNone(mesh2.textures)

    def test_save_load_with_normals(self):
        points = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32
        )
        normals = torch.tensor(
            [[0, 1, 0], [1, 0, 0], [1, 4, 1], [1, 0, 0]], dtype=torch.float32
        )
        features = torch.rand_like(points)

        for do_features, do_normals in itertools.product([True, False], [True, False]):
            cloud = Pointclouds(
                points=[points],
                features=[features] if do_features else None,
                normals=[normals] if do_normals else None,
            )
            device = torch.device("cuda:0")

            io = IO()
            with NamedTemporaryFile(mode="w", suffix=".ply") as f:
                io.save_pointcloud(cloud.cuda(), f.name)
                f.flush()
                cloud2 = io.load_pointcloud(f.name, device=device)
            self.assertEqual(cloud2.device, device)
            cloud2 = cloud2.cpu()
            self.assertClose(cloud2.points_padded(), cloud.points_padded())
            if do_normals:
                self.assertClose(cloud2.normals_padded(), cloud.normals_padded())
            else:
                self.assertIsNone(cloud.normals_padded())
                self.assertIsNone(cloud2.normals_padded())
            if do_features:
                self.assertClose(cloud2.features_packed(), features)
            else:
                self.assertIsNone(cloud2.features_packed())

    def test_save_ply_invalid_shapes(self):
        # Invalid vertices shape
        verts = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])  # (V, 4)
        faces = torch.LongTensor([[0, 1, 2]])
        with self.assertRaises(ValueError) as error:
            save_ply(BytesIO(), verts, faces)
        expected_message = (
            "Argument 'verts' should either be empty or of shape (num_verts, 3)."
        )
        self.assertTrue(expected_message, error.exception)

        # Invalid faces shape
        verts = torch.FloatTensor([[0.1, 0.2, 0.3]])
        faces = torch.LongTensor([[0, 1, 2, 3]])  # (F, 4)
        with self.assertRaises(ValueError) as error:
            save_ply(BytesIO(), verts, faces)
        expected_message = (
            "Argument 'faces' should either be empty or of shape (num_faces, 3)."
        )
        self.assertTrue(expected_message, error.exception)

    def test_save_ply_invalid_indices(self):
        message_regex = "Faces have invalid indices"
        verts = torch.FloatTensor([[0.1, 0.2, 0.3]])
        faces = torch.LongTensor([[0, 1, 2]])
        with self.assertWarnsRegex(UserWarning, message_regex):
            save_ply(BytesIO(), verts, faces)

        faces = torch.LongTensor([[-1, 0, 1]])
        with self.assertWarnsRegex(UserWarning, message_regex):
            save_ply(BytesIO(), verts, faces)

    def _test_save_load(self, verts, faces):
        f = BytesIO()
        save_ply(f, verts, faces)
        f.seek(0)
        # raise Exception(f.getvalue())
        expected_verts, expected_faces = verts, faces
        if not len(expected_verts):  # Always compare with a (V, 3) tensor
            expected_verts = torch.zeros(size=(0, 3), dtype=torch.float32)
        if not len(expected_faces):  # Always compare with an (F, 3) tensor
            expected_faces = torch.zeros(size=(0, 3), dtype=torch.int64)

        actual_verts, actual_faces = load_ply(f)
        self.assertClose(expected_verts, actual_verts)
        if len(actual_verts):
            self.assertClose(expected_faces, actual_faces)
        else:
            self.assertEqual(actual_faces.numel(), 0)

    def test_normals_save(self):
        verts = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32
        )
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]])
        normals = torch.tensor(
            [[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]], dtype=torch.float32
        )
        file = BytesIO()
        save_ply(file, verts=verts, faces=faces, verts_normals=normals)
        file.close()

    def test_contiguity_unimportant(self):
        verts = torch.rand(32, 3)
        self._test_save_load(verts, torch.randint(30, size=(10, 3)))
        self._test_save_load(verts, torch.randint(30, size=(3, 10)).T)

    def test_empty_save_load(self):
        # Vertices + empty faces
        verts = torch.tensor([[0.1, 0.2, 0.3]])
        faces = torch.LongTensor([])
        self._test_save_load(verts, faces)

        faces = torch.zeros(size=(0, 3), dtype=torch.int64)
        self._test_save_load(verts, faces)

        # Faces + empty vertices
        # => We don't save the faces
        verts = torch.FloatTensor([])
        faces = torch.LongTensor([[0, 1, 2]])
        message_regex = "Empty 'verts' provided"
        with self.assertWarnsRegex(UserWarning, message_regex):
            self._test_save_load(verts, faces)

        verts = torch.zeros(size=(0, 3), dtype=torch.float32)
        with self.assertWarnsRegex(UserWarning, message_regex):
            self._test_save_load(verts, faces)

        # Empty vertices + empty faces
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

    def test_simple_save(self):
        verts = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 2, 0]], dtype=torch.float32
        )
        faces = torch.tensor([[0, 1, 2], [0, 3, 4]])
        for filetype in BytesIO, TemporaryFile:
            lengths = {}
            for ascii in [True, False]:
                file = filetype()
                save_ply(file, verts=verts, faces=faces, ascii=ascii)
                lengths[ascii] = file.tell()

                file.seek(0)
                verts2, faces2 = load_ply(file)
                self.assertClose(verts, verts2)
                self.assertClose(faces, faces2)

                file.seek(0)
                if ascii:
                    file.read().decode("ascii")
                else:
                    with self.assertRaises(UnicodeDecodeError):
                        file.read().decode("ascii")

                if filetype is TemporaryFile:
                    file.close()
            self.assertLess(lengths[False], lengths[True], "ascii should be longer")

    def test_heterogeneous_property(self):
        ply_file_ascii = "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 8",
                "property float x",
                "property int y",
                "property int z",
                "end_header",
                "0 0 0",
                "0 0 1",
                "0 1 1",
                "0 1 0",
                "1 0 0",
                "1 0 1",
                "1 1 1",
                "1 1 0",
            ]
        )
        ply_file_binary = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 8",
                "property uchar x",
                "property char y",
                "property char z",
                "end_header",
                "",
            ]
        )
        data = [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0]
        stream_ascii = StringIO(ply_file_ascii)
        stream_binary = BytesIO(ply_file_binary.encode("ascii") + bytes(data))
        X = np.array([[0, 0, 0, 0, 1, 1, 1, 1]]).T
        YZ = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0])
        for stream in (stream_ascii, stream_binary):
            header, elements = _load_ply_raw(stream)
            [x, yz] = elements["vertex"]
            self.assertClose(x, X)
            self.assertClose(yz, YZ.reshape(8, 2))

    def test_load_cloudcompare_pointcloud(self):
        """
        Test loading a pointcloud styled like some cloudcompare output.
        cloudcompare is an open source 3D point cloud processing software.
        """
        header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "obj_info Not a key-value pair!",
                "element vertex 8",
                "property double x",
                "property double y",
                "property double z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property float my_Favorite",
                "end_header",
                "",
            ]
        ).encode("ascii")
        data = struct.pack("<" + "dddBBBf" * 8, *range(56))
        io = IO()
        with NamedTemporaryFile(mode="wb", suffix=".ply") as f:
            f.write(header)
            f.write(data)
            f.flush()
            pointcloud = io.load_pointcloud(f.name)

        self.assertClose(
            pointcloud.points_padded()[0],
            torch.FloatTensor([0, 1, 2]) + 7 * torch.arange(8)[:, None],
        )
        self.assertClose(
            pointcloud.features_padded()[0] * 255,
            torch.FloatTensor([3, 4, 5]) + 7 * torch.arange(8)[:, None],
        )

    def test_load_open3d_mesh(self):
        # Header based on issue #1104
        header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "comment Created by Open3D",
                "element vertex 3",
                "property double x",
                "property double y",
                "property double z",
                "property double nx",
                "property double ny",
                "property double nz",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "element face 1",
                "property list uchar uint vertex_indices",
                "end_header",
                "",
            ]
        ).encode("ascii")
        vert_data = struct.pack("<" + "ddddddBBB" * 3, *range(9 * 3))
        face_data = struct.pack("<" + "BIII", 3, 0, 1, 2)
        io = IO()
        with NamedTemporaryFile(mode="wb", suffix=".ply") as f:
            f.write(header)
            f.write(vert_data)
            f.write(face_data)
            f.flush()
            mesh = io.load_mesh(f.name)

        self.assertClose(mesh.faces_padded(), torch.arange(3)[None, None])
        self.assertClose(
            mesh.verts_padded(),
            (torch.arange(3) + 9.0 * torch.arange(3)[:, None])[None],
        )

    def test_save_pointcloud(self):
        header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 8",
                "property float x",
                "property float y",
                "property float z",
                "property float red",
                "property float green",
                "property float blue",
                "end_header",
                "",
            ]
        ).encode("ascii")
        data = struct.pack("<" + "f" * 48, *range(48))
        points = torch.FloatTensor([0, 1, 2]) + 6 * torch.arange(8)[:, None]
        features_large = torch.FloatTensor([3, 4, 5]) + 6 * torch.arange(8)[:, None]
        features = features_large / 255.0
        pointcloud_largefeatures = Pointclouds(
            points=[points], features=[features_large]
        )
        pointcloud = Pointclouds(points=[points], features=[features])

        io = IO()
        with NamedTemporaryFile(mode="rb", suffix=".ply") as f:
            io.save_pointcloud(data=pointcloud_largefeatures, path=f.name)
            f.flush()
            f.seek(0)
            actual_data = f.read()
            reloaded_pointcloud = io.load_pointcloud(f.name)

        self.assertEqual(header + data, actual_data)
        self.assertClose(reloaded_pointcloud.points_list()[0], points)
        self.assertClose(reloaded_pointcloud.features_list()[0], features_large)
        # Test the load-save cycle leaves file completely unchanged
        with NamedTemporaryFile(mode="rb", suffix=".ply") as f:
            io.save_pointcloud(
                data=reloaded_pointcloud,
                path=f.name,
            )
            f.flush()
            f.seek(0)
            data2 = f.read()
            self.assertEqual(data2, actual_data)

        with NamedTemporaryFile(mode="r", suffix=".ply") as f:
            io.save_pointcloud(
                data=pointcloud, path=f.name, binary=False, decimal_places=9
            )
            reloaded_pointcloud2 = io.load_pointcloud(f.name)
            self.assertEqual(f.readline(), "ply\n")
            self.assertEqual(f.readline(), "format ascii 1.0\n")
        self.assertClose(reloaded_pointcloud2.points_list()[0], points)
        self.assertClose(reloaded_pointcloud2.features_list()[0], features)

        for binary in [True, False]:
            with NamedTemporaryFile(mode="rb", suffix=".ply") as f:
                io.save_pointcloud(
                    data=pointcloud, path=f.name, colors_as_uint8=True, binary=binary
                )
                f.flush()
                f.seek(0)
                actual_data = f.read()
                reloaded_pointcloud3 = io.load_pointcloud(f.name)
            self.assertClose(reloaded_pointcloud3.features_list()[0], features)
            self.assertIn(b"property uchar green", actual_data)

            # Test the load-save cycle leaves file completely unchanged
            with NamedTemporaryFile(mode="rb", suffix=".ply") as f:
                io.save_pointcloud(
                    data=reloaded_pointcloud3,
                    path=f.name,
                    binary=binary,
                    colors_as_uint8=True,
                )
                f.flush()
                f.seek(0)
                data2 = f.read()
                self.assertEqual(data2, actual_data)

    def test_load_pointcloud_bad_order(self):
        """
        Ply file with a strange property order
        """
        file = "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property uchar green",
                "property float x",
                "property float z",
                "property uchar red",
                "property float y",
                "property uchar blue",
                "end_header",
                "1 2 3 4 5 6",
            ]
        )

        io = IO()
        pointcloud_gpu = io.load_pointcloud(StringIO(file), device="cuda:0")
        self.assertEqual(pointcloud_gpu.device, torch.device("cuda:0"))
        pointcloud = pointcloud_gpu.to(torch.device("cpu"))
        expected_points = torch.tensor([[[2, 5, 3]]], dtype=torch.float32)
        expected_features = torch.tensor([[[4, 1, 6]]], dtype=torch.float32) / 255.0
        self.assertClose(pointcloud.points_padded(), expected_points)
        self.assertClose(pointcloud.features_padded(), expected_features)

    def test_load_simple_binary(self):
        for big_endian in [True, False]:
            verts = (
                "0 0 0 " "0 0 1 " "0 1 1 " "0 1 0 " "1 0 0 " "1 0 1 " "1 1 1 " "1 1 0"
            ).split()
            faces = (
                "4 0 1 2 3 "
                "4 7 6 5 4 "
                "4 0 4 5 1 "
                "4 1 5 6 2 "
                "4 2 6 7 3 "
                "4 3 7 4 0 "  # end of first 6
                "4 0 1 2 3 "
                "4 7 6 5 4 "
                "3 4 5 1"
            ).split()
            short_one = b"\00\01" if big_endian else b"\01\00"
            mixed_data = b"\00\00" b"\03\03" + (short_one + b"\00\01\01\01" b"\00\02")
            minus_one_data = b"\xff" * 14
            endian_char = ">" if big_endian else "<"
            format = (
                "format binary_big_endian 1.0"
                if big_endian
                else "format binary_little_endian 1.0"
            )
            vertex_pattern = endian_char + "24f"
            vertex_data = struct.pack(vertex_pattern, *map(float, verts))
            vertex1_pattern = endian_char + "fdffdffdffdffdffdffdffdf"
            vertex1_data = struct.pack(vertex1_pattern, *map(float, verts))
            face_char_pattern = endian_char + "44b"
            face_char_data = struct.pack(face_char_pattern, *map(int, faces))
            header = "\n".join(
                [
                    "ply",
                    format,
                    "element vertex 8",
                    "property float x",
                    "property float32 y",
                    "property float z",
                    "element vertex1 8",
                    "property float x",
                    "property double y",
                    "property float z",
                    "element face 6",
                    "property list uchar uchar vertex_index",
                    "element irregular_list 3",
                    "property list uchar uchar vertex_index",
                    "element mixed 2",
                    "property list short uint foo",
                    "property short bar",
                    "element minus_ones 1",
                    "property char 1",
                    "property uchar 2",
                    "property short 3",
                    "property ushort 4",
                    "property int 5",
                    "property uint 6",
                    "end_header\n",
                ]
            )
            ply_file = b"".join(
                [
                    header.encode("ascii"),
                    vertex_data,
                    vertex1_data,
                    face_char_data,
                    mixed_data,
                    minus_one_data,
                ]
            )
            metadata, data = _load_ply_raw(BytesIO(ply_file))
            self.assertFalse(metadata.ascii)
            self.assertEqual(len(data), 6)
            self.assertTupleEqual(data["face"].shape, (6, 4))
            self.assertClose([0, 1, 2, 3], data["face"][0])
            self.assertClose([3, 7, 4, 0], data["face"][5])

            [vertex0] = data["vertex"]
            self.assertTupleEqual(vertex0.shape, (8, 3))
            self.assertEqual(len(data["vertex1"]), 3)
            self.assertClose(vertex0, np.column_stack(data["vertex1"]))
            self.assertClose(vertex0.flatten(), list(map(float, verts)))

            irregular = data["irregular_list"]
            self.assertEqual(len(irregular), 3)
            self.assertEqual(type(irregular), list)
            [x] = irregular[0]
            self.assertClose(x, [0, 1, 2, 3])
            [x] = irregular[1]
            self.assertClose(x, [7, 6, 5, 4])
            [x] = irregular[2]
            self.assertClose(x, [4, 5, 1])

            mixed = data["mixed"]
            self.assertEqual(len(mixed), 2)
            self.assertEqual(len(mixed[0]), 2)
            self.assertEqual(len(mixed[1]), 2)
            self.assertEqual(mixed[0][1], 3 * 256 + 3)
            self.assertEqual(len(mixed[0][0]), 0)
            self.assertEqual(mixed[1][1], (2 if big_endian else 2 * 256))
            base = 1 + 256 + 256 * 256
            self.assertEqual(len(mixed[1][0]), 1)
            self.assertEqual(mixed[1][0][0], base if big_endian else 256 * base)

            self.assertListEqual(
                data["minus_ones"], [-1, 255, -1, 65535, -1, 4294967295]
            )

    def test_load_uvs(self):
        io = IO()
        mesh = io.load_mesh(DATA_DIR / "uvs.ply")
        self.assertEqual(mesh.textures.verts_uvs_padded().shape, (1, 8, 2))
        self.assertClose(
            mesh.textures.verts_uvs_padded()[0],
            torch.tensor([[0, 0]] + [[0.2, 0.3]] * 6 + [[0.4, 0.5]]),
        )
        self.assertEqual(
            mesh.textures.faces_uvs_padded().shape, mesh.faces_padded().shape
        )
        self.assertEqual(mesh.textures.maps_padded().shape, (1, 512, 512, 3))

    def test_bad_ply_syntax(self):
        """Some syntactically bad ply files."""
        lines = [
            "ply",
            "format ascii 1.0",
            "comment dashfadskfj;k",
            "element vertex 1",
            "property float x",
            "element listy 1",
            "property list uint int x",
            "end_header",
            "0",
            "0",
        ]
        lines2 = lines.copy()
        # this is ok
        _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2[0] = "PLY"
        with self.assertRaisesRegex(ValueError, "Invalid file header."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2[2] = "#this is a comment"
        with self.assertRaisesRegex(ValueError, "Invalid line.*"):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2[3] = lines[4]
        lines2[4] = lines[3]
        with self.assertRaisesRegex(
            ValueError, "Encountered property before any element."
        ):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2[8] = "1 2"
        with self.assertRaisesRegex(ValueError, "Inconsistent data for vertex."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines[:-1]
        with self.assertRaisesRegex(ValueError, "Not enough data for listy."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2[5] = "element listy 2"
        with self.assertRaisesRegex(ValueError, "Not enough data for listy."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2.insert(4, "property short x")
        with self.assertRaisesRegex(
            ValueError, "Cannot have two properties called x in vertex."
        ):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2.insert(4, "property zz short")
        with self.assertRaisesRegex(ValueError, "Invalid datatype: zz"):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2.append("3")
        with self.assertRaisesRegex(ValueError, "Extra data at end of file."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2.append("comment foo")
        with self.assertRaisesRegex(ValueError, "Extra data at end of file."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2.insert(4, "element bad 1")
        with self.assertRaisesRegex(ValueError, "Found an element with no properties."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2[-1] = "3 2 3 3"
        _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2[-1] = "3 1 2 3 4"
        msg = "A line of listy data did not have the specified length."
        with self.assertRaisesRegex(ValueError, msg):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2 = lines.copy()
        lines2[3] = "element vertex one"
        msg = "Number of items for vertex was not a number."
        with self.assertRaisesRegex(ValueError, msg):
            _load_ply_raw(StringIO("\n".join(lines2)))

        # Heterogeneous cases
        lines2 = lines.copy()
        lines2.insert(4, "property double y")

        with self.assertRaisesRegex(ValueError, "Inconsistent data for vertex."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2[-2] = "3.3 4.2"
        _load_ply_raw(StringIO("\n".join(lines2)))

        lines2[-2] = "3.3 4.3 2"
        with self.assertRaisesRegex(ValueError, "Inconsistent data for vertex."):
            _load_ply_raw(StringIO("\n".join(lines2)))

        with self.assertRaisesRegex(ValueError, "Invalid vertices in file."):
            load_ply(StringIO("\n".join(lines)))

        lines2 = lines.copy()
        lines2[5] = "element face 1"
        with self.assertRaisesRegex(ValueError, "Invalid vertices in file."):
            load_ply(StringIO("\n".join(lines2)))

        lines2.insert(5, "property float z")
        lines2.insert(5, "property float y")
        lines2[-2] = "0 0 0"
        lines2[-1] = ""
        with self.assertRaisesRegex(ValueError, "Not enough data for face."):
            load_ply(StringIO("\n".join(lines2)))

        lines2[-1] = "2 0 0"
        with self.assertRaisesRegex(ValueError, "Faces must have at least 3 vertices."):
            load_ply(StringIO("\n".join(lines2)))

        # Good one
        lines2[-1] = "3 0 0 0"
        load_ply(StringIO("\n".join(lines2)))

    @staticmethod
    def _bm_save_ply(verts: torch.Tensor, faces: torch.Tensor, decimal_places: int):
        return lambda: save_ply(
            BytesIO(),
            verts=verts,
            faces=faces,
            ascii=True,
            decimal_places=decimal_places,
        )

    @staticmethod
    def _bm_load_ply(verts: torch.Tensor, faces: torch.Tensor, decimal_places: int):
        f = BytesIO()
        save_ply(f, verts=verts, faces=faces, ascii=True, decimal_places=decimal_places)
        s = f.getvalue()
        # Recreate stream so it's unaffected by how it was created.
        return lambda: load_ply(BytesIO(s))

    @staticmethod
    def bm_save_simple_ply_with_init(V: int, F: int):
        verts = torch.tensor(V * [[0.11, 0.22, 0.33]]).view(-1, 3)
        faces = torch.tensor(F * [[0, 1, 2]]).view(-1, 3)
        return TestMeshPlyIO._bm_save_ply(verts, faces, decimal_places=2)

    @staticmethod
    def bm_load_simple_ply_with_init(V: int, F: int):
        verts = torch.tensor([[0.1, 0.2, 0.3]]).expand(V, 3)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64).expand(F, 3)
        return TestMeshPlyIO._bm_load_ply(verts, faces, decimal_places=2)

    @staticmethod
    def bm_save_complex_ply(N: int):
        meshes = torus(r=0.25, R=1.0, sides=N, rings=2 * N)
        [verts], [faces] = meshes.verts_list(), meshes.faces_list()
        return TestMeshPlyIO._bm_save_ply(verts, faces, decimal_places=5)

    @staticmethod
    def bm_load_complex_ply(N: int):
        meshes = torus(r=0.25, R=1.0, sides=N, rings=2 * N)
        [verts], [faces] = meshes.verts_list(), meshes.faces_list()
        return TestMeshPlyIO._bm_load_ply(verts, faces, decimal_places=5)
