#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import struct
import unittest
from io import BytesIO, StringIO
import torch

from pytorch3d.io.ply_io import _load_ply_raw, load_ply, save_ply

from common_testing import TestCaseMixin


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
            self.assertTupleEqual(data["vertex"].shape, (8, 3))
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
        )
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
            verts_expected = [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 0],
            ]
            self.assertClose(verts, torch.FloatTensor(verts_expected))
            faces_expected = [
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
            self.assertClose(faces, torch.LongTensor(faces_expected))

    def test_simple_save(self):
        verts = torch.tensor(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float32
        )
        faces = torch.tensor([[0, 1, 2], [0, 3, 4]])
        file = StringIO()
        save_ply(file, verts=verts, faces=faces)
        file.seek(0)
        verts2, faces2 = load_ply(file)
        self.assertClose(verts, verts2)
        self.assertClose(faces, faces2)

    def test_load_simple_binary(self):
        for big_endian in [True, False]:
            verts = (
                "0 0 0 "
                "0 0 1 "
                "0 1 1 "
                "0 1 0 "
                "1 0 0 "
                "1 0 1 "
                "1 1 1 "
                "1 1 0"
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
            mixed_data = b"\00\00" b"\03\03" + (
                short_one + b"\00\01\01\01" b"\00\02"
            )
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
                    "property float y",
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

            self.assertTupleEqual(data["vertex"].shape, (8, 3))
            self.assertEqual(len(data["vertex1"]), 8)
            self.assertClose(data["vertex"], data["vertex1"])
            self.assertClose(data["vertex"].flatten(), list(map(float, verts)))

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
                data["minus_ones"], [(-1, 255, -1, 65535, -1, 4294967295)]
            )

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
        with self.assertRaisesRegex(
            ValueError, "Inconsistent data for vertex."
        ):
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
        with self.assertRaisesRegex(
            ValueError, "Found an element with no properties."
        ):
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

        # Heterogenous cases
        lines2 = lines.copy()
        lines2.insert(4, "property double y")

        with self.assertRaisesRegex(
            ValueError, "Too little data for an element."
        ):
            _load_ply_raw(StringIO("\n".join(lines2)))

        lines2[-2] = "3.3 4.2"
        _load_ply_raw(StringIO("\n".join(lines2)))

        lines2[-2] = "3.3 4.3 2"
        with self.assertRaisesRegex(
            ValueError, "Too much data for an element."
        ):
            _load_ply_raw(StringIO("\n".join(lines2)))

        # Now make the ply file actually be readable as a Mesh

        with self.assertRaisesRegex(
            ValueError, "The ply file has no face element."
        ):
            load_ply(StringIO("\n".join(lines)))

        lines2 = lines.copy()
        lines2[5] = "element face 1"
        with self.assertRaisesRegex(ValueError, "Invalid vertices in file."):
            load_ply(StringIO("\n".join(lines2)))

        lines2.insert(5, "property float z")
        lines2.insert(5, "property float y")
        lines2[-2] = "0 0 0"
        with self.assertRaisesRegex(
            ValueError, "Faces must have at least 3 vertices."
        ):
            load_ply(StringIO("\n".join(lines2)))

        # Good one
        lines2[-1] = "3 0 0 0"
        load_ply(StringIO("\n".join(lines2)))

    @staticmethod
    def save_ply_bm(V: int, F: int):
        verts_list = torch.tensor(V * [[0.11, 0.22, 0.33]]).view(-1, 3)
        faces_list = torch.tensor(F * [[0, 1, 2]]).view(-1, 3)

        def save_mesh():
            file = StringIO()
            save_ply(file, verts_list, faces_list, 2)

        return save_mesh

    @staticmethod
    def load_ply_bm(V: int, F: int):
        verts = torch.tensor([[0.1, 0.2, 0.3]]).expand(V, 3)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64).expand(F, 3)
        ply_file = StringIO()
        save_ply(ply_file, verts=verts, faces=faces)
        ply = ply_file.getvalue()
        # Recreate stream so it's unaffected by how it was created.

        def load_mesh():
            ply_file = StringIO(ply)
            verts, faces = load_ply(ply_file)

        return load_mesh
