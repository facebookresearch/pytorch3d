# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from tempfile import NamedTemporaryFile

import torch
from pytorch3d.io import IO
from pytorch3d.renderer import TexturesAtlas, TexturesVertex
from pytorch3d.utils import ico_sphere

from .common_testing import TestCaseMixin


CUBE_FACES = [
    [0, 1, 2],
    [7, 4, 0],
    [4, 5, 1],
    [5, 6, 2],
    [3, 2, 6],
    [6, 5, 4],
    [0, 2, 3],
    [7, 0, 3],
    [4, 1, 0],
    [5, 2, 1],
    [3, 6, 7],
    [6, 4, 7],
]


class TestMeshOffIO(TestCaseMixin, unittest.TestCase):
    def test_load_face_colors(self):
        # Example from wikipedia
        off_file_lines = [
            "OFF",
            "# cube.off",
            "# A cube",
            " ",
            "8 6 12",
            " 1.0  0.0 1.4142",
            " 0.0  1.0 1.4142",
            "-1.0  0.0 1.4142",
            " 0.0 -1.0 1.4142",
            " 1.0  0.0 0.0",
            " 0.0  1.0 0.0",
            "-1.0  0.0 0.0",
            " 0.0 -1.0 0.0",
            "4  0 1 2 3  255 0 0 #red",
            "4  7 4 0 3  0 255 0 #green",
            "4  4 5 1 0  0 0 255 #blue",
            "4  5 6 2 1  0 255 0 ",
            "4  3 2 6 7  0 0 255",
            "4  6 5 4 7  255 0 0",
        ]
        off_file = "\n".join(off_file_lines)
        io = IO()
        with NamedTemporaryFile(mode="w", suffix=".off") as f:
            f.write(off_file)
            f.flush()
            mesh = io.load_mesh(f.name)

        self.assertEqual(mesh.verts_padded().shape, (1, 8, 3))
        verts_str = " ".join(off_file_lines[5:13])
        verts_data = torch.tensor([float(i) for i in verts_str.split()])
        self.assertClose(mesh.verts_padded().flatten(), verts_data)
        self.assertClose(mesh.faces_padded(), torch.tensor(CUBE_FACES)[None])

        faces_colors_full = mesh.textures.atlas_padded()
        self.assertEqual(faces_colors_full.shape, (1, 12, 1, 1, 3))
        faces_colors = faces_colors_full[0, :, 0, 0]
        max_color = faces_colors.max()
        self.assertEqual(max_color, 1)

        # Every face has one color 1, the rest 0.
        total_color = faces_colors.sum(dim=1)
        self.assertEqual(total_color.max(), max_color)
        self.assertEqual(total_color.min(), max_color)

    def test_load_vertex_colors(self):
        # Example with no faces and with integer vertex colors
        off_file_lines = [
            "8 1 12",
            " 1.0  0.0 1.4142 0 1 0",
            " 0.0  1.0 1.4142 0 1 0",
            "-1.0  0.0 1.4142 0 1 0",
            " 0.0 -1.0 1.4142 0 1 0",
            " 1.0  0.0 0.0 0 1 0",
            " 0.0  1.0 0.0 0 1 0",
            "-1.0  0.0 0.0 0 1 0",
            " 0.0 -1.0 0.0 0 1 0",
            "3 0 1 2",
        ]
        off_file = "\n".join(off_file_lines)
        io = IO()
        with NamedTemporaryFile(mode="w", suffix=".off") as f:
            f.write(off_file)
            f.flush()
            mesh = io.load_mesh(f.name)

        self.assertEqual(mesh.verts_padded().shape, (1, 8, 3))
        verts_lines = (line.split()[:3] for line in off_file_lines[1:9])
        verts_data = [[[float(x) for x in line] for line in verts_lines]]
        self.assertClose(mesh.verts_padded(), torch.tensor(verts_data))
        self.assertClose(mesh.faces_padded(), torch.tensor([[[0, 1, 2]]]))

        self.assertIsInstance(mesh.textures, TexturesVertex)
        colors = mesh.textures.verts_features_padded()

        self.assertEqual(colors.shape, (1, 8, 3))
        self.assertClose(colors[0, :, [0, 2]], torch.zeros(8, 2))
        self.assertClose(colors[0, :, 1], torch.full((8,), 1.0 / 255))

    def test_load_lumpy(self):
        # Example off file whose faces have different numbers of vertices.
        off_file_lines = [
            "8 3 12",
            " 1.0  0.0 1.4142",
            " 0.0  1.0 1.4142",
            "-1.0  0.0 1.4142",
            " 0.0 -1.0 1.4142",
            " 1.0  0.0 0.0",
            " 0.0  1.0 0.0",
            "-1.0  0.0 0.0",
            " 0.0 -1.0 0.0",
            "3  0 1 2    255 0 0 #red",
            "4  7 4 0 3  0 255 0 #green",
            "4  4 5 1 0  0 0 255 #blue",
        ]
        off_file = "\n".join(off_file_lines)
        io = IO()
        with NamedTemporaryFile(mode="w", suffix=".off") as f:
            f.write(off_file)
            f.flush()
            mesh = io.load_mesh(f.name)

        self.assertEqual(mesh.verts_padded().shape, (1, 8, 3))
        verts_str = " ".join(off_file_lines[1:9])
        verts_data = torch.tensor([float(i) for i in verts_str.split()])
        self.assertClose(mesh.verts_padded().flatten(), verts_data)

        self.assertEqual(mesh.faces_padded().shape, (1, 5, 3))
        faces_expected = [[0, 1, 2], [7, 4, 0], [7, 0, 3], [4, 5, 1], [4, 1, 0]]
        self.assertClose(mesh.faces_padded()[0], torch.tensor(faces_expected))

    def test_save_load_icosphere(self):
        # Test that saving a mesh as an off file and loading it results in the
        # same data on the correct device, for all permitted types of textures.
        # Standard test is for random colors, but also check totally white,
        # because there's a different in OFF semantics between "1.0" color (=full)
        # and "1" (= 1/255 color)
        sphere = ico_sphere(0)
        io = IO()
        device = torch.device("cuda:0")

        atlas_padded = torch.rand(1, sphere.faces_list()[0].shape[0], 1, 1, 3)
        atlas = TexturesAtlas(atlas_padded)

        atlas_padded_white = torch.ones(1, sphere.faces_list()[0].shape[0], 1, 1, 3)
        atlas_white = TexturesAtlas(atlas_padded_white)

        verts_colors_padded = torch.rand(1, sphere.verts_list()[0].shape[0], 3)
        vertex_texture = TexturesVertex(verts_colors_padded)

        verts_colors_padded_white = torch.ones(1, sphere.verts_list()[0].shape[0], 3)
        vertex_texture_white = TexturesVertex(verts_colors_padded_white)

        # No colors case
        with NamedTemporaryFile(mode="w", suffix=".off") as f:
            io.save_mesh(sphere, f.name)
            f.flush()
            mesh1 = io.load_mesh(f.name, device=device)
        self.assertEqual(mesh1.device, device)
        mesh1 = mesh1.cpu()
        self.assertClose(mesh1.verts_padded(), sphere.verts_padded())
        self.assertClose(mesh1.faces_padded(), sphere.faces_padded())
        self.assertIsNone(mesh1.textures)

        # Atlas case
        sphere.textures = atlas
        with NamedTemporaryFile(mode="w", suffix=".off") as f:
            io.save_mesh(sphere, f.name)
            f.flush()
            mesh2 = io.load_mesh(f.name, device=device)

        self.assertEqual(mesh2.device, device)
        mesh2 = mesh2.cpu()
        self.assertClose(mesh2.verts_padded(), sphere.verts_padded())
        self.assertClose(mesh2.faces_padded(), sphere.faces_padded())
        self.assertClose(mesh2.textures.atlas_padded(), atlas_padded, atol=1e-4)

        # White atlas case
        sphere.textures = atlas_white
        with NamedTemporaryFile(mode="w", suffix=".off") as f:
            io.save_mesh(sphere, f.name)
            f.flush()
            mesh3 = io.load_mesh(f.name)

        self.assertClose(mesh3.textures.atlas_padded(), atlas_padded_white, atol=1e-4)

        # TexturesVertex case
        sphere.textures = vertex_texture
        with NamedTemporaryFile(mode="w", suffix=".off") as f:
            io.save_mesh(sphere, f.name)
            f.flush()
            mesh4 = io.load_mesh(f.name, device=device)

        self.assertEqual(mesh4.device, device)
        mesh4 = mesh4.cpu()
        self.assertClose(mesh4.verts_padded(), sphere.verts_padded())
        self.assertClose(mesh4.faces_padded(), sphere.faces_padded())
        self.assertClose(
            mesh4.textures.verts_features_padded(), verts_colors_padded, atol=1e-4
        )

        # white TexturesVertex case
        sphere.textures = vertex_texture_white
        with NamedTemporaryFile(mode="w", suffix=".off") as f:
            io.save_mesh(sphere, f.name)
            f.flush()
            mesh5 = io.load_mesh(f.name)

        self.assertClose(
            mesh5.textures.verts_features_padded(), verts_colors_padded_white, atol=1e-4
        )

    def test_bad(self):
        # Test errors from various invalid OFF files.
        io = IO()

        def load(lines):
            off_file = "\n".join(lines)
            with NamedTemporaryFile(mode="w", suffix=".off") as f:
                f.write(off_file)
                f.flush()
                io.load_mesh(f.name)

        # First a good example
        lines = [
            "4 2 12",
            " 1.0  0.0 1.4142",
            " 0.0  1.0 1.4142",
            " 1.0  0.0 0.4142",
            " 0.0  1.0 0.4142",
            "3  0 1 2 ",
            "3  1 3 0 ",
        ]

        # This example passes.
        load(lines)

        # OFF can occur on the first line separately
        load(["OFF"] + lines)

        # OFF line can be merged in to the first line
        lines2 = lines.copy()
        lines2[0] = "OFF " + lines[0]
        load(lines2)

        # OFF line can be merged in to the first line with no space
        lines2 = lines.copy()
        lines2[0] = "OFF" + lines[0]
        load(lines2)

        with self.assertRaisesRegex(ValueError, "Not enough face data."):
            load(lines[:-1])

        lines2 = lines.copy()
        lines2[0] = "4 1 12"
        with self.assertRaisesRegex(ValueError, "Extra data at end of file:"):
            load(lines2)

        lines2 = lines.copy()
        lines2[-1] = "2 1 3"
        with self.assertRaisesRegex(ValueError, "Faces must have at least 3 vertices."):
            load(lines2)

        lines2 = lines.copy()
        lines2[-1] = "4 1 3 0"
        with self.assertRaisesRegex(
            ValueError, "A line of face data did not have the specified length."
        ):
            load(lines2)

        lines2 = lines.copy()
        lines2[0] = "6 2 0"
        with self.assertRaisesRegex(ValueError, "number of columns"):
            load(lines2)

        lines2[0] = "5 1 0"
        with self.assertRaisesRegex(ValueError, "number of columns"):
            load(lines2)

        lines2[0] = "16 2 0"
        with self.assertRaisesRegex(ValueError, "number of columns"):
            load(lines2)

        lines2[0] = "3 3 0"
        # This is a bit of a special case because the last vertex could be a face
        with self.assertRaisesRegex(ValueError, "Faces must have at least 3 vertices."):
            load(lines2)

        lines2[4] = "7.3 4.2 8.3"
        with self.assertRaisesRegex(
            ValueError, "A line of face data did not have the specified length."
        ):
            load(lines2)

        # Now try bad number of colors

        lines2 = lines.copy()
        lines2[2] = "7.3 4.2 8.3 932"
        with self.assertRaisesRegex(ValueError, "number of columns"):
            load(lines2)

        lines2[1] = "7.3 4.2 8.3 932"
        lines2[3] = "7.3 4.2 8.3 932"
        lines2[4] = "7.3 4.2 8.3 932"
        with self.assertRaisesRegex(ValueError, "Bad vertex data."):
            load(lines2)

        lines2 = lines.copy()
        lines2[5] = "3  0 1 2 0.9"
        lines2[6] = "3  0 3 0 0.9"
        with self.assertRaisesRegex(ValueError, "Unexpected number of colors."):
            load(lines2)

        lines2 = lines.copy()
        for i in range(1, 7):
            lines2[i] = lines2[i] + " 4 4 4 4"
        msg = "Faces colors ignored because vertex colors provided too."
        with self.assertWarnsRegex(UserWarning, msg):
            load(lines2)
