# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import torch
import torch.nn.functional as F
from common_testing import TestCaseMixin
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.texturing import (
    interpolate_face_attributes,
    interpolate_texture_map,
    interpolate_vertex_colors,
)
from pytorch3d.structures import Meshes, Textures
from pytorch3d.structures.utils import list_to_padded
from test_meshes import TestMeshes


class TestTexturing(TestCaseMixin, unittest.TestCase):
    def test_interpolate_attributes(self):
        """
        This tests both interpolate_vertex_colors as well as
        interpolate_face_attributes.
        """
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        vert_tex = torch.tensor(
            [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32
        )
        tex = Textures(verts_rgb=vert_tex[None, :])
        mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        pix_to_face = torch.tensor([0, 1], dtype=torch.int64).view(1, 1, 1, 2)
        barycentric_coords = torch.tensor(
            [[0.5, 0.3, 0.2], [0.3, 0.6, 0.1]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        expected_vals = torch.tensor(
            [[0.5, 1.0, 0.3], [0.3, 1.0, 0.9]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=barycentric_coords,
            zbuf=torch.ones_like(pix_to_face),
            dists=torch.ones_like(pix_to_face),
        )
        texels = interpolate_vertex_colors(fragments, mesh)
        self.assertTrue(torch.allclose(texels, expected_vals[None, :]))

    def test_interpolate_attributes_grad(self):
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        vert_tex = torch.tensor(
            [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]],
            dtype=torch.float32,
            requires_grad=True,
        )
        tex = Textures(verts_rgb=vert_tex[None, :])
        mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        pix_to_face = torch.tensor([0, 1], dtype=torch.int64).view(1, 1, 1, 2)
        barycentric_coords = torch.tensor(
            [[0.5, 0.3, 0.2], [0.3, 0.6, 0.1]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=barycentric_coords,
            zbuf=torch.ones_like(pix_to_face),
            dists=torch.ones_like(pix_to_face),
        )
        grad_vert_tex = torch.tensor(
            [[0.3, 0.3, 0.3], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5], [0.3, 0.3, 0.3]],
            dtype=torch.float32,
        )
        texels = interpolate_vertex_colors(fragments, mesh)
        texels.sum().backward()
        self.assertTrue(hasattr(vert_tex, "grad"))
        self.assertTrue(torch.allclose(vert_tex.grad, grad_vert_tex[None, :]))

    def test_interpolate_face_attributes_fail(self):
        # 1. A face can only have 3 verts
        #   i.e. face_attributes must have shape (F, 3, D)
        face_attributes = torch.ones(1, 4, 3)
        pix_to_face = torch.ones((1, 1, 1, 1))
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=pix_to_face[..., None].expand(-1, -1, -1, -1, 3),
            zbuf=pix_to_face,
            dists=pix_to_face,
        )
        with self.assertRaises(ValueError):
            interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, face_attributes
            )

        # 2. pix_to_face must have shape (N, H, W, K)
        pix_to_face = torch.ones((1, 1, 1, 1, 3))
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=pix_to_face,
            zbuf=pix_to_face,
            dists=pix_to_face,
        )
        with self.assertRaises(ValueError):
            interpolate_face_attributes(
                fragments.pix_to_face, fragments.bary_coords, face_attributes
            )

    def test_interpolate_texture_map(self):
        barycentric_coords = torch.tensor(
            [[0.5, 0.3, 0.2], [0.3, 0.6, 0.1]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        dummy_verts = torch.zeros(4, 3)
        vert_uvs = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32)
        face_uvs = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
        interpolated_uvs = torch.tensor(
            [[0.5 + 0.2, 0.3 + 0.2], [0.6, 0.3 + 0.6]], dtype=torch.float32
        )

        # Create a dummy texture map
        H = 2
        W = 2
        x = torch.linspace(0, 1, W).view(1, W).expand(H, W)
        y = torch.linspace(0, 1, H).view(H, 1).expand(H, W)
        tex_map = torch.stack([x, y], dim=2).view(1, H, W, 2)
        pix_to_face = torch.tensor([0, 1], dtype=torch.int64).view(1, 1, 1, 2)
        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=barycentric_coords,
            zbuf=pix_to_face,
            dists=pix_to_face,
        )
        tex = Textures(
            maps=tex_map, faces_uvs=face_uvs[None, ...], verts_uvs=vert_uvs[None, ...]
        )
        meshes = Meshes(verts=[dummy_verts], faces=[face_uvs], textures=tex)
        texels = interpolate_texture_map(fragments, meshes)

        # Expected output
        pixel_uvs = interpolated_uvs * 2.0 - 1.0
        pixel_uvs = pixel_uvs.view(2, 1, 1, 2)
        tex_map = torch.flip(tex_map, [1])
        tex_map = tex_map.permute(0, 3, 1, 2)
        tex_map = torch.cat([tex_map, tex_map], dim=0)
        expected_out = F.grid_sample(tex_map, pixel_uvs, align_corners=False)
        self.assertTrue(torch.allclose(texels.squeeze(), expected_out.squeeze()))

    def test_init_rgb_uv_fail(self):
        V = 20
        # Maps has wrong shape
        with self.assertRaisesRegex(ValueError, "maps"):
            Textures(
                maps=torch.ones((5, 16, 16, 3, 4)),
                faces_uvs=torch.randint(size=(5, 10, 3), low=0, high=V),
                verts_uvs=torch.ones((5, V, 2)),
            )
        # faces_uvs has wrong shape
        with self.assertRaisesRegex(ValueError, "faces_uvs"):
            Textures(
                maps=torch.ones((5, 16, 16, 3)),
                faces_uvs=torch.randint(size=(5, 10, 3, 3), low=0, high=V),
                verts_uvs=torch.ones((5, V, 2)),
            )
        # verts_uvs has wrong shape
        with self.assertRaisesRegex(ValueError, "verts_uvs"):
            Textures(
                maps=torch.ones((5, 16, 16, 3)),
                faces_uvs=torch.randint(size=(5, 10, 3), low=0, high=V),
                verts_uvs=torch.ones((5, V, 2, 3)),
            )
        # verts_rgb has wrong shape
        with self.assertRaisesRegex(ValueError, "verts_rgb"):
            Textures(verts_rgb=torch.ones((5, 16, 16, 3)))

        # maps provided without verts/faces uvs
        with self.assertRaisesRegex(ValueError, "faces_uvs and verts_uvs are required"):
            Textures(maps=torch.ones((5, 16, 16, 3)))

    def test_padded_to_packed(self):
        N = 2
        # Case where each face in the mesh has 3 unique uv vertex indices
        # - i.e. even if a vertex is shared between multiple faces it will
        # have a unique uv coordinate for each face.
        faces_uvs_list = [
            torch.tensor([[0, 1, 2], [3, 5, 4], [7, 6, 8]]),
            torch.tensor([[0, 1, 2], [3, 4, 5]]),
        ]  # (N, 3, 3)
        verts_uvs_list = [torch.ones(9, 2), torch.ones(6, 2)]
        faces_uvs_padded = list_to_padded(faces_uvs_list, pad_value=-1)
        verts_uvs_padded = list_to_padded(verts_uvs_list)
        tex = Textures(
            maps=torch.ones((N, 16, 16, 3)),
            faces_uvs=faces_uvs_padded,
            verts_uvs=verts_uvs_padded,
        )

        # This is set inside Meshes when textures is passed as an input.
        # Here we set _num_faces_per_mesh and _num_verts_per_mesh explicity.
        tex1 = tex.clone()
        tex1._num_faces_per_mesh = faces_uvs_padded.gt(-1).all(-1).sum(-1).tolist()
        tex1._num_verts_per_mesh = torch.tensor([5, 4])
        faces_packed = tex1.faces_uvs_packed()
        verts_packed = tex1.verts_uvs_packed()
        faces_list = tex1.faces_uvs_list()
        verts_list = tex1.verts_uvs_list()

        for f1, f2 in zip(faces_uvs_list, faces_list):
            self.assertTrue((f1 == f2).all().item())

        for f, v1, v2 in zip(faces_list, verts_list, verts_uvs_list):
            idx = f.unique()
            self.assertTrue((v1[idx] == v2).all().item())

        self.assertTrue(faces_packed.shape == (3 + 2, 3))

        # verts_packed is just flattened verts_padded.
        # split sizes are not used for verts_uvs.
        self.assertTrue(verts_packed.shape == (9 * 2, 2))

        # Case where num_faces_per_mesh is not set
        tex2 = tex.clone()
        faces_packed = tex2.faces_uvs_packed()
        verts_packed = tex2.verts_uvs_packed()
        faces_list = tex2.faces_uvs_list()
        verts_list = tex2.verts_uvs_list()

        # Packed is just flattened padded as num_faces_per_mesh
        # has not been provided.
        self.assertTrue(verts_packed.shape == (9 * 2, 2))
        self.assertTrue(faces_packed.shape == (3 * 2, 3))

        for i in range(N):
            self.assertTrue(
                (faces_list[i] == faces_uvs_padded[i, ...].squeeze()).all().item()
            )

        for i in range(N):
            self.assertTrue(
                (verts_list[i] == verts_uvs_padded[i, ...].squeeze()).all().item()
            )

    def test_clone(self):
        V = 20
        tex = Textures(
            maps=torch.ones((5, 16, 16, 3)),
            faces_uvs=torch.randint(size=(5, 10, 3), low=0, high=V),
            verts_uvs=torch.ones((5, V, 2)),
        )
        tex_cloned = tex.clone()
        self.assertSeparate(tex._faces_uvs_padded, tex_cloned._faces_uvs_padded)
        self.assertSeparate(tex._verts_uvs_padded, tex_cloned._verts_uvs_padded)
        self.assertSeparate(tex._maps_padded, tex_cloned._maps_padded)

    def test_getitem(self):
        N = 5
        V = 20
        source = {
            "maps": torch.rand(size=(N, 16, 16, 3)),
            "faces_uvs": torch.randint(size=(N, 10, 3), low=0, high=V),
            "verts_uvs": torch.rand((N, V, 2)),
        }
        tex = Textures(
            maps=source["maps"],
            faces_uvs=source["faces_uvs"],
            verts_uvs=source["verts_uvs"],
        )

        verts = torch.rand(size=(N, V, 3))
        faces = torch.randint(size=(N, 10, 3), high=V)
        meshes = Meshes(verts=verts, faces=faces, textures=tex)

        def tryindex(index):
            tex2 = tex[index]
            meshes2 = meshes[index]
            tex_from_meshes = meshes2.textures
            for item in source:
                basic = source[item][index]
                from_texture = getattr(tex2, item + "_padded")()
                from_meshes = getattr(tex_from_meshes, item + "_padded")()
                if isinstance(index, int):
                    basic = basic[None]
                self.assertClose(basic, from_texture)
                self.assertClose(basic, from_meshes)
                self.assertEqual(
                    from_texture.ndim, getattr(tex, item + "_padded")().ndim
                )
                if item == "faces_uvs":
                    faces_uvs_list = tex_from_meshes.faces_uvs_list()
                    self.assertEqual(basic.shape[0], len(faces_uvs_list))
                    for i, faces_uvs in enumerate(faces_uvs_list):
                        self.assertClose(faces_uvs, basic[i])

        tryindex(2)
        tryindex(slice(0, 2, 1))
        index = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
        tryindex(index)
        index = torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool)
        tryindex(index)
        index = torch.tensor([1, 2], dtype=torch.int64)
        tryindex(index)
        tryindex([2, 4])

    def test_to(self):
        V = 20
        tex = Textures(
            maps=torch.ones((5, 16, 16, 3)),
            faces_uvs=torch.randint(size=(5, 10, 3), low=0, high=V),
            verts_uvs=torch.ones((5, V, 2)),
        )
        device = torch.device("cuda:0")
        tex = tex.to(device)
        self.assertTrue(tex._faces_uvs_padded.device == device)
        self.assertTrue(tex._verts_uvs_padded.device == device)
        self.assertTrue(tex._maps_padded.device == device)

    def test_extend(self):
        B = 10
        mesh = TestMeshes.init_mesh(B, 30, 50)
        V = mesh._V
        F = mesh._F

        # 1. Texture uvs
        tex_uv = Textures(
            maps=torch.randn((B, 16, 16, 3)),
            faces_uvs=torch.randint(size=(B, F, 3), low=0, high=V),
            verts_uvs=torch.randn((B, V, 2)),
        )
        tex_mesh = Meshes(
            verts=mesh.verts_padded(), faces=mesh.faces_padded(), textures=tex_uv
        )
        N = 20
        new_mesh = tex_mesh.extend(N)

        self.assertEqual(len(tex_mesh) * N, len(new_mesh))

        tex_init = tex_mesh.textures
        new_tex = new_mesh.textures

        for i in range(len(tex_mesh)):
            for n in range(N):
                self.assertClose(
                    tex_init.faces_uvs_list()[i], new_tex.faces_uvs_list()[i * N + n]
                )
                self.assertClose(
                    tex_init.verts_uvs_list()[i], new_tex.verts_uvs_list()[i * N + n]
                )
        self.assertAllSeparate(
            [
                tex_init.faces_uvs_padded(),
                new_tex.faces_uvs_padded(),
                tex_init.verts_uvs_padded(),
                new_tex.verts_uvs_padded(),
                tex_init.maps_padded(),
                new_tex.maps_padded(),
            ]
        )

        self.assertIsNone(new_tex.verts_rgb_list())
        self.assertIsNone(new_tex.verts_rgb_padded())
        self.assertIsNone(new_tex.verts_rgb_packed())

        # 2. Texture vertex RGB
        tex_rgb = Textures(verts_rgb=torch.randn((B, V, 3)))
        tex_mesh_rgb = Meshes(
            verts=mesh.verts_padded(), faces=mesh.faces_padded(), textures=tex_rgb
        )
        N = 20
        new_mesh_rgb = tex_mesh_rgb.extend(N)

        self.assertEqual(len(tex_mesh_rgb) * N, len(new_mesh_rgb))

        tex_init = tex_mesh_rgb.textures
        new_tex = new_mesh_rgb.textures

        for i in range(len(tex_mesh_rgb)):
            for n in range(N):
                self.assertClose(
                    tex_init.verts_rgb_list()[i], new_tex.verts_rgb_list()[i * N + n]
                )
        self.assertAllSeparate(
            [tex_init.verts_rgb_padded(), new_tex.verts_rgb_padded()]
        )

        self.assertIsNone(new_tex.verts_uvs_padded())
        self.assertIsNone(new_tex.verts_uvs_list())
        self.assertIsNone(new_tex.verts_uvs_packed())
        self.assertIsNone(new_tex.faces_uvs_padded())
        self.assertIsNone(new_tex.faces_uvs_list())
        self.assertIsNone(new_tex.faces_uvs_packed())

        # 3. Error
        with self.assertRaises(ValueError):
            tex_mesh.extend(N=-1)
