#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest
import torch
import torch.nn.functional as F

from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.texturing import (
    interpolate_face_attributes,
    interpolate_texture_map,
    interpolate_vertex_colors,
)
from pytorch3d.structures import Meshes, Textures

from common_testing import TestCaseMixin
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
            [
                [0.3, 0.3, 0.3],
                [0.9, 0.9, 0.9],
                [0.5, 0.5, 0.5],
                [0.3, 0.3, 0.3],
            ],
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
        vert_uvs = torch.tensor(
            [[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float32
        )
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
            maps=tex_map,
            faces_uvs=face_uvs[None, ...],
            verts_uvs=vert_uvs[None, ...],
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
        self.assertTrue(
            torch.allclose(texels.squeeze(), expected_out.squeeze())
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
        tex = Textures(
            maps=torch.randn((B, 16, 16, 3)),
            faces_uvs=torch.randint(size=(B, F, 3), low=0, high=V),
            verts_uvs=torch.randn((B, V, 2)),
        )
        tex_mesh = Meshes(
            verts=mesh.verts_padded(), faces=mesh.faces_padded(), textures=tex
        )
        N = 20
        new_mesh = tex_mesh.extend(N)

        self.assertEqual(len(tex_mesh) * N, len(new_mesh))

        tex_init = tex_mesh.textures
        new_tex = new_mesh.textures

        for i in range(len(tex_mesh)):
            for n in range(N):
                self.assertClose(
                    tex_init.faces_uvs_list()[i],
                    new_tex.faces_uvs_list()[i * N + n],
                )
                self.assertClose(
                    tex_init.verts_uvs_list()[i],
                    new_tex.verts_uvs_list()[i * N + n],
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
        with self.assertRaises(ValueError):
            tex_mesh.extend(N=-1)
