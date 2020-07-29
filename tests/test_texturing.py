# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import torch
import torch.nn.functional as F
from common_testing import TestCaseMixin
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.textures import (
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
    _list_to_padded_wrapper,
)
from pytorch3d.structures import Meshes, list_to_packed, packed_to_list
from test_meshes import TestMeshes


def tryindex(self, index, tex, meshes, source):
    tex2 = tex[index]
    meshes2 = meshes[index]
    tex_from_meshes = meshes2.textures
    for item in source:
        basic = source[item][index]
        from_texture = getattr(tex2, item + "_padded")()
        from_meshes = getattr(tex_from_meshes, item + "_padded")()
        if isinstance(index, int):
            basic = basic[None]

        if len(basic) == 0:
            self.assertEquals(len(from_texture), 0)
            self.assertEquals(len(from_meshes), 0)
        else:
            self.assertClose(basic, from_texture)
            self.assertClose(basic, from_meshes)
            self.assertEqual(from_texture.ndim, getattr(tex, item + "_padded")().ndim)
            item_list = getattr(tex_from_meshes, item + "_list")()
            self.assertEqual(basic.shape[0], len(item_list))
            for i, elem in enumerate(item_list):
                self.assertClose(elem, basic[i])


class TestTexturesVertex(TestCaseMixin, unittest.TestCase):
    def test_sample_vertex_textures(self):
        """
        This tests both interpolate_vertex_colors as well as
        interpolate_face_attributes.
        """
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        vert_tex = torch.tensor(
            [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]], dtype=torch.float32
        )
        verts_features = vert_tex
        tex = TexturesVertex(verts_features=[verts_features])
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
        # sample_textures calls interpolate_vertex_colors
        texels = mesh.sample_textures(fragments)
        self.assertTrue(torch.allclose(texels, expected_vals[None, :]))

    def test_sample_vertex_textures_grad(self):
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        vert_tex = torch.tensor(
            [[0, 1, 0], [0, 1, 1], [1, 1, 0], [1, 1, 1]],
            dtype=torch.float32,
            requires_grad=True,
        )
        verts_features = vert_tex
        tex = TexturesVertex(verts_features=[verts_features])
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
        texels = mesh.sample_textures(fragments)
        texels.sum().backward()
        self.assertTrue(hasattr(vert_tex, "grad"))
        self.assertTrue(torch.allclose(vert_tex.grad, grad_vert_tex[None, :]))

    def test_textures_vertex_init_fail(self):
        # Incorrect sized tensors
        with self.assertRaisesRegex(ValueError, "verts_features"):
            TexturesVertex(verts_features=torch.rand(size=(5, 10)))

        # Not a list or a tensor
        with self.assertRaisesRegex(ValueError, "verts_features"):
            TexturesVertex(verts_features=(1, 1, 1))

    def test_clone(self):
        tex = TexturesVertex(verts_features=torch.rand(size=(10, 100, 128)))
        tex_cloned = tex.clone()
        self.assertSeparate(
            tex._verts_features_padded, tex_cloned._verts_features_padded
        )
        self.assertSeparate(tex.valid, tex_cloned.valid)

    def test_extend(self):
        B = 10
        mesh = TestMeshes.init_mesh(B, 30, 50)
        V = mesh._V
        tex_uv = TexturesVertex(verts_features=torch.randn((B, V, 3)))
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
                    tex_init.verts_features_list()[i],
                    new_tex.verts_features_list()[i * N + n],
                )
                self.assertClose(
                    tex_init._num_faces_per_mesh[i],
                    new_tex._num_faces_per_mesh[i * N + n],
                )

        self.assertAllSeparate(
            [tex_init.verts_features_padded(), new_tex.verts_features_padded()]
        )

        with self.assertRaises(ValueError):
            tex_mesh.extend(N=-1)

    def test_padded_to_packed(self):
        # Case where each face in the mesh has 3 unique uv vertex indices
        # - i.e. even if a vertex is shared between multiple faces it will
        # have a unique uv coordinate for each face.
        num_verts_per_mesh = [9, 6]
        D = 10
        verts_features_list = [torch.rand(v, D) for v in num_verts_per_mesh]
        verts_features_packed = list_to_packed(verts_features_list)[0]
        verts_features_list = packed_to_list(verts_features_packed, num_verts_per_mesh)
        tex = TexturesVertex(verts_features=verts_features_list)

        # This is set inside Meshes when textures is passed as an input.
        # Here we set _num_faces_per_mesh and _num_verts_per_mesh explicity.
        tex1 = tex.clone()
        tex1._num_verts_per_mesh = num_verts_per_mesh
        verts_packed = tex1.verts_features_packed()
        verts_verts_list = tex1.verts_features_list()
        verts_padded = tex1.verts_features_padded()

        for f1, f2 in zip(verts_verts_list, verts_features_list):
            self.assertTrue((f1 == f2).all().item())

        self.assertTrue(verts_packed.shape == (sum(num_verts_per_mesh), D))
        self.assertTrue(verts_padded.shape == (2, 9, D))

        # Case where num_verts_per_mesh is not set and textures
        # are initialized with a padded tensor.
        tex2 = TexturesVertex(verts_features=verts_padded)
        verts_packed = tex2.verts_features_packed()
        verts_list = tex2.verts_features_list()

        # Packed is just flattened padded as num_verts_per_mesh
        # has not been provided.
        self.assertTrue(verts_packed.shape == (9 * 2, D))

        for i, (f1, f2) in enumerate(zip(verts_list, verts_features_list)):
            n = num_verts_per_mesh[i]
            self.assertTrue((f1[:n] == f2).all().item())

    def test_getitem(self):
        N = 5
        V = 20
        source = {"verts_features": torch.randn(size=(N, 10, 128))}
        tex = TexturesVertex(verts_features=source["verts_features"])

        verts = torch.rand(size=(N, V, 3))
        faces = torch.randint(size=(N, 10, 3), high=V)
        meshes = Meshes(verts=verts, faces=faces, textures=tex)

        tryindex(self, 2, tex, meshes, source)
        tryindex(self, slice(0, 2, 1), tex, meshes, source)
        index = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
        tryindex(self, index, tex, meshes, source)
        index = torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool)
        tryindex(self, index, tex, meshes, source)
        index = torch.tensor([1, 2], dtype=torch.int64)
        tryindex(self, index, tex, meshes, source)
        tryindex(self, [2, 4], tex, meshes, source)


class TestTexturesAtlas(TestCaseMixin, unittest.TestCase):
    def test_sample_texture_atlas(self):
        N, F, R = 1, 2, 2
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        faces_atlas = torch.rand(size=(N, F, R, R, 3))
        tex = TexturesAtlas(atlas=faces_atlas)
        mesh = Meshes(verts=[verts], faces=[faces], textures=tex)
        pix_to_face = torch.tensor([0, 1], dtype=torch.int64).view(1, 1, 1, 2)
        barycentric_coords = torch.tensor(
            [[0.5, 0.3, 0.2], [0.3, 0.6, 0.1]], dtype=torch.float32
        ).view(1, 1, 1, 2, -1)
        expected_vals = torch.tensor(
            [[0.5, 1.0, 0.3], [0.3, 1.0, 0.9]], dtype=torch.float32
        )
        expected_vals = torch.zeros((1, 1, 1, 2, 3), dtype=torch.float32)
        expected_vals[..., 0, :] = faces_atlas[0, 0, 0, 1, ...]
        expected_vals[..., 1, :] = faces_atlas[0, 1, 1, 0, ...]

        fragments = Fragments(
            pix_to_face=pix_to_face,
            bary_coords=barycentric_coords,
            zbuf=torch.ones_like(pix_to_face),
            dists=torch.ones_like(pix_to_face),
        )
        texels = mesh.textures.sample_textures(fragments)
        self.assertTrue(torch.allclose(texels, expected_vals))

    def test_textures_atlas_grad(self):
        N, F, R = 1, 2, 2
        verts = torch.randn((4, 3), dtype=torch.float32)
        faces = torch.tensor([[2, 1, 0], [3, 1, 0]], dtype=torch.int64)
        faces_atlas = torch.rand(size=(N, F, R, R, 3), requires_grad=True)
        tex = TexturesAtlas(atlas=faces_atlas)
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
        texels = mesh.textures.sample_textures(fragments)
        grad_tex = torch.rand_like(texels)
        grad_expected = torch.zeros_like(faces_atlas)
        grad_expected[0, 0, 0, 1, :] = grad_tex[..., 0:1, :]
        grad_expected[0, 1, 1, 0, :] = grad_tex[..., 1:2, :]
        texels.backward(grad_tex)
        self.assertTrue(hasattr(faces_atlas, "grad"))
        self.assertTrue(torch.allclose(faces_atlas.grad, grad_expected))

    def test_textures_atlas_init_fail(self):
        # Incorrect sized tensors
        with self.assertRaisesRegex(ValueError, "atlas"):
            TexturesAtlas(atlas=torch.rand(size=(5, 10, 3)))

        # Not a list or a tensor
        with self.assertRaisesRegex(ValueError, "atlas"):
            TexturesAtlas(atlas=(1, 1, 1))

    def test_clone(self):
        tex = TexturesAtlas(atlas=torch.rand(size=(1, 10, 2, 2, 3)))
        tex_cloned = tex.clone()
        self.assertSeparate(tex._atlas_padded, tex_cloned._atlas_padded)
        self.assertSeparate(tex.valid, tex_cloned.valid)

    def test_extend(self):
        B = 10
        mesh = TestMeshes.init_mesh(B, 30, 50)
        F = mesh._F
        tex_uv = TexturesAtlas(atlas=torch.randn((B, F, 2, 2, 3)))
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
                    tex_init.atlas_list()[i], new_tex.atlas_list()[i * N + n]
                )
                self.assertClose(
                    tex_init._num_faces_per_mesh[i],
                    new_tex._num_faces_per_mesh[i * N + n],
                )

        self.assertAllSeparate([tex_init.atlas_padded(), new_tex.atlas_padded()])

        with self.assertRaises(ValueError):
            tex_mesh.extend(N=-1)

    def test_padded_to_packed(self):
        # Case where each face in the mesh has 3 unique uv vertex indices
        # - i.e. even if a vertex is shared between multiple faces it will
        # have a unique uv coordinate for each face.
        R = 2
        N = 20
        num_faces_per_mesh = torch.randint(size=(N,), low=0, high=30)
        atlas_list = [torch.rand(f, R, R, 3) for f in num_faces_per_mesh]
        tex = TexturesAtlas(atlas=atlas_list)

        # This is set inside Meshes when textures is passed as an input.
        # Here we set _num_faces_per_mesh explicity.
        tex1 = tex.clone()
        tex1._num_faces_per_mesh = num_faces_per_mesh.tolist()
        atlas_packed = tex1.atlas_packed()
        atlas_list_new = tex1.atlas_list()
        atlas_padded = tex1.atlas_padded()

        for f1, f2 in zip(atlas_list_new, atlas_list):
            self.assertTrue((f1 == f2).all().item())

        sum_F = num_faces_per_mesh.sum()
        max_F = num_faces_per_mesh.max().item()
        self.assertTrue(atlas_packed.shape == (sum_F, R, R, 3))
        self.assertTrue(atlas_padded.shape == (N, max_F, R, R, 3))

        # Case where num_faces_per_mesh is not set and textures
        # are initialized with a padded tensor.
        atlas_list_padded = _list_to_padded_wrapper(atlas_list)
        tex2 = TexturesAtlas(atlas=atlas_list_padded)
        atlas_packed = tex2.atlas_packed()
        atlas_list_new = tex2.atlas_list()

        # Packed is just flattened padded as num_faces_per_mesh
        # has not been provided.
        self.assertTrue(atlas_packed.shape == (N * max_F, R, R, 3))

        for i, (f1, f2) in enumerate(zip(atlas_list_new, atlas_list)):
            n = num_faces_per_mesh[i]
            self.assertTrue((f1[:n] == f2).all().item())

    def test_getitem(self):
        N = 5
        V = 20
        source = {"atlas": torch.randn(size=(N, 10, 4, 4, 3))}
        tex = TexturesAtlas(atlas=source["atlas"])

        verts = torch.rand(size=(N, V, 3))
        faces = torch.randint(size=(N, 10, 3), high=V)
        meshes = Meshes(verts=verts, faces=faces, textures=tex)

        tryindex(self, 2, tex, meshes, source)
        tryindex(self, slice(0, 2, 1), tex, meshes, source)
        index = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
        tryindex(self, index, tex, meshes, source)
        index = torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool)
        tryindex(self, index, tex, meshes, source)
        index = torch.tensor([1, 2], dtype=torch.int64)
        tryindex(self, index, tex, meshes, source)
        tryindex(self, [2, 4], tex, meshes, source)


class TestTexturesUV(TestCaseMixin, unittest.TestCase):
    def test_sample_textures_uv(self):
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

        tex = TexturesUV(maps=tex_map, faces_uvs=[face_uvs], verts_uvs=[vert_uvs])
        meshes = Meshes(verts=[dummy_verts], faces=[face_uvs], textures=tex)
        mesh_textures = meshes.textures
        texels = mesh_textures.sample_textures(fragments)

        # Expected output
        pixel_uvs = interpolated_uvs * 2.0 - 1.0
        pixel_uvs = pixel_uvs.view(2, 1, 1, 2)
        tex_map = torch.flip(tex_map, [1])
        tex_map = tex_map.permute(0, 3, 1, 2)
        tex_map = torch.cat([tex_map, tex_map], dim=0)
        expected_out = F.grid_sample(tex_map, pixel_uvs, align_corners=False)
        self.assertTrue(torch.allclose(texels.squeeze(), expected_out.squeeze()))

    def test_textures_uv_init_fail(self):
        # Maps has wrong shape
        with self.assertRaisesRegex(ValueError, "maps"):
            TexturesUV(
                maps=torch.ones((5, 16, 16, 3, 4)),
                faces_uvs=torch.rand(size=(5, 10, 3)),
                verts_uvs=torch.rand(size=(5, 15, 2)),
            )

        # faces_uvs has wrong shape
        with self.assertRaisesRegex(ValueError, "faces_uvs"):
            TexturesUV(
                maps=torch.ones((5, 16, 16, 3)),
                faces_uvs=torch.rand(size=(5, 10, 3, 3)),
                verts_uvs=torch.rand(size=(5, 15, 2)),
            )

        # verts_uvs has wrong shape
        with self.assertRaisesRegex(ValueError, "verts_uvs"):
            TexturesUV(
                maps=torch.ones((5, 16, 16, 3)),
                faces_uvs=torch.rand(size=(5, 10, 3)),
                verts_uvs=torch.rand(size=(5, 15, 2, 3)),
            )

        # verts has different batch dim to faces
        with self.assertRaisesRegex(ValueError, "verts_uvs"):
            TexturesUV(
                maps=torch.ones((5, 16, 16, 3)),
                faces_uvs=torch.rand(size=(5, 10, 3)),
                verts_uvs=torch.rand(size=(8, 15, 2)),
            )

        # maps has different batch dim to faces
        with self.assertRaisesRegex(ValueError, "maps"):
            TexturesUV(
                maps=torch.ones((8, 16, 16, 3)),
                faces_uvs=torch.rand(size=(5, 10, 3)),
                verts_uvs=torch.rand(size=(5, 15, 2)),
            )

        # verts on different device to faces
        with self.assertRaisesRegex(ValueError, "verts_uvs"):
            TexturesUV(
                maps=torch.ones((5, 16, 16, 3)),
                faces_uvs=torch.rand(size=(5, 10, 3)),
                verts_uvs=torch.rand(size=(5, 15, 2, 3), device="cuda"),
            )

        # maps on different device to faces
        with self.assertRaisesRegex(ValueError, "map"):
            TexturesUV(
                maps=torch.ones((5, 16, 16, 3), device="cuda"),
                faces_uvs=torch.rand(size=(5, 10, 3)),
                verts_uvs=torch.rand(size=(5, 15, 2)),
            )

    def test_clone(self):
        tex = TexturesUV(
            maps=torch.ones((5, 16, 16, 3)),
            faces_uvs=torch.rand(size=(5, 10, 3)),
            verts_uvs=torch.rand(size=(5, 15, 2)),
        )
        tex_cloned = tex.clone()
        self.assertSeparate(tex._faces_uvs_padded, tex_cloned._faces_uvs_padded)
        self.assertSeparate(tex._verts_uvs_padded, tex_cloned._verts_uvs_padded)
        self.assertSeparate(tex._maps_padded, tex_cloned._maps_padded)
        self.assertSeparate(tex.valid, tex_cloned.valid)

    def test_extend(self):
        B = 5
        mesh = TestMeshes.init_mesh(B, 30, 50)
        V = mesh._V
        num_faces = mesh.num_faces_per_mesh()
        num_verts = mesh.num_verts_per_mesh()
        faces_uvs_list = [torch.randint(size=(f, 3), low=0, high=V) for f in num_faces]
        verts_uvs_list = [torch.rand(v, 2) for v in num_verts]
        tex_uv = TexturesUV(
            maps=torch.ones((B, 16, 16, 3)),
            faces_uvs=faces_uvs_list,
            verts_uvs=verts_uvs_list,
        )
        tex_mesh = Meshes(
            verts=mesh.verts_list(), faces=mesh.faces_list(), textures=tex_uv
        )
        N = 2
        new_mesh = tex_mesh.extend(N)

        self.assertEqual(len(tex_mesh) * N, len(new_mesh))

        tex_init = tex_mesh.textures
        new_tex = new_mesh.textures

        for i in range(len(tex_mesh)):
            for n in range(N):
                self.assertClose(
                    tex_init.verts_uvs_list()[i], new_tex.verts_uvs_list()[i * N + n]
                )
                self.assertClose(
                    tex_init.faces_uvs_list()[i], new_tex.faces_uvs_list()[i * N + n]
                )
                self.assertClose(
                    tex_init.maps_padded()[i, ...], new_tex.maps_padded()[i * N + n]
                )
                self.assertClose(
                    tex_init._num_faces_per_mesh[i],
                    new_tex._num_faces_per_mesh[i * N + n],
                )

        self.assertAllSeparate(
            [
                tex_init.faces_uvs_padded(),
                new_tex.faces_uvs_padded(),
                tex_init.faces_uvs_packed(),
                new_tex.faces_uvs_packed(),
                tex_init.verts_uvs_padded(),
                new_tex.verts_uvs_padded(),
                tex_init.verts_uvs_packed(),
                new_tex.verts_uvs_packed(),
                tex_init.maps_padded(),
                new_tex.maps_padded(),
            ]
        )

        with self.assertRaises(ValueError):
            tex_mesh.extend(N=-1)

    def test_padded_to_packed(self):
        # Case where each face in the mesh has 3 unique uv vertex indices
        # - i.e. even if a vertex is shared between multiple faces it will
        # have a unique uv coordinate for each face.
        N = 2
        faces_uvs_list = [
            torch.tensor([[0, 1, 2], [3, 5, 4], [7, 6, 8]]),
            torch.tensor([[0, 1, 2], [3, 4, 5]]),
        ]  # (N, 3, 3)
        verts_uvs_list = [torch.ones(9, 2), torch.ones(6, 2)]

        num_faces_per_mesh = [f.shape[0] for f in faces_uvs_list]
        num_verts_per_mesh = [v.shape[0] for v in verts_uvs_list]
        tex = TexturesUV(
            maps=torch.ones((N, 16, 16, 3)),
            faces_uvs=faces_uvs_list,
            verts_uvs=verts_uvs_list,
        )

        # This is set inside Meshes when textures is passed as an input.
        # Here we set _num_faces_per_mesh and _num_verts_per_mesh explicity.
        tex1 = tex.clone()
        tex1._num_faces_per_mesh = num_faces_per_mesh
        tex1._num_verts_per_mesh = num_verts_per_mesh
        verts_packed = tex1.verts_uvs_packed()
        verts_list = tex1.verts_uvs_list()
        verts_padded = tex1.verts_uvs_padded()

        faces_packed = tex1.faces_uvs_packed()
        faces_list = tex1.faces_uvs_list()
        faces_padded = tex1.faces_uvs_padded()

        for f1, f2 in zip(faces_list, faces_uvs_list):
            self.assertTrue((f1 == f2).all().item())

        for f1, f2 in zip(verts_list, verts_uvs_list):
            self.assertTrue((f1 == f2).all().item())

        self.assertTrue(faces_packed.shape == (3 + 2, 3))
        self.assertTrue(faces_padded.shape == (2, 3, 3))
        self.assertTrue(verts_packed.shape == (9 + 6, 2))
        self.assertTrue(verts_padded.shape == (2, 9, 2))

        # Case where num_faces_per_mesh is not set and faces_verts_uvs
        # are initialized with a padded tensor.
        tex2 = TexturesUV(
            maps=torch.ones((N, 16, 16, 3)),
            verts_uvs=verts_padded,
            faces_uvs=faces_padded,
        )
        faces_packed = tex2.faces_uvs_packed()
        faces_list = tex2.faces_uvs_list()
        verts_packed = tex2.verts_uvs_packed()
        verts_list = tex2.verts_uvs_list()

        # Packed is just flattened padded as num_faces_per_mesh
        # has not been provided.
        self.assertTrue(faces_packed.shape == (3 * 2, 3))
        self.assertTrue(verts_packed.shape == (9 * 2, 2))

        for i, (f1, f2) in enumerate(zip(faces_list, faces_uvs_list)):
            n = num_faces_per_mesh[i]
            self.assertTrue((f1[:n] == f2).all().item())

        for i, (f1, f2) in enumerate(zip(verts_list, verts_uvs_list)):
            n = num_verts_per_mesh[i]
            self.assertTrue((f1[:n] == f2).all().item())

    def test_to(self):
        tex = TexturesUV(
            maps=torch.ones((5, 16, 16, 3)),
            faces_uvs=torch.randint(size=(5, 10, 3), high=15),
            verts_uvs=torch.rand(size=(5, 15, 2)),
        )
        device = torch.device("cuda:0")
        tex = tex.to(device)
        self.assertTrue(tex._faces_uvs_padded.device == device)
        self.assertTrue(tex._verts_uvs_padded.device == device)
        self.assertTrue(tex._maps_padded.device == device)

    def test_getitem(self):
        N = 5
        V = 20
        source = {
            "maps": torch.rand(size=(N, 1, 1, 3)),
            "faces_uvs": torch.randint(size=(N, 10, 3), high=V),
            "verts_uvs": torch.randn(size=(N, V, 2)),
        }
        tex = TexturesUV(
            maps=source["maps"],
            faces_uvs=source["faces_uvs"],
            verts_uvs=source["verts_uvs"],
        )

        verts = torch.rand(size=(N, V, 3))
        faces = torch.randint(size=(N, 10, 3), high=V)
        meshes = Meshes(verts=verts, faces=faces, textures=tex)

        tryindex(self, 2, tex, meshes, source)
        tryindex(self, slice(0, 2, 1), tex, meshes, source)
        index = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
        tryindex(self, index, tex, meshes, source)
        index = torch.tensor([0, 0, 0, 0, 0], dtype=torch.bool)
        tryindex(self, index, tex, meshes, source)
        index = torch.tensor([1, 2], dtype=torch.int64)
        tryindex(self, index, tex, meshes, source)
        tryindex(self, [2, 4], tex, meshes, source)
