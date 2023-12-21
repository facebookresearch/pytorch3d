# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
import torch.nn.functional as F
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.textures import (
    _list_to_padded_wrapper,
    TexturesAtlas,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.utils import (
    pack_rectangles,
    pack_unique_rectangles,
    Rectangle,
)
from pytorch3d.structures import list_to_packed, Meshes, packed_to_list

from .common_testing import TestCaseMixin
from .test_meshes import init_mesh


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
            self.assertEqual(len(from_texture), 0)
            self.assertEqual(len(from_meshes), 0)
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

    def test_faces_verts_textures(self):
        device = torch.device("cuda:0")
        verts = torch.randn((2, 4, 3), dtype=torch.float32, device=device)
        faces = torch.tensor(
            [[[2, 1, 0], [3, 1, 0]], [[1, 3, 0], [2, 1, 3]]],
            dtype=torch.int64,
            device=device,
        )

        # define TexturesVertex
        verts_texture = torch.rand(verts.shape, device=device)
        textures = TexturesVertex(verts_features=verts_texture)

        # compute packed faces
        ff = faces.unbind(0)
        faces_packed = torch.cat([ff[0], ff[1] + verts.shape[1]])

        # face verts textures
        faces_verts_texts = textures.faces_verts_textures_packed(faces_packed)

        verts_texts_packed = torch.cat(verts_texture.unbind(0))
        faces_verts_texts_packed = verts_texts_packed[faces_packed]

        self.assertClose(faces_verts_texts_packed, faces_verts_texts)

    def test_submeshes(self):
        # define TexturesVertex
        verts_features = torch.tensor(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=torch.float32,
        )

        textures = TexturesVertex(
            verts_features=[verts_features, verts_features, verts_features]
        )
        subtextures = textures.submeshes(
            [
                [
                    torch.LongTensor([0, 2, 3]),
                    torch.LongTensor(list(range(8))),
                ],
                [],
                [
                    torch.LongTensor([4]),
                ],
            ],
            None,
        )

        subtextures_features = subtextures.verts_features_list()

        self.assertEqual(len(subtextures_features), 3)
        self.assertTrue(
            torch.equal(
                subtextures_features[0],
                torch.FloatTensor([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
            )
        )
        self.assertTrue(torch.equal(subtextures_features[1], verts_features))
        self.assertTrue(
            torch.equal(subtextures_features[2], torch.FloatTensor([[0, 1, 0]]))
        )

    def test_clone(self):
        tex = TexturesVertex(verts_features=torch.rand(size=(10, 100, 128)))
        tex.verts_features_list()
        tex_cloned = tex.clone()
        self.assertSeparate(
            tex._verts_features_padded, tex_cloned._verts_features_padded
        )
        self.assertClose(tex._verts_features_padded, tex_cloned._verts_features_padded)
        self.assertSeparate(tex.valid, tex_cloned.valid)
        self.assertTrue(tex.valid.eq(tex_cloned.valid).all())
        for i in range(tex._N):
            self.assertSeparate(
                tex._verts_features_list[i], tex_cloned._verts_features_list[i]
            )
            self.assertClose(
                tex._verts_features_list[i], tex_cloned._verts_features_list[i]
            )

    def test_detach(self):
        tex = TexturesVertex(
            verts_features=torch.rand(size=(10, 100, 128), requires_grad=True)
        )
        tex.verts_features_list()
        tex_detached = tex.detach()
        self.assertFalse(tex_detached._verts_features_padded.requires_grad)
        self.assertClose(
            tex_detached._verts_features_padded, tex._verts_features_padded
        )
        for i in range(tex._N):
            self.assertClose(
                tex._verts_features_list[i], tex_detached._verts_features_list[i]
            )
            self.assertFalse(tex_detached._verts_features_list[i].requires_grad)

    def test_extend(self):
        B = 10
        mesh = init_mesh(B, 30, 50)
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
        source = {"verts_features": torch.randn(size=(N, V, 128))}
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

    def test_sample_textures_error(self):
        N = 5
        V = 20
        verts = torch.rand(size=(N, V, 3))
        faces = torch.randint(size=(N, 10, 3), high=V)
        tex = TexturesVertex(verts_features=torch.randn(size=(N, 10, 128)))

        # Verts features have the wrong number of verts
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            Meshes(verts=verts, faces=faces, textures=tex)

        # Verts features have the wrong batch dim
        tex = TexturesVertex(verts_features=torch.randn(size=(1, V, 128)))
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            Meshes(verts=verts, faces=faces, textures=tex)

        meshes = Meshes(verts=verts, faces=faces)
        meshes.textures = tex

        # Cannot use the texture attribute set on meshes for sampling
        # textures if the dimensions don't match
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            meshes.sample_textures(None)


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

    def test_faces_verts_textures(self):
        device = torch.device("cuda:0")
        N, F, R = 2, 2, 8
        num_faces = torch.randint(low=1, high=F, size=(N,))
        faces_atlas = [
            torch.rand(size=(num_faces[i].item(), R, R, 3), device=device)
            for i in range(N)
        ]
        tex = TexturesAtlas(atlas=faces_atlas)

        # faces_verts naive
        faces_verts = []
        for n in range(N):
            ff = num_faces[n].item()
            temp = torch.zeros(ff, 3, 3)
            for f in range(ff):
                t0 = faces_atlas[n][f, 0, -1]  # for v0, bary = (1, 0)
                t1 = faces_atlas[n][f, -1, 0]  # for v1, bary = (0, 1)
                t2 = faces_atlas[n][f, 0, 0]  # for v2, bary = (0, 0)
                temp[f, 0] = t0
                temp[f, 1] = t1
                temp[f, 2] = t2
            faces_verts.append(temp)
        faces_verts = torch.cat(faces_verts, 0)

        self.assertClose(faces_verts, tex.faces_verts_textures_packed().cpu())

    def test_clone(self):
        tex = TexturesAtlas(atlas=torch.rand(size=(1, 10, 2, 2, 3)))
        tex.atlas_list()
        tex_cloned = tex.clone()
        self.assertSeparate(tex._atlas_padded, tex_cloned._atlas_padded)
        self.assertClose(tex._atlas_padded, tex_cloned._atlas_padded)
        self.assertSeparate(tex.valid, tex_cloned.valid)
        self.assertTrue(tex.valid.eq(tex_cloned.valid).all())
        for i in range(tex._N):
            self.assertSeparate(tex._atlas_list[i], tex_cloned._atlas_list[i])
            self.assertClose(tex._atlas_list[i], tex_cloned._atlas_list[i])

    def test_detach(self):
        tex = TexturesAtlas(atlas=torch.rand(size=(1, 10, 2, 2, 3), requires_grad=True))
        tex.atlas_list()
        tex_detached = tex.detach()
        self.assertFalse(tex_detached._atlas_padded.requires_grad)
        self.assertClose(tex_detached._atlas_padded, tex._atlas_padded)
        for i in range(tex._N):
            self.assertFalse(tex_detached._atlas_list[i].requires_grad)
            self.assertClose(tex._atlas_list[i], tex_detached._atlas_list[i])

    def test_extend(self):
        B = 10
        mesh = init_mesh(B, 30, 50)
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
        F = 10
        source = {"atlas": torch.randn(size=(N, F, 4, 4, 3))}
        tex = TexturesAtlas(atlas=source["atlas"])

        verts = torch.rand(size=(N, V, 3))
        faces = torch.randint(size=(N, F, 3), high=V)
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

    def test_sample_textures_error(self):
        N = 1
        V = 20
        F = 10
        verts = torch.rand(size=(5, V, 3))
        faces = torch.randint(size=(5, F, 3), high=V)
        meshes = Meshes(verts=verts, faces=faces)

        # TexturesAtlas have the wrong batch dim
        tex = TexturesAtlas(atlas=torch.randn(size=(1, F, 4, 4, 3)))
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            Meshes(verts=verts, faces=faces, textures=tex)

        # TexturesAtlas have the wrong number of faces
        tex = TexturesAtlas(atlas=torch.randn(size=(N, 15, 4, 4, 3)))
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            Meshes(verts=verts, faces=faces, textures=tex)

        meshes = Meshes(verts=verts, faces=faces)
        meshes.textures = tex

        # Cannot use the texture attribute set on meshes for sampling
        # textures if the dimensions don't match
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            meshes.sample_textures(None)

    def test_submeshes(self):
        N = 2
        V = 5
        F = 5
        tex = TexturesAtlas(
            atlas=torch.arange(N * F * 4 * 4 * 3, dtype=torch.float32).reshape(
                N, F, 4, 4, 3
            )
        )

        verts = torch.rand(size=(N, V, 3))
        faces = torch.randint(size=(N, F, 3), high=V)
        mesh = Meshes(verts=verts, faces=faces, textures=tex)

        sub_faces = [
            [torch.tensor([0, 2]), torch.tensor([1, 2])],
            [],
        ]
        subtex = mesh.submeshes(sub_faces).textures
        subtex_faces = subtex.atlas_list()

        self.assertEqual(len(subtex_faces), 2)
        self.assertClose(
            subtex_faces[0].flatten().msort(),
            torch.cat(
                (
                    torch.arange(4 * 4 * 3, dtype=torch.float32),
                    torch.arange(96, 96 + 4 * 4 * 3, dtype=torch.float32),
                ),
                0,
            ),
        )


class TestTexturesUV(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

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

        for align_corners in [True, False]:
            tex = TexturesUV(
                maps=tex_map,
                faces_uvs=[face_uvs],
                verts_uvs=[vert_uvs],
                align_corners=align_corners,
            )
            meshes = Meshes(verts=[dummy_verts], faces=[face_uvs], textures=tex)
            mesh_textures = meshes.textures
            texels = mesh_textures.sample_textures(fragments)

            # Expected output
            pixel_uvs = interpolated_uvs * 2.0 - 1.0
            pixel_uvs = pixel_uvs.view(2, 1, 1, 2)
            tex_map_ = torch.flip(tex_map, [1]).permute(0, 3, 1, 2)
            tex_map_ = torch.cat([tex_map_, tex_map_], dim=0)
            expected_out = F.grid_sample(
                tex_map_, pixel_uvs, align_corners=align_corners, padding_mode="border"
            )
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

    def test_faces_verts_textures(self):
        device = torch.device("cuda:0")
        N, V, F, H, W = 2, 5, 12, 8, 8
        vert_uvs = torch.rand((N, V, 2), dtype=torch.float32, device=device)
        face_uvs = torch.randint(
            high=V, size=(N, F, 3), dtype=torch.int64, device=device
        )
        maps = torch.rand((N, H, W, 3), dtype=torch.float32, device=device)

        tex = TexturesUV(maps=maps, verts_uvs=vert_uvs, faces_uvs=face_uvs)

        # naive faces_verts_textures
        faces_verts_texs = []
        for n in range(N):
            temp = torch.zeros((F, 3, 3), device=device, dtype=torch.float32)
            for f in range(F):
                uv0 = vert_uvs[n, face_uvs[n, f, 0]]
                uv1 = vert_uvs[n, face_uvs[n, f, 1]]
                uv2 = vert_uvs[n, face_uvs[n, f, 2]]

                idx = torch.stack((uv0, uv1, uv2), dim=0).view(1, 1, 3, 2)  # 1x1x3x2
                idx = idx * 2.0 - 1.0
                imap = maps[n].view(1, H, W, 3).permute(0, 3, 1, 2)  # 1x3xHxW
                imap = torch.flip(imap, [2])

                texts = torch.nn.functional.grid_sample(
                    imap,
                    idx,
                    align_corners=tex.align_corners,
                    padding_mode=tex.padding_mode,
                )  # 1x3x1x3
                temp[f] = texts[0, :, 0, :].permute(1, 0)
            faces_verts_texs.append(temp)
        faces_verts_texs = torch.cat(faces_verts_texs, 0)

        self.assertClose(faces_verts_texs, tex.faces_verts_textures_packed())

    def test_clone(self):
        tex = TexturesUV(
            maps=torch.ones((5, 16, 16, 3)),
            faces_uvs=torch.rand(size=(5, 10, 3)),
            verts_uvs=torch.rand(size=(5, 15, 2)),
        )
        tex.faces_uvs_list()
        tex.verts_uvs_list()
        tex_cloned = tex.clone()
        self.assertSeparate(tex._faces_uvs_padded, tex_cloned._faces_uvs_padded)
        self.assertClose(tex._faces_uvs_padded, tex_cloned._faces_uvs_padded)
        self.assertSeparate(tex._verts_uvs_padded, tex_cloned._verts_uvs_padded)
        self.assertClose(tex._verts_uvs_padded, tex_cloned._verts_uvs_padded)
        self.assertSeparate(tex._maps_padded, tex_cloned._maps_padded)
        self.assertClose(tex._maps_padded, tex_cloned._maps_padded)
        self.assertSeparate(tex.valid, tex_cloned.valid)
        self.assertTrue(tex.valid.eq(tex_cloned.valid).all())
        for i in range(tex._N):
            self.assertSeparate(tex._faces_uvs_list[i], tex_cloned._faces_uvs_list[i])
            self.assertClose(tex._faces_uvs_list[i], tex_cloned._faces_uvs_list[i])
            self.assertSeparate(tex._verts_uvs_list[i], tex_cloned._verts_uvs_list[i])
            self.assertClose(tex._verts_uvs_list[i], tex_cloned._verts_uvs_list[i])
            # tex._maps_list is not use anywhere so it's not stored. We call it explicitly
            self.assertSeparate(tex.maps_list()[i], tex_cloned.maps_list()[i])
            self.assertClose(tex.maps_list()[i], tex_cloned.maps_list()[i])

    def test_detach(self):
        tex = TexturesUV(
            maps=torch.ones((5, 16, 16, 3), requires_grad=True),
            faces_uvs=torch.rand(size=(5, 10, 3)),
            verts_uvs=torch.rand(size=(5, 15, 2)),
        )
        tex.faces_uvs_list()
        tex.verts_uvs_list()
        tex_detached = tex.detach()
        self.assertFalse(tex_detached._maps_padded.requires_grad)
        self.assertClose(tex._maps_padded, tex_detached._maps_padded)
        self.assertFalse(tex_detached._verts_uvs_padded.requires_grad)
        self.assertClose(tex._verts_uvs_padded, tex_detached._verts_uvs_padded)
        self.assertFalse(tex_detached._faces_uvs_padded.requires_grad)
        self.assertClose(tex._faces_uvs_padded, tex_detached._faces_uvs_padded)
        for i in range(tex._N):
            self.assertFalse(tex_detached._verts_uvs_list[i].requires_grad)
            self.assertClose(tex._verts_uvs_list[i], tex_detached._verts_uvs_list[i])
            self.assertFalse(tex_detached._faces_uvs_list[i].requires_grad)
            self.assertClose(tex._faces_uvs_list[i], tex_detached._faces_uvs_list[i])
            # tex._maps_list is not use anywhere so it's not stored. We call it explicitly
            self.assertFalse(tex_detached.maps_list()[i].requires_grad)
            self.assertClose(tex.maps_list()[i], tex_detached.maps_list()[i])

    def test_extend(self):
        B = 5
        mesh = init_mesh(B, 30, 50)
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

        new_tex_num_verts = new_mesh.num_verts_per_mesh()
        for i in range(len(tex_mesh)):
            for n in range(N):
                tex_nv = new_tex_num_verts[i * N + n]
                self.assertClose(
                    # The original textures were initialized using
                    # verts uvs list
                    tex_init.verts_uvs_list()[i],
                    # In the new textures, the verts_uvs are initialized
                    # from padded. The verts per mesh are not used to
                    # convert from padded to list. See TexturesUV for an
                    # explanation.
                    new_tex.verts_uvs_list()[i * N + n][:tex_nv, ...],
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
                tex_init.verts_uvs_padded(),
                new_tex.verts_uvs_padded(),
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
        verts_list = tex1.verts_uvs_list()
        verts_padded = tex1.verts_uvs_padded()

        faces_list = tex1.faces_uvs_list()
        faces_padded = tex1.faces_uvs_padded()

        for f1, f2 in zip(faces_list, faces_uvs_list):
            self.assertTrue((f1 == f2).all().item())

        for f1, f2 in zip(verts_list, verts_uvs_list):
            self.assertTrue((f1 == f2).all().item())

        self.assertTrue(faces_padded.shape == (2, 3, 3))
        self.assertTrue(verts_padded.shape == (2, 9, 2))

        # Case where num_faces_per_mesh is not set and faces_verts_uvs
        # are initialized with a padded tensor.
        tex2 = TexturesUV(
            maps=torch.ones((N, 16, 16, 3)),
            verts_uvs=verts_padded,
            faces_uvs=faces_padded,
        )
        faces_list = tex2.faces_uvs_list()
        verts_list = tex2.verts_uvs_list()

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
        self.assertEqual(tex._faces_uvs_padded.device, device)
        self.assertEqual(tex._verts_uvs_padded.device, device)
        self.assertEqual(tex._maps_padded.device, device)

    def test_mesh_to(self):
        tex_cpu = TexturesUV(
            maps=torch.ones((5, 16, 16, 3)),
            faces_uvs=torch.randint(size=(5, 10, 3), high=15),
            verts_uvs=torch.rand(size=(5, 15, 2)),
        )
        verts = torch.rand(size=(5, 15, 3))
        faces = torch.randint(size=(5, 10, 3), high=15)
        mesh_cpu = Meshes(faces=faces, verts=verts, textures=tex_cpu)
        cpu = torch.device("cpu")
        device = torch.device("cuda:0")
        tex = mesh_cpu.to(device).textures
        self.assertEqual(tex._faces_uvs_padded.device, device)
        self.assertEqual(tex._verts_uvs_padded.device, device)
        self.assertEqual(tex._maps_padded.device, device)
        self.assertEqual(tex_cpu._verts_uvs_padded.device, cpu)

        self.assertEqual(tex_cpu.device, cpu)
        self.assertEqual(tex.device, device)

    def test_getitem(self):
        N = 5
        V = 20
        F = 10
        source = {
            "maps": torch.rand(size=(N, 1, 1, 3)),
            "faces_uvs": torch.randint(size=(N, F, 3), high=V),
            "verts_uvs": torch.randn(size=(N, V, 2)),
        }
        tex = TexturesUV(
            maps=source["maps"],
            faces_uvs=source["faces_uvs"],
            verts_uvs=source["verts_uvs"],
        )

        verts = torch.rand(size=(N, V, 3))
        faces = torch.randint(size=(N, F, 3), high=V)
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

    def test_centers_for_image(self):
        maps = torch.rand(size=(1, 257, 129, 3))
        verts_uvs = torch.FloatTensor([[[0.25, 0.125], [0.5, 0.625], [0.5, 0.5]]])
        faces_uvs = torch.zeros(size=(1, 0, 3), dtype=torch.int64)
        tex = TexturesUV(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs)

        expected = torch.FloatTensor([[32, 224], [64, 96], [64, 128]])
        self.assertClose(tex.centers_for_image(0), expected)

    def test_sample_textures_error(self):
        N = 1
        V = 20
        F = 10
        maps = torch.rand(size=(N, 1, 1, 3))
        verts_uvs = torch.randn(size=(N, V, 2))
        tex = TexturesUV(
            maps=maps,
            faces_uvs=torch.randint(size=(N, 15, 3), high=V),
            verts_uvs=verts_uvs,
        )
        verts = torch.rand(size=(5, V, 3))
        faces = torch.randint(size=(5, 10, 3), high=V)
        meshes = Meshes(verts=verts, faces=faces)

        # Wrong number of faces
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            Meshes(verts=verts, faces=faces, textures=tex)

        # Wrong batch dim for faces
        tex = TexturesUV(
            maps=maps,
            faces_uvs=torch.randint(size=(1, F, 3), high=V),
            verts_uvs=verts_uvs,
        )
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            Meshes(verts=verts, faces=faces, textures=tex)

        # Wrong batch dim for verts_uvs is not necessary to check as
        # there is already a check inside TexturesUV for a batch dim
        # mismatch with faces_uvs

        meshes = Meshes(verts=verts, faces=faces)
        meshes.textures = tex

        # Cannot use the texture attribute set on meshes for sampling
        # textures if the dimensions don't match
        with self.assertRaisesRegex(ValueError, "do not match the dimensions"):
            meshes.sample_textures(None)

    def test_submeshes(self):
        N = 2
        faces_uvs_list = [
            torch.LongTensor([[0, 1, 2], [3, 5, 4], [7, 6, 8]]),
            torch.LongTensor([[0, 1, 2], [3, 4, 5]]),
        ]
        verts_uvs_list = [
            torch.arange(18, dtype=torch.float32).reshape(9, 2),
            torch.ones(6, 2),
        ]
        tex = TexturesUV(
            maps=torch.rand((N, 16, 16, 3)),
            faces_uvs=faces_uvs_list,
            verts_uvs=verts_uvs_list,
        )

        sub_faces = [
            [torch.tensor([0, 1]), torch.tensor([1, 2])],
            [],
        ]

        mesh = Meshes(
            verts=[torch.rand(9, 3), torch.rand(6, 3)],
            faces=faces_uvs_list,
            textures=tex,
        )
        subtex = mesh.submeshes(sub_faces).textures
        subtex_faces = subtex.faces_uvs_padded()
        self.assertEqual(len(subtex_faces), 2)
        self.assertClose(
            subtex_faces[0],
            torch.tensor([[0, 1, 2], [3, 5, 4]]),
        )
        self.assertClose(
            subtex.verts_uvs_list()[0][subtex.faces_uvs_list()[0].flatten()]
            .flatten()
            .msort(),
            torch.arange(12, dtype=torch.float32),
        )
        self.assertClose(
            subtex.maps_padded(), tex.maps_padded()[:1].expand(2, -1, -1, -1)
        )


class TestRectanglePacking(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    def wrap_pack(self, sizes):
        """
        Call the pack_rectangles function, which we want to test,
        and return its outputs.
        Additionally makes some sanity checks on the output.
        """
        res = pack_rectangles(sizes)
        total = res.total_size
        self.assertGreaterEqual(total[0], 0)
        self.assertGreaterEqual(total[1], 0)
        mask = torch.zeros(total, dtype=torch.bool)
        seen_x_bound = False
        seen_y_bound = False
        for (in_x, in_y), (out_x, out_y, flipped, is_first) in zip(
            sizes, res.locations
        ):
            self.assertTrue(is_first)
            self.assertGreaterEqual(out_x, 0)
            self.assertGreaterEqual(out_y, 0)
            placed_x, placed_y = (in_y, in_x) if flipped else (in_x, in_y)
            upper_x = placed_x + out_x
            upper_y = placed_y + out_y
            self.assertGreaterEqual(total[0], upper_x)
            if total[0] == upper_x:
                seen_x_bound = True
            self.assertGreaterEqual(total[1], upper_y)
            if total[1] == upper_y:
                seen_y_bound = True
            already_taken = torch.sum(mask[out_x:upper_x, out_y:upper_y])
            self.assertEqual(already_taken, 0)
            mask[out_x:upper_x, out_y:upper_y] = 1
        self.assertTrue(seen_x_bound)
        self.assertTrue(seen_y_bound)

        self.assertTrue(torch.all(torch.sum(mask, dim=0, dtype=torch.int32) > 0))
        self.assertTrue(torch.all(torch.sum(mask, dim=1, dtype=torch.int32) > 0))
        return res

    def assert_bb(self, sizes, expected):
        """
        Apply the pack_rectangles function to sizes and verify the
        bounding box dimensions are expected.
        """
        self.assertSetEqual(set(self.wrap_pack(sizes).total_size), expected)

    def test_simple(self):
        self.assert_bb([(3, 4), (4, 3)], {6, 4})
        self.assert_bb([(2, 2), (2, 4), (2, 2)], {4})

        # many squares
        self.assert_bb([(2, 2)] * 9, {2, 18})

        # One big square and many small ones.
        self.assert_bb([(3, 3)] + [(1, 1)] * 2, {3, 4})
        self.assert_bb([(3, 3)] + [(1, 1)] * 3, {3, 4})
        self.assert_bb([(3, 3)] + [(1, 1)] * 4, {3, 5})
        self.assert_bb([(3, 3)] + [(1, 1)] * 5, {3, 5})
        self.assert_bb([(1, 1)] * 6 + [(3, 3)], {3, 5})
        self.assert_bb([(3, 3)] + [(1, 1)] * 7, {3, 6})

        # many identical rectangles
        self.assert_bb([(7, 190)] * 4 + [(190, 7)] * 4, {190, 56})

        # require placing the flipped version of a rectangle
        self.assert_bb([(1, 100), (5, 96), (4, 5)], {100, 6})

    def test_random(self):
        for _ in range(5):
            vals = torch.randint(size=(20, 2), low=1, high=18)
            sizes = []
            for j in range(vals.shape[0]):
                sizes.append((int(vals[j, 0]), int(vals[j, 1])))
            self.wrap_pack(sizes)

    def test_all_identical(self):
        sizes = [Rectangle(xsize=61, ysize=82, identifier=1729)] * 3
        total_size, locations = pack_unique_rectangles(sizes)
        self.assertEqual(total_size, (61, 82))
        self.assertEqual(len(locations), 3)
        for i, (x, y, is_flipped, is_first) in enumerate(locations):
            self.assertEqual(x, 0)
            self.assertEqual(y, 0)
            self.assertFalse(is_flipped)
            self.assertEqual(is_first, i == 0)

    def test_one_different_id(self):
        sizes = [Rectangle(xsize=61, ysize=82, identifier=220)] * 3
        sizes.extend([Rectangle(xsize=61, ysize=82, identifier=284)] * 3)
        total_size, locations = pack_unique_rectangles(sizes)
        self.assertEqual(total_size, (82, 122))
        self.assertEqual(len(locations), 6)
        for i, (x, y, is_flipped, is_first) in enumerate(locations):
            self.assertTrue(is_flipped)
            self.assertEqual(is_first, i % 3 == 0)
            self.assertEqual(x, 0)
            if i < 3:
                self.assertEqual(y, 61)
            else:
                self.assertEqual(y, 0)
