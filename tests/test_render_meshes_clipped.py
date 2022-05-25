# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Checks for mesh rasterization in the case where the camera enters the
inside of the mesh and some mesh faces are partially
behind the image plane. These faces are clipped and then rasterized.
See pytorch3d/renderer/mesh/clip.py for more details about the
clipping process.
"""
import unittest

import imageio
import numpy as np
import torch
from pytorch3d.io import save_obj
from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PerspectiveCameras,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.mesh import (
    clip_faces,
    ClipFrustum,
    convert_clipped_rasterization_to_original_faces,
    TexturesUV,
)
from pytorch3d.renderer.mesh.rasterize_meshes import _RasterizeFaceVerts
from pytorch3d.renderer.mesh.rasterizer import MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.mesh.renderer import MeshRenderer
from pytorch3d.renderer.mesh.shader import SoftPhongShader
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.structures.meshes import Meshes
from pytorch3d.utils import torus

from .common_testing import get_tests_dir, load_rgb_image, TestCaseMixin


# If DEBUG=True, save out images generated in the tests for debugging.
# All saved images have prefix DEBUG_
DEBUG = False
DATA_DIR = get_tests_dir() / "data"


class TestRenderMeshesClipping(TestCaseMixin, unittest.TestCase):
    def load_cube_mesh_with_texture(self, device="cpu", with_grad: bool = False):
        verts = torch.tensor(
            [
                [-1, 1, 1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, -1],
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
            ],
            device=device,
            dtype=torch.float32,
            requires_grad=with_grad,
        )

        # all faces correctly wound
        faces = torch.tensor(
            [
                [0, 1, 4],
                [4, 1, 5],
                [1, 2, 5],
                [5, 2, 6],
                [2, 7, 6],
                [2, 3, 7],
                [3, 4, 7],
                [0, 4, 3],
                [4, 5, 6],
                [4, 6, 7],
            ],
            device=device,
            dtype=torch.int64,
        )

        verts_uvs = torch.tensor(
            [
                [
                    [0, 1],
                    [1, 1],
                    [1, 0],
                    [0, 0],
                    [0.204, 0.743],
                    [0.781, 0.743],
                    [0.781, 0.154],
                    [0.204, 0.154],
                ]
            ],
            device=device,
            dtype=torch.float,
        )
        texture_map = load_rgb_image("room.jpg", DATA_DIR).to(device)
        textures = TexturesUV(
            maps=[texture_map], faces_uvs=faces.unsqueeze(0), verts_uvs=verts_uvs
        )
        mesh = Meshes([verts], [faces], textures=textures)
        if with_grad:
            return mesh, verts
        return mesh

    def debug_cube_mesh_render(self):
        """
        End-End debug run of rendering a cube mesh with texture
        from decreasing camera distances. The camera starts
        outside the cube and enters the inside of the cube.
        """
        device = torch.device("cuda:0")
        mesh = self.load_cube_mesh_with_texture(device)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=1e-8,
            faces_per_pixel=5,
            z_clip_value=1e-2,
            perspective_correct=True,
            bin_size=0,
        )

        # Only ambient, no diffuse or specular
        lights = PointLights(
            device=device,
            ambient_color=((1.0, 1.0, 1.0),),
            diffuse_color=((0.0, 0.0, 0.0),),
            specular_color=((0.0, 0.0, 0.0),),
            location=[[0.0, 0.0, -3.0]],
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, lights=lights),
        )

        # Render the cube by decreasing the distance from the camera until
        # the camera enters the cube. Check the output looks correct.
        images_list = []
        dists = np.linspace(0.1, 2.5, 20)[::-1]

        for d in dists:
            R, T = look_at_view_transform(d, 0, 0)
            T[0, 1] -= 0.1  # move down in the y axis
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=90)
            images = renderer(mesh, cameras=cameras)
            rgb = images[0, ..., :3].cpu().detach()
            im = (rgb.numpy() * 255).astype(np.uint8)
            images_list.append(im)

        # Save a gif of the output - this should show
        # the camera moving inside the cube.
        if DEBUG:
            gif_filename = (
                "room_original.gif"
                if raster_settings.z_clip_value is None
                else "room_clipped.gif"
            )
            imageio.mimsave(DATA_DIR / gif_filename, images_list, fps=2)
            save_obj(
                f=DATA_DIR / "cube.obj",
                verts=mesh.verts_packed().cpu(),
                faces=mesh.faces_packed().cpu(),
            )

    @staticmethod
    def clip_faces(meshes):
        verts_packed = meshes.verts_packed()
        faces_packed = meshes.faces_packed()
        face_verts = verts_packed[faces_packed]
        mesh_to_face_first_idx = meshes.mesh_to_faces_packed_first_idx()
        num_faces_per_mesh = meshes.num_faces_per_mesh()

        frustum = ClipFrustum(
            left=-1,
            right=1,
            top=-1,
            bottom=1,
            # In the unit tests for each case below the triangles are asummed
            #  to have already been projected onto the image plane.
            perspective_correct=False,
            z_clip_value=1e-2,
            cull=True,  # Cull to frustrum
        )

        clipped_faces = clip_faces(
            face_verts, mesh_to_face_first_idx, num_faces_per_mesh, frustum
        )
        return clipped_faces

    def test_grad(self):
        """
        Check that gradient flow is unaffected when the camera is inside the mesh
        """
        device = torch.device("cuda:0")
        mesh, verts = self.load_cube_mesh_with_texture(device=device, with_grad=True)
        raster_settings = RasterizationSettings(
            image_size=512,
            blur_radius=1e-5,
            faces_per_pixel=5,
            z_clip_value=1e-2,
            perspective_correct=True,
            bin_size=0,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(device=device),
        )
        dist = 0.4  # Camera is inside the cube
        R, T = look_at_view_transform(dist, 0, 0)
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=90)
        images = renderer(mesh, cameras=cameras)
        images.sum().backward()

        # Check gradients exist
        self.assertIsNotNone(verts.grad)

    def test_case_1(self):
        """
        Case 1: Single triangle fully in front of the image plane (z=0)
        Triangle is not clipped or culled. The triangle is asummed to have
        already been projected onto the image plane so no perspective
        correction is needed.
        """
        device = "cuda:0"
        verts = torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        clipped_faces = self.clip_faces(meshes)

        self.assertClose(clipped_faces.face_verts, verts[faces])
        self.assertEqual(clipped_faces.mesh_to_face_first_idx.item(), 0)
        self.assertEqual(clipped_faces.num_faces_per_mesh.item(), 1)
        self.assertIsNone(clipped_faces.faces_clipped_to_unclipped_idx)
        self.assertIsNone(clipped_faces.faces_clipped_to_conversion_idx)
        self.assertIsNone(clipped_faces.clipped_faces_neighbor_idx)
        self.assertIsNone(clipped_faces.barycentric_conversion)

    def test_case_2(self):
        """
        Case 2 triangles are fully behind the image plane (z=0) so are completely culled.
        Test with a single triangle behind the image plane.
        """

        device = "cuda:0"
        verts = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        clipped_faces = self.clip_faces(meshes)

        zero_t = torch.zeros(size=(1,), dtype=torch.int64, device=device)
        self.assertClose(
            clipped_faces.face_verts, torch.empty(device=device, size=(0, 3, 3))
        )
        self.assertClose(clipped_faces.mesh_to_face_first_idx, zero_t)
        self.assertClose(clipped_faces.num_faces_per_mesh, zero_t)
        self.assertClose(
            clipped_faces.faces_clipped_to_unclipped_idx,
            torch.empty(device=device, dtype=torch.int64, size=(0,)),
        )
        self.assertIsNone(clipped_faces.faces_clipped_to_conversion_idx)
        self.assertIsNone(clipped_faces.clipped_faces_neighbor_idx)
        self.assertIsNone(clipped_faces.barycentric_conversion)

    def test_case_3(self):
        """
        Case 3 triangles have exactly two vertices behind the clipping plane (z=0) so are
        clipped into a smaller triangle.

        Test with a single triangle parallel to the z axis which intersects with
        the image plane.
        """

        device = "cuda:0"
        verts = torch.tensor(
            [[-1.0, 0.0, -1.0], [0.0, 0.0, 1.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        clipped_faces = self.clip_faces(meshes)

        zero_t = torch.zeros(size=(1,), dtype=torch.int64, device=device)
        clipped_face_verts = torch.tensor(
            [
                [
                    [0.4950, 0.0000, 0.0100],
                    [-0.4950, 0.0000, 0.0100],
                    [0.0000, 0.0000, 1.0000],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )

        # barycentric_conversion[i, :, k] stores the barycentric weights
        # in terms of the world coordinates of the original
        # (big) triangle for the kth vertex in the clipped (small) triangle.
        barycentric_conversion = torch.tensor(
            [
                [
                    [0.0000, 0.4950, 0.0000],
                    [0.5050, 0.5050, 1.0000],
                    [0.4950, 0.0000, 0.0000],
                ]
            ],
            device=device,
            dtype=torch.float32,
        )

        self.assertClose(clipped_faces.face_verts, clipped_face_verts)
        self.assertEqual(clipped_faces.mesh_to_face_first_idx.item(), 0)
        self.assertEqual(clipped_faces.num_faces_per_mesh.item(), 1)
        self.assertClose(clipped_faces.faces_clipped_to_unclipped_idx, zero_t)
        self.assertClose(clipped_faces.faces_clipped_to_conversion_idx, zero_t)
        self.assertClose(
            clipped_faces.clipped_faces_neighbor_idx,
            zero_t - 1,  # default is -1
        )
        self.assertClose(clipped_faces.barycentric_conversion, barycentric_conversion)

    def test_case_4(self):
        """
        Case 4 triangles have exactly 1 vertex behind the clipping plane (z=0) so
        are clipped into a smaller quadrilateral and then divided into two triangles.

        Test with a single triangle parallel to the z axis which intersects with
        the image plane.
        """

        device = "cuda:0"
        verts = torch.tensor(
            [[0.0, 0.0, -1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0, 1, 2],
            ],
            dtype=torch.int64,
            device=device,
        )
        meshes = Meshes(verts=[verts], faces=[faces])
        clipped_faces = self.clip_faces(meshes)

        clipped_face_verts = torch.tensor(
            [
                # t1
                [
                    [-0.5050, 0.0000, 0.0100],
                    [-1.0000, 0.0000, 1.0000],
                    [0.5050, 0.0000, 0.0100],
                ],
                # t2
                [
                    [0.5050, 0.0000, 0.0100],
                    [-1.0000, 0.0000, 1.0000],
                    [1.0000, 0.0000, 1.0000],
                ],
            ],
            device=device,
            dtype=torch.float32,
        )

        barycentric_conversion = torch.tensor(
            [
                [
                    [0.4950, 0.0000, 0.4950],
                    [0.5050, 1.0000, 0.0000],
                    [0.0000, 0.0000, 0.5050],
                ],
                [
                    [0.4950, 0.0000, 0.0000],
                    [0.0000, 1.0000, 0.0000],
                    [0.5050, 0.0000, 1.0000],
                ],
            ],
            device=device,
            dtype=torch.float32,
        )

        self.assertClose(clipped_faces.face_verts, clipped_face_verts)
        self.assertEqual(clipped_faces.mesh_to_face_first_idx.item(), 0)
        self.assertEqual(
            clipped_faces.num_faces_per_mesh.item(), 2
        )  # now two faces instead of 1
        self.assertClose(
            clipped_faces.faces_clipped_to_unclipped_idx,
            torch.tensor([0, 0], device=device, dtype=torch.int64),
        )
        # Neighboring face for each of the sub triangles e.g. for t1, neighbor is t2,
        # and for t2, neighbor is t1
        self.assertClose(
            clipped_faces.clipped_faces_neighbor_idx,
            torch.tensor([1, 0], device=device, dtype=torch.int64),
        )
        # barycentric_conversion is of shape (F_clipped)
        self.assertEqual(clipped_faces.barycentric_conversion.shape[0], 2)
        self.assertClose(clipped_faces.barycentric_conversion, barycentric_conversion)
        # Index into barycentric_conversion for each clipped face.
        self.assertClose(
            clipped_faces.faces_clipped_to_conversion_idx,
            torch.tensor([0, 1], device=device, dtype=torch.int64),
        )

    def test_mixture_of_cases(self):
        """
        Test with two meshes composed of different cases to check all the
        indexing is correct.
        Case 4 faces are subdivided into two faces which are referred
        to as t1 and t2.
        """
        device = "cuda:0"
        # fmt: off
        verts = [
            torch.tensor(
                [
                    [-1.0,  0.0, -1.0],  # noqa: E241, E201
                    [ 0.0,  1.0, -1.0],  # noqa: E241, E201
                    [ 1.0,  0.0, -1.0],  # noqa: E241, E201
                    [ 0.0, -1.0, -1.0],  # noqa: E241, E201
                    [-1.0,  0.5,  0.5],  # noqa: E241, E201
                    [ 1.0,  1.0,  1.0],  # noqa: E241, E201
                    [ 0.0, -1.0,  1.0],  # noqa: E241, E201
                    [-1.0,  0.5, -0.5],  # noqa: E241, E201
                    [ 1.0,  1.0, -1.0],  # noqa: E241, E201
                    [-1.0,  0.0,  1.0],  # noqa: E241, E201
                    [ 0.0,  1.0,  1.0],  # noqa: E241, E201
                    [ 1.0,  0.0,  1.0],  # noqa: E241, E201
                ],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [
                    [ 0.0, -1.0, -1.0],  # noqa: E241, E201
                    [-1.0,  0.5,  0.5],  # noqa: E241, E201
                    [ 1.0,  1.0,  1.0],  # noqa: E241, E201
                ],
                dtype=torch.float32,
                device=device
            )
        ]
        faces = [
            torch.tensor(
                [
                    [0,  1,  2],  # noqa: E241, E201  Case 2 fully clipped
                    [3,  4,  5],  # noqa: E241, E201  Case 4 clipped and subdivided
                    [5,  4,  3],  # noqa: E241, E201  Repeat of Case 4
                    [6,  7,  8],  # noqa: E241, E201  Case 3 clipped
                    [9, 10, 11],  # noqa: E241, E201  Case 1 untouched
                ],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor(
                [
                    [0,  1,  2],  # noqa: E241, E201  Case 4
                ],
                dtype=torch.int64,
                device=device,
            ),
        ]
        # fmt: on
        meshes = Meshes(verts=verts, faces=faces)

        # Clip meshes
        clipped_faces = self.clip_faces(meshes)

        # mesh 1: 4x faces (from Case 4) + 1 (from Case 3) + 1 (from Case 1)
        # mesh 2: 2x faces (from Case 4)
        self.assertEqual(clipped_faces.face_verts.shape[0], 6 + 2)

        # dummy idx type tensor to avoid having to initialize the dype/device each time
        idx = torch.empty(size=(1,), dtype=torch.int64, device=device)
        unclipped_idx = idx.new_tensor([1, 1, 2, 2, 3, 4, 5, 5])
        neighbors = idx.new_tensor([1, 0, 3, 2, -1, -1, 7, 6])
        first_idx = idx.new_tensor([0, 6])
        num_faces = idx.new_tensor([6, 2])

        self.assertClose(clipped_faces.clipped_faces_neighbor_idx, neighbors)
        self.assertClose(clipped_faces.faces_clipped_to_unclipped_idx, unclipped_idx)
        self.assertClose(clipped_faces.mesh_to_face_first_idx, first_idx)
        self.assertClose(clipped_faces.num_faces_per_mesh, num_faces)

        # faces_clipped_to_conversion_idx maps each output face to the
        # corresponding row of the barycentric_conversion matrix.
        # The barycentric_conversion matrix is composed by
        # finding the barycentric conversion weights for case 3 faces
        # case 4 (t1) faces and case 4 (t2) faces. These are then
        # concatenated. Therefore case 3 faces will be the first rows of
        # the barycentric_conversion matrix followed by t1 and then t2.
        # Case type of all faces: [4 (t1), 4 (t2), 4 (t1), 4 (t2), 3, 1, 4 (t1), 4 (t2)]
        # Based on this information we can calculate the indices into the
        # barycentric conversion matrix.
        bary_idx = idx.new_tensor([1, 4, 2, 5, 0, -1, 3, 6])
        self.assertClose(clipped_faces.faces_clipped_to_conversion_idx, bary_idx)

    def test_convert_clipped_to_unclipped_case_4(self):
        """
        Test with a single case 4 triangle which is clipped into
        a quadrilateral and subdivided.
        """
        device = "cuda:0"
        # fmt: off
        verts = torch.tensor(
            [
                [-1.0,  0.0, -1.0],  # noqa: E241, E201
                [ 0.0,  1.0, -1.0],  # noqa: E241, E201
                [ 1.0,  0.0, -1.0],  # noqa: E241, E201
                [ 0.0, -1.0, -1.0],  # noqa: E241, E201
                [-1.0,  0.5,  0.5],  # noqa: E241, E201
                [ 1.0,  1.0,  1.0],  # noqa: E241, E201
                [ 0.0, -1.0,  1.0],  # noqa: E241, E201
                [-1.0,  0.5, -0.5],  # noqa: E241, E201
                [ 1.0,  1.0, -1.0],  # noqa: E241, E201
                [-1.0,  0.0,  1.0],  # noqa: E241, E201
                [ 0.0,  1.0,  1.0],  # noqa: E241, E201
                [ 1.0,  0.0,  1.0],  # noqa: E241, E201
            ],
            dtype=torch.float32,
            device=device,
        )
        faces = torch.tensor(
            [
                [0,  1,  2],  # noqa: E241, E201  Case 2 fully clipped
                [3,  4,  5],  # noqa: E241, E201  Case 4 clipped and subdivided
                [5,  4,  3],  # noqa: E241, E201  Repeat of Case 4
                [6,  7,  8],  # noqa: E241, E201  Case 3 clipped
                [9, 10, 11],  # noqa: E241, E201  Case 1 untouched
            ],
            dtype=torch.int64,
            device=device,
        )
        # fmt: on
        meshes = Meshes(verts=[verts], faces=[faces])

        # Clip meshes
        clipped_faces = self.clip_faces(meshes)

        # 4x faces (from Case 4) + 1 (from Case 3) + 1 (from Case 1)
        self.assertEqual(clipped_faces.face_verts.shape[0], 6)

        image_size = (10, 10)
        blur_radius = 0.05
        faces_per_pixel = 2
        perspective_correct = True
        bin_size = 0
        max_faces_per_bin = 20
        clip_barycentric_coords = False
        cull_backfaces = False

        # Rasterize clipped mesh
        pix_to_face, zbuf, barycentric_coords, dists = _RasterizeFaceVerts.apply(
            clipped_faces.face_verts,
            clipped_faces.mesh_to_face_first_idx,
            clipped_faces.num_faces_per_mesh,
            clipped_faces.clipped_faces_neighbor_idx,
            image_size,
            blur_radius,
            faces_per_pixel,
            bin_size,
            max_faces_per_bin,
            perspective_correct,
            clip_barycentric_coords,
            cull_backfaces,
        )

        # Convert outputs so they are in terms of the unclipped mesh.
        outputs = convert_clipped_rasterization_to_original_faces(
            pix_to_face,
            barycentric_coords,
            clipped_faces,
        )
        pix_to_face_unclipped, barycentric_coords_unclipped = outputs

        # In the clipped mesh there are more faces than in the unclipped mesh
        self.assertTrue(pix_to_face.max() > pix_to_face_unclipped.max())
        # Unclipped pix_to_face indices must be in the limit of the number
        # of faces in the unclipped mesh.
        self.assertTrue(pix_to_face_unclipped.max() < faces.shape[0])

    def test_case_4_no_duplicates(self):
        """
        In the case of an simple mesh with one face that is cut by the image
        plane into a quadrilateral, there shouldn't be duplicates indices of
        the face in the pix_to_face output of rasterization.
        """
        for (device, bin_size) in [("cpu", 0), ("cuda:0", 0), ("cuda:0", None)]:
            verts = torch.tensor(
                [[0.0, -10.0, 1.0], [-1.0, 2.0, -2.0], [1.0, 5.0, -10.0]],
                dtype=torch.float32,
                device=device,
            )
            faces = torch.tensor(
                [
                    [0, 1, 2],
                ],
                dtype=torch.int64,
                device=device,
            )
            meshes = Meshes(verts=[verts], faces=[faces])
            k = 3
            settings = RasterizationSettings(
                image_size=10,
                blur_radius=0.05,
                faces_per_pixel=k,
                z_clip_value=1e-2,
                perspective_correct=True,
                cull_to_frustum=True,
                bin_size=bin_size,
            )

            # The camera is positioned so that the image plane cuts
            # the mesh face into a quadrilateral.
            R, T = look_at_view_transform(0.2, 0, 0)
            cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=90)
            rasterizer = MeshRasterizer(raster_settings=settings, cameras=cameras)
            fragments = rasterizer(meshes)

            p2f = fragments.pix_to_face.reshape(-1, k)
            unique_vals, idx_counts = p2f.unique(dim=0, return_counts=True)
            # There is only one face in this mesh so if it hits a pixel
            # it can only be at position k = 0
            # For any pixel, the values [0, 0, 1] for the top K faces cannot be possible
            double_hit = torch.tensor([0, 0, -1], device=device)
            check_double_hit = any(torch.allclose(i, double_hit) for i in unique_vals)
            self.assertFalse(check_double_hit)

    def test_mesh_outside_frustrum(self):
        """
        Test cases:
        1. Where the mesh is completely outside the view
        frustrum so all faces are culled and z_clip_value = None.
        2. Where the part of the mesh is in the view frustrum but
        the z_clip value = 5.0 so all the visible faces are behind
        the clip plane so are culled instead of clipped.
        """
        device = "cuda:0"
        mesh1 = torus(20.0, 85.0, 32, 16, device=device)
        mesh2 = torus(2.0, 3.0, 32, 16, device=device)
        for (mesh, z_clip) in [(mesh1, None), (mesh2, 5.0)]:
            tex = TexturesVertex(verts_features=torch.rand_like(mesh.verts_padded()))
            mesh.textures = tex
            raster_settings = RasterizationSettings(
                image_size=512, cull_to_frustum=True, z_clip_value=z_clip
            )
            R, T = look_at_view_transform(3.0, 0.0, 0.0)
            cameras = PerspectiveCameras(device=device, R=R, T=T)
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, raster_settings=raster_settings
                ),
                shader=SoftPhongShader(cameras=cameras, device=device),
            )
            images = renderer(mesh)
            # The image should be white.
            self.assertClose(images[0, ..., :3], torch.ones_like(images[0, ..., :3]))
