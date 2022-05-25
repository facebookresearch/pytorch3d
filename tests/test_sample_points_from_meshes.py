# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.mesh.rasterize_meshes import barycentric_coordinates
from pytorch3d.renderer.points import (
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils.ico_sphere import ico_sphere

from .common_testing import (
    get_pytorch3d_dir,
    get_random_cuda_device,
    get_tests_dir,
    TestCaseMixin,
)


# If DEBUG=True, save out images generated in the tests for debugging.
# All saved images have prefix DEBUG_
DEBUG = False
DATA_DIR = get_tests_dir() / "data"


def init_meshes(
    num_meshes: int = 10,
    num_verts: int = 1000,
    num_faces: int = 3000,
    device: str = "cpu",
    add_texture: bool = False,
):
    device = torch.device(device)
    verts_list = []
    faces_list = []
    texts_list = []
    for _ in range(num_meshes):
        verts = torch.rand((num_verts, 3), dtype=torch.float32, device=device)
        faces = torch.randint(
            num_verts, size=(num_faces, 3), dtype=torch.int64, device=device
        )
        texts = torch.rand((num_verts, 3), dtype=torch.float32, device=device)
        verts_list.append(verts)
        faces_list.append(faces)
        texts_list.append(texts)

    # create textures
    textures = None
    if add_texture:
        textures = TexturesVertex(texts_list)
    meshes = Meshes(verts=verts_list, faces=faces_list, textures=textures)

    return meshes


class TestSamplePoints(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_all_empty_meshes(self):
        """
        Check sample_points_from_meshes raises an exception if all meshes are
        invalid.
        """
        device = get_random_cuda_device()
        verts1 = torch.tensor([], dtype=torch.float32, device=device)
        faces1 = torch.tensor([], dtype=torch.int64, device=device)
        meshes = Meshes(verts=[verts1, verts1, verts1], faces=[faces1, faces1, faces1])
        with self.assertRaises(ValueError) as err:
            sample_points_from_meshes(meshes, num_samples=100, return_normals=True)
        self.assertTrue("Meshes are empty." in str(err.exception))

    def test_sampling_output(self):
        """
        Check outputs of sampling are correct for different meshes.
        For an ico_sphere, the sampled vertices should lie on a unit sphere.
        For an empty mesh, the samples and normals should be 0.
        """
        device = get_random_cuda_device()

        # Unit simplex.
        verts_pyramid = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        faces_pyramid = torch.tensor(
            [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]],
            dtype=torch.int64,
            device=device,
        )
        sphere_mesh = ico_sphere(9, device)
        verts_sphere, faces_sphere = sphere_mesh.get_mesh_verts_faces(0)
        verts_empty = torch.tensor([], dtype=torch.float32, device=device)
        faces_empty = torch.tensor([], dtype=torch.int64, device=device)
        num_samples = 10
        meshes = Meshes(
            verts=[verts_empty, verts_sphere, verts_pyramid],
            faces=[faces_empty, faces_sphere, faces_pyramid],
        )
        samples, normals = sample_points_from_meshes(
            meshes, num_samples=num_samples, return_normals=True
        )
        samples = samples.cpu()
        normals = normals.cpu()

        self.assertEqual(samples.shape, (3, num_samples, 3))
        self.assertEqual(normals.shape, (3, num_samples, 3))

        # Empty meshes: should have all zeros for samples and normals.
        self.assertClose(samples[0, :], torch.zeros((num_samples, 3)))
        self.assertClose(normals[0, :], torch.zeros((num_samples, 3)))

        # Sphere: points should have radius 1.
        x, y, z = samples[1, :].unbind(1)
        radius = torch.sqrt(x**2 + y**2 + z**2)

        self.assertClose(radius, torch.ones(num_samples))

        # Pyramid: points shoudl lie on one of the faces.
        pyramid_verts = samples[2, :]
        pyramid_normals = normals[2, :]

        self.assertClose(pyramid_verts.lt(1).float(), torch.ones_like(pyramid_verts))
        self.assertClose((pyramid_verts >= 0).float(), torch.ones_like(pyramid_verts))

        # Face 1: z = 0,  x + y <= 1, normals = (0, 0, 1).
        face_1_idxs = pyramid_verts[:, 2] == 0
        face_1_verts, face_1_normals = (
            pyramid_verts[face_1_idxs, :],
            pyramid_normals[face_1_idxs, :],
        )
        self.assertTrue(torch.all((face_1_verts[:, 0] + face_1_verts[:, 1]) <= 1))
        self.assertClose(
            face_1_normals,
            torch.tensor([0, 0, 1], dtype=torch.float32).expand(face_1_normals.size()),
        )

        # Face 2: x = 0,  z + y <= 1, normals = (1, 0, 0).
        face_2_idxs = pyramid_verts[:, 0] == 0
        face_2_verts, face_2_normals = (
            pyramid_verts[face_2_idxs, :],
            pyramid_normals[face_2_idxs, :],
        )
        self.assertTrue(torch.all((face_2_verts[:, 1] + face_2_verts[:, 2]) <= 1))
        self.assertClose(
            face_2_normals,
            torch.tensor([1, 0, 0], dtype=torch.float32).expand(face_2_normals.size()),
        )

        # Face 3: y = 0, x + z <= 1, normals = (0, -1, 0).
        face_3_idxs = pyramid_verts[:, 1] == 0
        face_3_verts, face_3_normals = (
            pyramid_verts[face_3_idxs, :],
            pyramid_normals[face_3_idxs, :],
        )
        self.assertTrue(torch.all((face_3_verts[:, 0] + face_3_verts[:, 2]) <= 1))
        self.assertClose(
            face_3_normals,
            torch.tensor([0, -1, 0], dtype=torch.float32).expand(face_3_normals.size()),
        )

        # Face 4: x + y + z = 1, normals = (1, 1, 1)/sqrt(3).
        face_4_idxs = pyramid_verts.gt(0).all(1)
        face_4_verts, face_4_normals = (
            pyramid_verts[face_4_idxs, :],
            pyramid_normals[face_4_idxs, :],
        )
        self.assertClose(face_4_verts.sum(1), torch.ones(face_4_verts.size(0)))
        self.assertClose(
            face_4_normals,
            (
                torch.tensor([1, 1, 1], dtype=torch.float32)
                / torch.sqrt(torch.tensor(3, dtype=torch.float32))
            ).expand(face_4_normals.size()),
        )

    def test_multinomial(self):
        """
        Confirm that torch.multinomial does not sample elements which have
        zero probability.
        """
        freqs = torch.cuda.FloatTensor(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.03178183361887932,
                0.027680952101945877,
                0.033176131546497345,
                0.046052902936935425,
                0.07742464542388916,
                0.11543981730937958,
                0.14148041605949402,
                0.15784293413162231,
                0.13180233538150787,
                0.08271478116512299,
                0.049702685326337814,
                0.027557924389839172,
                0.018125897273421288,
                0.011851548217236996,
                0.010252203792333603,
                0.007422595750540495,
                0.005372154992073774,
                0.0045109698548913,
                0.0036087757907807827,
                0.0035267581697553396,
                0.0018864056328311563,
                0.0024605290964245796,
                0.0022964938543736935,
                0.0018453967059031129,
                0.0010662291897460818,
                0.0009842115687206388,
                0.00045109697384759784,
                0.0007791675161570311,
                0.00020504408166743815,
                0.00020504408166743815,
                0.00020504408166743815,
                0.00012302644609007984,
                0.0,
                0.00012302644609007984,
                4.100881778867915e-05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        sample = []
        for _ in range(1000):
            torch.cuda.get_rng_state()
            sample = torch.multinomial(freqs, 1000, True)
            if freqs[sample].min() == 0:
                sample_idx = (freqs[sample] == 0).nonzero()[0][0]
                sampled = sample[sample_idx]
                print(
                    "%s th element of last sample was %s, which has probability %s"
                    % (sample_idx, sampled, freqs[sampled])
                )
                return False
        return True

    def test_multinomial_weights(self):
        """
        Confirm that torch.multinomial does not sample elements which have
        zero probability using a real example of input from a training run.
        """
        weights = torch.load(get_tests_dir() / "weights.pt")
        S = 4096
        num_trials = 100
        for _ in range(0, num_trials):
            weights[weights < 0] = 0.0
            samples = weights.multinomial(S, replacement=True)
            sampled_weights = weights[samples]
            assert sampled_weights.min() > 0
            if sampled_weights.min() <= 0:
                return False
        return True

    def test_verts_nan(self):
        num_verts = 30
        num_faces = 50
        for device in ["cpu", "cuda:0"]:
            for invalid in ["nan", "inf"]:
                verts = torch.rand((num_verts, 3), dtype=torch.float32, device=device)
                # randomly assign an invalid type
                verts[torch.randperm(num_verts)[:10]] = float(invalid)
                faces = torch.randint(
                    num_verts, size=(num_faces, 3), dtype=torch.int64, device=device
                )
                meshes = Meshes(verts=[verts], faces=[faces])

                with self.assertRaisesRegex(ValueError, "Meshes contain nan or inf."):
                    sample_points_from_meshes(
                        meshes, num_samples=100, return_normals=True
                    )

    def test_outputs(self):

        for add_texture in (True, False):
            meshes = init_meshes(device=torch.device("cuda:0"), add_texture=add_texture)
            out1 = sample_points_from_meshes(meshes, num_samples=100)
            self.assertTrue(torch.is_tensor(out1))

            out2 = sample_points_from_meshes(
                meshes, num_samples=100, return_normals=True
            )
            self.assertTrue(isinstance(out2, tuple) and len(out2) == 2)

            if add_texture:
                out3 = sample_points_from_meshes(
                    meshes, num_samples=100, return_textures=True
                )
                self.assertTrue(isinstance(out3, tuple) and len(out3) == 2)

                out4 = sample_points_from_meshes(
                    meshes, num_samples=100, return_normals=True, return_textures=True
                )
                self.assertTrue(isinstance(out4, tuple) and len(out4) == 3)
            else:
                with self.assertRaisesRegex(
                    ValueError, "Meshes do not contain textures."
                ):
                    sample_points_from_meshes(
                        meshes, num_samples=100, return_textures=True
                    )

                with self.assertRaisesRegex(
                    ValueError, "Meshes do not contain textures."
                ):
                    sample_points_from_meshes(
                        meshes,
                        num_samples=100,
                        return_normals=True,
                        return_textures=True,
                    )

    def test_texture_sampling(self):
        device = torch.device("cuda:0")
        batch_size = 6
        # verts
        verts = torch.rand((batch_size, 6, 3), device=device, dtype=torch.float32)
        verts[:, :3, 2] = 1.0
        verts[:, 3:, 2] = -1.0
        # textures
        texts = torch.rand((batch_size, 6, 3), device=device, dtype=torch.float32)
        # faces
        faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device, dtype=torch.int64)
        faces = faces.view(1, 2, 3).expand(batch_size, -1, -1)

        meshes = Meshes(verts=verts, faces=faces, textures=TexturesVertex(texts))

        num_samples = 24
        samples, normals, textures = sample_points_from_meshes(
            meshes, num_samples=num_samples, return_normals=True, return_textures=True
        )

        textures_naive = torch.zeros(
            (batch_size, num_samples, 3), dtype=torch.float32, device=device
        )
        for n in range(batch_size):
            for i in range(num_samples):
                p = samples[n, i]
                if p[2] > 0.0:  # sampled from 1st face
                    v0, v1, v2 = verts[n, 0, :2], verts[n, 1, :2], verts[n, 2, :2]
                    w0, w1, w2 = barycentric_coordinates(p[:2], v0, v1, v2)
                    t0, t1, t2 = texts[n, 0], texts[n, 1], texts[n, 2]
                else:  # sampled from 2nd face
                    v0, v1, v2 = verts[n, 3, :2], verts[n, 4, :2], verts[n, 5, :2]
                    w0, w1, w2 = barycentric_coordinates(p[:2], v0, v1, v2)
                    t0, t1, t2 = texts[n, 3], texts[n, 4], texts[n, 5]

                tt = w0 * t0 + w1 * t1 + w2 * t2
                textures_naive[n, i] = tt

        self.assertClose(textures, textures_naive)

    def test_texture_sampling_cow(self):
        # test texture sampling for the cow example by converting
        # the cow mesh and its texture uv to a pointcloud with texture

        device = torch.device("cuda:0")
        obj_dir = get_pytorch3d_dir() / "docs/tutorials/data"
        obj_filename = obj_dir / "cow_mesh/cow.obj"

        for text_type in ("uv", "atlas"):
            # Load mesh + texture
            if text_type == "uv":
                mesh = load_objs_as_meshes(
                    [obj_filename], device=device, load_textures=True, texture_wrap=None
                )
            elif text_type == "atlas":
                mesh = load_objs_as_meshes(
                    [obj_filename],
                    device=device,
                    load_textures=True,
                    create_texture_atlas=True,
                    texture_atlas_size=8,
                    texture_wrap=None,
                )

            points, normals, textures = sample_points_from_meshes(
                mesh, num_samples=50000, return_normals=True, return_textures=True
            )
            pointclouds = Pointclouds(points, normals=normals, features=textures)

            for pos in ("front", "back"):
                # Init rasterizer settings
                if pos == "back":
                    azim = 0.0
                elif pos == "front":
                    azim = 180
                R, T = look_at_view_transform(2.7, 0, azim)
                cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

                raster_settings = PointsRasterizationSettings(
                    image_size=512, radius=1e-2, points_per_pixel=1
                )

                rasterizer = PointsRasterizer(
                    cameras=cameras, raster_settings=raster_settings
                )
                compositor = NormWeightedCompositor()
                renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
                images = renderer(pointclouds)

                rgb = images[0, ..., :3].squeeze().cpu()
                if DEBUG:
                    filename = "DEBUG_cow_mesh_to_pointcloud_%s_%s.png" % (
                        text_type,
                        pos,
                    )
                    Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                        DATA_DIR / filename
                    )

    @staticmethod
    def sample_points_with_init(
        num_meshes: int,
        num_verts: int,
        num_faces: int,
        num_samples: int,
        device: str = "cpu",
    ):
        verts_list = []
        faces_list = []
        for _ in range(num_meshes):
            verts = torch.rand((num_verts, 3), dtype=torch.float32, device=device)
            faces = torch.randint(
                num_verts, size=(num_faces, 3), dtype=torch.int64, device=device
            )
            verts_list.append(verts)
            faces_list.append(faces)
        meshes = Meshes(verts_list, faces_list)
        torch.cuda.synchronize()

        def sample_points():
            sample_points_from_meshes(
                meshes, num_samples=num_samples, return_normals=True
            )
            torch.cuda.synchronize()

        return sample_points
