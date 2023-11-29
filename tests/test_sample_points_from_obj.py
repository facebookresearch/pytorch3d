# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.io.obj_io import _Faces, _Aux
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import sample_points_from_obj
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, look_at_view_transform
from pytorch3d.renderer.points import (
    NormWeightedCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
)
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils.ico_sphere import ico_sphere
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals

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


class TestSamplePoints(TestCaseMixin, unittest.TestCase):
    """This test class mirrors core tests from test_sample_points_from meshes but focus on 
    new features in sample_points_from_obj since it leverages the same functions.
    """
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_all_empty_obj(self):
        """
        Check sample_points_from_obj raises an exception if the input OBJ is
        invalid.
        """
        device = get_random_cuda_device()
        verts1 = torch.tensor([], dtype=torch.float32, device=device)
        faces1 = torch.tensor([], dtype=torch.int64, device=device)
    
        aux = _Aux(normals=None, verts_uvs=None, material_colors=None, texture_images=None, texture_atlas=None)
        faces = _Faces(verts_idx=faces1, normals_idx=None, textures_idx=None, materials_idx=None)
        
        obj = (verts1, faces, aux)
        # checking test condition against obj sampler
        with self.assertRaises(ValueError) as err:
            sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx
            )

        self.assertTrue("OBJ is empty." in str(err.exception))

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

                aux = _Aux(normals=None, verts_uvs=None, material_colors=None, texture_images=None, texture_atlas=None)
                faces = _Faces(verts_idx=faces, normals_idx=None, textures_idx=None, materials_idx=None)

                obj = (verts, faces, aux)

                with self.assertRaisesRegex(ValueError, "Verts contain nan or inf."):

                    sample_points_from_obj(
                        verts=obj[0],
                        faces=obj[1].verts_idx,
                        verts_uvs=obj[2].verts_uvs,
                        faces_uvs=obj[1].textures_idx,
                        texture_images=obj[2].texture_images,
                        materials_idx=obj[1].materials_idx,
                        texture_atlas=obj[2].texture_atlas,
                        num_samples=100,
                        sample_all_faces=False,
                        return_mappers=False, 
                        return_textures=False, 
                        return_normals=True
                    )

    def test_relative_sampling_output(self):
        """
        Check outputs of sampling are correct for objs.
        Relative to the baseline from sample_points_from_meshes, 
        sample_points_from_obj should produce similar point clouds.
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

        num_samples = 10
        chamfer_loss_threshold = 1
        
        meshes_sphere = Meshes(
            verts=[verts_sphere],
            faces=[faces_sphere],
        )

        meshes_pyramid = Meshes(
            verts=[verts_pyramid],
            faces=[faces_pyramid],
        )

        aux = _Aux(normals=None, verts_uvs=None, material_colors=None, texture_images=None, texture_atlas=None)
        
        faces = _Faces(verts_idx=faces_sphere, normals_idx=None, textures_idx=None, materials_idx=None)
        verts = verts_sphere
        obj_sphere = (verts, faces, aux)

        faces = _Faces(verts_idx=faces_pyramid, normals_idx=None, textures_idx=None, materials_idx=None)
        verts = verts_pyramid
        obj_pyramid = (verts, faces, aux)

        data = dict(
            sphere=[meshes_sphere, obj_sphere],
            pyramid=[meshes_pyramid, obj_pyramid]
        )
        
        for name, meshes in data.items():
            
            mesh = meshes[0]
            obj = meshes[1]
            # sample points from sample_points_from_meshes and compare chamfer loss to sample_points_from_obj
            base_sample = sample_points_from_meshes(
                mesh, num_samples=num_samples, return_normals=False
            )

            samples2, normals2, textures2, _ = sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx,
                verts_uvs=obj[2].verts_uvs,
                faces_uvs=obj[1].textures_idx,
                texture_images=obj[2].texture_images,
                materials_idx=obj[1].materials_idx,
                texture_atlas=obj[2].texture_atlas,
                num_samples=num_samples,
                sample_all_faces=False,
                return_mappers=False, 
                return_textures=True, # expect to return None since no textures are provided
                return_normals=True
            )

            with torch.no_grad():
                chamfer_loss, _ = chamfer_distance(base_sample, samples2)
                self.assertTrue(chamfer_loss.item() <= chamfer_loss_threshold)
            # nubmer of points in sample2 should be the same as in the baseline 
            self.assertTrue(base_sample.squeeze().shape[0] == samples2.squeeze().shape[0])
            # textures should return None if input has no textures, regardless of return_textures == True
            self.assertTrue(None in [textures2])

            samples2 = samples2.cpu()
            normals2 = normals2.cpu()
 
            # apply same tests as in test_sample_points_from_meshes
            if name == 'sphere':
                # Sphere: points should have radius 1.
                x, y, z = samples2[0, :].unbind(1)
                radius = torch.sqrt(x**2 + y**2 + z**2)

                self.assertClose(radius, torch.ones(num_samples))
            
            if name == 'pyramid':
                # Pyramid: points shoudl lie on one of the faces.
                pyramid_verts = samples2[0, :]
                pyramid_normals = normals2[0, :]

                self.assertClose(pyramid_verts.lt(1).float(), torch.ones_like(pyramid_verts))
                self.assertClose((pyramid_verts >= 0).float(), torch.ones_like(pyramid_verts))

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
            
            del samples2, normals2

            samples3, _, _, _ = sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx,
                verts_uvs=obj[2].verts_uvs,
                faces_uvs=obj[1].textures_idx,
                texture_images=obj[2].texture_images,
                materials_idx=obj[1].materials_idx,
                texture_atlas=obj[2].texture_atlas,
                num_samples=num_samples,
                sample_all_faces=True, # forces at least one sample per face
                return_mappers=False, 
                return_textures=False, 
                return_normals=False
            )

            with torch.no_grad():
                chamfer_loss, _ = chamfer_distance(base_sample, samples3)
                self.assertTrue(chamfer_loss.item() <= chamfer_loss_threshold)
            # number of points in sample 3 should be greater than or equal to number of input faces
            self.assertTrue(samples3.squeeze().shape[0] >= obj[1].verts_idx.shape[0])

            del samples3

            samples4, _, _, mappers4 = sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx,
                verts_uvs=obj[2].verts_uvs,
                faces_uvs=obj[1].textures_idx,
                texture_images=obj[2].texture_images,
                materials_idx=obj[1].materials_idx,
                texture_atlas=obj[2].texture_atlas,
                num_samples=None, # auto sampling enabled
                sample_all_faces=True,
                return_mappers=True, 
                return_textures=False, 
                return_normals=False
            )

            with torch.no_grad():
                chamfer_loss, _ = chamfer_distance(base_sample, samples4)
                self.assertTrue(chamfer_loss.item() <= chamfer_loss_threshold)
            
            samples4 = samples4.cpu()
            mappers4 = mappers4.cpu()

            # number of points in sample 4 should be greater than or equal to number of input faces
            self.assertTrue(samples4.squeeze().shape[0] >= obj[1].verts_idx.shape[0])
            # largest mapper value plus one should be the size of the input faces
            self.assertTrue(mappers4.max() + 1 == obj[1].verts_idx.shape[0])
            # mapper should have the same first two dims as the sampled points
            self.assertTrue(samples4.shape[0] == mappers4.shape[0])
            self.assertTrue(samples4.shape[1] == mappers4.shape[1])
            
            # check that the mappers links back to points that belong to each origin face
            # randomly select and test indices for 1% of sampled points
            mappers_idxs = torch.randint(mappers4.shape[1], (int(mappers4.shape[1] * .01), ))
            # get face areas to determine relative tolerances per face
            areas, _ = mesh_face_areas_normals(obj[0], obj[1].verts_idx)
            # establish tolerance for distance from face centroid to point by face area
            areas_max = areas.max()
            tolerance = areas_max  * 2
            target_loss = torch.tensor(0.0)

            for mapper_idx in mappers_idxs:
                # mapper values index into mappers and sampled points
                curr_face_idx = mappers4[:, mapper_idx]
                # the value is the face index
                curr_face = obj[1].verts_idx[curr_face_idx]  
                # index into verts with faces
                curr_verts = obj[0][curr_face]
                # use face verts mean as an approximate centroid of the face
                curr_centroid = curr_verts.mean(dim=1)[None]
                # use the mapper to index into the sampled point
                curr_sample = samples4[:, mapper_idx][None]
                
                with torch.no_grad():
                    # check that the sampled point is close to the centroid of the face
                    chamfer_loss, _ = chamfer_distance(curr_centroid.cpu(), curr_sample)
                    result = torch.allclose(chamfer_loss.cpu(), target_loss, rtol=tolerance, atol=1)
                    self.assertTrue(result)
            
            del samples4, mappers4

    def test_texture_sampling_cow(self):
        # test texture sampling for the cow example by converting
        # the cow mesh and its texture uv to a pointcloud with texture

        device = torch.device("cuda:0")
        obj_dir = get_pytorch3d_dir() / "docs/tutorials/data"
        obj_filename = obj_dir / "cow_mesh/cow.obj"
        num_samples = None # force auto sampling
        min_sampling_factor = 10000 # sample at least n times the surface area of each face
        sample_all_faces = False # sure at least one point per face, regardless of area
        expected_min_point_num = 50000

        for text_type in ("uv", "atlas"):
            # Load mesh + texture
            if text_type == "uv":
                obj = load_obj(
                    f=obj_filename,
                    load_textures=True,
                    texture_wrap=None,
                    device=device
                )

                points, normals, textures, _ = sample_points_from_obj(
                    verts=obj[0],
                    faces=obj[1].verts_idx,
                    verts_uvs=obj[2].verts_uvs,
                    faces_uvs=obj[1].textures_idx,
                    texture_images=obj[2].texture_images,
                    materials_idx=obj[1].materials_idx,
                    texture_atlas=obj[2].texture_atlas,
                    num_samples=num_samples,
                    sample_all_faces=sample_all_faces,
                    min_sampling_factor=min_sampling_factor,
                    return_mappers=False, 
                    return_textures=True, 
                    return_normals=True
                )

            elif text_type == "atlas":
                obj = load_obj(
                    f=obj_filename,
                    load_textures=True,
                    texture_wrap=None,
                    texture_atlas_size=8,
                    create_texture_atlas=True,
                    device=device
                )
                 
                points, normals, textures, _ = sample_points_from_obj(
                    verts=obj[0],
                    faces=obj[1].verts_idx,
                    verts_uvs=obj[2].verts_uvs,
                    faces_uvs=obj[1].textures_idx,
                    texture_images=obj[2].texture_images,
                    materials_idx=obj[1].materials_idx,
                    texture_atlas=obj[2].texture_atlas,
                    num_samples=num_samples,
                    sample_all_faces=sample_all_faces,
                    min_sampling_factor=min_sampling_factor,
                    return_mappers=False, 
                    return_textures=True, 
                    return_normals=True,
                    use_texture_atlas=True
                )
            
            self.assertTrue(points.shape[1] > expected_min_point_num)

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
                    filename = "DEBUG_cow_obj_to_pointcloud_%s_%s.png" % (
                        text_type,
                        pos,
                    )
                    Image.fromarray((rgb.numpy() * 255).astype(np.uint8)).save(
                        DATA_DIR / filename
                    )

            del points, normals, textures

        # check bad sampling factor input
        with self.assertRaises(ValueError) as err:
            sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx,
                verts_uvs=obj[2].verts_uvs,
                faces_uvs=obj[1].textures_idx,
                texture_images=obj[2].texture_images,
                materials_idx=obj[1].materials_idx,
                texture_atlas=obj[2].texture_atlas,
                num_samples=10,
                sample_all_faces=False,
                return_mappers=False, 
                return_textures=False, 
                return_normals=False,
                sampling_factors=torch.tensor([100, 100]).to(device)
            )
            message = "sampling_sizes.shape[0] != len(meshes); check sampling_factors"
            self.assertTrue(message in str(err.exception))

            # check params conflict, if num samples given plus other params for size, num_samples takes precedence
            num_samples = 10
            points, _, _, _ = sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx,
                verts_uvs=obj[2].verts_uvs,
                faces_uvs=obj[1].textures_idx,
                texture_images=obj[2].texture_images,
                materials_idx=obj[1].materials_idx,
                texture_atlas=obj[2].texture_atlas,
                num_samples=num_samples,
                sample_all_faces=False,
                return_mappers=False, 
                return_textures=False, 
                return_normals=False,
                sampling_factors=torch.tensor([100]).to(device)
            )

            self.assertTrue(points.shape[1] == num_samples)
        
            # check output size, given array of sampling factors
            num_samples = None
            points, _, _, _= sample_points_from_obj(
                verts=obj[0],
                faces=obj[1].verts_idx,
                verts_uvs=obj[2].verts_uvs,
                faces_uvs=obj[1].textures_idx,
                texture_images=obj[2].texture_images,
                materials_idx=obj[1].materials_idx,
                texture_atlas=obj[2].texture_atlas,
                num_samples=num_samples,
                sample_all_faces=False,
                return_mappers=False, 
                return_textures=False, 
                return_normals=False,
                sampling_factors=torch.tensor([1000]).to(device)
            )

            self.assertTrue(points.shape[1] > 5000)