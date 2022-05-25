# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sanity checks for loading R2N2.
"""
import json
import os
import unittest

import numpy as np
import torch
from PIL import Image
from pytorch3d.datasets import (
    BlenderCamera,
    collate_batched_R2N2,
    R2N2,
    render_cubified_voxels,
)
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
)
from pytorch3d.renderer.cameras import get_world_to_view_transform
from pytorch3d.transforms import Transform3d
from pytorch3d.transforms.so3 import so3_exp_map
from torch.utils.data import DataLoader

from .common_testing import get_tests_dir, load_rgb_image, TestCaseMixin


# Set these paths in order to run the tests.
R2N2_PATH = None
SHAPENET_PATH = None
SPLITS_PATH = None
VOXELS_REL_PATH = "ShapeNetVox"


DEBUG = False
DATA_DIR = get_tests_dir() / "data"


class TestR2N2(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        """
        Check if the data paths are given otherwise skip tests.
        """
        if SHAPENET_PATH is None or not os.path.exists(SHAPENET_PATH):
            url = "https://www.shapenet.org/"
            msg = (
                "ShapeNet data not found, download from %s, update "
                "SHAPENET_PATH at the top of the file, and rerun."
            )
            self.skipTest(msg % url)
        if R2N2_PATH is None or not os.path.exists(R2N2_PATH):
            url = "http://3d-r2n2.stanford.edu/"
            msg = (
                "R2N2 data not found, download from %s, update "
                "R2N2_PATH at the top of the file, and rerun."
            )
            self.skipTest(msg % url)
        if SPLITS_PATH is None or not os.path.exists(SPLITS_PATH):
            msg = """Splits file not found, update SPLITS_PATH at the top
                of the file, and rerun."""
            self.skipTest(msg)

    def test_load_R2N2(self):
        """
        Test the loaded train split of R2N2 return items of the correct shapes and types. Also
        check the first image returned is correct.
        """
        # Load dataset in the train split.
        r2n2_dataset = R2N2(
            "test",
            SHAPENET_PATH,
            R2N2_PATH,
            SPLITS_PATH,
            return_voxels=True,
            voxels_rel_path=VOXELS_REL_PATH,
        )

        # Check total number of objects in the dataset is correct.
        with open(SPLITS_PATH) as splits:
            split_dict = json.load(splits)["test"]
        model_nums = [len(split_dict[synset]) for synset in split_dict]
        self.assertEqual(len(r2n2_dataset), sum(model_nums))

        # Check the numbers of loaded instances for each category are correct.
        for synset in split_dict:
            split_synset_nums = sum(
                len(split_dict[synset][model]) for model in split_dict[synset]
            )
            idx_start = r2n2_dataset.synset_start_idxs[synset]
            idx_end = idx_start + r2n2_dataset.synset_num_models[synset]
            synset_views_list = r2n2_dataset.views_per_model_list[idx_start:idx_end]
            loaded_synset_views = sum(len(views) for views in synset_views_list)
            self.assertEqual(loaded_synset_views, split_synset_nums)

        # Retrieve an object from the dataset.
        r2n2_obj = r2n2_dataset[39]
        # Check that verts and faces returned by __getitem__ have the correct shapes and types.
        verts, faces = r2n2_obj["verts"], r2n2_obj["faces"]
        self.assertTrue(verts.dtype == torch.float32)
        self.assertTrue(faces.dtype == torch.int64)
        self.assertEqual(verts.ndim, 2)
        self.assertEqual(verts.shape[-1], 3)
        self.assertEqual(faces.ndim, 2)
        self.assertEqual(faces.shape[-1], 3)

        # Check that the intrinsic matrix and extrinsic matrix have the
        # correct shapes.
        self.assertEqual(r2n2_obj["R"].shape[0], 24)
        self.assertEqual(r2n2_obj["R"].shape[1:], (3, 3))
        self.assertEqual(r2n2_obj["T"].ndim, 2)
        self.assertEqual(r2n2_obj["T"].shape[1], 3)
        self.assertEqual(r2n2_obj["K"].ndim, 3)
        self.assertEqual(r2n2_obj["K"].shape[1:], (4, 4))

        # Check that image batch returned by __getitem__ has the correct shape.
        self.assertEqual(r2n2_obj["images"].shape[0], 24)
        self.assertEqual(r2n2_obj["images"].shape[1:-1], (137, 137))
        self.assertEqual(r2n2_obj["images"].shape[-1], 3)
        self.assertEqual(r2n2_dataset[39, [21]]["images"].shape[0], 1)
        self.assertEqual(r2n2_dataset[39, torch.tensor([12, 21])]["images"].shape[0], 2)

        # Check models with total view counts less than 24 return image batches
        # of the correct shapes.
        self.assertEqual(r2n2_dataset[635]["images"].shape[0], 5)
        self.assertEqual(r2n2_dataset[8369]["images"].shape[0], 10)

        # Check that the voxel tensor returned by __getitem__ has the correct shape.
        self.assertEqual(r2n2_obj["voxels"].ndim, 4)
        self.assertEqual(r2n2_obj["voxels"].shape, (24, 128, 128, 128))

    def test_collate_models(self):
        """
        Test collate_batched_meshes returns items of the correct shapes and types.
        Check that when collate_batched_meshes is passed to Dataloader, batches of
        the correct shapes and types are returned.
        """
        # Load dataset in the train split.
        r2n2_dataset = R2N2(
            "val",
            SHAPENET_PATH,
            R2N2_PATH,
            SPLITS_PATH,
            return_voxels=True,
            voxels_rel_path=VOXELS_REL_PATH,
        )

        # Randomly retrieve several objects from the dataset and collate them.
        collated_meshes = collate_batched_R2N2(
            [r2n2_dataset[idx] for idx in torch.randint(len(r2n2_dataset), (6,))]
        )
        # Check the collated verts and faces have the correct shapes.
        verts, faces = collated_meshes["verts"], collated_meshes["faces"]
        self.assertEqual(len(verts), 6)
        self.assertEqual(len(faces), 6)
        self.assertEqual(verts[0].shape[-1], 3)
        self.assertEqual(faces[0].shape[-1], 3)

        # Check the collated mesh has the correct shape.
        mesh = collated_meshes["mesh"]
        self.assertEqual(mesh.verts_padded().shape[0], 6)
        self.assertEqual(mesh.verts_padded().shape[-1], 3)
        self.assertEqual(mesh.faces_padded().shape[0], 6)
        self.assertEqual(mesh.faces_padded().shape[-1], 3)

        # Pass the custom collate_fn function to DataLoader and check elements
        # in batch have the correct shape.
        batch_size = 12
        r2n2_loader = DataLoader(
            r2n2_dataset, batch_size=batch_size, collate_fn=collate_batched_R2N2
        )
        it = iter(r2n2_loader)
        object_batch = next(it)
        self.assertEqual(len(object_batch["synset_id"]), batch_size)
        self.assertEqual(len(object_batch["model_id"]), batch_size)
        self.assertEqual(len(object_batch["label"]), batch_size)
        self.assertEqual(object_batch["mesh"].verts_padded().shape[0], batch_size)
        self.assertEqual(object_batch["mesh"].faces_padded().shape[0], batch_size)
        self.assertEqual(object_batch["images"].shape[0], batch_size)
        self.assertEqual(object_batch["R"].shape[0], batch_size)
        self.assertEqual(object_batch["T"].shape[0], batch_size)
        self.assertEqual(object_batch["K"].shape[0], batch_size)
        self.assertEqual(len(object_batch["voxels"]), batch_size)

    def test_catch_render_arg_errors(self):
        """
        Test rendering R2N2 with an invalid model_id, category or index, and
        catch corresponding errors.
        """
        # Load dataset in the train split.
        r2n2_dataset = R2N2("train", SHAPENET_PATH, R2N2_PATH, SPLITS_PATH)

        # Try loading with an invalid model_id and catch error.
        with self.assertRaises(ValueError) as err:
            r2n2_dataset.render(model_ids=["lamp0"])
        self.assertTrue("not found in the loaded dataset" in str(err.exception))

        # Try loading with an index out of bounds and catch error.
        with self.assertRaises(IndexError) as err:
            r2n2_dataset.render(idxs=[1000000])
        self.assertTrue("are out of bounds" in str(err.exception))

        blend_cameras = BlenderCamera(
            R=torch.rand((3, 3, 3)), T=torch.rand((3, 3)), K=torch.rand((3, 4, 4))
        )
        with self.assertRaises(ValueError) as err:
            r2n2_dataset.render(idxs=[10, 11], cameras=blend_cameras)
        self.assertTrue("Mismatch between batch dims" in str(err.exception))

    def test_render_r2n2(self):
        """
        Test rendering objects from R2N2 selected both by indices and model_ids.
        """
        # Set up device and seed for random selections.
        device = torch.device("cuda:0")
        torch.manual_seed(39)

        # Load dataset in the train split.
        r2n2_dataset = R2N2("train", SHAPENET_PATH, R2N2_PATH, SPLITS_PATH)

        # Render first three models in the dataset.
        R, T = look_at_view_transform(1.0, 1.0, 90)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        raster_settings = RasterizationSettings(image_size=512)
        lights = PointLights(
            location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],
            # TODO: debug the source of the discrepancy in two images when rendering on GPU.
            diffuse_color=((0, 0, 0),),
            specular_color=((0, 0, 0),),
            device=device,
        )

        r2n2_by_idxs = r2n2_dataset.render(
            idxs=list(range(3)),
            device=device,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
        )
        # Check that there are three images in the batch.
        self.assertEqual(r2n2_by_idxs.shape[0], 3)

        # Compare the rendered models to the reference images.
        for idx in range(3):
            r2n2_by_idxs_rgb = r2n2_by_idxs[idx, ..., :3].squeeze().cpu()
            if DEBUG:
                Image.fromarray((r2n2_by_idxs_rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / ("DEBUG_r2n2_render_by_idxs_%s.png" % idx)
                )
            image_ref = load_rgb_image(
                "test_r2n2_render_by_idxs_and_ids_%s.png" % idx, DATA_DIR
            )
            self.assertClose(r2n2_by_idxs_rgb, image_ref, atol=0.05)

        # Render the same models but by model_ids this time.
        r2n2_by_model_ids = r2n2_dataset.render(
            model_ids=[
                "1a4a8592046253ab5ff61a3a2a0e2484",
                "1a04dcce7027357ab540cc4083acfa57",
                "1a9d0480b74d782698f5bccb3529a48d",
            ],
            device=device,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
        )

        # Compare the rendered models to the reference images.
        for idx in range(3):
            r2n2_by_model_ids_rgb = r2n2_by_model_ids[idx, ..., :3].squeeze().cpu()
            if DEBUG:
                Image.fromarray(
                    (r2n2_by_model_ids_rgb.numpy() * 255).astype(np.uint8)
                ).save(DATA_DIR / ("DEBUG_r2n2_render_by_model_ids_%s.png" % idx))
            image_ref = load_rgb_image(
                "test_r2n2_render_by_idxs_and_ids_%s.png" % idx, DATA_DIR
            )
            self.assertClose(r2n2_by_model_ids_rgb, image_ref, atol=0.05)

        ###############################
        # Test rendering by categories
        ###############################

        # Render a mixture of categories.
        categories = ["chair", "lamp"]
        mixed_objs = r2n2_dataset.render(
            categories=categories,
            sample_nums=[1, 2],
            device=device,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
        )
        # Compare the rendered models to the reference images.
        for idx in range(3):
            mixed_rgb = mixed_objs[idx, ..., :3].squeeze().cpu()
            if DEBUG:
                Image.fromarray((mixed_rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / ("DEBUG_r2n2_render_by_categories_%s.png" % idx)
                )
            image_ref = load_rgb_image(
                "test_r2n2_render_by_categories_%s.png" % idx, DATA_DIR
            )
            self.assertClose(mixed_rgb, image_ref, atol=0.05)

    def test_blender_camera(self):
        """
        Test BlenderCamera.
        """
        # Test get_world_to_view_transform.
        T = torch.randn(10, 3)
        R = so3_exp_map(torch.randn(10, 3) * 3.0)
        RT = get_world_to_view_transform(R=R, T=T)
        cam = BlenderCamera(R=R, T=T)
        RT_class = cam.get_world_to_view_transform()
        self.assertTrue(torch.allclose(RT.get_matrix(), RT_class.get_matrix()))
        self.assertTrue(isinstance(RT, Transform3d))

        # Test getting camera center.
        C = cam.get_camera_center()
        C_ = -torch.bmm(R, T[:, :, None])[:, :, 0]
        self.assertTrue(torch.allclose(C, C_, atol=1e-05))

    def test_render_by_r2n2_calibration(self):
        """
        Test rendering R2N2 models with calibration matrices from R2N2's own Blender
        in batches.
        """
        # Set up device and seed for random selections.
        device = torch.device("cuda:0")
        torch.manual_seed(39)

        # Load dataset in the train split.
        r2n2_dataset = R2N2("train", SHAPENET_PATH, R2N2_PATH, SPLITS_PATH)
        model_idxs = torch.randint(1000, (2,)).tolist()
        view_idxs = torch.randint(24, (2,)).tolist()
        raster_settings = RasterizationSettings(image_size=512)
        lights = PointLights(
            location=torch.tensor([0.0, 1.0, -2.0], device=device)[None],
            # TODO(nikhilar): debug the source of the discrepancy in two images when
            # rendering on GPU.
            diffuse_color=((0, 0, 0),),
            specular_color=((0, 0, 0),),
            device=device,
        )
        r2n2_batch = r2n2_dataset.render(
            idxs=model_idxs,
            view_idxs=view_idxs,
            device=device,
            raster_settings=raster_settings,
            lights=lights,
        )
        for idx in range(4):
            r2n2_batch_rgb = r2n2_batch[idx, ..., :3].squeeze().cpu()
            if DEBUG:
                Image.fromarray((r2n2_batch_rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR
                    / ("DEBUG_r2n2_render_with_blender_calibrations_%s.png" % idx)
                )
            image_ref = load_rgb_image(
                "test_r2n2_render_with_blender_calibrations_%s.png" % idx, DATA_DIR
            )
            self.assertClose(r2n2_batch_rgb, image_ref, atol=0.05)

    def test_render_voxels(self):
        """
        Test rendering meshes formed from voxels.
        """
        # Set up device and seed for random selections.
        device = torch.device("cuda:0")

        # Load dataset in the train split with only a single view returned for each model.
        r2n2_dataset = R2N2(
            "train",
            SHAPENET_PATH,
            R2N2_PATH,
            SPLITS_PATH,
            return_voxels=True,
            voxels_rel_path=VOXELS_REL_PATH,
        )
        r2n2_model = r2n2_dataset[6, [5]]
        vox_render = render_cubified_voxels(r2n2_model["voxels"], device=device)
        vox_render_rgb = vox_render[0, ..., :3].squeeze().cpu()
        if DEBUG:
            Image.fromarray((vox_render_rgb.numpy() * 255).astype(np.uint8)).save(
                DATA_DIR / ("DEBUG_r2n2_voxel_to_mesh_render.png")
            )
        image_ref = load_rgb_image("test_r2n2_voxel_to_mesh_render.png", DATA_DIR)
        self.assertClose(vox_render_rgb, image_ref, atol=0.05)
