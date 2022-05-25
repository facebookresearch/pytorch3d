# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Sanity checks for loading ShapeNetCore.
"""
import os
import unittest

import numpy as np
import torch
from PIL import Image
from pytorch3d.datasets import collate_batched_meshes, ShapeNetCore
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
)
from torch.utils.data import DataLoader

from .common_testing import get_tests_dir, load_rgb_image, TestCaseMixin


# Set the SHAPENET_PATH to the local path to the dataset
SHAPENET_PATH = None
VERSION = 1
# If DEBUG=True, save out images generated in the tests for debugging.
# All saved images have prefix DEBUG_
DEBUG = False
DATA_DIR = get_tests_dir() / "data"


class TestShapenetCore(TestCaseMixin, unittest.TestCase):
    def setUp(self):
        """
        Check if the ShapeNet dataset is provided in the repo.
        If not, download this separately and update the shapenet_path`
        with the location of the dataset in order to run the tests.
        """
        if SHAPENET_PATH is None or not os.path.exists(SHAPENET_PATH):
            url = "https://www.shapenet.org/"
            msg = (
                "ShapeNet data not found, download from %s, update "
                "SHAPENET_PATH at the top of the file, and rerun."
            )

            self.skipTest(msg % url)

    def test_load_shapenet_core(self):
        """
        Test loading both the entire ShapeNetCore dataset and a subset of the ShapeNetCore
        dataset. Check the loaded datasets return items of the correct shapes and types.
        """
        # Try loading ShapeNetCore with an invalid version number and catch error.
        with self.assertRaises(ValueError) as err:
            ShapeNetCore(SHAPENET_PATH, version=3)
        self.assertTrue("Version number must be either 1 or 2." in str(err.exception))

        # Load ShapeNetCore without specifying any particular categories.
        shapenet_dataset = ShapeNetCore(SHAPENET_PATH, version=VERSION)

        # Count the number of grandchildren directories (which should be equal to
        # the total number of objects in the dataset) by walking through the given
        # directory.
        wnsynset_list = [
            wnsynset
            for wnsynset in os.listdir(SHAPENET_PATH)
            if os.path.isdir(os.path.join(SHAPENET_PATH, wnsynset))
        ]
        model_num_list = [
            (len(next(os.walk(os.path.join(SHAPENET_PATH, wnsynset)))[1]))
            for wnsynset in wnsynset_list
        ]
        # Check total number of objects in the dataset is correct.
        self.assertEqual(len(shapenet_dataset), sum(model_num_list))

        # Randomly retrieve an object from the dataset.
        rand_obj = shapenet_dataset[torch.randint(len(shapenet_dataset), (1,))]
        # Check that data types and shapes of items returned by __getitem__ are correct.
        verts, faces = rand_obj["verts"], rand_obj["faces"]
        self.assertTrue(verts.dtype == torch.float32)
        self.assertTrue(faces.dtype == torch.int64)
        self.assertEqual(verts.ndim, 2)
        self.assertEqual(verts.shape[-1], 3)
        self.assertEqual(faces.ndim, 2)
        self.assertEqual(faces.shape[-1], 3)

        # Load six categories from ShapeNetCore.
        # Specify categories with a combination of offsets and labels.
        shapenet_subset = ShapeNetCore(
            SHAPENET_PATH,
            synsets=[
                "04330267",
                "guitar",
                "02801938",
                "birdhouse",
                "03991062",
                "tower",
            ],
            version=1,
        )
        subset_offsets = [
            "04330267",
            "03467517",
            "02801938",
            "02843684",
            "03991062",
            "04460130",
        ]
        subset_model_nums = [
            (len(next(os.walk(os.path.join(SHAPENET_PATH, offset)))[1]))
            for offset in subset_offsets
        ]
        self.assertEqual(len(shapenet_subset), sum(subset_model_nums))

    def test_collate_models(self):
        """
        Test collate_batched_meshes returns items of the correct shapes and types.
        Check that when collate_batched_meshes is passed to Dataloader, batches of
        the correct shapes and types are returned.
        """
        # Load ShapeNetCore without specifying any particular categories.
        shapenet_dataset = ShapeNetCore(SHAPENET_PATH)
        # Randomly retrieve several objects from the dataset.
        rand_idxs = torch.randint(len(shapenet_dataset), (6,))
        rand_objs = [shapenet_dataset[idx] for idx in rand_idxs]

        # Collate the randomly selected objects
        collated_meshes = collate_batched_meshes(rand_objs)
        verts, faces = (collated_meshes["verts"], collated_meshes["faces"])
        self.assertEqual(len(verts), 6)
        self.assertEqual(len(faces), 6)

        # Pass the custom collate_fn function to DataLoader and check elements
        # in batch have the correct shape.
        batch_size = 12
        shapenet_core_loader = DataLoader(
            shapenet_dataset, batch_size=batch_size, collate_fn=collate_batched_meshes
        )
        it = iter(shapenet_core_loader)
        object_batch = next(it)
        self.assertEqual(len(object_batch["synset_id"]), batch_size)
        self.assertEqual(len(object_batch["model_id"]), batch_size)
        self.assertEqual(len(object_batch["label"]), batch_size)
        self.assertEqual(object_batch["mesh"].verts_padded().shape[0], batch_size)
        self.assertEqual(object_batch["mesh"].faces_padded().shape[0], batch_size)

    def test_catch_render_arg_errors(self):
        """
        Test rendering ShapeNetCore with invalid model_ids, categories or indices,
        and catch corresponding errors.
        """
        # Load ShapeNetCore.
        shapenet_dataset = ShapeNetCore(SHAPENET_PATH)

        # Try loading with an invalid model_id and catch error.
        with self.assertRaises(ValueError) as err:
            shapenet_dataset.render(model_ids=["piano0"])
        self.assertTrue("not found in the loaded dataset" in str(err.exception))

        # Try loading with an index out of bounds and catch error.
        with self.assertRaises(IndexError) as err:
            shapenet_dataset.render(idxs=[100000])
        self.assertTrue("are out of bounds" in str(err.exception))

    def test_render_shapenet_core(self):
        """
        Test rendering objects from ShapeNetCore.
        """
        # Setup device and seed for random selections.
        device = torch.device("cuda:0")
        torch.manual_seed(39)

        # Load category piano from ShapeNetCore.
        piano_dataset = ShapeNetCore(SHAPENET_PATH, synsets=["piano"])

        # Rendering settings.
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

        # Render first three models in the piano category.
        pianos = piano_dataset.render(
            idxs=list(range(3)),
            device=device,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
        )
        # Check that there are three images in the batch.
        self.assertEqual(pianos.shape[0], 3)

        # Compare the rendered models to the reference images.
        for idx in range(3):
            piano_rgb = pianos[idx, ..., :3].squeeze().cpu()
            if DEBUG:
                Image.fromarray((piano_rgb.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / ("DEBUG_shapenet_core_render_piano_by_idxs_%s.png" % idx)
                )
            image_ref = load_rgb_image(
                "test_shapenet_core_render_piano_%s.png" % idx, DATA_DIR
            )
            self.assertClose(piano_rgb, image_ref, atol=0.05)

        # Render the same piano models but by model_ids this time.
        pianos_2 = piano_dataset.render(
            model_ids=[
                "13394ca47c89f91525a3aaf903a41c90",
                "14755c2ee8e693aba508f621166382b0",
                "156c4207af6d2c8f1fdc97905708b8ea",
            ],
            device=device,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
        )

        # Compare the rendered models to the reference images.
        for idx in range(3):
            piano_rgb_2 = pianos_2[idx, ..., :3].squeeze().cpu()
            if DEBUG:
                Image.fromarray((piano_rgb_2.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR / ("DEBUG_shapenet_core_render_piano_by_ids_%s.png" % idx)
                )
            image_ref = load_rgb_image(
                "test_shapenet_core_render_piano_%s.png" % idx, DATA_DIR
            )
            self.assertClose(piano_rgb_2, image_ref, atol=0.05)

        #######################
        # Render by categories
        #######################

        # Load ShapeNetCore.
        shapenet_dataset = ShapeNetCore(SHAPENET_PATH)

        # Render a mixture of categories and specify the number of models to be
        # randomly sampled from each category.
        mixed_objs = shapenet_dataset.render(
            categories=["faucet", "chair"],
            sample_nums=[2, 1],
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
                    DATA_DIR
                    / ("DEBUG_shapenet_core_render_mixed_by_categories_%s.png" % idx)
                )
            image_ref = load_rgb_image(
                "test_shapenet_core_render_mixed_by_categories_%s.png" % idx, DATA_DIR
            )
            self.assertClose(mixed_rgb, image_ref, atol=0.05)

        # Render a mixture of categories without specifying sample_nums.
        mixed_objs_2 = shapenet_dataset.render(
            categories=["faucet", "chair"],
            device=device,
            cameras=cameras,
            raster_settings=raster_settings,
            lights=lights,
        )
        # Compare the rendered models to the reference images.
        for idx in range(2):
            mixed_rgb_2 = mixed_objs_2[idx, ..., :3].squeeze().cpu()
            if DEBUG:
                Image.fromarray((mixed_rgb_2.numpy() * 255).astype(np.uint8)).save(
                    DATA_DIR
                    / ("DEBUG_shapenet_core_render_without_sample_nums_%s.png" % idx)
                )
            image_ref = load_rgb_image(
                "test_shapenet_core_render_without_sample_nums_%s.png" % idx, DATA_DIR
            )
            self.assertClose(mixed_rgb_2, image_ref, atol=0.05)

    def test_load_textures_false(self):
        shapenet_dataset = ShapeNetCore(
            SHAPENET_PATH, load_textures=False, version=VERSION
        )
        model = shapenet_dataset[0]
        self.assertIsNone(model["textures"])
