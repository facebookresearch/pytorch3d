# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Sanity checks for loading R2N2.
"""
import json
import os
import unittest
from pathlib import Path

import numpy as np
import torch
from common_testing import TestCaseMixin, load_rgb_image
from PIL import Image
from pytorch3d.datasets import R2N2, collate_batched_meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    look_at_view_transform,
)
from torch.utils.data import DataLoader


# Set these paths in order to run the tests.
R2N2_PATH = None
SHAPENET_PATH = None
SPLITS_PATH = None

DEBUG = False
DATA_DIR = Path(__file__).resolve().parent / "data"


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
        r2n2_dataset = R2N2("train", SHAPENET_PATH, R2N2_PATH, SPLITS_PATH)

        # Check total number of objects in the dataset is correct.
        with open(SPLITS_PATH) as splits:
            split_dict = json.load(splits)["train"]
        model_nums = [len(split_dict[synset].keys()) for synset in split_dict.keys()]
        self.assertEqual(len(r2n2_dataset), sum(model_nums))

        # Randomly retrieve an object from the dataset.
        rand_idx = torch.randint(len(r2n2_dataset), (1,))
        rand_obj = r2n2_dataset[rand_idx]
        # Check that verts and faces returned by __getitem__ have the correct shapes and types.
        verts, faces = rand_obj["verts"], rand_obj["faces"]
        self.assertTrue(verts.dtype == torch.float32)
        self.assertTrue(faces.dtype == torch.int64)
        self.assertEqual(verts.ndim, 2)
        self.assertEqual(verts.shape[-1], 3)
        self.assertEqual(faces.ndim, 2)
        self.assertEqual(faces.shape[-1], 3)

        # Check that image batch returned by __getitem__ has the correct shape.
        self.assertEqual(rand_obj["images"].shape[0], 24)
        self.assertEqual(rand_obj["images"].shape[1], 137)
        self.assertEqual(rand_obj["images"].shape[2], 137)
        self.assertEqual(rand_obj["images"].shape[-1], 3)
        self.assertEqual(r2n2_dataset[rand_idx, [21]]["images"].shape[0], 1)

    def test_collate_models(self):
        """
        Test collate_batched_meshes returns items of the correct shapes and types.
        Check that when collate_batched_meshes is passed to Dataloader, batches of
        the correct shapes and types are returned.
        """
        # Load dataset in the train split.
        r2n2_dataset = R2N2("train", SHAPENET_PATH, R2N2_PATH, SPLITS_PATH)

        # Randomly retrieve several objects from the dataset and collate them.
        collated_meshes = collate_batched_meshes(
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
            r2n2_dataset, batch_size=batch_size, collate_fn=collate_batched_meshes
        )
        it = iter(r2n2_loader)
        object_batch = next(it)
        self.assertEqual(len(object_batch["synset_id"]), batch_size)
        self.assertEqual(len(object_batch["model_id"]), batch_size)
        self.assertEqual(len(object_batch["label"]), batch_size)
        self.assertEqual(object_batch["mesh"].verts_padded().shape[0], batch_size)
        self.assertEqual(object_batch["mesh"].faces_padded().shape[0], batch_size)
        self.assertEqual(object_batch["images"].shape[0], batch_size)

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
        cameras = OpenGLPerspectiveCameras(R=R, T=T, device=device)
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
