# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Sanity checks for loading R2N2.
"""
import json
import os
import unittest

import torch
from common_testing import TestCaseMixin
from pytorch3d.datasets import R2N2, collate_batched_meshes
from torch.utils.data import DataLoader


# Set these paths in order to run the tests.
R2N2_PATH = None
SHAPENET_PATH = None
SPLITS_PATH = None


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
        Test loading the train split of R2N2. Check the loaded dataset return items
        of the correct shapes and types.
        """
        # Load dataset in the train split.
        split = "train"
        r2n2_dataset = R2N2(split, SHAPENET_PATH, R2N2_PATH, SPLITS_PATH)

        # Check total number of objects in the dataset is correct.
        with open(SPLITS_PATH) as splits:
            split_dict = json.load(splits)[split]
        model_nums = [len(split_dict[synset].keys()) for synset in split_dict.keys()]
        self.assertEqual(len(r2n2_dataset), sum(model_nums))

        # Randomly retrieve an object from the dataset.
        rand_obj = r2n2_dataset[torch.randint(len(r2n2_dataset), (1,))]
        # Check that data type and shape of the item returned by __getitem__ are correct.
        verts, faces = rand_obj["verts"], rand_obj["faces"]
        self.assertTrue(verts.dtype == torch.float32)
        self.assertTrue(faces.dtype == torch.int64)
        self.assertEqual(verts.ndim, 2)
        self.assertEqual(verts.shape[-1], 3)
        self.assertEqual(faces.ndim, 2)
        self.assertEqual(faces.shape[-1], 3)

    def test_collate_models(self):
        """
        Test collate_batched_meshes returns items of the correct shapes and types.
        Check that when collate_batched_meshes is passed to Dataloader, batches of
        the correct shapes and types are returned.
        """
        # Load dataset in the train split.
        split = "train"
        r2n2_dataset = R2N2(split, SHAPENET_PATH, R2N2_PATH, SPLITS_PATH)

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
