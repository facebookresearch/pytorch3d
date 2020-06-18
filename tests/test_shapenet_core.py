# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
"""
Sanity checks for loading ShapeNet Core v1.
"""
import os
import random
import unittest
import warnings

import torch
from common_testing import TestCaseMixin
from pytorch3d.datasets.shapenet.shapenet_core import ShapeNetCore


SHAPENET_PATH = None


class TestShapenetCore(TestCaseMixin, unittest.TestCase):
    def test_load_shapenet_core(self):

        # The ShapeNet dataset is not provided in the repo.
        # Download this separately and update the `shapenet_path`
        # with the location of the dataset in order to run this test.
        if SHAPENET_PATH is None or not os.path.exists(SHAPENET_PATH):
            url = "https://www.shapenet.org/"
            msg = """ShapeNet data not found, download from %s, save it at the path %s,
                update SHAPENET_PATH at the top of the file, and rerun""" % (
                url,
                SHAPENET_PATH,
            )
            warnings.warn(msg)
            return True

        # Load ShapeNetCore without specifying any particular categories.
        shapenet_dataset = ShapeNetCore(SHAPENET_PATH)

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
        rand_obj = random.choice(shapenet_dataset)
        self.assertEqual(len(rand_obj), 4)
        # Check that data types and shapes of items returned by __getitem__ are correct.
        verts, faces = rand_obj["verts"], rand_obj["faces"]
        self.assertTrue(verts.dtype == torch.float32)
        self.assertTrue(faces.dtype == torch.int64)
        self.assertEqual(verts.ndim, 2)
        self.assertEqual(verts.shape[-1], 3)
        self.assertEqual(faces.ndim, 2)
        self.assertEqual(faces.shape[-1], 3)
