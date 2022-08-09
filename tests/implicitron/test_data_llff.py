# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from pytorch3d.implicitron.dataset.blender_dataset_map_provider import (
    BlenderDatasetMapProvider,
)
from pytorch3d.implicitron.dataset.data_source import ImplicitronDataSource
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.llff_dataset_map_provider import (
    LlffDatasetMapProvider,
)
from pytorch3d.implicitron.tools.config import expand_args_fields, get_default_args
from pytorch3d.renderer import PerspectiveCameras
from tests.common_testing import TestCaseMixin


# These tests are only run internally, where the data is available.
internal = os.environ.get("FB_TEST", False)
inside_re_worker = os.environ.get("INSIDE_RE_WORKER", False)


@unittest.skipUnless(internal, "no data")
class TestDataLlff(TestCaseMixin, unittest.TestCase):
    def test_synthetic(self):
        if inside_re_worker:
            return
        expand_args_fields(BlenderDatasetMapProvider)

        provider = BlenderDatasetMapProvider(
            base_dir="manifold://co3d/tree/nerf_data/nerf_synthetic/lego",
            object_name="lego",
        )
        dataset_map = provider.get_dataset_map()
        known_matrix = torch.zeros(1, 4, 4)
        known_matrix[0, 0, 0] = 2.7778
        known_matrix[0, 1, 1] = 2.7778
        known_matrix[0, 2, 3] = 1
        known_matrix[0, 3, 2] = 1

        for name, length in [("train", 100), ("val", 100), ("test", 200)]:
            dataset = getattr(dataset_map, name)
            self.assertEqual(len(dataset), length)
            # try getting a value
            value = dataset[0]
            self.assertEqual(value.image_rgb.shape, (3, 800, 800))
            self.assertEqual(value.fg_probability.shape, (1, 800, 800))
            # corner of image is background
            self.assertEqual(value.fg_probability[0, 0, 0], 0)
            self.assertEqual(value.fg_probability.max(), 1.0)
            self.assertIsInstance(value.camera, PerspectiveCameras)
            self.assertEqual(len(value.camera), 1)
            self.assertIsNone(value.camera.K)
            matrix = value.camera.get_projection_transform().get_matrix()
            self.assertClose(matrix, known_matrix, atol=1e-4)
            self.assertIsInstance(value, FrameData)

    def test_llff(self):
        if inside_re_worker:
            return
        expand_args_fields(LlffDatasetMapProvider)

        provider = LlffDatasetMapProvider(
            base_dir="manifold://co3d/tree/nerf_data/nerf_llff_data/fern",
            object_name="fern",
            downscale_factor=8,
        )
        dataset_map = provider.get_dataset_map()
        known_matrix = torch.zeros(1, 4, 4)
        known_matrix[0, 0, 0] = 2.1564
        known_matrix[0, 1, 1] = 2.1564
        known_matrix[0, 2, 3] = 1
        known_matrix[0, 3, 2] = 1

        for name, length, frame_type in [
            ("train", 17, "known"),
            ("test", 3, "unseen"),
            ("val", 3, "unseen"),
        ]:
            dataset = getattr(dataset_map, name)
            self.assertEqual(len(dataset), length)
            # try getting a value
            value = dataset[0]
            self.assertIsInstance(value, FrameData)
            self.assertEqual(value.frame_type, frame_type)
            self.assertEqual(value.image_rgb.shape, (3, 378, 504))
            self.assertIsInstance(value.camera, PerspectiveCameras)
            self.assertEqual(len(value.camera), 1)
            self.assertIsNone(value.camera.K)
            matrix = value.camera.get_projection_transform().get_matrix()
            self.assertClose(matrix, known_matrix, atol=1e-4)

        self.assertEqual(len(dataset_map.test.get_eval_batches()), 3)
        for batch in dataset_map.test.get_eval_batches():
            self.assertEqual(len(batch), 1)
            self.assertEqual(dataset_map.test[batch[0]].frame_type, "unseen")

    def test_include_known_frames(self):
        if inside_re_worker:
            return
        expand_args_fields(LlffDatasetMapProvider)

        provider = LlffDatasetMapProvider(
            base_dir="manifold://co3d/tree/nerf_data/nerf_llff_data/fern",
            object_name="fern",
            n_known_frames_for_test=2,
        )
        dataset_map = provider.get_dataset_map()

        for name, types in [
            ("train", ["known"] * 17),
            ("val", ["unseen"] * 3 + ["known"] * 17),
            ("test", ["unseen"] * 3 + ["known"] * 17),
        ]:
            dataset = getattr(dataset_map, name)
            self.assertEqual(len(dataset), len(types))
            for i, frame_type in enumerate(types):
                value = dataset[i]
                self.assertEqual(value.frame_type, frame_type)
                self.assertIsNone(value.fg_probability)

        self.assertEqual(len(dataset_map.test.get_eval_batches()), 3)
        for batch in dataset_map.test.get_eval_batches():
            self.assertEqual(len(batch), 3)
            self.assertEqual(dataset_map.test[batch[0]].frame_type, "unseen")
            for i in batch[1:]:
                self.assertEqual(dataset_map.test[i].frame_type, "known")

    def test_loaders(self):
        if inside_re_worker:
            return
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "BlenderDatasetMapProvider"
        dataset_args = args.dataset_map_provider_BlenderDatasetMapProvider_args
        dataset_args.object_name = "lego"
        dataset_args.base_dir = "manifold://co3d/tree/nerf_data/nerf_synthetic/lego"

        data_source = ImplicitronDataSource(**args)
        _, data_loaders = data_source.get_datasets_and_dataloaders()
        for i in data_loaders.train:
            self.assertEqual(i.frame_type, ["known"])
            self.assertEqual(i.image_rgb.shape, (1, 3, 800, 800))
        for i in data_loaders.val:
            self.assertEqual(i.frame_type, ["unseen"])
            self.assertEqual(i.image_rgb.shape, (1, 3, 800, 800))
        for i in data_loaders.test:
            self.assertEqual(i.frame_type, ["unseen"])
            self.assertEqual(i.image_rgb.shape, (1, 3, 800, 800))

        cameras = data_source.all_train_cameras
        self.assertIsInstance(cameras, PerspectiveCameras)
        self.assertEqual(len(cameras), 100)
