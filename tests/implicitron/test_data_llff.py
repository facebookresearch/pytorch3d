# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from pytorch3d.implicitron.dataset.blender_dataset_map_provider import (
    BlenderDatasetMapProvider,
)
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.llff_dataset_map_provider import (
    LlffDatasetMapProvider,
)
from pytorch3d.implicitron.tools.config import expand_args_fields
from tests.common_testing import TestCaseMixin


# These tests are only run internally, where the data is available.
internal = os.environ.get("FB_TEST", False)
inside_re_worker = os.environ.get("INSIDE_RE_WORKER", False)
skip_tests = not internal or inside_re_worker


@unittest.skipIf(skip_tests, "no data")
class TestDataLlff(TestCaseMixin, unittest.TestCase):
    def test_synthetic(self):
        expand_args_fields(BlenderDatasetMapProvider)

        provider = BlenderDatasetMapProvider(
            base_dir="manifold://co3d/tree/nerf_data/nerf_synthetic/lego",
            object_name="lego",
        )
        dataset_map = provider.get_dataset_map()

        for name, length in [("train", 100), ("val", 100), ("test", 200)]:
            dataset = getattr(dataset_map, name)
            self.assertEqual(len(dataset), length)
            # try getting a value
            value = dataset[0]
            self.assertIsInstance(value, FrameData)

    def test_llff(self):
        expand_args_fields(LlffDatasetMapProvider)

        provider = LlffDatasetMapProvider(
            base_dir="manifold://co3d/tree/nerf_data/nerf_llff_data/fern",
            object_name="fern",
        )
        dataset_map = provider.get_dataset_map()

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

        self.assertEqual(len(dataset_map.test.get_eval_batches()), 3)
        for batch in dataset_map.test.get_eval_batches():
            self.assertEqual(len(batch), 1)
            self.assertEqual(dataset_map.test[batch[0]].frame_type, "unseen")

    def test_include_known_frames(self):
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

        self.assertEqual(len(dataset_map.test.get_eval_batches()), 3)
        for batch in dataset_map.test.get_eval_batches():
            self.assertEqual(len(batch), 3)
            self.assertEqual(dataset_map.test[batch[0]].frame_type, "unseen")
            for i in batch[1:]:
                self.assertEqual(dataset_map.test[i].frame_type, "known")
