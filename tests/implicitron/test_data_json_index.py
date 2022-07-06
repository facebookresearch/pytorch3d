# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

from pytorch3d.implicitron.dataset.data_source import ImplicitronDataSource
from pytorch3d.implicitron.tools.config import get_default_args
from pytorch3d.renderer import PerspectiveCameras
from tests.common_testing import TestCaseMixin

# These tests are only run internally, where the data is available.
internal = os.environ.get("FB_TEST", False)
inside_re_worker = os.environ.get("INSIDE_RE_WORKER", False)
skip_tests = not internal or inside_re_worker


@unittest.skipIf(skip_tests, "no data")
class TestDataJsonIndex(TestCaseMixin, unittest.TestCase):
    def test_loaders(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "JsonIndexDatasetMapProvider"
        dataset_args = args.dataset_map_provider_JsonIndexDatasetMapProvider_args
        dataset_args.category = "skateboard"
        dataset_args.dataset_root = "manifold://co3d/tree/extracted"
        dataset_args.test_restrict_sequence_id = 0
        dataset_args.dataset_JsonIndexDataset_args.limit_sequences_to = 1

        data_source = ImplicitronDataSource(**args)

        cameras = data_source.get_all_train_cameras()
        self.assertIsInstance(cameras, PerspectiveCameras)
        self.assertEqual(len(cameras), 81)

        data_sets, data_loaders = data_source.get_datasets_and_dataloaders()

        self.assertEqual(len(data_sets.train), 81)
        self.assertEqual(len(data_sets.val), 102)
        self.assertEqual(len(data_sets.test), 102)
