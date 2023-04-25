# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import unittest

import torch

from pytorch3d.implicitron.dataset.data_loader_map_provider import (  # noqa
    SequenceDataLoaderMapProvider,
    SimpleDataLoaderMapProvider,
)
from pytorch3d.implicitron.dataset.data_source import ImplicitronDataSource
from pytorch3d.implicitron.dataset.sql_dataset import SqlIndexDataset  # noqa
from pytorch3d.implicitron.dataset.sql_dataset_provider import (  # noqa
    SqlIndexDatasetMapProvider,
)
from pytorch3d.implicitron.dataset.train_eval_data_loader_provider import (
    TrainEvalDataLoaderMapProvider,
)
from pytorch3d.implicitron.tools.config import get_default_args

logger = logging.getLogger("pytorch3d.implicitron.dataset.sql_dataset")
sh = logging.StreamHandler()
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)

_CO3D_SQL_DATASET_ROOT: str = os.getenv("CO3D_SQL_DATASET_ROOT", "")


@unittest.skipUnless(_CO3D_SQL_DATASET_ROOT, "Run only if CO3D is available")
class TestCo3dSqlDataSource(unittest.TestCase):
    def test_no_subsets(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "SqlIndexDatasetMapProvider"
        args.data_loader_map_provider_class_type = "TrainEvalDataLoaderMapProvider"
        provider_args = args.dataset_map_provider_SqlIndexDatasetMapProvider_args
        provider_args.ignore_subsets = True

        dataset_args = provider_args.dataset_SqlIndexDataset_args
        dataset_args.pick_categories = ["skateboard"]
        dataset_args.limit_sequences_to = 1

        data_source = ImplicitronDataSource(**args)
        self.assertIsInstance(
            data_source.data_loader_map_provider, TrainEvalDataLoaderMapProvider
        )
        _, data_loaders = data_source.get_datasets_and_dataloaders()
        self.assertEqual(len(data_loaders.train), 202)
        for frame in data_loaders.train:
            self.assertIsNone(frame.frame_type)
            self.assertEqual(frame.image_rgb.shape[-1], 800)  # check loading blobs
            break

    def test_subsets(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "SqlIndexDatasetMapProvider"
        provider_args = args.dataset_map_provider_SqlIndexDatasetMapProvider_args
        provider_args.subset_lists_path = (
            "skateboard/set_lists/set_lists_manyview_dev_0.json"
        )
        # this will naturally limit to one sequence (no need to limit by cat/sequence)

        dataset_args = provider_args.dataset_SqlIndexDataset_args
        dataset_args.remove_empty_masks = True

        for sampler_type in [
            "SimpleDataLoaderMapProvider",
            "SequenceDataLoaderMapProvider",
            "TrainEvalDataLoaderMapProvider",
        ]:
            args.data_loader_map_provider_class_type = sampler_type
            data_source = ImplicitronDataSource(**args)
            _, data_loaders = data_source.get_datasets_and_dataloaders()
            self.assertEqual(len(data_loaders.train), 102)
            self.assertEqual(len(data_loaders.val), 100)
            self.assertEqual(len(data_loaders.test), 100)
            for split in ["train", "val", "test"]:
                for frame in data_loaders[split]:
                    self.assertEqual(frame.frame_type, [split])
                    # check loading blobs
                    self.assertEqual(frame.image_rgb.shape[-1], 800)
                    break

    def test_sql_subsets(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "SqlIndexDatasetMapProvider"
        provider_args = args.dataset_map_provider_SqlIndexDatasetMapProvider_args
        provider_args.subset_lists_path = "set_lists/set_lists_manyview_dev_0.sqlite"

        dataset_args = provider_args.dataset_SqlIndexDataset_args
        dataset_args.remove_empty_masks = True
        dataset_args.pick_categories = ["skateboard"]

        for sampler_type in [
            "SimpleDataLoaderMapProvider",
            "SequenceDataLoaderMapProvider",
            "TrainEvalDataLoaderMapProvider",
        ]:
            args.data_loader_map_provider_class_type = sampler_type
            data_source = ImplicitronDataSource(**args)
            _, data_loaders = data_source.get_datasets_and_dataloaders()
            self.assertEqual(len(data_loaders.train), 102)
            self.assertEqual(len(data_loaders.val), 100)
            self.assertEqual(len(data_loaders.test), 100)
            for split in ["train", "val", "test"]:
                for frame in data_loaders[split]:
                    self.assertEqual(frame.frame_type, [split])
                    self.assertEqual(
                        frame.image_rgb.shape[-1], 800
                    )  # check loading blobs
                    break

    @unittest.skip("It takes 75 seconds; skipping by default")
    def test_huge_subsets(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "SqlIndexDatasetMapProvider"
        args.data_loader_map_provider_class_type = "TrainEvalDataLoaderMapProvider"
        provider_args = args.dataset_map_provider_SqlIndexDatasetMapProvider_args
        provider_args.subset_lists_path = "set_lists/set_lists_fewview_dev.sqlite"

        dataset_args = provider_args.dataset_SqlIndexDataset_args
        dataset_args.remove_empty_masks = True

        data_source = ImplicitronDataSource(**args)
        _, data_loaders = data_source.get_datasets_and_dataloaders()
        self.assertEqual(len(data_loaders.train), 3158974)
        self.assertEqual(len(data_loaders.val), 518417)
        self.assertEqual(len(data_loaders.test), 518417)
        for split in ["train", "val", "test"]:
            for frame in data_loaders[split]:
                self.assertEqual(frame.frame_type, [split])
                self.assertEqual(frame.image_rgb.shape[-1], 800)  # check loading blobs
                break

    def test_broken_subsets(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "SqlIndexDatasetMapProvider"
        args.data_loader_map_provider_class_type = "TrainEvalDataLoaderMapProvider"
        provider_args = args.dataset_map_provider_SqlIndexDatasetMapProvider_args
        provider_args.subset_lists_path = "et_non_est"
        provider_args.dataset_SqlIndexDataset_args.pick_categories = ["skateboard"]
        with self.assertRaises(FileNotFoundError) as err:
            ImplicitronDataSource(**args)

        # check the hint text
        self.assertIn("Subset lists path given but not found", str(err.exception))

    def test_eval_batches(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "SqlIndexDatasetMapProvider"
        args.data_loader_map_provider_class_type = "TrainEvalDataLoaderMapProvider"
        provider_args = args.dataset_map_provider_SqlIndexDatasetMapProvider_args
        provider_args.subset_lists_path = "set_lists/set_lists_manyview_dev_0.sqlite"
        provider_args.eval_batches_path = (
            "skateboard/eval_batches/eval_batches_manyview_dev_0.json"
        )

        dataset_args = provider_args.dataset_SqlIndexDataset_args
        dataset_args.remove_empty_masks = True
        dataset_args.pick_categories = ["skateboard"]

        data_source = ImplicitronDataSource(**args)
        _, data_loaders = data_source.get_datasets_and_dataloaders()
        self.assertEqual(len(data_loaders.train), 102)
        self.assertEqual(len(data_loaders.val), 100)
        self.assertEqual(len(data_loaders.test), 50)
        for split in ["train", "val", "test"]:
            for frame in data_loaders[split]:
                self.assertEqual(frame.frame_type, [split])
                self.assertEqual(frame.image_rgb.shape[-1], 800)  # check loading blobs
                break

    def test_eval_batches_from_subset_list_name(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "SqlIndexDatasetMapProvider"
        args.data_loader_map_provider_class_type = "TrainEvalDataLoaderMapProvider"
        provider_args = args.dataset_map_provider_SqlIndexDatasetMapProvider_args
        provider_args.subset_list_name = "manyview_dev_0"
        provider_args.category = "skateboard"

        dataset_args = provider_args.dataset_SqlIndexDataset_args
        dataset_args.remove_empty_masks = True

        data_source = ImplicitronDataSource(**args)
        dataset, data_loaders = data_source.get_datasets_and_dataloaders()
        self.assertListEqual(list(dataset.train.pick_categories), ["skateboard"])
        self.assertEqual(len(data_loaders.train), 102)
        self.assertEqual(len(data_loaders.val), 100)
        self.assertEqual(len(data_loaders.test), 50)
        for split in ["train", "val", "test"]:
            for frame in data_loaders[split]:
                self.assertEqual(frame.frame_type, [split])
                self.assertEqual(frame.image_rgb.shape[-1], 800)  # check loading blobs
                break

    def test_frame_access(self):
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "SqlIndexDatasetMapProvider"
        args.data_loader_map_provider_class_type = "TrainEvalDataLoaderMapProvider"
        provider_args = args.dataset_map_provider_SqlIndexDatasetMapProvider_args
        provider_args.subset_lists_path = "set_lists/set_lists_manyview_dev_0.sqlite"

        dataset_args = provider_args.dataset_SqlIndexDataset_args
        dataset_args.remove_empty_masks = True
        dataset_args.pick_categories = ["skateboard"]
        frame_builder_args = dataset_args.frame_data_builder_FrameDataBuilder_args
        frame_builder_args.load_point_clouds = True
        frame_builder_args.box_crop = False  # required for .meta

        data_source = ImplicitronDataSource(**args)
        dataset_map, _ = data_source.get_datasets_and_dataloaders()
        dataset = dataset_map["train"]

        for idx in [10, ("245_26182_52130", 22)]:
            example_meta = dataset.meta[idx]
            example = dataset[idx]

            self.assertIsNone(example_meta.image_rgb)
            self.assertIsNone(example_meta.fg_probability)
            self.assertIsNone(example_meta.depth_map)
            self.assertIsNone(example_meta.sequence_point_cloud)
            self.assertIsNotNone(example_meta.camera)

            self.assertIsNotNone(example.image_rgb)
            self.assertIsNotNone(example.fg_probability)
            self.assertIsNotNone(example.depth_map)
            self.assertIsNotNone(example.sequence_point_cloud)
            self.assertIsNotNone(example.camera)

            self.assertEqual(example_meta.sequence_name, example.sequence_name)
            self.assertEqual(example_meta.frame_number, example.frame_number)
            self.assertEqual(example_meta.frame_timestamp, example.frame_timestamp)
            self.assertEqual(example_meta.sequence_category, example.sequence_category)
            torch.testing.assert_close(example_meta.camera.R, example.camera.R)
            torch.testing.assert_close(example_meta.camera.T, example.camera.T)
            torch.testing.assert_close(
                example_meta.camera.focal_length, example.camera.focal_length
            )
            torch.testing.assert_close(
                example_meta.camera.principal_point, example.camera.principal_point
            )
