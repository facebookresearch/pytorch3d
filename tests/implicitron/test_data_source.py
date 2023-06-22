# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import unittest
import unittest.mock

import torch
from omegaconf import OmegaConf
from pytorch3d.implicitron.dataset.data_loader_map_provider import (
    SequenceDataLoaderMapProvider,
    SimpleDataLoaderMapProvider,
)
from pytorch3d.implicitron.dataset.data_source import ImplicitronDataSource
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.tools.config import get_default_args
from tests.common_testing import get_tests_dir
from tests.implicitron.common_resources import get_skateboard_data

DATA_DIR = get_tests_dir() / "implicitron/data"
DEBUG: bool = False


class TestDataSource(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        torch.manual_seed(42)

        stack = contextlib.ExitStack()
        self.dataset_root, self.path_manager = stack.enter_context(
            get_skateboard_data()
        )
        self.addCleanup(stack.close)

    def _test_omegaconf_generic_failure(self):
        # OmegaConf possible bug - this is why we need _GenericWorkaround
        from dataclasses import dataclass

        import torch

        @dataclass
        class D(torch.utils.data.Dataset[int]):
            a: int = 3

        OmegaConf.structured(D)

    def _test_omegaconf_ListList(self):
        # Demo that OmegaConf doesn't support nested lists
        from dataclasses import dataclass
        from typing import Sequence

        @dataclass
        class A:
            a: Sequence[Sequence[int]] = ((32,),)

        OmegaConf.structured(A)

    def test_JsonIndexDataset_args(self):
        # test that JsonIndexDataset works with get_default_args
        get_default_args(JsonIndexDataset)

    def test_one(self):
        cfg = get_default_args(ImplicitronDataSource)
        # making the test invariant to env variables
        cfg.dataset_map_provider_JsonIndexDatasetMapProvider_args.dataset_root = ""
        cfg.dataset_map_provider_JsonIndexDatasetMapProviderV2_args.dataset_root = ""
        # making the test invariant to the presence of SQL dataset
        if "dataset_map_provider_SqlIndexDatasetMapProvider_args" in cfg:
            del cfg.dataset_map_provider_SqlIndexDatasetMapProvider_args
        yaml = OmegaConf.to_yaml(cfg, sort_keys=False)
        if DEBUG:
            (DATA_DIR / "data_source.yaml").write_text(yaml)
        self.assertEqual(yaml, (DATA_DIR / "data_source.yaml").read_text())

    def test_default(self):
        if os.environ.get("INSIDE_RE_WORKER") is not None:
            return
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "JsonIndexDatasetMapProvider"
        dataset_args = args.dataset_map_provider_JsonIndexDatasetMapProvider_args
        dataset_args.category = "skateboard"
        dataset_args.test_restrict_sequence_id = 0
        dataset_args.n_frames_per_sequence = -1

        dataset_args.dataset_root = self.dataset_root

        data_source = ImplicitronDataSource(**args)
        self.assertIsInstance(
            data_source.data_loader_map_provider, SequenceDataLoaderMapProvider
        )
        _, data_loaders = data_source.get_datasets_and_dataloaders()
        self.assertEqual(len(data_loaders.train), 81)
        for i in data_loaders.train:
            self.assertEqual(i.frame_type, ["test_known"])
            break

    def test_simple(self):
        if os.environ.get("INSIDE_RE_WORKER") is not None:
            return
        args = get_default_args(ImplicitronDataSource)
        args.dataset_map_provider_class_type = "JsonIndexDatasetMapProvider"
        args.data_loader_map_provider_class_type = "SimpleDataLoaderMapProvider"
        dataset_args = args.dataset_map_provider_JsonIndexDatasetMapProvider_args
        dataset_args.category = "skateboard"
        dataset_args.test_restrict_sequence_id = 0
        dataset_args.n_frames_per_sequence = -1

        dataset_args.dataset_root = self.dataset_root

        data_source = ImplicitronDataSource(**args)
        self.assertIsInstance(
            data_source.data_loader_map_provider, SimpleDataLoaderMapProvider
        )
        _, data_loaders = data_source.get_datasets_and_dataloaders()

        self.assertEqual(len(data_loaders.train), 81)
        for i in data_loaders.train:
            self.assertEqual(i.frame_type, ["test_known"])
            break
