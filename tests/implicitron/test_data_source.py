# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import unittest.mock

from omegaconf import OmegaConf
from pytorch3d.implicitron.dataset.data_source import ImplicitronDataSource
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.tools.config import get_default_args
from tests.common_testing import get_tests_dir

DATA_DIR = get_tests_dir() / "implicitron/data"
DEBUG: bool = False


class TestDataSource(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

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
        with unittest.mock.patch.dict(os.environ, {"CO3D_DATASET_ROOT": ""}):
            cfg = get_default_args(ImplicitronDataSource)
            yaml = OmegaConf.to_yaml(cfg, sort_keys=False)
            if DEBUG:
                (DATA_DIR / "data_source.yaml").write_text(yaml)
            self.assertEqual(yaml, (DATA_DIR / "data_source.yaml").read_text())
