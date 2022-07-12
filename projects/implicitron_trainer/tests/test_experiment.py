# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .. import experiment


def interactive_testing_requested() -> bool:
    """
    Certain tests are only useful when run interactively, and so are not regularly run.
    These are activated by this funciton returning True, which the user requests by
    setting the environment variable `PYTORCH3D_INTERACTIVE_TESTING` to 1.
    """
    return os.environ.get("PYTORCH3D_INTERACTIVE_TESTING", "") == "1"


internal = os.environ.get("FB_TEST", False)


DATA_DIR = Path(__file__).resolve().parent
IMPLICITRON_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEBUG: bool = False

# TODO:
# - add enough files to skateboard_first_5 that this works on RE.
# - share common code with PyTorch3D tests?
# - deal with the temporary output files this test creates


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_from_defaults(self):
        # Test making minimal changes to the dataclass defaults.
        if not interactive_testing_requested() or not internal:
            return
        cfg = OmegaConf.structured(experiment.ExperimentConfig)
        cfg.data_source_args.dataset_map_provider_class_type = (
            "JsonIndexDatasetMapProvider"
        )
        dataset_args = (
            cfg.data_source_args.dataset_map_provider_JsonIndexDatasetMapProvider_args
        )
        dataloader_args = (
            cfg.data_source_args.data_loader_map_provider_SequenceDataLoaderMapProvider_args
        )
        dataset_args.category = "skateboard"
        dataset_args.test_restrict_sequence_id = 0
        dataset_args.dataset_root = "manifold://co3d/tree/extracted"
        dataset_args.dataset_JsonIndexDataset_args.limit_sequences_to = 5
        dataset_args.dataset_JsonIndexDataset_args.image_height = 80
        dataset_args.dataset_JsonIndexDataset_args.image_width = 80
        dataloader_args.dataset_length_train = 1
        dataloader_args.dataset_length_val = 1
        cfg.solver_args.max_epochs = 2

        experiment.run_training(cfg)

    def test_yaml_contents(self):
        cfg = OmegaConf.structured(experiment.ExperimentConfig)
        yaml = OmegaConf.to_yaml(cfg, sort_keys=False)
        if DEBUG:
            (DATA_DIR / "experiment.yaml").write_text(yaml)
        self.assertEqual(yaml, (DATA_DIR / "experiment.yaml").read_text())

    def test_load_configs(self):
        config_files = []

        for pattern in ("repro_singleseq*.yaml", "repro_multiseq*.yaml"):
            config_files.extend(
                [
                    f
                    for f in IMPLICITRON_CONFIGS_DIR.glob(pattern)
                    if not f.name.endswith("_base.yaml")
                ]
            )

        for file in config_files:
            with self.subTest(file.name):
                with initialize_config_dir(config_dir=str(IMPLICITRON_CONFIGS_DIR)):
                    compose(file.name)
