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
from .utils import intercept_logs


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


def _parse_float_from_log(line):
    return float(line.split()[-1])


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_from_defaults(self):
        # Test making minimal changes to the dataclass defaults.
        if not interactive_testing_requested() or not internal:
            return

        # Manually override config values. Note that this is not necessary out-
        # side of the tests!
        cfg = OmegaConf.structured(experiment.Experiment)
        cfg.data_source_ImplicitronDataSource_args.dataset_map_provider_class_type = (
            "JsonIndexDatasetMapProvider"
        )
        dataset_args = (
            cfg.data_source_ImplicitronDataSource_args.dataset_map_provider_JsonIndexDatasetMapProvider_args
        )
        dataloader_args = (
            cfg.data_source_ImplicitronDataSource_args.data_loader_map_provider_SequenceDataLoaderMapProvider_args
        )
        dataset_args.category = "skateboard"
        dataset_args.test_restrict_sequence_id = 0
        dataset_args.dataset_root = "manifold://co3d/tree/extracted"
        dataset_args.dataset_JsonIndexDataset_args.limit_sequences_to = 5
        dataset_args.dataset_JsonIndexDataset_args.image_height = 80
        dataset_args.dataset_JsonIndexDataset_args.image_width = 80
        dataloader_args.dataset_length_train = 1
        dataloader_args.dataset_length_val = 1
        cfg.training_loop_ImplicitronTrainingLoop_args.max_epochs = 2
        cfg.training_loop_ImplicitronTrainingLoop_args.store_checkpoints = False
        cfg.optimizer_factory_ImplicitronOptimizerFactory_args.multistep_lr_milestones = [
            0,
            1,
        ]

        if DEBUG:
            experiment.dump_cfg(cfg)
        with intercept_logs(
            logger_name="projects.implicitron_trainer.impl.training_loop",
            regexp="LR change!",
        ) as intercepted_logs:
            experiment_runner = experiment.Experiment(**cfg)
            experiment_runner.run()

            # Make sure LR decreased on 0th and 1st epoch 10fold.
            self.assertEqual(intercepted_logs[0].split()[-1], "5e-06")

    def test_exponential_lr(self):
        # Test making minimal changes to the dataclass defaults.
        if not interactive_testing_requested():
            return
        cfg = OmegaConf.structured(experiment.Experiment)
        cfg.data_source_ImplicitronDataSource_args.dataset_map_provider_class_type = (
            "JsonIndexDatasetMapProvider"
        )
        dataset_args = (
            cfg.data_source_ImplicitronDataSource_args.dataset_map_provider_JsonIndexDatasetMapProvider_args
        )
        dataloader_args = (
            cfg.data_source_ImplicitronDataSource_args.data_loader_map_provider_SequenceDataLoaderMapProvider_args
        )
        dataset_args.category = "skateboard"
        dataset_args.test_restrict_sequence_id = 0
        dataset_args.dataset_root = "manifold://co3d/tree/extracted"
        dataset_args.dataset_JsonIndexDataset_args.limit_sequences_to = 5
        dataset_args.dataset_JsonIndexDataset_args.image_height = 80
        dataset_args.dataset_JsonIndexDataset_args.image_width = 80
        dataloader_args.dataset_length_train = 1
        dataloader_args.dataset_length_val = 1
        cfg.training_loop_ImplicitronTrainingLoop_args.max_epochs = 2
        cfg.training_loop_ImplicitronTrainingLoop_args.store_checkpoints = False
        cfg.optimizer_factory_ImplicitronOptimizerFactory_args.lr_policy = "Exponential"
        cfg.optimizer_factory_ImplicitronOptimizerFactory_args.exponential_lr_step_size = (
            2
        )

        if DEBUG:
            experiment.dump_cfg(cfg)
        with intercept_logs(
            logger_name="projects.implicitron_trainer.impl.training_loop",
            regexp="LR change!",
        ) as intercepted_logs:
            experiment_runner = experiment.Experiment(**cfg)
            experiment_runner.run()

            # Make sure we followed the exponential lr schedule with gamma=0.1,
            # exponential_lr_step_size=2 -- so after two epochs, should
            # decrease lr 10x to 5e-5.
            self.assertEqual(intercepted_logs[0].split()[-1], "0.00015811388300841897")
            self.assertEqual(intercepted_logs[1].split()[-1], "5e-05")

    def test_yaml_contents(self):
        # Check that the default config values, defined by Experiment and its
        # members, is what we expect it to be.
        cfg = OmegaConf.structured(experiment.Experiment)
        yaml = OmegaConf.to_yaml(cfg, sort_keys=False)
        if DEBUG:
            (DATA_DIR / "experiment.yaml").write_text(yaml)
        self.assertEqual(yaml, (DATA_DIR / "experiment.yaml").read_text())

    def test_load_configs(self):
        # Check that all the pre-prepared configs are valid.
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


class TestNerfRepro(unittest.TestCase):
    @unittest.skip("This runs full NeRF training on Blender data.")
    def test_nerf_blender(self):
        # Train vanilla NERF.
        # Set env vars BLENDER_DATASET_ROOT and BLENDER_SINGLESEQ_CLASS first!
        if not interactive_testing_requested():
            return
        with initialize_config_dir(config_dir=str(IMPLICITRON_CONFIGS_DIR)):
            cfg = compose(config_name="repro_singleseq_nerf_blender", overrides=[])
            experiment_runner = experiment.Experiment(**cfg)
            experiment.dump_cfg(cfg)
            experiment_runner.run()
