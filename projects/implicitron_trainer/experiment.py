#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

""""
This file is the entry point for launching experiments with Implicitron.

Launch Training
---------------
Experiment config .yaml files are located in the
`projects/implicitron_trainer/configs` folder. To launch an experiment,
specify the name of the file. Specific config values can also be overridden
from the command line, for example:

```
./experiment.py --config-name base_config.yaml override.param.one=42 override.param.two=84
```

Main functions
---------------
- The Experiment class defines `run` which creates the model, optimizer, and other
  objects used in training, then starts TrainingLoop's `run` function.
- TrainingLoop takes care of the actual training logic: forward and backward passes,
  evaluation and testing, as well as model checkpointing, visualization, and metric
  printing.

Outputs
--------
The outputs of the experiment are saved and logged in multiple ways:
  - Checkpoints:
        Model, optimizer and stats are stored in the directory
        named by the `exp_dir` key from the config file / CLI parameters.
  - Stats
        Stats are logged and plotted to the file "train_stats.pdf" in the
        same directory. The stats are also saved as part of the checkpoint file.
  - Visualizations
        Predictions are plotted to a visdom server running at the
        port specified by the `visdom_server` and `visdom_port` keys in the
        config file.

"""
import logging
import os
import warnings

from dataclasses import field

import hydra

import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from packaging import version

from pytorch3d.implicitron.dataset.data_source import (
    DataSourceBase,
    ImplicitronDataSource,
)
from pytorch3d.implicitron.models.base_model import ImplicitronModelBase

from pytorch3d.implicitron.models.renderer.multipass_ea import (
    MultiPassEmissionAbsorptionRenderer,
)
from pytorch3d.implicitron.models.renderer.ray_sampler import AdaptiveRaySampler
from pytorch3d.implicitron.tools.config import (
    Configurable,
    expand_args_fields,
    remove_unused_components,
    run_auto_creation,
)

from .impl.model_factory import ModelFactoryBase
from .impl.optimizer_factory import OptimizerFactoryBase
from .impl.training_loop import TrainingLoopBase
from .impl.utils import seed_all_random_engines

logger = logging.getLogger(__name__)

# workaround for https://github.com/facebookresearch/hydra/issues/2262
_RUN = hydra.types.RunMode.RUN

if version.parse(hydra.__version__) < version.Version("1.1"):
    raise ValueError(
        f"Hydra version {hydra.__version__} is too old."
        " (Implicitron requires version 1.1 or later.)"
    )

try:
    # only makes sense in FAIR cluster
    import pytorch3d.implicitron.fair_cluster.slurm  # noqa: F401
except ModuleNotFoundError:
    pass

no_accelerate = os.environ.get("PYTORCH3D_NO_ACCELERATE") is not None


class Experiment(Configurable):  # pyre-ignore: 13
    """
    This class is at the top level of Implicitron's config hierarchy. Its
    members are high-level components necessary for training an implicit rende-
    ring network.

    Members:
        data_source: An object that produces datasets and dataloaders.
        model_factory: An object that produces an implicit rendering model as
            well as its corresponding Stats object.
        optimizer_factory: An object that produces the optimizer and lr
            scheduler.
        training_loop: An object that runs training given the outputs produced
            by the data_source, model_factory and optimizer_factory.
        seed: A random seed to ensure reproducibility.
        detect_anomaly: Whether torch.autograd should detect anomalies. Useful
            for debugging, but might slow down the training.
        exp_dir: Root experimentation directory. Checkpoints and training stats
            will be saved here.
    """

    data_source: DataSourceBase
    data_source_class_type: str = "ImplicitronDataSource"
    model_factory: ModelFactoryBase
    model_factory_class_type: str = "ImplicitronModelFactory"
    optimizer_factory: OptimizerFactoryBase
    optimizer_factory_class_type: str = "ImplicitronOptimizerFactory"
    training_loop: TrainingLoopBase
    training_loop_class_type: str = "ImplicitronTrainingLoop"

    seed: int = 42
    detect_anomaly: bool = False
    exp_dir: str = "./data/default_experiment/"

    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},  # Make hydra not change the working dir.
            "output_subdir": None,  # disable storing the .hydra logs
            "mode": _RUN,
        }
    )

    def __post_init__(self):
        seed_all_random_engines(
            self.seed
        )  # Set all random engine seeds for reproducibility

        run_auto_creation(self)

    def run(self) -> None:
        # Initialize the accelerator if desired.
        if no_accelerate:
            accelerator = None
            device = torch.device("cuda:0")
        else:
            accelerator = Accelerator(device_placement=False)
            logger.info(accelerator.state)
            device = accelerator.device

        logger.info(f"Running experiment on device: {device}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # set the debug mode
        if self.detect_anomaly:
            logger.info("Anomaly detection!")
        torch.autograd.set_detect_anomaly(self.detect_anomaly)

        # Initialize the datasets and dataloaders.
        datasets, dataloaders = self.data_source.get_datasets_and_dataloaders()

        # Init the model and the corresponding Stats object.
        model = self.model_factory(
            accelerator=accelerator,
            exp_dir=self.exp_dir,
        )

        stats = self.training_loop.load_stats(
            log_vars=model.log_vars,
            exp_dir=self.exp_dir,
            resume=self.model_factory.resume,
            resume_epoch=self.model_factory.resume_epoch,  # pyre-ignore [16]
        )
        start_epoch = stats.epoch + 1

        model.to(device)

        # Init the optimizer and LR scheduler.
        optimizer, scheduler = self.optimizer_factory(
            accelerator=accelerator,
            exp_dir=self.exp_dir,
            last_epoch=start_epoch,
            model=model,
            resume=self.model_factory.resume,
            resume_epoch=self.model_factory.resume_epoch,
        )

        # Wrap all modules in the distributed library
        # Note: we don't pass the scheduler to prepare as it
        # doesn't need to be stepped at each optimizer step
        train_loader = dataloaders.train
        val_loader = dataloaders.val
        test_loader = dataloaders.test
        if accelerator is not None:
            (
                model,
                optimizer,
                train_loader,
                val_loader,
            ) = accelerator.prepare(model, optimizer, train_loader, val_loader)

        # Enter the main training loop.
        self.training_loop.run(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            # pyre-ignore[6]
            train_dataset=datasets.train,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            accelerator=accelerator,
            device=device,
            exp_dir=self.exp_dir,
            stats=stats,
            seed=self.seed,
        )


def _setup_envvars_for_cluster() -> bool:
    """
    Prepares to run on cluster if relevant.
    Returns whether FAIR cluster in use.
    """
    # TODO: How much of this is needed in general?

    try:
        import submitit
    except ImportError:
        return False

    try:
        # Only needed when launching on cluster with slurm and submitit
        job_env = submitit.JobEnvironment()
    except RuntimeError:
        return False

    os.environ["LOCAL_RANK"] = str(job_env.local_rank)
    os.environ["RANK"] = str(job_env.global_rank)
    os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "42918"
    logger.info(
        "Num tasks %s, global_rank %s"
        % (str(job_env.num_tasks), str(job_env.global_rank))
    )

    return True


def dump_cfg(cfg: DictConfig) -> None:
    remove_unused_components(cfg)
    # dump the exp config to the exp dir
    os.makedirs(cfg.exp_dir, exist_ok=True)
    try:
        cfg_filename = os.path.join(cfg.exp_dir, "expconfig.yaml")
        OmegaConf.save(config=cfg, f=cfg_filename)
    except PermissionError:
        warnings.warn("Can't dump config due to insufficient permissions!")


expand_args_fields(Experiment)
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config", node=Experiment)


@hydra.main(config_path="./configs/", config_name="default_config")
def experiment(cfg: DictConfig) -> None:
    # CUDA_VISIBLE_DEVICES must have been set.

    if "CUDA_DEVICE_ORDER" not in os.environ:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if not _setup_envvars_for_cluster():
        logger.info("Running locally")

    # TODO: The following may be needed for hydra/submitit it to work
    expand_args_fields(ImplicitronModelBase)
    expand_args_fields(AdaptiveRaySampler)
    expand_args_fields(MultiPassEmissionAbsorptionRenderer)
    expand_args_fields(ImplicitronDataSource)

    experiment = Experiment(**cfg)
    dump_cfg(cfg)
    experiment.run()


if __name__ == "__main__":
    experiment()
