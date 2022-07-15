# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import field
from typing import Any, Dict, Tuple

from omegaconf import DictConfig
from pytorch3d.implicitron.dataset.data_source import ImplicitronDataSource
from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.tools.config import Configurable, get_default_args_field

from .optimization import init_optimizer


class ExperimentConfig(Configurable):
    generic_model_args: DictConfig = get_default_args_field(GenericModel)
    solver_args: DictConfig = get_default_args_field(init_optimizer)
    data_source_args: DictConfig = get_default_args_field(ImplicitronDataSource)
    architecture: str = "generic"
    detect_anomaly: bool = False
    eval_only: bool = False
    exp_dir: str = "./data/default_experiment/"
    exp_idx: int = 0
    gpu_idx: int = 0
    metric_print_interval: int = 5
    resume: bool = True
    resume_epoch: int = -1
    seed: int = 0
    store_checkpoints: bool = True
    store_checkpoints_purge: int = 1
    test_interval: int = -1
    test_when_finished: bool = False
    validation_interval: int = 1
    visdom_env: str = ""
    visdom_port: int = 8097
    visdom_server: str = "http://127.0.0.1"
    visualize_interval: int = 1000
    clip_grad: float = 0.0
    camera_difficulty_bin_breaks: Tuple[float, ...] = 0.97, 0.98

    hydra: Dict[str, Any] = field(
        default_factory=lambda: {
            "run": {"dir": "."},  # Make hydra not change the working dir.
            "output_subdir": None,  # disable storing the .hydra logs
        }
    )
