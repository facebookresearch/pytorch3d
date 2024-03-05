#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Script to visualize a previously trained model. Example call:

    pytorch3d_implicitron_visualizer \
    exp_dir='./exps/checkpoint_dir' visdom_show_preds=True visdom_port=8097 \
    n_eval_cameras=40 render_size="[64,64]" video_size="[256,256]"
"""

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch3d.implicitron.models.visualization.render_flyaround import render_flyaround
from pytorch3d.implicitron.tools.config import enable_get_default_args, get_default_args

from .experiment import Experiment


def visualize_reconstruction(
    exp_dir: str = "",
    restrict_sequence_name: Optional[str] = None,
    output_directory: Optional[str] = None,
    render_size: Tuple[int, int] = (512, 512),
    video_size: Optional[Tuple[int, int]] = None,
    split: str = "train",
    n_source_views: int = 9,
    n_eval_cameras: int = 40,
    visdom_show_preds: bool = False,
    visdom_server: str = "http://127.0.0.1",
    visdom_port: int = 8097,
    visdom_env: Optional[str] = None,
    **render_flyaround_kwargs,
) -> None:
    """
    Given an `exp_dir` containing a trained Implicitron model, generates videos consisting
    of renderes of sequences from the dataset used to train and evaluate the trained
    Implicitron model.

    Args:
        exp_dir: Implicitron experiment directory.
        restrict_sequence_name: If set, defines the list of sequences to visualize.
        output_directory: If set, defines a custom directory to output visualizations to.
        render_size: The size (HxW) of the generated renders.
        video_size: The size (HxW) of the output video.
        split: The dataset split to use for visualization.
            Can be "train" / "val" / "test".
        n_source_views: The number of source views added to each rendered batch. These
            views are required inputs for models such as NeRFormer / NeRF-WCE.
        n_eval_cameras: The number of cameras each fly-around trajectory.
        visdom_show_preds: If `True`, outputs visualizations to visdom.
        visdom_server: The address of the visdom server.
        visdom_port: The port of the visdom server.
        visdom_env: If set, defines a custom name for the visdom environment.
        render_flyaround_kwargs: Keyword arguments passed to the invoked `render_flyaround`
            function (see `pytorch3d.implicitron.models.visualization.render_flyaround`).
    """

    # In case an output directory is specified use it. If no output_directory
    # is specified create a vis folder inside the experiment directory
    if output_directory is None:
        output_directory = os.path.join(exp_dir, "vis")
    os.makedirs(output_directory, exist_ok=True)

    # Set the random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Get the config from the experiment_directory,
    # and overwrite relevant fields
    config = _get_config_from_experiment_directory(exp_dir)
    config.exp_dir = exp_dir
    # important so that the CO3D dataset gets loaded in full
    data_source_args = config.data_source_ImplicitronDataSource_args
    if "dataset_map_provider_JsonIndexDatasetMapProvider_args" in data_source_args:
        dataset_args = (
            data_source_args.dataset_map_provider_JsonIndexDatasetMapProvider_args
        )
        dataset_args.test_on_train = False
        if restrict_sequence_name is not None:
            dataset_args.restrict_sequence_name = restrict_sequence_name

    # Set the rendering image size
    model_factory_args = config.model_factory_ImplicitronModelFactory_args
    model_factory_args.force_resume = True
    model_args = model_factory_args.model_GenericModel_args
    model_args.render_image_width = render_size[0]
    model_args.render_image_height = render_size[1]

    # Load the previously trained model
    experiment = Experiment(**config)
    model = experiment.model_factory(exp_dir=exp_dir)
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    # Setup the dataset
    data_source = experiment.data_source
    dataset_map, _ = data_source.get_datasets_and_dataloaders()
    dataset = dataset_map[split]
    if dataset is None:
        raise ValueError(f"{split} dataset not provided")

    if visdom_env is None:
        visdom_env = (
            "visualizer_" + config.training_loop_ImplicitronTrainingLoop_args.visdom_env
        )

    # iterate over the sequences in the dataset
    for sequence_name in dataset.sequence_names():
        with torch.no_grad():
            render_kwargs = {
                "dataset": dataset,
                "sequence_name": sequence_name,
                "model": model,
                "output_video_path": os.path.join(output_directory, "video"),
                "n_source_views": n_source_views,
                "visdom_show_preds": visdom_show_preds,
                "n_flyaround_poses": n_eval_cameras,
                "visdom_server": visdom_server,
                "visdom_port": visdom_port,
                "visdom_environment": visdom_env,
                "video_resize": video_size,
                "device": device,
                **render_flyaround_kwargs,
            }
            render_flyaround(**render_kwargs)


enable_get_default_args(visualize_reconstruction)


def _get_config_from_experiment_directory(experiment_directory) -> DictConfig:
    cfg_file = os.path.join(experiment_directory, "expconfig.yaml")
    config = OmegaConf.load(cfg_file)
    # pyre-ignore[7]
    return OmegaConf.merge(get_default_args(Experiment), config)


def main(argv=sys.argv) -> None:
    # automatically parses arguments of visualize_reconstruction
    cfg = OmegaConf.create(get_default_args(visualize_reconstruction))
    cfg.update(OmegaConf.from_cli(argv))
    with torch.no_grad():
        visualize_reconstruction(**cfg)


if __name__ == "__main__":
    main()
