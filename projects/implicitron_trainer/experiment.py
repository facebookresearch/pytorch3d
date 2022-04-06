#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

""""
This file is the entry point for launching experiments with Implicitron.

Main functions
---------------
- `run_training` is the wrapper for the train, val, test loops
    and checkpointing
- `trainvalidate` is the inner loop which runs the model forward/backward
    pass, visualizations and metric printing

Launch Training
---------------
Experiment config .yaml files are located in the
`projects/implicitron_trainer/configs` folder. To launch
an experiment, specify the name of the file. Specific config values can
also be overridden from the command line, for example:

```
./experiment.py --config-name base_config.yaml override.param.one=42 override.param.two=84
```

To run an experiment on a specific GPU, specify the `gpu_idx` key
in the config file / CLI. To run on a different device, specify the
device in `run_training`.

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
        Prredictions are plotted to a visdom server running at the
        port specified by the `visdom_server` and `visdom_port` keys in the
        config file.

"""

import copy
import json
import logging
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import hydra
import lpips
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf
from packaging import version
from pytorch3d.implicitron.dataset import utils as ds_utils
from pytorch3d.implicitron.dataset.dataloader_zoo import dataloader_zoo
from pytorch3d.implicitron.dataset.dataset_zoo import dataset_zoo
from pytorch3d.implicitron.dataset.implicitron_dataset import (
    FrameData,
    ImplicitronDataset,
)
from pytorch3d.implicitron.evaluation import evaluate_new_view_synthesis as evaluate
from pytorch3d.implicitron.models.base import EvaluationMode, GenericModel
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.implicitron.tools.config import (
    enable_get_default_args,
    get_default_args_field,
    remove_unused_components,
)
from pytorch3d.implicitron.tools.stats import Stats
from pytorch3d.renderer.cameras import CamerasBase


logger = logging.getLogger(__name__)

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


def init_model(
    cfg: DictConfig,
    force_load: bool = False,
    clear_stats: bool = False,
    load_model_only: bool = False,
) -> Tuple[GenericModel, Stats, Optional[Dict[str, Any]]]:
    """
    Returns an instance of `GenericModel`.

    If `cfg.resume` is set or `force_load` is true,
    attempts to load the last checkpoint from `cfg.exp_dir`. Failure to do so
    will return the model with initial weights, unless `force_load` is passed,
    in which case a FileNotFoundError is raised.

    Args:
        force_load: If true, force load model from checkpoint even if
            cfg.resume is false.
        clear_stats: If true, clear the stats object loaded from checkpoint
        load_model_only: If true, load only the model weights from checkpoint
            and do not load the state of the optimizer and stats.

    Returns:
        model: The model with optionally loaded weights from checkpoint
        stats: The stats structure (optionally loaded from checkpoint)
        optimizer_state: The optimizer state dict containing
            `state` and `param_groups` keys (optionally loaded from checkpoint)

    Raise:
        FileNotFoundError if `force_load` is passed but checkpoint is not found.
    """

    # Initialize the model
    if cfg.architecture == "generic":
        model = GenericModel(**cfg.generic_model_args)
    else:
        raise ValueError(f"No such arch {cfg.architecture}.")

    # Determine the network outputs that should be logged
    if hasattr(model, "log_vars"):
        log_vars = copy.deepcopy(list(model.log_vars))
    else:
        log_vars = ["objective"]

    visdom_env_charts = vis_utils.get_visdom_env(cfg) + "_charts"

    # Init the stats struct
    stats = Stats(
        log_vars,
        visdom_env=visdom_env_charts,
        verbose=False,
        visdom_server=cfg.visdom_server,
        visdom_port=cfg.visdom_port,
    )

    # Retrieve the last checkpoint
    if cfg.resume_epoch > 0:
        model_path = model_io.get_checkpoint(cfg.exp_dir, cfg.resume_epoch)
    else:
        model_path = model_io.find_last_checkpoint(cfg.exp_dir)

    optimizer_state = None
    if model_path is not None:
        logger.info("found previous model %s" % model_path)
        if force_load or cfg.resume:
            logger.info("   -> resuming")
            if load_model_only:
                model_state_dict = torch.load(model_io.get_model_path(model_path))
                stats_load, optimizer_state = None, None
            else:
                model_state_dict, stats_load, optimizer_state = model_io.load_model(
                    model_path
                )

                # Determine if stats should be reset
                if not clear_stats:
                    if stats_load is None:
                        logger.info("\n\n\n\nCORRUPT STATS -> clearing stats\n\n\n\n")
                        last_epoch = model_io.parse_epoch_from_model_path(model_path)
                        logger.info(f"Estimated resume epoch = {last_epoch}")

                        # Reset the stats struct
                        for _ in range(last_epoch + 1):
                            stats.new_epoch()
                        assert last_epoch == stats.epoch
                    else:
                        stats = stats_load

                    # Update stats properties incase it was reset on load
                    stats.visdom_env = visdom_env_charts
                    stats.visdom_server = cfg.visdom_server
                    stats.visdom_port = cfg.visdom_port
                    stats.plot_file = os.path.join(cfg.exp_dir, "train_stats.pdf")
                    stats.synchronize_logged_vars(log_vars)
                else:
                    logger.info("   -> clearing stats")

            try:
                # TODO: fix on creation of the buffers
                # after the hack above, this will not pass in most cases
                # ... but this is fine for now
                model.load_state_dict(model_state_dict, strict=True)
            except RuntimeError as e:
                logger.error(e)
                logger.info("Cant load state dict in strict mode! -> trying non-strict")
                model.load_state_dict(model_state_dict, strict=False)
            model.log_vars = log_vars
        else:
            logger.info("   -> but not resuming -> starting from scratch")
    elif force_load:
        raise FileNotFoundError(f"Cannot find a checkpoint in {cfg.exp_dir}!")

    return model, stats, optimizer_state


def init_optimizer(
    model: GenericModel,
    optimizer_state: Optional[Dict[str, Any]],
    last_epoch: int,
    breed: str = "adam",
    weight_decay: float = 0.0,
    lr_policy: str = "multistep",
    lr: float = 0.0005,
    gamma: float = 0.1,
    momentum: float = 0.9,
    betas: Tuple[float] = (0.9, 0.999),
    milestones: tuple = (),
    max_epochs: int = 1000,
):
    """
    Initialize the optimizer (optionally from checkpoint state)
    and the learning rate scheduler.

    Args:
        model: The model with optionally loaded weights
        optimizer_state: The state dict for the optimizer. If None
            it has not been loaded from checkpoint
        last_epoch: If the model was loaded from checkpoint this will be the
            number of the last epoch that was saved
        breed: The type of optimizer to use e.g. adam
        weight_decay: The optimizer weight_decay (L2 penalty on model weights)
        lr_policy: The policy to use for learning rate. Currently, only "multistep:
            is supported.
        lr: The value for the initial learning rate
        gamma: Multiplicative factor of learning rate decay
        momentum: Momentum factor for SGD optimizer
        betas: Coefficients used for computing running averages of gradient and its square
            in the Adam optimizer
        milestones: List of increasing epoch indices at which the learning rate is
            modified
        max_epochs: The maximum number of epochs to run the optimizer for

    Returns:
        optimizer: Optimizer module, optionally loaded from checkpoint
        scheduler: Learning rate scheduler module

    Raise:
        ValueError if `breed` or `lr_policy` are not supported.
    """

    # Get the parameters to optimize
    if hasattr(model, "_get_param_groups"):  # use the model function
        p_groups = model._get_param_groups(lr, wd=weight_decay)
    else:
        allprm = [prm for prm in model.parameters() if prm.requires_grad]
        p_groups = [{"params": allprm, "lr": lr}]

    # Intialize the optimizer
    if breed == "sgd":
        optimizer = torch.optim.SGD(
            p_groups, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif breed == "adagrad":
        optimizer = torch.optim.Adagrad(p_groups, lr=lr, weight_decay=weight_decay)
    elif breed == "adam":
        optimizer = torch.optim.Adam(
            p_groups, lr=lr, betas=betas, weight_decay=weight_decay
        )
    else:
        raise ValueError("no such solver type %s" % breed)
    logger.info("  -> solver type = %s" % breed)

    # Load state from checkpoint
    if optimizer_state is not None:
        logger.info("  -> setting loaded optimizer state")
        optimizer.load_state_dict(optimizer_state)

    # Initialize the learning rate scheduler
    if lr_policy == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    else:
        raise ValueError("no such lr policy %s" % lr_policy)

    # When loading from checkpoint, this will make sure that the
    # lr is correctly set even after returning
    for _ in range(last_epoch):
        scheduler.step()

    # Add the max epochs here
    scheduler.max_epochs = max_epochs

    optimizer.zero_grad()
    return optimizer, scheduler


enable_get_default_args(init_optimizer)


def trainvalidate(
    model,
    stats,
    epoch,
    loader,
    optimizer,
    validation,
    bp_var: str = "objective",
    metric_print_interval: int = 5,
    visualize_interval: int = 100,
    visdom_env_root: str = "trainvalidate",
    clip_grad: float = 0.0,
    device: str = "cuda:0",
    **kwargs,
) -> None:
    """
    This is the main loop for training and evaluation including:
    model forward pass, loss computation, backward pass and visualization.

    Args:
        model: The model module optionally loaded from checkpoint
        stats: The stats struct, also optionally loaded from checkpoint
        epoch: The index of the current epoch
        loader: The dataloader to use for the loop
        optimizer: The optimizer module optionally loaded from checkpoint
        validation: If true, run the loop with the model in eval mode
            and skip the backward pass
        bp_var: The name of the key in the model output `preds` dict which
            should be used as the loss for the backward pass.
        metric_print_interval: The batch interval at which the stats should be
            logged.
        visualize_interval: The batch interval at which the visualizations
            should be plotted
        visdom_env_root: The name of the visdom environment to use for plotting
        clip_grad: Optionally clip the gradient norms.
            If set to a value <=0.0, no clipping
        device: The device on which to run the model.

    Returns:
        None
    """

    if validation:
        model.eval()
        trainmode = "val"
    else:
        model.train()
        trainmode = "train"

    t_start = time.time()

    # get the visdom env name
    visdom_env_imgs = visdom_env_root + "_images_" + trainmode
    viz = vis_utils.get_visdom_connection(
        server=stats.visdom_server,
        port=stats.visdom_port,
    )

    # Iterate through the batches
    n_batches = len(loader)
    for it, batch in enumerate(loader):
        last_iter = it == n_batches - 1

        # move to gpu where possible (in place)
        net_input = batch.to(device)

        # run the forward pass
        if not validation:
            optimizer.zero_grad()
            preds = model(**{**net_input, "evaluation_mode": EvaluationMode.TRAINING})
        else:
            with torch.no_grad():
                preds = model(
                    **{**net_input, "evaluation_mode": EvaluationMode.EVALUATION}
                )

        # make sure we dont overwrite something
        assert all(k not in preds for k in net_input.keys())
        # merge everything into one big dict
        preds.update(net_input)

        # update the stats logger
        stats.update(preds, time_start=t_start, stat_set=trainmode)
        assert stats.it[trainmode] == it, "inconsistent stat iteration number!"

        # print textual status update
        if it % metric_print_interval == 0 or last_iter:
            stats.print(stat_set=trainmode, max_it=n_batches)

        # visualize results
        if visualize_interval > 0 and it % visualize_interval == 0:
            prefix = f"e{stats.epoch}_it{stats.it[trainmode]}"

            model.visualize(
                viz,
                visdom_env_imgs,
                preds,
                prefix,
            )

        # optimizer step
        if not validation:
            loss = preds[bp_var]
            assert torch.isfinite(loss).all(), "Non-finite loss!"
            # backprop
            loss.backward()
            if clip_grad > 0.0:
                # Optionally clip the gradient norms.
                total_norm = torch.nn.utils.clip_grad_norm(
                    model.parameters(), clip_grad
                )
                if total_norm > clip_grad:
                    logger.info(
                        f"Clipping gradient: {total_norm}"
                        + f" with coef {clip_grad / total_norm}."
                    )

            optimizer.step()


def run_training(cfg: DictConfig, device: str = "cpu"):
    """
    Entry point to run the training and validation loops
    based on the specified config file.
    """

    # set the debug mode
    if cfg.detect_anomaly:
        logger.info("Anomaly detection!")
    torch.autograd.set_detect_anomaly(cfg.detect_anomaly)

    # create the output folder
    os.makedirs(cfg.exp_dir, exist_ok=True)
    _seed_all_random_engines(cfg.seed)
    remove_unused_components(cfg)

    # dump the exp config to the exp dir
    try:
        cfg_filename = os.path.join(cfg.exp_dir, "expconfig.yaml")
        OmegaConf.save(config=cfg, f=cfg_filename)
    except PermissionError:
        warnings.warn("Cant dump config due to insufficient permissions!")

    # setup datasets
    datasets = dataset_zoo(**cfg.dataset_args)
    cfg.dataloader_args["dataset_name"] = cfg.dataset_args["dataset_name"]
    dataloaders = dataloader_zoo(datasets, **cfg.dataloader_args)

    # init the model
    model, stats, optimizer_state = init_model(cfg)
    start_epoch = stats.epoch + 1

    # move model to gpu
    model.to(device)

    # only run evaluation on the test dataloader
    if cfg.eval_only:
        _eval_and_dump(cfg, datasets, dataloaders, model, stats, device=device)
        return

    # init the optimizer
    optimizer, scheduler = init_optimizer(
        model,
        optimizer_state=optimizer_state,
        last_epoch=start_epoch,
        **cfg.solver_args,
    )

    # check the scheduler and stats have been initialized correctly
    assert scheduler.last_epoch == stats.epoch + 1
    assert scheduler.last_epoch == start_epoch

    past_scheduler_lrs = []
    # loop through epochs
    for epoch in range(start_epoch, cfg.solver_args.max_epochs):
        # automatic new_epoch and plotting of stats at every epoch start
        with stats:

            # Make sure to re-seed random generators to ensure reproducibility
            # even after restart.
            _seed_all_random_engines(cfg.seed + epoch)

            cur_lr = float(scheduler.get_last_lr()[-1])
            logger.info(f"scheduler lr = {cur_lr:1.2e}")
            past_scheduler_lrs.append(cur_lr)

            # train loop
            trainvalidate(
                model,
                stats,
                epoch,
                dataloaders["train"],
                optimizer,
                False,
                visdom_env_root=vis_utils.get_visdom_env(cfg),
                device=device,
                **cfg,
            )

            # val loop (optional)
            if "val" in dataloaders and epoch % cfg.validation_interval == 0:
                trainvalidate(
                    model,
                    stats,
                    epoch,
                    dataloaders["val"],
                    optimizer,
                    True,
                    visdom_env_root=vis_utils.get_visdom_env(cfg),
                    device=device,
                    **cfg,
                )

            # eval loop (optional)
            if (
                "test" in dataloaders
                and cfg.test_interval > 0
                and epoch % cfg.test_interval == 0
            ):
                run_eval(cfg, model, stats, dataloaders["test"], device=device)

            assert stats.epoch == epoch, "inconsistent stats!"

            # delete previous models if required
            # save model
            if cfg.store_checkpoints:
                if cfg.store_checkpoints_purge > 0:
                    for prev_epoch in range(epoch - cfg.store_checkpoints_purge):
                        model_io.purge_epoch(cfg.exp_dir, prev_epoch)
                outfile = model_io.get_checkpoint(cfg.exp_dir, epoch)
                model_io.safe_save_model(model, stats, outfile, optimizer=optimizer)

            scheduler.step()

            new_lr = float(scheduler.get_last_lr()[-1])
            if new_lr != cur_lr:
                logger.info(f"LR change! {cur_lr} -> {new_lr}")

    if cfg.test_when_finished:
        _eval_and_dump(cfg, datasets, dataloaders, model, stats, device=device)


def _eval_and_dump(cfg, datasets, dataloaders, model, stats, device):
    """
    Run the evaluation loop with the test data loader and
    save the predictions to the `exp_dir`.
    """

    if "test" not in dataloaders:
        raise ValueError('Dataloaders have to contain the "test" entry for eval!')

    eval_task = cfg.dataset_args["dataset_name"].split("_")[-1]
    all_source_cameras = (
        _get_all_source_cameras(datasets["train"])
        if eval_task == "singlesequence"
        else None
    )
    results = run_eval(
        cfg, model, all_source_cameras, dataloaders["test"], eval_task, device=device
    )

    # add the evaluation epoch to the results
    for r in results:
        r["eval_epoch"] = int(stats.epoch)

    logger.info("Evaluation results")
    evaluate.pretty_print_nvs_metrics(results)

    with open(os.path.join(cfg.exp_dir, "results_test.json"), "w") as f:
        json.dump(results, f)


def _get_eval_frame_data(frame_data):
    """
    Masks the unknown image data to make sure we cannot use it at model evaluation time.
    """
    frame_data_for_eval = copy.deepcopy(frame_data)
    is_known = ds_utils.is_known_frame(frame_data.frame_type).type_as(
        frame_data.image_rgb
    )[:, None, None, None]
    for k in ("image_rgb", "depth_map", "fg_probability", "mask_crop"):
        value_masked = getattr(frame_data_for_eval, k).clone() * is_known
        setattr(frame_data_for_eval, k, value_masked)
    return frame_data_for_eval


def run_eval(cfg, model, all_source_cameras, loader, task, device):
    """
    Run the evaluation loop on the test dataloader
    """
    lpips_model = lpips.LPIPS(net="vgg")
    lpips_model = lpips_model.to(device)

    model.eval()

    per_batch_eval_results = []
    logger.info("Evaluating model ...")
    for frame_data in tqdm.tqdm(loader):
        frame_data = frame_data.to(device)

        # mask out the unknown images so that the model does not see them
        frame_data_for_eval = _get_eval_frame_data(frame_data)

        with torch.no_grad():
            preds = model(
                **{**frame_data_for_eval, "evaluation_mode": EvaluationMode.EVALUATION}
            )
            nvs_prediction = copy.deepcopy(preds["nvs_prediction"])
            per_batch_eval_results.append(
                evaluate.eval_batch(
                    frame_data,
                    nvs_prediction,
                    bg_color="black",
                    lpips_model=lpips_model,
                    source_cameras=all_source_cameras,
                )
            )

    _, category_result = evaluate.summarize_nvs_eval_results(
        per_batch_eval_results, task
    )

    return category_result["results"]


def _get_all_source_cameras(
    dataset: ImplicitronDataset,
    num_workers: int = 8,
) -> CamerasBase:
    """
    Load and return all the source cameras in the training dataset
    """

    all_frame_data = next(
        iter(
            torch.utils.data.DataLoader(
                dataset,
                shuffle=False,
                batch_size=len(dataset),
                num_workers=num_workers,
                collate_fn=FrameData.collate,
            )
        )
    )

    is_source = ds_utils.is_known_frame(all_frame_data.frame_type)
    source_cameras = all_frame_data.camera[torch.where(is_source)[0]]
    return source_cameras


def _seed_all_random_engines(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


@dataclass(eq=False)
class ExperimentConfig:
    generic_model_args: DictConfig = get_default_args_field(GenericModel)
    solver_args: DictConfig = get_default_args_field(init_optimizer)
    dataset_args: DictConfig = get_default_args_field(dataset_zoo)
    dataloader_args: DictConfig = get_default_args_field(dataloader_zoo)
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

    hydra: dict = field(
        default_factory=lambda: {
            "run": {"dir": "."},  # Make hydra not change the working dir.
            "output_subdir": None,  # disable storing the .hydra logs
        }
    )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="default_config", node=ExperimentConfig)


@hydra.main(config_path="./configs/", config_name="default_config")
def experiment(cfg: DictConfig) -> None:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_idx)
    # Set the device
    device = "cpu"
    if torch.cuda.is_available() and cfg.gpu_idx < torch.cuda.device_count():
        device = f"cuda:{cfg.gpu_idx}"
    logger.info(f"Running experiment on device: {device}")
    run_training(cfg, device)


if __name__ == "__main__":
    experiment()
