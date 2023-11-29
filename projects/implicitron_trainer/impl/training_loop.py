# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from typing import Any, List, Optional

import torch
from accelerate import Accelerator
from pytorch3d.implicitron.evaluation.evaluator import EvaluatorBase
from pytorch3d.implicitron.models.base_model import ImplicitronModelBase
from pytorch3d.implicitron.models.generic_model import EvaluationMode
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from pytorch3d.implicitron.tools.stats import Stats
from torch.utils.data import DataLoader, Dataset

from .utils import seed_all_random_engines

logger = logging.getLogger(__name__)


# pyre-fixme[13]: Attribute `evaluator` is never initialized.
class TrainingLoopBase(ReplaceableBase):
    """
    Members:
        evaluator: An EvaluatorBase instance, used to evaluate training results.
    """

    evaluator: Optional[EvaluatorBase]
    evaluator_class_type: Optional[str] = "ImplicitronEvaluator"

    def run(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        train_dataset: Dataset,
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        **kwargs,
    ) -> None:
        raise NotImplementedError()

    def load_stats(
        self,
        log_vars: List[str],
        exp_dir: str,
        resume: bool = True,
        resume_epoch: int = -1,
        **kwargs,
    ) -> Stats:
        raise NotImplementedError()


@registry.register
class ImplicitronTrainingLoop(TrainingLoopBase):
    """
    Members:
        eval_only: If True, only run evaluation using the test dataloader.
        max_epochs: Train for this many epochs. Note that if the model was
            loaded from a checkpoint, we will restart training at the appropriate
            epoch and run for (max_epochs - checkpoint_epoch) epochs.
        store_checkpoints: If True, store model and optimizer state checkpoints.
        store_checkpoints_purge: If >= 0, remove any checkpoints older or equal
            to this many epochs.
        test_interval: Evaluate on a test dataloader each `test_interval` epochs.
        test_when_finished: If True, evaluate on a test dataloader when training
            completes.
        validation_interval: Validate each `validation_interval` epochs.
        clip_grad: Optionally clip the gradient norms.
            If set to a value <=0.0, no clipping
        metric_print_interval: The batch interval at which the stats should be
            logged.
        visualize_interval: The batch interval at which the visualizations
            should be plotted
        visdom_env: The name of the Visdom environment to use for plotting.
        visdom_port: The Visdom port.
        visdom_server: Address of the Visdom server.
    """

    # Parameters of the outer training loop.
    eval_only: bool = False
    max_epochs: int = 1000
    store_checkpoints: bool = True
    store_checkpoints_purge: int = 1
    test_interval: int = -1
    test_when_finished: bool = False
    validation_interval: int = 1

    # Gradient clipping.
    clip_grad: float = 0.0

    # Visualization/logging parameters.
    metric_print_interval: int = 5
    visualize_interval: int = 1000
    visdom_env: str = ""
    visdom_port: int = int(os.environ.get("VISDOM_PORT", 8097))
    visdom_server: str = "http://127.0.0.1"

    def __post_init__(self):
        run_auto_creation(self)

    # pyre-fixme[14]: `run` overrides method defined in `TrainingLoopBase`
    #  inconsistently.
    def run(
        self,
        *,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        train_dataset: Dataset,
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        accelerator: Optional[Accelerator],
        device: torch.device,
        exp_dir: str,
        stats: Stats,
        seed: int,
        **kwargs,
    ):
        """
        Entry point to run the training and validation loops
        based on the specified config file.
        """
        start_epoch = stats.epoch + 1
        assert scheduler.last_epoch == stats.epoch + 1
        assert scheduler.last_epoch == start_epoch

        # only run evaluation on the test dataloader
        if self.eval_only:
            if test_loader is not None:
                # pyre-fixme[16]: `Optional` has no attribute `run`.
                self.evaluator.run(
                    dataloader=test_loader,
                    device=device,
                    dump_to_json=True,
                    epoch=stats.epoch,
                    exp_dir=exp_dir,
                    model=model,
                )
                return
            else:
                raise ValueError(
                    "Cannot evaluate and dump results to json, no test data provided."
                )

        # loop through epochs
        for epoch in range(start_epoch, self.max_epochs):
            # automatic new_epoch and plotting of stats at every epoch start
            with stats:

                # Make sure to re-seed random generators to ensure reproducibility
                # even after restart.
                seed_all_random_engines(seed + epoch)

                cur_lr = float(scheduler.get_last_lr()[-1])
                logger.debug(f"scheduler lr = {cur_lr:1.2e}")

                # train loop
                self._training_or_validation_epoch(
                    accelerator=accelerator,
                    device=device,
                    epoch=epoch,
                    loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    stats=stats,
                    validation=False,
                )

                # val loop (optional)
                if val_loader is not None and epoch % self.validation_interval == 0:
                    self._training_or_validation_epoch(
                        accelerator=accelerator,
                        device=device,
                        epoch=epoch,
                        loader=val_loader,
                        model=model,
                        optimizer=optimizer,
                        stats=stats,
                        validation=True,
                    )

                # eval loop (optional)
                if (
                    test_loader is not None
                    and self.test_interval > 0
                    and epoch % self.test_interval == 0
                ):
                    self.evaluator.run(
                        device=device,
                        dataloader=test_loader,
                        model=model,
                    )

                assert stats.epoch == epoch, "inconsistent stats!"
                self._checkpoint(accelerator, epoch, exp_dir, model, optimizer, stats)

                scheduler.step()
                new_lr = float(scheduler.get_last_lr()[-1])
                if new_lr != cur_lr:
                    logger.info(f"LR change! {cur_lr} -> {new_lr}")

        if self.test_when_finished:
            if test_loader is not None:
                self.evaluator.run(
                    device=device,
                    dump_to_json=True,
                    epoch=stats.epoch,
                    exp_dir=exp_dir,
                    dataloader=test_loader,
                    model=model,
                )
            else:
                raise ValueError(
                    "Cannot evaluate and dump results to json, no test data provided."
                )

    def load_stats(
        self,
        log_vars: List[str],
        exp_dir: str,
        resume: bool = True,
        resume_epoch: int = -1,
        **kwargs,
    ) -> Stats:
        """
        Load Stats that correspond to the model's log_vars and resume_epoch.

        Args:
            log_vars: A list of variable names to log. Should be a subset of the
                `preds` returned by the forward function of the corresponding
                ImplicitronModelBase instance.
            exp_dir: Root experiment directory.
            resume: If False, do not load stats from the checkpoint speci-
                fied by resume and resume_epoch; instead, create a fresh stats object.

        stats: The stats structure (optionally loaded from checkpoint)
        """
        # Init the stats struct
        visdom_env_charts = (
            vis_utils.get_visdom_env(self.visdom_env, exp_dir) + "_charts"
        )
        stats = Stats(
            # log_vars should be a list, but OmegaConf might load them as ListConfig
            list(log_vars),
            plot_file=os.path.join(exp_dir, "train_stats.pdf"),
            visdom_env=visdom_env_charts,
            visdom_server=self.visdom_server,
            visdom_port=self.visdom_port,
        )

        model_path = None
        if resume:
            if resume_epoch > 0:
                model_path = model_io.get_checkpoint(exp_dir, resume_epoch)
                if not os.path.isfile(model_path):
                    raise FileNotFoundError(
                        f"Cannot find stats from epoch {resume_epoch}."
                    )
            else:
                model_path = model_io.find_last_checkpoint(exp_dir)

        if model_path is not None:
            stats_path = model_io.get_stats_path(model_path)
            stats_load = model_io.load_stats(stats_path)

            # Determine if stats should be reset
            if resume:
                if stats_load is None:
                    logger.warning("\n\n\n\nCORRUPT STATS -> clearing stats\n\n\n\n")
                    last_epoch = model_io.parse_epoch_from_model_path(model_path)
                    logger.info(f"Estimated resume epoch = {last_epoch}")

                    # Reset the stats struct
                    for _ in range(last_epoch + 1):
                        stats.new_epoch()
                    assert last_epoch == stats.epoch
                else:
                    logger.info(f"Found previous stats in {stats_path} -> resuming.")
                    stats = stats_load

                # Update stats properties incase it was reset on load
                stats.visdom_env = visdom_env_charts
                stats.visdom_server = self.visdom_server
                stats.visdom_port = self.visdom_port
                stats.plot_file = os.path.join(exp_dir, "train_stats.pdf")
                stats.synchronize_logged_vars(log_vars)
            else:
                logger.info("Clearing stats")

        return stats

    def _training_or_validation_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        stats: Stats,
        validation: bool,
        *,
        accelerator: Optional[Accelerator],
        bp_var: str = "objective",
        device: torch.device,
        **kwargs,
    ) -> None:
        """
        This is the main loop for training and evaluation including:
        model forward pass, loss computation, backward pass and visualization.

        Args:
            epoch: The index of the current epoch
            loader: The dataloader to use for the loop
            model: The model module optionally loaded from checkpoint
            optimizer: The optimizer module optionally loaded from checkpoint
            stats: The stats struct, also optionally loaded from checkpoint
            validation: If true, run the loop with the model in eval mode
                and skip the backward pass
            accelerator: An optional Accelerator instance.
            bp_var: The name of the key in the model output `preds` dict which
                should be used as the loss for the backward pass.
            device: The device on which to run the model.
        """

        if validation:
            model.eval()
            trainmode = "val"
        else:
            model.train()
            trainmode = "train"

        t_start = time.time()

        # get the visdom env name
        visdom_env_imgs = stats.visdom_env + "_images_" + trainmode
        viz = vis_utils.get_visdom_connection(
            server=stats.visdom_server,
            port=stats.visdom_port,
        )

        # Iterate through the batches
        n_batches = len(loader)
        for it, net_input in enumerate(loader):
            last_iter = it == n_batches - 1

            # move to gpu where possible (in place)
            net_input = net_input.to(device)

            # run the forward pass
            if not validation:
                optimizer.zero_grad()
                preds = model(
                    **{**net_input, "evaluation_mode": EvaluationMode.TRAINING}
                )
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
            # pyre-ignore [16]
            assert stats.it[trainmode] == it, "inconsistent stat iteration number!"

            # print textual status update
            if it % self.metric_print_interval == 0 or last_iter:
                std_out = stats.get_status_string(stat_set=trainmode, max_it=n_batches)
                logger.info(std_out)

            # visualize results
            if (
                (accelerator is None or accelerator.is_local_main_process)
                and self.visualize_interval > 0
                and it % self.visualize_interval == 0
            ):
                prefix = f"e{stats.epoch}_it{stats.it[trainmode]}"
                if hasattr(model, "visualize"):
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
                if accelerator is None:
                    loss.backward()
                else:
                    accelerator.backward(loss)
                if self.clip_grad > 0.0:
                    # Optionally clip the gradient norms.
                    total_norm = torch.nn.utils.clip_grad_norm(
                        model.parameters(), self.clip_grad
                    )
                    if total_norm > self.clip_grad:
                        logger.debug(
                            f"Clipping gradient: {total_norm}"
                            + f" with coef {self.clip_grad / float(total_norm)}."
                        )

                optimizer.step()

    def _checkpoint(
        self,
        accelerator: Optional[Accelerator],
        epoch: int,
        exp_dir: str,
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        stats: Stats,
    ):
        """
        Save a model and its corresponding Stats object to a file, if
        `self.store_checkpoints` is True. In addition, if
        `self.store_checkpoints_purge` is True, remove any checkpoints older
        than `self.store_checkpoints_purge` epochs old.
        """
        if self.store_checkpoints and (
            accelerator is None or accelerator.is_local_main_process
        ):
            if self.store_checkpoints_purge > 0:
                for prev_epoch in range(epoch - self.store_checkpoints_purge):
                    model_io.purge_epoch(exp_dir, prev_epoch)
            outfile = model_io.get_checkpoint(exp_dir, epoch)
            unwrapped_model = (
                model if accelerator is None else accelerator.unwrap_model(model)
            )
            model_io.safe_save_model(
                unwrapped_model, stats, outfile, optimizer=optimizer
            )
