# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import random
import time
from typing import Any, Optional

import numpy as np
import torch
from accelerate import Accelerator
from pytorch3d.implicitron.dataset.data_source import Task
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
from pytorch3d.renderer.cameras import CamerasBase
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TrainingLoopBase(ReplaceableBase):
    def run(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        **kwargs,
    ) -> None:
        raise NotImplementedError()


@registry.register
class ImplicitronTrainingLoop(TrainingLoopBase):  # pyre-ignore [13]
    """
    Members:
        eval_only: If True, only run evaluation using the test dataloader.
        evaluator: An EvaluatorBase instance, used to evaluate training results.
        max_epochs: Train for this many epochs. Note that if the model was
            loaded from a checkpoint, we will restart training at the appropriate
            epoch and run for (max_epochs - checkpoint_epoch) epochs.
        seed: A random seed to ensure reproducibility.
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
    """

    # Parameters of the outer training loop.
    eval_only: bool = False
    evaluator: EvaluatorBase
    evaluator_class_type: str = "ImplicitronEvaluator"
    max_epochs: int = 1000
    seed: int = 0
    store_checkpoints: bool = True
    store_checkpoints_purge: int = 1
    test_interval: int = -1
    test_when_finished: bool = False
    validation_interval: int = 1

    # Parameters of a single training-validation step.
    clip_grad: float = 0.0
    metric_print_interval: int = 5
    visualize_interval: int = 1000

    def __post_init__(self):
        run_auto_creation(self)

    def run(
        self,
        *,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        model: ImplicitronModelBase,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        accelerator: Optional[Accelerator],
        all_train_cameras: Optional[CamerasBase],
        device: torch.device,
        exp_dir: str,
        stats: Stats,
        task: Task,
        **kwargs,
    ):
        """
        Entry point to run the training and validation loops
        based on the specified config file.
        """
        _seed_all_random_engines(self.seed)
        start_epoch = stats.epoch + 1
        assert scheduler.last_epoch == stats.epoch + 1
        assert scheduler.last_epoch == start_epoch

        # only run evaluation on the test dataloader
        if self.eval_only:
            if test_loader is not None:
                self.evaluator.run(
                    all_train_cameras=all_train_cameras,
                    dataloader=test_loader,
                    device=device,
                    dump_to_json=True,
                    epoch=stats.epoch,
                    exp_dir=exp_dir,
                    model=model,
                    task=task,
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
                _seed_all_random_engines(self.seed + epoch)

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
                        all_train_cameras=all_train_cameras,
                        device=device,
                        dataloader=test_loader,
                        model=model,
                        task=task,
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
                    all_train_cameras=all_train_cameras,
                    device=device,
                    dump_to_json=True,
                    epoch=stats.epoch,
                    exp_dir=exp_dir,
                    dataloader=test_loader,
                    model=model,
                    task=task,
                )
            else:
                raise ValueError(
                    "Cannot evaluate and dump results to json, no test data provided."
                )

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
                stats.print(stat_set=trainmode, max_it=n_batches)

            # visualize results
            if (
                (accelerator is None or accelerator.is_local_main_process)
                and self.visualize_interval > 0
                and it % self.visualize_interval == 0
            ):
                prefix = f"e{stats.epoch}_it{stats.it[trainmode]}"
                if hasattr(model, "visualize"):
                    # pyre-ignore [29]
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


def _seed_all_random_engines(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
