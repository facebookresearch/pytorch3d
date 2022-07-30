# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch.optim

from accelerate import Accelerator

from pytorch3d.implicitron.models.base_model import ImplicitronModelBase
from pytorch3d.implicitron.tools import model_io
from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)

logger = logging.getLogger(__name__)


class OptimizerFactoryBase(ReplaceableBase):
    def __call__(
        self, model: ImplicitronModelBase, **kwargs
    ) -> Tuple[torch.optim.Optimizer, Any]:
        """
        Initialize the optimizer and lr scheduler.

        Args:
            model: The model with optionally loaded weights.

        Returns:
            An optimizer module (optionally loaded from a checkpoint) and
            a learning rate scheduler module (should be a subclass of torch.optim's
            lr_scheduler._LRScheduler).
        """
        raise NotImplementedError()


@registry.register
class ImplicitronOptimizerFactory(OptimizerFactoryBase):
    """
    A factory that initializes the optimizer and lr scheduler.

    Members:
        betas: Beta parameters for the Adam optimizer.
        breed: The type of optimizer to use. We currently support SGD, Adagrad
            and Adam.
        exponential_lr_step_size: With Exponential policy only,
            lr = lr * gamma ** (epoch/step_size)
        gamma: Multiplicative factor of learning rate decay.
        lr: The value for the initial learning rate.
        lr_policy: The policy to use for learning rate. We currently support
            MultiStepLR and Exponential policies.
        momentum: A momentum value (for SGD only).
        multistep_lr_milestones: With MultiStepLR policy only: list of
            increasing epoch indices at which the learning rate is modified.
        momentum: Momentum factor for SGD optimizer.
        resume: If True, attempt to load the last checkpoint from `exp_dir`
            passed to __call__. Failure to do so will return a newly initialized
            optimizer.
        resume_epoch: If `resume` is True: Resume optimizer at this epoch. If
            `resume_epoch` <= 0, then resume from the latest checkpoint.
        weight_decay: The optimizer weight_decay (L2 penalty on model weights).
    """

    betas: Tuple[float, ...] = (0.9, 0.999)
    breed: str = "Adam"
    exponential_lr_step_size: int = 250
    gamma: float = 0.1
    lr: float = 0.0005
    lr_policy: str = "MultiStepLR"
    momentum: float = 0.9
    multistep_lr_milestones: tuple = ()
    resume: bool = False
    resume_epoch: int = -1
    weight_decay: float = 0.0

    def __post_init__(self):
        run_auto_creation(self)

    def __call__(
        self,
        last_epoch: int,
        model: ImplicitronModelBase,
        accelerator: Optional[Accelerator] = None,
        exp_dir: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.optim.Optimizer, Any]:
        """
        Initialize the optimizer (optionally from a checkpoint) and the lr scheduluer.

        Args:
            last_epoch: If the model was loaded from checkpoint this will be the
                number of the last epoch that was saved.
            model: The model with optionally loaded weights.
            accelerator: An optional Accelerator instance.
            exp_dir: Root experiment directory.

        Returns:
            An optimizer module (optionally loaded from a checkpoint) and
            a learning rate scheduler module (should be a subclass of torch.optim's
            lr_scheduler._LRScheduler).
        """
        # Get the parameters to optimize
        if hasattr(model, "_get_param_groups"):  # use the model function
            # pyre-ignore[29]
            p_groups = model._get_param_groups(self.lr, wd=self.weight_decay)
        else:
            allprm = [prm for prm in model.parameters() if prm.requires_grad]
            p_groups = [{"params": allprm, "lr": self.lr}]

        # Intialize the optimizer
        if self.breed == "SGD":
            optimizer = torch.optim.SGD(
                p_groups,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.breed == "Adagrad":
            optimizer = torch.optim.Adagrad(
                p_groups, lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.breed == "Adam":
            optimizer = torch.optim.Adam(
                p_groups, lr=self.lr, betas=self.betas, weight_decay=self.weight_decay
            )
        else:
            raise ValueError("no such solver type %s" % self.breed)
        logger.info("  -> solver type = %s" % self.breed)

        # Load state from checkpoint
        optimizer_state = self._get_optimizer_state(exp_dir, accelerator)
        if optimizer_state is not None:
            logger.info("  -> setting loaded optimizer state")
            optimizer.load_state_dict(optimizer_state)

        # Initialize the learning rate scheduler
        if self.lr_policy.casefold() == "MultiStepLR".casefold():
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.multistep_lr_milestones,
                gamma=self.gamma,
            )
        elif self.lr_policy.casefold() == "Exponential".casefold():
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda epoch: self.gamma ** (epoch / self.exponential_lr_step_size),
                verbose=False,
            )
        else:
            raise ValueError("no such lr policy %s" % self.lr_policy)

        # When loading from checkpoint, this will make sure that the
        # lr is correctly set even after returning.
        for _ in range(last_epoch):
            scheduler.step()

        optimizer.zero_grad()

        return optimizer, scheduler

    def _get_optimizer_state(
        self,
        exp_dir: Optional[str],
        accelerator: Optional[Accelerator] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load an optimizer state from a checkpoint.
        """
        if exp_dir is None or not self.resume:
            return None
        if self.resume_epoch > 0:
            save_path = model_io.get_checkpoint(exp_dir, self.resume_epoch)
        else:
            save_path = model_io.find_last_checkpoint(exp_dir)
        optimizer_state = None
        if save_path is not None:
            logger.info(f"Found previous optimizer state {save_path}.")
            logger.info("   -> resuming")
            opt_path = model_io.get_optimizer_path(save_path)

            if os.path.isfile(opt_path):
                map_location = None
                if accelerator is not None and not accelerator.is_local_main_process:
                    map_location = {
                        "cuda:%d" % 0: "cuda:%d" % accelerator.local_process_index
                    }
                optimizer_state = torch.load(opt_path, map_location)
            else:
                optimizer_state = None
        return optimizer_state
