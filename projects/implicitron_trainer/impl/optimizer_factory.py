# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import inspect
import logging
import os
from collections import defaultdict
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple

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
        weight_decay: The optimizer weight_decay (L2 penalty on model weights).
        foreach: Whether to use new "foreach" implementation of optimizer where
            available (e.g. requires PyTorch 1.12.0 for Adam)
        group_learning_rates: Parameters or modules can be assigned to parameter
            groups. This dictionary has names of those parameter groups as keys
            and learning rates as values. All parameter group names have to be
            defined in this dictionary. Parameters which do not have predefined
            parameter group are put into "default" parameter group which has
            `lr` as its learning rate.
    """

    betas: Tuple[float, ...] = (0.9, 0.999)
    breed: str = "Adam"
    exponential_lr_step_size: int = 250
    gamma: float = 0.1
    lr: float = 0.0005
    lr_policy: str = "MultiStepLR"
    momentum: float = 0.9
    multistep_lr_milestones: tuple = ()
    weight_decay: float = 0.0
    linear_exponential_lr_milestone: int = 200
    linear_exponential_start_gamma: float = 0.1
    foreach: Optional[bool] = True
    group_learning_rates: Dict[str, float] = field(default_factory=lambda: {})

    def __post_init__(self):
        run_auto_creation(self)

    def __call__(
        self,
        last_epoch: int,
        model: ImplicitronModelBase,
        accelerator: Optional[Accelerator] = None,
        exp_dir: Optional[str] = None,
        resume: bool = True,
        resume_epoch: int = -1,
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
            resume: If True, attempt to load optimizer checkpoint from exp_dir.
                Failure to do so will return a newly initialized optimizer.
            resume_epoch: If `resume` is True: Resume optimizer at this epoch. If
                `resume_epoch` <= 0, then resume from the latest checkpoint.
        Returns:
            An optimizer module (optionally loaded from a checkpoint) and
            a learning rate scheduler module (should be a subclass of torch.optim's
            lr_scheduler._LRScheduler).
        """
        # Get the parameters to optimize
        if hasattr(model, "_get_param_groups"):  # use the model function
            # pyre-fixme[29]: `Union[Tensor, Module]` is not a function.
            p_groups = model._get_param_groups(self.lr, wd=self.weight_decay)
        else:
            p_groups = [
                {"params": params, "lr": self._get_group_learning_rate(group)}
                for group, params in self._get_param_groups(model).items()
            ]

        # Intialize the optimizer
        optimizer_kwargs: Dict[str, Any] = {
            "lr": self.lr,
            "weight_decay": self.weight_decay,
        }
        if self.breed == "SGD":
            optimizer_class = torch.optim.SGD
            optimizer_kwargs["momentum"] = self.momentum
        elif self.breed == "Adagrad":
            optimizer_class = torch.optim.Adagrad
        elif self.breed == "Adam":
            optimizer_class = torch.optim.Adam
            optimizer_kwargs["betas"] = self.betas
        else:
            raise ValueError(f"No such solver type {self.breed}")

        if "foreach" in inspect.signature(optimizer_class.__init__).parameters:
            optimizer_kwargs["foreach"] = self.foreach
        optimizer = optimizer_class(p_groups, **optimizer_kwargs)
        logger.info(f"Solver type = {self.breed}")

        # Load state from checkpoint
        optimizer_state = self._get_optimizer_state(
            exp_dir,
            accelerator,
            resume_epoch=resume_epoch,
            resume=resume,
        )
        if optimizer_state is not None:
            logger.info("Setting loaded optimizer state.")
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
        elif self.lr_policy.casefold() == "LinearExponential".casefold():
            # linear learning rate progression between epochs 0 to
            # self.linear_exponential_lr_milestone, followed by exponential
            # lr decay for the rest of the epochs
            def _get_lr(epoch: int):
                m = self.linear_exponential_lr_milestone
                if epoch < m:
                    w = (m - epoch) / m
                    gamma = w * self.linear_exponential_start_gamma + (1 - w)
                else:
                    epoch_rest = epoch - m
                    gamma = self.gamma ** (epoch_rest / self.exponential_lr_step_size)
                return gamma

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, _get_lr, verbose=False
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
        resume: bool = True,
        resume_epoch: int = -1,
    ) -> Optional[Dict[str, Any]]:
        """
        Load an optimizer state from a checkpoint.

        resume: If True, attempt to load the last checkpoint from `exp_dir`
            passed to __call__. Failure to do so will return a newly initialized
            optimizer.
        resume_epoch: If `resume` is True: Resume optimizer at this epoch. If
            `resume_epoch` <= 0, then resume from the latest checkpoint.
        """
        if exp_dir is None or not resume:
            return None
        if resume_epoch > 0:
            save_path = model_io.get_checkpoint(exp_dir, resume_epoch)
            if not os.path.isfile(save_path):
                raise FileNotFoundError(
                    f"Cannot find optimizer from epoch {resume_epoch}."
                )
        else:
            save_path = model_io.find_last_checkpoint(exp_dir)
        optimizer_state = None
        if save_path is not None:
            logger.info(f"Found previous optimizer state {save_path} -> resuming.")
            opt_path = model_io.get_optimizer_path(save_path)

            if os.path.isfile(opt_path):
                map_location = None
                if accelerator is not None and not accelerator.is_local_main_process:
                    map_location = {
                        "cuda:%d" % 0: "cuda:%d" % accelerator.local_process_index
                    }
                optimizer_state = torch.load(opt_path, map_location, weights_only=True)
            else:
                raise FileNotFoundError(f"Optimizer state {opt_path} does not exist.")
        return optimizer_state

    def _get_param_groups(
        self, module: torch.nn.Module
    ) -> Dict[str, List[torch.nn.Parameter]]:
        """
        Recursively visits all the modules inside the `module` and sorts all the
        parameters in parameter groups.

        Uses `param_groups` dictionary member, where keys are names of individual
        parameters or module members and values are the names of the parameter groups
        for those parameters or members. "self" key is used to denote the parameter groups
        at the module level. Possible keys, including the "self" key do not have to
        be defined. By default all parameters have the learning rate defined in the
        optimizer. This can be overridden by setting the parameter group in `param_groups`
        member of a specific module. Values are a parameter group name. The keys
        specify what parameters will be affected as follows:
            - “self”: All the parameters of the module and its child modules
            - name of a parameter: A parameter with that name.
            - name of a module member: All the parameters of the module and its
                child modules.
                This is useful if members do not have `param_groups`, for
                example torch.nn.Linear.
            - <name of module member>.<something>: recursive. Same as if <something>
                was used in param_groups of that submodule/member.

        Args:
            module: module from which to extract the parameters and their parameter
                groups
        Returns:
            dictionary with parameter groups as keys and lists of parameters as values
        """

        param_groups = defaultdict(list)

        def traverse(module, default_group: str, mapping: Dict[str, str]) -> None:
            """
            Visitor for module to assign its parameters to the relevant member of
            param_groups.

            Args:
                module: the module being visited in a depth-first search
                default_group: the param group to assign parameters to unless
                                otherwise overriden.
                mapping: known mappings of parameters to groups for this module,
                    destructively modified by this function.
            """
            # If key self is defined in param_groups then chenge the default param
            # group for all parameters and children in the module.
            if hasattr(module, "param_groups") and "self" in module.param_groups:
                default_group = module.param_groups["self"]

            # Collect all the parameters that are directly inside the `module`,
            # they will be in the default param group if they don't have
            # defined group.
            if hasattr(module, "param_groups"):
                mapping.update(module.param_groups)

            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    group_name = mapping.get(name, default_group)
                    logger.debug(f"Assigning {name} to param_group {group_name}")
                    param_groups[group_name].append(param)

            # If children have defined default param group then use it else pass
            # own default.
            for child_name, child in module.named_children():
                mapping_to_add = {
                    name[len(child_name) + 1 :]: group
                    for name, group in mapping.items()
                    if name.startswith(child_name + ".")
                }
                traverse(child, mapping.get(child_name, default_group), mapping_to_add)

        traverse(module, "default", {})
        return param_groups

    def _get_group_learning_rate(self, group_name: str) -> float:
        """
        Wraps the `group_learning_rates` dictionary providing errors and returns
        `self.lr` for "default" group_name.

        Args:
            group_name: a string representing the name of the group
        Returns:
            learning rate for a specific group
        """
        if group_name == "default":
            return self.lr
        lr = self.group_learning_rates.get(group_name, None)
        if lr is None:
            raise ValueError(f"no learning rate given for group {group_name}")
        return lr
