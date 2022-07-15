# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional, Tuple

import torch
from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.tools.config import enable_get_default_args

logger = logging.getLogger(__name__)


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
    betas: Tuple[float, ...] = (0.9, 0.999),
    milestones: Tuple[int, ...] = (),
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
        # pyre-ignore[29]
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

    optimizer.zero_grad()
    return optimizer, scheduler


enable_get_default_args(init_optimizer)
