# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import os
from typing import Optional

import torch.optim

from accelerate import Accelerator
from pytorch3d.implicitron.models.base_model import ImplicitronModelBase
from pytorch3d.implicitron.tools import model_io
from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from pytorch3d.implicitron.tools.stats import Stats

logger = logging.getLogger(__name__)


class ModelFactoryBase(ReplaceableBase):
    resume: bool = True  # resume from the last checkpoint

    def __call__(self, **kwargs) -> ImplicitronModelBase:
        """
        Initialize the model (possibly from a previously saved state).

        Returns: An instance of ImplicitronModelBase.
        """
        raise NotImplementedError()

    def load_stats(self, **kwargs) -> Stats:
        """
        Initialize or load a Stats object.
        """
        raise NotImplementedError()


@registry.register
class ImplicitronModelFactory(ModelFactoryBase):
    """
    A factory class that initializes an implicit rendering model.

    Members:
        model: An ImplicitronModelBase object.
        resume: If True, attempt to load the last checkpoint from `exp_dir`
            passed to __call__. Failure to do so will return a model with ini-
            tial weights unless `force_resume` is True.
        resume_epoch: If `resume` is True: Resume a model at this epoch, or if
            `resume_epoch` <= 0, then resume from the latest checkpoint.
        force_resume: If True, throw a FileNotFoundError if `resume` is True but
            a model checkpoint cannot be found.

    """

    # pyre-fixme[13]: Attribute `model` is never initialized.
    model: ImplicitronModelBase
    model_class_type: str = "GenericModel"
    resume: bool = True
    resume_epoch: int = -1
    force_resume: bool = False

    def __post_init__(self):
        run_auto_creation(self)

    def __call__(
        self,
        exp_dir: str,
        accelerator: Optional[Accelerator] = None,
    ) -> ImplicitronModelBase:
        """
        Returns an instance of `ImplicitronModelBase`, possibly loaded from a
        checkpoint (if self.resume, self.resume_epoch specify so).

        Args:
            exp_dir: Root experiment directory.
            accelerator: An Accelerator object.

        Returns:
            model: The model with optionally loaded weights from checkpoint

        Raise:
            FileNotFoundError if `force_resume` is True but checkpoint not found.
        """
        # Determine the network outputs that should be logged
        if hasattr(self.model, "log_vars"):
            log_vars = list(self.model.log_vars)
        else:
            log_vars = ["objective"]

        if self.resume_epoch > 0:
            # Resume from a certain epoch
            model_path = model_io.get_checkpoint(exp_dir, self.resume_epoch)
            if not os.path.isfile(model_path):
                raise ValueError(f"Cannot find model from epoch {self.resume_epoch}.")
        else:
            # Retrieve the last checkpoint
            model_path = model_io.find_last_checkpoint(exp_dir)

        if model_path is not None:
            logger.info(f"Found previous model {model_path}")
            if self.force_resume or self.resume:
                logger.info("Resuming.")

                map_location = None
                if accelerator is not None and not accelerator.is_local_main_process:
                    map_location = {
                        "cuda:%d" % 0: "cuda:%d" % accelerator.local_process_index
                    }
                model_state_dict = torch.load(
                    model_io.get_model_path(model_path),
                    map_location=map_location,
                    weights_only=True,
                )

                try:
                    self.model.load_state_dict(model_state_dict, strict=True)
                except RuntimeError as e:
                    logger.error(e)
                    logger.info(
                        "Cannot load state dict in strict mode! -> trying non-strict"
                    )
                    self.model.load_state_dict(model_state_dict, strict=False)
                self.model.log_vars = log_vars
            else:
                logger.info("Not resuming -> starting from scratch.")
        elif self.force_resume:
            raise FileNotFoundError(f"Cannot find a checkpoint in {exp_dir}!")

        return self.model
