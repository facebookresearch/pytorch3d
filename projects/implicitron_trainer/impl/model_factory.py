# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import List, Optional

import torch.optim

from accelerate import Accelerator
from pytorch3d.implicitron.models.base_model import ImplicitronModelBase
from pytorch3d.implicitron.tools import model_io, vis_utils
from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from pytorch3d.implicitron.tools.stats import Stats

logger = logging.getLogger(__name__)


class ModelFactoryBase(ReplaceableBase):
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
class ImplicitronModelFactory(ModelFactoryBase):  # pyre-ignore [13]
    """
    A factory class that initializes an implicit rendering model.

    Members:
        force_load: If True, throw a FileNotFoundError if `resume` is True but
            a model checkpoint cannot be found.
        model: An ImplicitronModelBase object.
        resume: If True, attempt to load the last checkpoint from `exp_dir`
            passed to __call__. Failure to do so will return a model with ini-
            tial weights unless `force_load` is True.
        resume_epoch: If `resume` is True: Resume a model at this epoch, or if
            `resume_epoch` <= 0, then resume from the latest checkpoint.
        visdom_env: The name of the Visdom environment to use for plotting.
        visdom_port: The Visdom port.
        visdom_server: Address of the Visdom server.
    """

    force_load: bool = False
    model: ImplicitronModelBase
    model_class_type: str = "GenericModel"
    resume: bool = False
    resume_epoch: int = -1
    visdom_env: str = ""
    visdom_port: int = int(os.environ.get("VISDOM_PORT", 8097))
    visdom_server: str = "http://127.0.0.1"

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
            FileNotFoundError if `force_load` is True but checkpoint not found.
        """
        # Determine the network outputs that should be logged
        if hasattr(self.model, "log_vars"):
            log_vars = list(self.model.log_vars)  # pyre-ignore [6]
        else:
            log_vars = ["objective"]

        # Retrieve the last checkpoint
        if self.resume_epoch > 0:
            model_path = model_io.get_checkpoint(exp_dir, self.resume_epoch)
        else:
            model_path = model_io.find_last_checkpoint(exp_dir)

        if model_path is not None:
            logger.info("found previous model %s" % model_path)
            if self.force_load or self.resume:
                logger.info("   -> resuming")

                map_location = None
                if accelerator is not None and not accelerator.is_local_main_process:
                    map_location = {
                        "cuda:%d" % 0: "cuda:%d" % accelerator.local_process_index
                    }
                model_state_dict = torch.load(
                    model_io.get_model_path(model_path), map_location=map_location
                )

                try:
                    self.model.load_state_dict(model_state_dict, strict=True)
                except RuntimeError as e:
                    logger.error(e)
                    logger.info(
                        "Cant load state dict in strict mode! -> trying non-strict"
                    )
                    self.model.load_state_dict(model_state_dict, strict=False)
                self.model.log_vars = log_vars  # pyre-ignore [16]
            else:
                logger.info("   -> but not resuming -> starting from scratch")
        elif self.force_load:
            raise FileNotFoundError(f"Cannot find a checkpoint in {exp_dir}!")

        return self.model

    def load_stats(
        self,
        log_vars: List[str],
        exp_dir: str,
        clear_stats: bool = False,
        **kwargs,
    ) -> Stats:
        """
        Load Stats that correspond to the model's log_vars.

        Args:
            log_vars: A list of variable names to log. Should be a subset of the
                `preds` returned by the forward function of the corresponding
                ImplicitronModelBase instance.
            exp_dir: Root experiment directory.
            clear_stats: If True, do not load stats from the checkpoint speci-
                fied by self.resume and self.resume_epoch; instead, create a
                fresh stats object.

        stats: The stats structure (optionally loaded from checkpoint)
        """
        # Init the stats struct
        visdom_env_charts = (
            vis_utils.get_visdom_env(self.visdom_env, exp_dir) + "_charts"
        )
        stats = Stats(
            # log_vars should be a list, but OmegaConf might load them as ListConfig
            list(log_vars),
            visdom_env=visdom_env_charts,
            verbose=False,
            visdom_server=self.visdom_server,
            visdom_port=self.visdom_port,
        )
        if self.resume_epoch > 0:
            model_path = model_io.get_checkpoint(exp_dir, self.resume_epoch)
        else:
            model_path = model_io.find_last_checkpoint(exp_dir)

        if model_path is not None:
            stats_path = model_io.get_stats_path(model_path)
            stats_load = model_io.load_stats(stats_path)

            # Determine if stats should be reset
            if not clear_stats:
                if stats_load is None:
                    logger.warning("\n\n\n\nCORRUPT STATS -> clearing stats\n\n\n\n")
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
                stats.visdom_server = self.visdom_server
                stats.visdom_port = self.visdom_port
                stats.plot_file = os.path.join(exp_dir, "train_stats.pdf")
                stats.synchronize_logged_vars(log_vars)
            else:
                logger.info("   -> clearing stats")

        return stats
