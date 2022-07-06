# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

from iopath.common.file_io import PathManager
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from pytorch3d.renderer.cameras import CamerasBase

from .dataset_base import DatasetBase


@dataclass
class DatasetMap:
    """
    A collection of datasets for implicitron.

    Members:

        train: a dataset for training
        val: a dataset for validating during training
        test: a dataset for final evaluation
    """

    train: Optional[DatasetBase]
    val: Optional[DatasetBase]
    test: Optional[DatasetBase]

    def __getitem__(self, split: str) -> Optional[DatasetBase]:
        """
        Get one of the datasets by key (name of data split)
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"{split} was not a valid split name (train/val/test)")
        return getattr(self, split)

    def iter_datasets(self) -> Iterator[DatasetBase]:
        """
        Iterator over all datasets.
        """
        if self.train is not None:
            yield self.train
        if self.val is not None:
            yield self.val
        if self.test is not None:
            yield self.test


class Task(Enum):
    SINGLE_SEQUENCE = "singlesequence"
    MULTI_SEQUENCE = "multisequence"


class DatasetMapProviderBase(ReplaceableBase):
    """
    Base class for a provider of training / validation and testing
    dataset objects.
    """

    def get_dataset_map(self) -> DatasetMap:
        """
        Returns:
            An object containing the torch.Dataset objects in train/val/test fields.
        """
        raise NotImplementedError()

    def get_task(self) -> Task:
        raise NotImplementedError()

    def get_all_train_cameras(self) -> Optional[CamerasBase]:
        """
        If the data is all for a single scene, returns a list
        of the known training cameras for that scene, which is
        used for evaluating the difficulty of the unknown
        cameras. Otherwise return None.
        """
        raise NotImplementedError()


@registry.register
class PathManagerFactory(ReplaceableBase):
    """
    Base class and default implementation of a tool which dataset_map_provider implementations
    may use to construct a path manager if needed.

    Args:
        silence_logs: Whether to reduce log output from iopath library.
    """

    silence_logs: bool = True

    def get(self) -> Optional[PathManager]:
        """
        Makes a PathManager if needed.
        For open source users, this function should always return None.
        Internally, this allows manifold access.
        """
        if os.environ.get("INSIDE_RE_WORKER", False):
            return None

        try:
            from iopath.fb.manifold import ManifoldPathHandler
        except ImportError:
            return None

        if self.silence_logs:
            logging.getLogger("iopath.fb.manifold").setLevel(logging.CRITICAL)
            logging.getLogger("iopath.common.file_io").setLevel(logging.CRITICAL)

        path_manager = PathManager()
        path_manager.register_handler(ManifoldPathHandler())

        return path_manager
