# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

from pytorch3d.implicitron.tools.config import ReplaceableBase

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
