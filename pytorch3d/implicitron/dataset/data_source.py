# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Tuple

from omegaconf import DictConfig
from pytorch3d.implicitron.tools.config import get_default_args_field, ReplaceableBase

from .dataloader_zoo import dataloader_zoo, Dataloaders
from .dataset_zoo import dataset_zoo, Datasets


class Task(Enum):
    SINGLE_SEQUENCE = "singlesequence"
    MULTI_SEQUENCE = "multisequence"


class DataSourceBase(ReplaceableBase):
    """
    Base class for a data source in Implicitron. It encapsulates Dataset
    and DataLoader configuration.
    """

    def get_datasets_and_dataloaders(self) -> Tuple[Datasets, Dataloaders]:
        raise NotImplementedError()


class ImplicitronDataSource(DataSourceBase):
    """
    Represents the data used in Implicitron. This is the only implementation
    of DataSourceBase provided.
    """

    dataset_args: DictConfig = get_default_args_field(dataset_zoo)
    dataloader_args: DictConfig = get_default_args_field(dataloader_zoo)

    def get_datasets_and_dataloaders(self) -> Tuple[Datasets, Dataloaders]:
        datasets = dataset_zoo(**self.dataset_args)
        dataloaders = dataloader_zoo(datasets, **self.dataloader_args)
        return datasets, dataloaders

    def get_task(self) -> Task:
        eval_task = self.dataset_args["dataset_name"].split("_")[-1]
        return Task(eval_task)
