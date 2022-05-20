# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

from omegaconf import DictConfig
from pytorch3d.implicitron.tools.config import (
    get_default_args_field,
    ReplaceableBase,
    run_auto_creation,
)

from . import json_index_dataset_map_provider  # noqa
from .dataloader_zoo import dataloader_zoo, Dataloaders
from .dataset_map_provider import DatasetMap, DatasetMapProviderBase, Task


class DataSourceBase(ReplaceableBase):
    """
    Base class for a data source in Implicitron. It encapsulates Dataset
    and DataLoader configuration.
    """

    def get_datasets_and_dataloaders(self) -> Tuple[DatasetMap, Dataloaders]:
        raise NotImplementedError()


class ImplicitronDataSource(DataSourceBase):  # pyre-ignore[13]
    """
    Represents the data used in Implicitron. This is the only implementation
    of DataSourceBase provided.

    Members:
        dataset_map_provider_class_type: identifies type for dataset_map_provider.
            e.g. JsonIndexDatasetMapProvider for Co3D.
    """

    dataset_map_provider: DatasetMapProviderBase
    dataset_map_provider_class_type: str
    dataloader_args: DictConfig = get_default_args_field(dataloader_zoo)

    def __post_init__(self):
        run_auto_creation(self)

    def get_datasets_and_dataloaders(self) -> Tuple[DatasetMap, Dataloaders]:
        datasets = self.dataset_map_provider.get_dataset_map()
        dataloaders = dataloader_zoo(datasets, **self.dataloader_args)
        return datasets, dataloaders

    def get_task(self) -> Task:
        return self.dataset_map_provider.get_task()
