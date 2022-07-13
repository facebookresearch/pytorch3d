# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from pytorch3d.renderer.cameras import CamerasBase

from .blender_dataset_map_provider import BlenderDatasetMapProvider  # noqa
from .data_loader_map_provider import DataLoaderMap, DataLoaderMapProviderBase
from .dataset_map_provider import DatasetMap, DatasetMapProviderBase, Task
from .json_index_dataset_map_provider import JsonIndexDatasetMapProvider  # noqa
from .json_index_dataset_map_provider_v2 import JsonIndexDatasetMapProviderV2  # noqa
from .llff_dataset_map_provider import LlffDatasetMapProvider  # noqa


class DataSourceBase(ReplaceableBase):
    """
    Base class for a data source in Implicitron. It encapsulates Dataset
    and DataLoader configuration.
    """

    def get_datasets_and_dataloaders(self) -> Tuple[DatasetMap, DataLoaderMap]:
        raise NotImplementedError()

    def get_all_train_cameras(self) -> Optional[CamerasBase]:
        """
        If the data is all for a single scene, returns a list
        of the known training cameras for that scene, which is
        used for evaluating the viewpoint difficulty of the
        unseen cameras.
        """
        raise NotImplementedError()


@registry.register
class ImplicitronDataSource(DataSourceBase):  # pyre-ignore[13]
    """
    Represents the data used in Implicitron. This is the only implementation
    of DataSourceBase provided.

    Members:
        dataset_map_provider_class_type: identifies type for dataset_map_provider.
            e.g. JsonIndexDatasetMapProvider for Co3D.
        data_loader_map_provider_class_type: identifies type for data_loader_map_provider.
    """

    dataset_map_provider: DatasetMapProviderBase
    dataset_map_provider_class_type: str
    data_loader_map_provider: DataLoaderMapProviderBase
    data_loader_map_provider_class_type: str = "SequenceDataLoaderMapProvider"

    def __post_init__(self):
        run_auto_creation(self)

    def get_datasets_and_dataloaders(self) -> Tuple[DatasetMap, DataLoaderMap]:
        datasets = self.dataset_map_provider.get_dataset_map()
        dataloaders = self.data_loader_map_provider.get_data_loader_map(datasets)
        return datasets, dataloaders

    def get_task(self) -> Task:
        return self.dataset_map_provider.get_task()

    def get_all_train_cameras(self) -> Optional[CamerasBase]:
        return self.dataset_map_provider.get_all_train_cameras()
