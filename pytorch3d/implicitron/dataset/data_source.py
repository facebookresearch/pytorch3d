# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Tuple

from pytorch3d.implicitron.tools.config import (
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from pytorch3d.renderer.cameras import CamerasBase

from .data_loader_map_provider import DataLoaderMap, DataLoaderMapProviderBase
from .dataset_map_provider import DatasetMap, DatasetMapProviderBase


class DataSourceBase(ReplaceableBase):
    """
    Base class for a data source in Implicitron. It encapsulates Dataset
    and DataLoader configuration.
    """

    def get_datasets_and_dataloaders(self) -> Tuple[DatasetMap, DataLoaderMap]:
        raise NotImplementedError()

    @property
    def all_train_cameras(self) -> Optional[CamerasBase]:
        """
        DEPRECATED! The property will be removed in future versions.
        If the data is all for a single scene, a list
        of the known training cameras for that scene, which is
        used for evaluating the viewpoint difficulty of the
        unseen cameras.
        """
        raise NotImplementedError()


@registry.register
class ImplicitronDataSource(DataSourceBase):
    """
    Represents the data used in Implicitron. This is the only implementation
    of DataSourceBase provided.

    Members:
        dataset_map_provider_class_type: identifies type for dataset_map_provider.
            e.g. JsonIndexDatasetMapProvider for Co3D.
        data_loader_map_provider_class_type: identifies type for data_loader_map_provider.
    """

    # pyre-fixme[13]: Attribute `dataset_map_provider` is never initialized.
    dataset_map_provider: DatasetMapProviderBase
    # pyre-fixme[13]: Attribute `dataset_map_provider_class_type` is never initialized.
    dataset_map_provider_class_type: str
    # pyre-fixme[13]: Attribute `data_loader_map_provider` is never initialized.
    data_loader_map_provider: DataLoaderMapProviderBase
    data_loader_map_provider_class_type: str = "SequenceDataLoaderMapProvider"

    @classmethod
    def pre_expand(cls) -> None:
        # use try/finally to bypass cinder's lazy imports
        try:
            from .blender_dataset_map_provider import (  # noqa: F401
                BlenderDatasetMapProvider,
            )
            from .json_index_dataset_map_provider import (  # noqa: F401
                JsonIndexDatasetMapProvider,
            )
            from .json_index_dataset_map_provider_v2 import (  # noqa: F401
                JsonIndexDatasetMapProviderV2,
            )
            from .llff_dataset_map_provider import LlffDatasetMapProvider  # noqa: F401
            from .rendered_mesh_dataset_map_provider import (  # noqa: F401
                RenderedMeshDatasetMapProvider,
            )
            from .train_eval_data_loader_provider import (  # noqa: F401
                TrainEvalDataLoaderMapProvider,
            )

            try:
                from .sql_dataset_provider import (  # noqa: F401  # pyre-ignore
                    SqlIndexDatasetMapProvider,
                )
            except ModuleNotFoundError:
                pass  # environment without SQL dataset
        finally:
            pass

    def __post_init__(self):
        run_auto_creation(self)
        self._all_train_cameras_cache: Optional[Tuple[Optional[CamerasBase]]] = None

    def get_datasets_and_dataloaders(self) -> Tuple[DatasetMap, DataLoaderMap]:
        datasets = self.dataset_map_provider.get_dataset_map()
        dataloaders = self.data_loader_map_provider.get_data_loader_map(datasets)
        return datasets, dataloaders

    @property
    def all_train_cameras(self) -> Optional[CamerasBase]:
        """
        DEPRECATED! The property will be removed in future versions.
        """
        if self._all_train_cameras_cache is None:  # pyre-ignore[16]
            all_train_cameras = self.dataset_map_provider.get_all_train_cameras()
            self._all_train_cameras_cache = (all_train_cameras,)

        return self._all_train_cameras_cache[0]
