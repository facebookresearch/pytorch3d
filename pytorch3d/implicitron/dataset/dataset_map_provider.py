# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

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

    def join(self, other_dataset_maps: Iterable["DatasetMap"]) -> None:
        """
        Joins the current DatasetMap with other dataset maps from the input list.

        For each subset of each dataset map (train/val/test), the function
        omits joining the subsets that are None.

        Note the train/val/test datasets of the current dataset map will be
        modified in-place.

        Args:
            other_dataset_maps: The list of dataset maps to be joined into the
                current dataset map.
        """
        for set_ in ["train", "val", "test"]:
            dataset_list = [
                getattr(self, set_),
                *[getattr(dmap, set_) for dmap in other_dataset_maps],
            ]
            dataset_list = [d for d in dataset_list if d is not None]
            if len(dataset_list) == 0:
                setattr(self, set_, None)
                continue
            d0 = dataset_list[0]
            if len(dataset_list) > 1:
                d0.join(dataset_list[1:])
            setattr(self, set_, d0)


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

    def get_all_train_cameras(self) -> Optional[CamerasBase]:
        """
        DEPRECATED! The function will be removed in future versions.
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
