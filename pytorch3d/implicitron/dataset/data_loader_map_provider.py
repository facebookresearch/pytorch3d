# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase

from .dataset_base import DatasetBase, FrameData
from .dataset_map_provider import DatasetMap
from .scene_batch_sampler import SceneBatchSampler


@dataclass
class DataLoaderMap:
    """
    A collection of data loaders for Implicitron.

    Members:

        train: a data loader for training
        val: a data loader for validating during training
        test: a data loader for final evaluation
    """

    train: Optional[torch.utils.data.DataLoader[FrameData]]
    val: Optional[torch.utils.data.DataLoader[FrameData]]
    test: Optional[torch.utils.data.DataLoader[FrameData]]

    def __getitem__(
        self, split: str
    ) -> Optional[torch.utils.data.DataLoader[FrameData]]:
        """
        Get one of the data loaders by key (name of data split)
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"{split} was not a valid split name (train/val/test)")
        return getattr(self, split)


class DataLoaderMapProviderBase(ReplaceableBase):
    """
    Provider of a collection of data loaders for a given collection of datasets.
    """

    def get_data_loader_map(self, datasets: DatasetMap) -> DataLoaderMap:
        """
        Returns a collection of data loaders for a given collection of datasets.
        """
        raise NotImplementedError()


@registry.register
class SequenceDataLoaderMapProvider(DataLoaderMapProviderBase):
    """
    The default implementation of DataLoaderMapProviderBase.

    Members:
        batch_size: The size of the batch of the data loader.
        num_workers: Number data-loading threads.
        dataset_len: The number of batches in a training epoch.
        dataset_len_val: The number of batches in a validation epoch.
        images_per_seq_options: Possible numbers of images sampled per sequence.
        sample_consecutive_frames: if True, will sample a contiguous interval of frames
            in the sequence. It first sorts the frames by timestimps when available,
            otherwise by frame numbers, finds the connected segments within the sequence
            of sufficient length, then samples a random pivot element among them and
            ideally uses it as a middle of the temporal window, shifting the borders
            where necessary. This strategy mitigates the bias against shorter segments
            and their boundaries.
        consecutive_frames_max_gap: if a number > 0, then used to define the maximum
            difference in frame_number of neighbouring frames when forming connected
            segments; if both this and consecutive_frames_max_gap_seconds are 0s,
            the whole sequence is considered a segment regardless of frame numbers.
        consecutive_frames_max_gap_seconds: if a number > 0.0, then used to define the
            maximum difference in frame_timestamp of neighbouring frames when forming
            connected segments; if both this and consecutive_frames_max_gap are 0s,
            the whole sequence is considered a segment regardless of frame timestamps.
    """

    batch_size: int = 1
    num_workers: int = 0
    dataset_len: int = 1000
    dataset_len_val: int = 1
    images_per_seq_options: Sequence[int] = (2,)
    sample_consecutive_frames: bool = False
    consecutive_frames_max_gap: int = 0
    consecutive_frames_max_gap_seconds: float = 0.1

    def get_data_loader_map(self, datasets: DatasetMap) -> DataLoaderMap:
        """
        Returns a collection of data loaders for a given collection of datasets.
        """

        data_loader_kwargs = {
            "num_workers": self.num_workers,
            "collate_fn": FrameData.collate,
        }

        def train_or_val_loader(
            dataset: Optional[DatasetBase], num_batches: int
        ) -> Optional[torch.utils.data.DataLoader]:
            if dataset is None:
                return None
            batch_sampler = SceneBatchSampler(
                dataset,
                self.batch_size,
                num_batches=len(dataset) if num_batches <= 0 else num_batches,
                images_per_seq_options=self.images_per_seq_options,
                sample_consecutive_frames=self.sample_consecutive_frames,
                consecutive_frames_max_gap=self.consecutive_frames_max_gap,
                consecutive_frames_max_gap_seconds=self.consecutive_frames_max_gap_seconds,
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                **data_loader_kwargs,
            )

        train_data_loader = train_or_val_loader(datasets.train, self.dataset_len)
        val_data_loader = train_or_val_loader(datasets.val, self.dataset_len_val)

        test_dataset = datasets.test
        if test_dataset is not None:
            test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_sampler=test_dataset.get_eval_batches(),
                **data_loader_kwargs,
            )
        else:
            test_data_loader = None

        return DataLoaderMap(
            train=train_data_loader, val=val_data_loader, test=test_data_loader
        )
