# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from pytorch3d.implicitron.tools.config import enable_get_default_args

from .dataset_zoo import Datasets
from .implicitron_dataset import FrameData, ImplicitronDatasetBase
from .scene_batch_sampler import SceneBatchSampler


@dataclass
class Dataloaders:
    """
    A provider of dataloaders for implicitron.

    Members:

        train: a dataloader for training
        val: a dataloader for validating during training
        test: a dataloader for final evaluation
    """

    train: Optional[torch.utils.data.DataLoader[FrameData]]
    val: Optional[torch.utils.data.DataLoader[FrameData]]
    test: Optional[torch.utils.data.DataLoader[FrameData]]


def dataloader_zoo(
    datasets: Datasets,
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_len: int = 1000,
    dataset_len_val: int = 1,
    images_per_seq_options: Sequence[int] = (2,),
    sample_consecutive_frames: bool = False,
    consecutive_frames_max_gap: int = 0,
    consecutive_frames_max_gap_seconds: float = 0.1,
) -> Dataloaders:
    """
    Returns a set of dataloaders for a given set of datasets.

    Args:
        datasets: A dictionary containing the
            `"dataset_subset_name": torch_dataset_object` key, value pairs.
        dataset_name: The name of the returned dataset.
        batch_size: The size of the batch of the dataloader.
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

    Returns:
        dataloaders: A dictionary containing the
            `"dataset_subset_name": torch_dataloader_object` key, value pairs.
    """

    dataloader_kwargs = {"num_workers": num_workers, "collate_fn": FrameData.collate}

    def train_or_val_loader(
        dataset: Optional[ImplicitronDatasetBase], num_batches: int
    ) -> Optional[torch.utils.data.DataLoader]:
        if dataset is None:
            return None
        batch_sampler = SceneBatchSampler(
            dataset,
            batch_size,
            num_batches=len(dataset) if num_batches <= 0 else num_batches,
            images_per_seq_options=images_per_seq_options,
            sample_consecutive_frames=sample_consecutive_frames,
            consecutive_frames_max_gap=consecutive_frames_max_gap,
            consecutive_frames_max_gap_seconds=consecutive_frames_max_gap_seconds,
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **dataloader_kwargs,
        )

    train_dataloader = train_or_val_loader(datasets.train, dataset_len)
    val_dataloader = train_or_val_loader(datasets.val, dataset_len_val)

    test_dataset = datasets.test
    if test_dataset is not None:
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_sampler=test_dataset.get_eval_batches(),
            **dataloader_kwargs,
        )
    else:
        test_dataloader = None

    return Dataloaders(train=train_dataloader, val=val_dataloader, test=test_dataloader)


enable_get_default_args(dataloader_zoo)
