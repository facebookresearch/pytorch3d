# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Sequence

import torch
from pytorch3d.implicitron.tools.config import enable_get_default_args

from .implicitron_dataset import FrameData, ImplicitronDatasetBase
from .scene_batch_sampler import SceneBatchSampler


def dataloader_zoo(
    datasets: Dict[str, ImplicitronDatasetBase],
    dataset_name: str = "co3d_singlesequence",
    batch_size: int = 1,
    num_workers: int = 0,
    dataset_len: int = 1000,
    dataset_len_val: int = 1,
    images_per_seq_options: Sequence[int] = (2,),
    sample_consecutive_frames: bool = False,
    consecutive_frames_max_gap: int = 0,
    consecutive_frames_max_gap_seconds: float = 0.1,
) -> Dict[str, torch.utils.data.DataLoader]:
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
    if dataset_name not in ["co3d_singlesequence", "co3d_multisequence"]:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloaders = {}

    if dataset_name in ["co3d_singlesequence", "co3d_multisequence"]:
        for dataset_set, dataset in datasets.items():
            num_samples = {
                "train": dataset_len,
                "val": dataset_len_val,
                "test": None,
            }[dataset_set]

            if dataset_set == "test":
                batch_sampler = dataset.get_eval_batches()
            else:
                assert num_samples is not None
                num_samples = len(dataset) if num_samples <= 0 else num_samples
                batch_sampler = SceneBatchSampler(
                    dataset,
                    batch_size,
                    num_batches=num_samples,
                    images_per_seq_options=images_per_seq_options,
                    sample_consecutive_frames=sample_consecutive_frames,
                    consecutive_frames_max_gap=consecutive_frames_max_gap,
                )

            dataloaders[dataset_set] = torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=FrameData.collate,
            )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataloaders


enable_get_default_args(dataloader_zoo)
