# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
from torch.utils.data.sampler import Sampler

from .dataset_base import DatasetBase


@dataclass(eq=False)  # TODO: do we need this if not init from config?
class SceneBatchSampler(Sampler[List[int]]):
    """
    A class for sampling training batches with a controlled composition
    of sequences.
    """

    dataset: DatasetBase
    batch_size: int
    num_batches: int
    # the sampler first samples a random element k from this list and then
    # takes k random frames per sequence
    images_per_seq_options: Sequence[int]

    # if True, will sample a contiguous interval of frames in the sequence
    # it first finds the connected segments within the sequence of sufficient length,
    # then samples a random pivot element among them and ideally uses it as a middle
    # of the temporal window, shifting the borders where necessary.
    # This strategy mitigates the bias against shorter segments and their boundaries.
    sample_consecutive_frames: bool = False
    # if a number > 0, then used to define the maximum difference in frame_number
    # of neighbouring frames when forming connected segments; otherwise the whole
    # sequence is considered a segment regardless of frame numbers
    consecutive_frames_max_gap: int = 0
    # same but for timestamps if they are available
    consecutive_frames_max_gap_seconds: float = 0.1

    # if True, the sampler first reads from the dataset the mapping between
    # sequence names and their categories.
    # During batch sampling, the sampler ensures uniform distribution over the categories
    # of the sampled sequences.
    category_aware: bool = True

    seq_names: List[str] = field(init=False)

    category_to_sequence_names: Dict[str, List[str]] = field(init=False)
    categories: List[str] = field(init=False)

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integral value, "
                f"but got batch_size={self.batch_size}"
            )

        if len(self.images_per_seq_options) < 1:
            raise ValueError("n_per_seq_posibilities list cannot be empty")

        self.seq_names = list(self.dataset.sequence_names())

        if self.category_aware:
            self.category_to_sequence_names = self.dataset.category_to_sequence_names()
            self.categories = list(self.category_to_sequence_names.keys())

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        for batch_idx in range(len(self)):
            batch = self._sample_batch(batch_idx)
            yield batch

    def _sample_batch(self, batch_idx) -> List[int]:
        n_per_seq = np.random.choice(self.images_per_seq_options)
        n_seqs = -(-self.batch_size // n_per_seq)  # round up

        if self.category_aware:
            # first sample categories at random, these can be repeated in the batch
            chosen_cat = _capped_random_choice(self.categories, n_seqs, replace=True)
            # then randomly sample a set of unique sequences within each category
            chosen_seq = []
            for cat, n_per_category in Counter(chosen_cat).items():
                category_chosen_seq = _capped_random_choice(
                    self.category_to_sequence_names[cat],
                    n_per_category,
                    replace=False,
                )
                chosen_seq.extend([str(s) for s in category_chosen_seq])
        else:
            chosen_seq = _capped_random_choice(
                self.seq_names,
                n_seqs,
                replace=False,
            )

        if self.sample_consecutive_frames:
            frame_idx = []
            for seq in chosen_seq:
                segment_index = self._build_segment_index(seq, n_per_seq)

                segment, idx = segment_index[np.random.randint(len(segment_index))]
                if len(segment) <= n_per_seq:
                    frame_idx.append(segment)
                else:
                    start = np.clip(idx - n_per_seq // 2, 0, len(segment) - n_per_seq)
                    frame_idx.append(segment[start : start + n_per_seq])

        else:
            frame_idx = [
                _capped_random_choice(
                    list(self.dataset.sequence_indices_in_order(seq)),
                    n_per_seq,
                    replace=False,
                )
                for seq in chosen_seq
            ]
        frame_idx = np.concatenate(frame_idx)[: self.batch_size].tolist()
        if len(frame_idx) < self.batch_size:
            warnings.warn(
                "Batch size smaller than self.batch_size!"
                + " (This is fine for experiments with a single scene and viewpooling)"
            )
        return frame_idx

    def _build_segment_index(self, seq: str, size: int) -> List[Tuple[List[int], int]]:
        """
        Returns a list of (segment, index) tuples, one per eligible frame, where
            segment is a list of frame indices in the contiguous segment the frame
            belongs to index is the frame's index within that segment.
        Segment references are repeated but the memory is shared.
        """
        if (
            self.consecutive_frames_max_gap > 0
            or self.consecutive_frames_max_gap_seconds > 0.0
        ):
            segments = self._split_to_segments(
                self.dataset.sequence_frames_in_order(seq)
            )
            segments = _cull_short_segments(segments, size)
            if not segments:
                raise AssertionError("Empty segments after culling")
        else:
            segments = [list(self.dataset.sequence_indices_in_order(seq))]

        # build an index of segment for random selection of a pivot frame
        segment_index = [
            (segment, i) for segment in segments for i in range(len(segment))
        ]

        return segment_index

    def _split_to_segments(
        self, sequence_timestamps: Iterable[Tuple[float, int, int]]
    ) -> List[List[int]]:
        if (
            self.consecutive_frames_max_gap <= 0
            and self.consecutive_frames_max_gap_seconds <= 0.0
        ):
            raise AssertionError("This function is only needed for non-trivial max_gap")

        segments = []
        last_no = -self.consecutive_frames_max_gap - 1  # will trigger a new segment
        last_ts = -self.consecutive_frames_max_gap_seconds - 1.0
        for ts, no, idx in sequence_timestamps:
            if ts <= 0.0 and no <= last_no:
                raise AssertionError(
                    "Sequence frames are not ordered while timestamps are not given"
                )

            if (
                no - last_no > self.consecutive_frames_max_gap > 0
                or ts - last_ts > self.consecutive_frames_max_gap_seconds > 0.0
            ):  # new group
                segments.append([idx])
            else:
                segments[-1].append(idx)

            last_no = no
            last_ts = ts

        return segments


def _cull_short_segments(segments: List[List[int]], min_size: int) -> List[List[int]]:
    lengths = [(len(segment), segment) for segment in segments]
    max_len, longest_segment = max(lengths)

    if max_len < min_size:
        return [longest_segment]

    return [segment for segment in segments if len(segment) >= min_size]


def _capped_random_choice(x, size, replace: bool = True):
    """
    if replace==True
        randomly chooses from x `size` elements without replacement if len(x)>size
        else allows replacement and selects `size` elements again.
    if replace==False
        randomly chooses from x `min(len(x), size)` elements without replacement
    """
    len_x = x if isinstance(x, int) else len(x)
    if replace:
        return np.random.choice(x, size=size, replace=len_x < size)
    else:
        return np.random.choice(x, size=min(size, len_x), replace=False)
