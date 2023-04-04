# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from dataclasses import dataclass
from typing import (
    ClassVar,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
)

import torch

from pytorch3d.implicitron.dataset.frame_data import FrameData
from pytorch3d.implicitron.dataset.utils import GenericWorkaround


@dataclass(eq=False)
class DatasetBase(GenericWorkaround, torch.utils.data.Dataset[FrameData]):
    """
    Base class to describe a dataset to be used with Implicitron.

    The dataset is made up of frames, and the frames are grouped into sequences.
    Each sequence has a name (a string).
    (A sequence could be a video, or a set of images of one scene.)

    This means they have a __getitem__ which returns an instance of a FrameData,
    which will describe one frame in one sequence.
    """

    # _seq_to_idx is a member which implementations can define.
    # It maps sequence name to the sequence's global frame indices.
    # It is used for the default implementations of some functions in this class.
    # Implementations which override them are free to ignore it.
    # _seq_to_idx: Dict[str, List[int]] = field(init=False)

    def __len__(self) -> int:
        raise NotImplementedError()

    def get_frame_numbers_and_timestamps(
        self, idxs: Sequence[int], subset_filter: Optional[Sequence[str]] = None
    ) -> List[Tuple[int, float]]:
        """
        If the sequences in the dataset are videos rather than
        unordered views, then the dataset should override this method to
        return the index and timestamp in their videos of the frames whose
        indices are given in `idxs`. In addition,
        the values in _seq_to_idx should be in ascending order.
        If timestamps are absent, they should be replaced with a constant.

        This is used for letting SceneBatchSampler identify consecutive
        frames.

        Args:
            idxs: frame index in self
            subset_filter: If given, an index in idxs is ignored if the
                corresponding frame is not in any of the named subsets.

        Returns:
            tuple of
                - frame index in video
                - timestamp of frame in video
        """
        raise ValueError("This dataset does not contain videos.")

    def join(self, other_datasets: Iterable["DatasetBase"]) -> None:
        """
        Joins the current dataset with a list of other datasets of the same type.
        """
        raise NotImplementedError()

    def get_eval_batches(self) -> Optional[List[List[int]]]:
        return None

    def sequence_names(self) -> Iterable[str]:
        """Returns an iterator over sequence names in the dataset."""
        # pyre-ignore[16]
        return self._seq_to_idx.keys()

    def category_to_sequence_names(self) -> Dict[str, List[str]]:
        """
        Returns a dict mapping from each dataset category to a list of its
        sequence names.

        Returns:
            category_to_sequence_names: Dict {category_i: [..., sequence_name_j, ...]}
        """
        c2seq = defaultdict(list)
        for sequence_name in self.sequence_names():
            first_frame_idx = next(self.sequence_indices_in_order(sequence_name))
            # crashes without overriding __getitem__
            sequence_category = self[first_frame_idx].sequence_category
            c2seq[sequence_category].append(sequence_name)
        return dict(c2seq)

    def sequence_frames_in_order(
        self, seq_name: str, subset_filter: Optional[Sequence[str]] = None
    ) -> Iterator[Tuple[float, int, int]]:
        """Returns an iterator over the frame indices in a given sequence.
        We attempt to first sort by timestamp (if they are available),
        then by frame number.

        Args:
            seq_name: the name of the sequence.

        Returns:
            an iterator over triplets `(timestamp, frame_no, dataset_idx)`,
                where `frame_no` is the index within the sequence, and
                `dataset_idx` is the index within the dataset.
                `None` timestamps are replaced with 0s.
        """
        # pyre-ignore[16]
        seq_frame_indices = self._seq_to_idx[seq_name]
        nos_timestamps = self.get_frame_numbers_and_timestamps(
            seq_frame_indices, subset_filter
        )

        yield from sorted(
            [
                (timestamp, frame_no, idx)
                for idx, (frame_no, timestamp) in zip(seq_frame_indices, nos_timestamps)
            ]
        )

    def sequence_indices_in_order(
        self, seq_name: str, subset_filter: Optional[Sequence[str]] = None
    ) -> Iterator[int]:
        """Same as `sequence_frames_in_order` but returns the iterator over
        only dataset indices.
        """
        for _, _, idx in self.sequence_frames_in_order(seq_name, subset_filter):
            yield idx

    # frame_data_type is the actual type of frames returned by the dataset.
    # Collation uses its classmethod `collate`
    frame_data_type: ClassVar[Type[FrameData]] = FrameData
