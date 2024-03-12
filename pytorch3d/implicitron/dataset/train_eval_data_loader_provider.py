# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
from typing import Any, Dict, Optional, Tuple

from pytorch3d.implicitron.dataset.data_loader_map_provider import (
    DataLoaderMap,
    SceneBatchSampler,
    SequenceDataLoaderMapProvider,
)
from pytorch3d.implicitron.dataset.dataset_base import DatasetBase
from pytorch3d.implicitron.dataset.dataset_map_provider import DatasetMap
from pytorch3d.implicitron.dataset.frame_data import FrameData
from pytorch3d.implicitron.tools.config import registry, run_auto_creation

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# TODO: we can merge it with SequenceDataLoaderMapProvider in PyTorch3D
# and support both eval_batches protocols
@registry.register
class TrainEvalDataLoaderMapProvider(SequenceDataLoaderMapProvider):
    """
    Implementation of DataLoaderMapProviderBase that may use internal eval batches for
    the test dataset. In particular, if `eval_batches_relpath` is set, it loads
    eval batches from that json file, otherwise test set is treated in the same way as
    train and val, i.e. the parameters `dataset_length_test` and `test_conditioning_type`
    are respected.

    If conditioning is not required, then the batch size should
    be set as 1, and most of the fields do not matter.

    If conditioning is required, each batch will contain one main
    frame first to predict and the, rest of the elements are for
    conditioning.

    If images_per_seq_options is left empty, the conditioning
    frames are picked according to the conditioning type given.
    This does not have regard to the order of frames in a
    scene, or which frames belong to what scene.

    If images_per_seq_options is given, then the conditioning types
    must be SAME and the remaining fields are used.

    Members:
        batch_size: The size of the batch of the data loader.
        num_workers: Number of data-loading threads in each data loader.
        dataset_length_train: The number of batches in a training epoch. Or 0 to mean
            an epoch is the length of the training set.
        dataset_length_val: The number of batches in a validation epoch. Or 0 to mean
            an epoch is the length of the validation set.
        dataset_length_test: used if test_dataset.eval_batches is NOT set. The number of
            batches in a testing epoch. Or 0 to mean an epoch is the length of the test
            set.
        images_per_seq_options: Possible numbers of frames sampled per sequence in a batch.
            If a conditioning_type is KNOWN or TRAIN, then this must be left at its initial
            value. Empty (the default) means that we are not careful about which frames
            come from which scene.
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

    dataset_length_train: int = 0
    dataset_length_val: int = 0
    dataset_length_test: int = 0

    images_per_seq_options: Tuple[int, ...] = ()
    sample_consecutive_frames: bool = False
    consecutive_frames_max_gap: int = 0
    consecutive_frames_max_gap_seconds: float = 0.1

    def __post_init__(self):
        run_auto_creation(self)

    def get_data_loader_map(self, datasets: DatasetMap) -> DataLoaderMap:
        """
        Returns a collection of data loaders for a given collection of datasets.
        """
        train = self._make_generic_data_loader(
            datasets.train,
            self.dataset_length_train,
            datasets.train,
        )

        val = self._make_generic_data_loader(
            datasets.val,
            self.dataset_length_val,
            datasets.train,
        )

        if datasets.test is not None and datasets.test.get_eval_batches() is not None:
            test = self._make_eval_data_loader(datasets.test)
        else:
            test = self._make_generic_data_loader(
                datasets.test,
                self.dataset_length_test,
                datasets.train,
            )

        return DataLoaderMap(train=train, val=val, test=test)

    def _make_eval_data_loader(
        self,
        dataset: Optional[DatasetBase],
    ) -> Optional[DataLoader[FrameData]]:
        if dataset is None:
            return None

        return DataLoader(
            dataset,
            batch_sampler=dataset.get_eval_batches(),
            **self._get_data_loader_common_kwargs(dataset),
        )

    def _make_generic_data_loader(
        self,
        dataset: Optional[DatasetBase],
        num_batches: int,
        train_dataset: Optional[DatasetBase],
    ) -> Optional[DataLoader[FrameData]]:
        """
        Returns the dataloader for a dataset.

        Args:
            dataset: the dataset
            num_batches: possible ceiling on number of batches per epoch
            train_dataset: the training dataset, used if conditioning_type==TRAIN
            conditioning_type: source for padding of batches
        """
        if dataset is None:
            return None

        data_loader_kwargs = self._get_data_loader_common_kwargs(dataset)

        if len(self.images_per_seq_options) > 0:
            # this is a typical few-view setup
            # conditioning comes from the same subset since subsets are split by seqs
            batch_sampler = SceneBatchSampler(
                dataset,
                self.batch_size,
                num_batches=len(dataset) if num_batches <= 0 else num_batches,
                images_per_seq_options=self.images_per_seq_options,
                sample_consecutive_frames=self.sample_consecutive_frames,
                consecutive_frames_max_gap=self.consecutive_frames_max_gap,
                consecutive_frames_max_gap_seconds=self.consecutive_frames_max_gap_seconds,
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                **data_loader_kwargs,
            )

        if self.batch_size == 1:
            # this is a typical many-view setup (without conditioning)
            return self._simple_loader(dataset, num_batches, data_loader_kwargs)

        # edge case: conditioning on train subset, typical for Nerformer-like many-view
        # there is only one sequence in all datasets, so we condition on another subset
        return self._train_loader(
            dataset, train_dataset, num_batches, data_loader_kwargs
        )

    def _get_data_loader_common_kwargs(self, dataset: DatasetBase) -> Dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "collate_fn": dataset.frame_data_type.collate,
        }
