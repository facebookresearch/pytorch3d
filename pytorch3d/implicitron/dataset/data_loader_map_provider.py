# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional, Tuple

import torch
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    RandomSampler,
    Sampler,
)

from .dataset_base import DatasetBase
from .dataset_map_provider import DatasetMap
from .frame_data import FrameData
from .scene_batch_sampler import SceneBatchSampler
from .utils import is_known_frame_scalar


@dataclass
class DataLoaderMap:
    """
    A collection of data loaders for Implicitron.

    Members:

        train: a data loader for training
        val: a data loader for validating during training
        test: a data loader for final evaluation
    """

    train: Optional[DataLoader[FrameData]]
    val: Optional[DataLoader[FrameData]]
    test: Optional[DataLoader[FrameData]]

    def __getitem__(self, split: str) -> Optional[DataLoader[FrameData]]:
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
class SimpleDataLoaderMapProvider(DataLoaderMapProviderBase):
    """
    Trivial implementation of DataLoaderMapProviderBase.

    If a dataset returns batches from get_eval_batches(), then
    they will be what the corresponding dataloader returns,
    independently of any of the fields on this class.

    Otherwise, returns shuffled batches.
    """

    batch_size: int = 1
    num_workers: int = 0
    dataset_length_train: int = 0
    dataset_length_val: int = 0
    dataset_length_test: int = 0

    def get_data_loader_map(self, datasets: DatasetMap) -> DataLoaderMap:
        """
        Returns a collection of data loaders for a given collection of datasets.
        """
        return DataLoaderMap(
            train=self._make_data_loader(
                datasets.train,
                self.dataset_length_train,
            ),
            val=self._make_data_loader(
                datasets.val,
                self.dataset_length_val,
            ),
            test=self._make_data_loader(
                datasets.test,
                self.dataset_length_test,
            ),
        )

    def _make_data_loader(
        self,
        dataset: Optional[DatasetBase],
        num_batches: int,
    ) -> Optional[DataLoader[FrameData]]:
        """
        Returns the dataloader for a dataset.

        Args:
            dataset: the dataset
            num_batches: possible ceiling on number of batches per epoch
        """
        if dataset is None:
            return None

        data_loader_kwargs = {
            "num_workers": self.num_workers,
            "collate_fn": dataset.frame_data_type.collate,
        }

        eval_batches = dataset.get_eval_batches()
        if eval_batches is not None:
            return DataLoader(
                dataset,
                batch_sampler=eval_batches,
                **data_loader_kwargs,
            )

        if num_batches > 0:
            num_samples = self.batch_size * num_batches
        else:
            num_samples = None

        # sample with replacement only if a custom number of samples is specified
        sampler = RandomSampler(
            dataset,
            replacement=num_samples is not None,
            num_samples=num_samples,
        )

        batch_sampler = BatchSampler(sampler, self.batch_size, drop_last=True)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **data_loader_kwargs,
        )


class DoublePoolBatchSampler(Sampler[List[int]]):
    """
    Batch sampler for making random batches of a single frame
    from one list and a number of known frames from another list.
    """

    def __init__(
        self,
        first_indices: List[int],
        rest_indices: List[int],
        batch_size: int,
        replacement: bool,
        num_batches: Optional[int] = None,
    ) -> None:
        """
        Args:
            first_indices: indexes of dataset items to use as the first element
                        of each batch.
            rest_indices: indexes of dataset items to use as the subsequent
                        elements of each batch. Not used if batch_size==1.
            batch_size: The common size of any batch.
            replacement: Whether the sampling of first items is with replacement.
            num_batches: The number of batches in an epoch. If 0 or None,
                        one epoch is the length of `first_indices`.
        """
        self.first_indices = first_indices
        self.rest_indices = rest_indices
        self.batch_size = batch_size
        self.replacement = replacement
        self.num_batches = None if num_batches == 0 else num_batches

        if batch_size - 1 > len(rest_indices):
            raise ValueError(
                f"Cannot make up ({batch_size})-batches from {len(self.rest_indices)}"
            )

        # copied from RandomSampler
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __len__(self) -> int:
        if self.num_batches is not None:
            return self.num_batches
        return len(self.first_indices)

    def __iter__(self) -> Iterator[List[int]]:
        num_batches = self.num_batches
        if self.replacement:
            i_first = torch.randint(
                len(self.first_indices),
                size=(len(self),),
                generator=self.generator,
            )
        elif num_batches is not None:
            n_copies = 1 + (num_batches - 1) // len(self.first_indices)
            raw_indices = [
                torch.randperm(len(self.first_indices), generator=self.generator)
                for _ in range(n_copies)
            ]
            i_first = torch.cat(raw_indices)[:num_batches]
        else:
            i_first = torch.randperm(len(self.first_indices), generator=self.generator)
        first_indices = [self.first_indices[i] for i in i_first]

        if self.batch_size == 1:
            for first_index in first_indices:
                yield [first_index]
            return

        for first_index in first_indices:
            # Consider using this class in a program which sets the seed. This use
            # of randperm means that rerunning with a higher batch_size
            # results in batches whose first elements as the first run.
            i_rest = torch.randperm(
                len(self.rest_indices),
                generator=self.generator,
            )[: self.batch_size - 1]
            yield [first_index] + [self.rest_indices[i] for i in i_rest]


class BatchConditioningType(Enum):
    """
    Ways to add conditioning frames for the val and test batches.

    SAME: Use the corresponding dataset for all elements of val batches
        without regard to frame type.
    TRAIN: Use the corresponding dataset for the first element of each
        batch, and the training dataset for the extra conditioning
            elements. No regard to frame type.
    KNOWN: Use frames from the corresponding dataset but separate them
        according to their frame_type. Each batch will contain one UNSEEN
        frame followed by many KNOWN frames.
    """

    SAME = "same"
    TRAIN = "train"
    KNOWN = "known"


@registry.register
class SequenceDataLoaderMapProvider(DataLoaderMapProviderBase):
    """
    Default implementation of DataLoaderMapProviderBase.

    If a dataset returns batches from get_eval_batches(), then
    they will be what the corresponding dataloader returns,
    independently of any of the fields on this class.

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
        dataset_length_test: The number of batches in a testing epoch. Or 0 to mean
            an epoch is the length of the test set.
        train_conditioning_type: Whether the train data loader should use
            only known frames for conditioning.
            Only used if batch_size>1 and train dataset is
            present and does not return eval_batches.
        val_conditioning_type: Whether the val data loader should use
            training frames or known frames for conditioning.
            Only used if batch_size>1 and val dataset is
            present and does not return eval_batches.
        test_conditioning_type: Whether the test data loader should use
            training frames or known frames for conditioning.
            Only used if batch_size>1 and test dataset is
            present and does not return eval_batches.
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
    train_conditioning_type: BatchConditioningType = BatchConditioningType.SAME
    val_conditioning_type: BatchConditioningType = BatchConditioningType.SAME
    test_conditioning_type: BatchConditioningType = BatchConditioningType.KNOWN
    images_per_seq_options: Tuple[int, ...] = ()
    sample_consecutive_frames: bool = False
    consecutive_frames_max_gap: int = 0
    consecutive_frames_max_gap_seconds: float = 0.1

    def get_data_loader_map(self, datasets: DatasetMap) -> DataLoaderMap:
        """
        Returns a collection of data loaders for a given collection of datasets.
        """
        return DataLoaderMap(
            train=self._make_data_loader(
                datasets.train,
                self.dataset_length_train,
                datasets.train,
                self.train_conditioning_type,
            ),
            val=self._make_data_loader(
                datasets.val,
                self.dataset_length_val,
                datasets.train,
                self.val_conditioning_type,
            ),
            test=self._make_data_loader(
                datasets.test,
                self.dataset_length_test,
                datasets.train,
                self.test_conditioning_type,
            ),
        )

    def _make_data_loader(
        self,
        dataset: Optional[DatasetBase],
        num_batches: int,
        train_dataset: Optional[DatasetBase],
        conditioning_type: BatchConditioningType,
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

        data_loader_kwargs = {
            "num_workers": self.num_workers,
            "collate_fn": dataset.frame_data_type.collate,
        }

        eval_batches = dataset.get_eval_batches()
        if eval_batches is not None:
            return DataLoader(
                dataset,
                batch_sampler=eval_batches,
                **data_loader_kwargs,
            )

        scenes_matter = len(self.images_per_seq_options) > 0
        if scenes_matter and conditioning_type != BatchConditioningType.SAME:
            raise ValueError(
                f"{conditioning_type} cannot be used with images_per_seq "
                + str(self.images_per_seq_options)
            )

        if self.batch_size == 1 or (
            not scenes_matter and conditioning_type == BatchConditioningType.SAME
        ):
            return self._simple_loader(dataset, num_batches, data_loader_kwargs)

        if scenes_matter:
            assert conditioning_type == BatchConditioningType.SAME
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

        if conditioning_type == BatchConditioningType.TRAIN:
            return self._train_loader(
                dataset, train_dataset, num_batches, data_loader_kwargs
            )

        assert conditioning_type == BatchConditioningType.KNOWN
        return self._known_loader(dataset, num_batches, data_loader_kwargs)

    def _simple_loader(
        self,
        dataset: DatasetBase,
        num_batches: int,
        data_loader_kwargs: dict,
    ) -> DataLoader[FrameData]:
        """
        Return a simple loader for frames in the dataset.

        This is equivalent to
            Dataloader(dataset, batch_size=self.batch_size, **data_loader_kwargs)
        except that num_batches is fixed.

        Args:
            dataset: the dataset
            num_batches: possible ceiling on number of batches per epoch
            data_loader_kwargs: common args for dataloader
        """
        if num_batches > 0:
            num_samples = self.batch_size * num_batches
            replacement = True
        else:
            num_samples = None
            replacement = False
        sampler = RandomSampler(
            dataset, replacement=replacement, num_samples=num_samples
        )
        batch_sampler = BatchSampler(sampler, self.batch_size, drop_last=True)
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **data_loader_kwargs,
        )

    def _train_loader(
        self,
        dataset: DatasetBase,
        train_dataset: Optional[DatasetBase],
        num_batches: int,
        data_loader_kwargs: dict,
    ) -> DataLoader[FrameData]:
        """
        Return the loader for TRAIN conditioning.

        Args:
            dataset: the dataset
            train_dataset: the training dataset
            num_batches: possible ceiling on number of batches per epoch
            data_loader_kwargs: common args for dataloader
        """
        if train_dataset is None:
            raise ValueError("No training data for conditioning.")
        length = len(dataset)
        first_indices = list(range(length))
        rest_indices = list(range(length, length + len(train_dataset)))
        sampler = DoublePoolBatchSampler(
            first_indices=first_indices,
            rest_indices=rest_indices,
            batch_size=self.batch_size,
            replacement=True,
            num_batches=num_batches,
        )
        return DataLoader(
            ConcatDataset([dataset, train_dataset]),
            batch_sampler=sampler,
            **data_loader_kwargs,
        )

    def _known_loader(
        self,
        dataset: DatasetBase,
        num_batches: int,
        data_loader_kwargs: dict,
    ) -> DataLoader[FrameData]:
        """
        Return the loader for KNOWN conditioning.

        Args:
            dataset: the dataset
            num_batches: possible ceiling on number of batches per epoch
            data_loader_kwargs: common args for dataloader
        """
        first_indices, rest_indices = [], []
        for idx in range(len(dataset)):
            frame_type = dataset[idx].frame_type
            assert isinstance(frame_type, str)
            if is_known_frame_scalar(frame_type):
                rest_indices.append(idx)
            else:
                first_indices.append(idx)
        sampler = DoublePoolBatchSampler(
            first_indices=first_indices,
            rest_indices=rest_indices,
            batch_size=self.batch_size,
            replacement=True,
            num_batches=num_batches,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            **data_loader_kwargs,
        )
