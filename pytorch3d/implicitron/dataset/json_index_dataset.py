# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import functools
import gzip
import hashlib
import json
import logging
import os
import random
import warnings
from collections import defaultdict
from itertools import islice
from typing import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.dataset_base import DatasetBase
from pytorch3d.implicitron.dataset.frame_data import FrameData, FrameDataBuilder
from pytorch3d.implicitron.dataset.utils import is_known_frame_scalar
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import CamerasBase

from tqdm import tqdm


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from typing import TypedDict

    class FrameAnnotsEntry(TypedDict):
        subset: Optional[str]
        frame_annotation: types.FrameAnnotation

else:
    FrameAnnotsEntry = dict


@registry.register
class JsonIndexDataset(DatasetBase, ReplaceableBase):
    """
    A dataset with annotations in json files like the Common Objects in 3D
    (CO3D) dataset.

    Metadata-related args::
        frame_annotations_file: A zipped json file containing metadata of the
            frames in the dataset, serialized List[types.FrameAnnotation].
        sequence_annotations_file: A zipped json file containing metadata of the
            sequences in the dataset, serialized List[types.SequenceAnnotation].
        subset_lists_file: A json file containing the lists of frames corresponding
            corresponding to different subsets (e.g. train/val/test) of the dataset;
            format: {subset: (sequence_name, frame_id, file_path)}.
        subsets: Restrict frames/sequences only to the given list of subsets
            as defined in subset_lists_file (see above).
        limit_to: Limit the dataset to the first #limit_to frames (after other
            filters have been applied).
        limit_sequences_to: Limit the dataset to the first
            #limit_sequences_to sequences (after other sequence filters have been
            applied but before frame-based filters).
        pick_sequence: A list of sequence names to restrict the dataset to.
        exclude_sequence: A list of the names of the sequences to exclude.
        limit_category_to: Restrict the dataset to the given list of categories.
        remove_empty_masks: Removes the frames with no active foreground pixels
            in the segmentation mask after thresholding (see box_crop_mask_thr).
        n_frames_per_sequence: If > 0, randomly samples #n_frames_per_sequence
            frames in each sequences uniformly without replacement if it has
            more frames than that; applied before other frame-level filters.
        seed: The seed of the random generator sampling #n_frames_per_sequence
            random frames per sequence.
        sort_frames: Enable frame annotations sorting to group frames from the
            same sequences together and order them by timestamps
        eval_batches: A list of batches that form the evaluation set;
            list of batch-sized lists of indices corresponding to __getitem__
            of this class, thus it can be used directly as a batch sampler.
        eval_batch_index:
            ( Optional[List[List[Union[Tuple[str, int, str], Tuple[str, int]]]] )
            A list of batches of frames described as (sequence_name, frame_idx)
            that can form the evaluation set, `eval_batches` will be set from this.

    Blob-loading parameters:
        dataset_root: The root folder of the dataset; all the paths in jsons are
            specified relative to this root (but not json paths themselves).
        load_images: Enable loading the frame RGB data.
        load_depths: Enable loading the frame depth maps.
        load_depth_masks: Enable loading the frame depth map masks denoting the
            depth values used for evaluation (the points consistent across views).
        load_masks: Enable loading frame foreground masks.
        load_point_clouds: Enable loading sequence-level point clouds.
        max_points: Cap on the number of loaded points in the point cloud;
            if reached, they are randomly sampled without replacement.
        mask_images: Whether to mask the images with the loaded foreground masks;
            0 value is used for background.
        mask_depths: Whether to mask the depth maps with the loaded foreground
            masks; 0 value is used for background.
        image_height: The height of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        image_width: The width of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        box_crop: Enable cropping of the image around the bounding box inferred
            from the foreground region of the loaded segmentation mask; masks
            and depth maps are cropped accordingly; cameras are corrected.
        box_crop_mask_thr: The threshold used to separate pixels into foreground
            and background based on the foreground_probability mask; if no value
            is greater than this threshold, the loader lowers it and repeats.
        box_crop_context: The amount of additional padding added to each
            dimension of the cropping bounding box, relative to box size.
    """

    frame_annotations_type: ClassVar[Type[types.FrameAnnotation]] = (
        types.FrameAnnotation
    )

    path_manager: Any = None
    frame_annotations_file: str = ""
    sequence_annotations_file: str = ""
    subset_lists_file: str = ""
    subsets: Optional[List[str]] = None
    limit_to: int = 0
    limit_sequences_to: int = 0
    pick_sequence: Tuple[str, ...] = ()
    exclude_sequence: Tuple[str, ...] = ()
    limit_category_to: Tuple[int, ...] = ()
    dataset_root: str = ""
    load_images: bool = True
    load_depths: bool = True
    load_depth_masks: bool = True
    load_masks: bool = True
    load_point_clouds: bool = False
    max_points: int = 0
    mask_images: bool = False
    mask_depths: bool = False
    image_height: Optional[int] = 800
    image_width: Optional[int] = 800
    box_crop: bool = True
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    remove_empty_masks: bool = True
    n_frames_per_sequence: int = -1
    seed: int = 0
    sort_frames: bool = False
    eval_batches: Any = None
    eval_batch_index: Any = None
    # initialised in __post_init__
    # commented because of OmegaConf (for tests to pass)
    # _frame_data_builder: FrameDataBuilder = field(init=False)
    # frame_annots: List[FrameAnnotsEntry] = field(init=False)
    # seq_annots: Dict[str, types.SequenceAnnotation] = field(init=False)
    # _seq_to_idx: Dict[str, List[int]] = field(init=False)

    def __post_init__(self) -> None:
        self._load_frames()
        self._load_sequences()
        if self.sort_frames:
            self._sort_frames()
        self._load_subset_lists()
        self._filter_db()  # also computes sequence indices
        self._extract_and_set_eval_batches()

        # pyre-ignore
        self._frame_data_builder = FrameDataBuilder(
            dataset_root=self.dataset_root,
            load_images=self.load_images,
            load_depths=self.load_depths,
            load_depth_masks=self.load_depth_masks,
            load_masks=self.load_masks,
            load_point_clouds=self.load_point_clouds,
            max_points=self.max_points,
            mask_images=self.mask_images,
            mask_depths=self.mask_depths,
            image_height=self.image_height,
            image_width=self.image_width,
            box_crop=self.box_crop,
            box_crop_mask_thr=self.box_crop_mask_thr,
            box_crop_context=self.box_crop_context,
            path_manager=self.path_manager,
        )
        logger.info(str(self))

    def _extract_and_set_eval_batches(self) -> None:
        """
        Sets eval_batches based on input eval_batch_index.
        """
        if self.eval_batch_index is not None:
            if self.eval_batches is not None:
                raise ValueError(
                    "Cannot define both eval_batch_index and eval_batches."
                )
            self.eval_batches = self.seq_frame_index_to_dataset_index(
                self.eval_batch_index
            )

    def join(self, other_datasets: Iterable[DatasetBase]) -> None:
        """
        Join the dataset with other JsonIndexDataset objects.

        Args:
            other_datasets: A list of JsonIndexDataset objects to be joined
                into the current dataset.
        """
        if not all(isinstance(d, JsonIndexDataset) for d in other_datasets):
            raise ValueError("This function can only join a list of JsonIndexDataset")
        # pyre-ignore[16]
        self.frame_annots.extend([fa for d in other_datasets for fa in d.frame_annots])
        # pyre-ignore[16]
        self.seq_annots.update(
            # https://gist.github.com/treyhunner/f35292e676efa0be1728
            functools.reduce(
                lambda a, b: {**a, **b},
                # pyre-ignore[16]
                [d.seq_annots for d in other_datasets],
            )
        )
        all_eval_batches = [
            self.eval_batches,
            *[d.eval_batches for d in other_datasets],  # pyre-ignore[16]
        ]
        if not (
            all(ba is None for ba in all_eval_batches)
            or all(ba is not None for ba in all_eval_batches)
        ):
            raise ValueError(
                "When joining datasets, either all joined datasets have to have their"
                " eval_batches defined, or all should have their eval batches undefined."
            )
        if self.eval_batches is not None:
            self.eval_batches = sum(all_eval_batches, [])
        self._invalidate_indexes(filter_seq_annots=True)

    def is_filtered(self) -> bool:
        """
        Returns `True` in case the dataset has been filtered and thus some frame annotations
        stored on the disk might be missing in the dataset object.

        Returns:
            is_filtered: `True` if the dataset has been filtered, else `False`.
        """
        return (
            self.remove_empty_masks
            or self.limit_to > 0
            or self.limit_sequences_to > 0
            or len(self.pick_sequence) > 0
            or len(self.exclude_sequence) > 0
            or len(self.limit_category_to) > 0
            or self.n_frames_per_sequence > 0
        )

    def seq_frame_index_to_dataset_index(
        self,
        seq_frame_index: List[List[Union[Tuple[str, int, str], Tuple[str, int]]]],
        allow_missing_indices: bool = False,
        remove_missing_indices: bool = False,
        suppress_missing_index_warning: bool = True,
    ) -> Union[List[List[Optional[int]]], List[List[int]]]:
        """
        Obtain indices into the dataset object given a list of frame ids.

        Args:
            seq_frame_index: The list of frame ids specified as
                `List[List[Tuple[sequence_name:str, frame_number:int]]]`. Optionally,
                Image paths relative to the dataset_root can be stored specified as well:
                `List[List[Tuple[sequence_name:str, frame_number:int, image_path:str]]]`
            allow_missing_indices: If `False`, throws an IndexError upon reaching the first
                entry from `seq_frame_index` which is missing in the dataset.
                Otherwise, depending on `remove_missing_indices`, either returns `None`
                in place of missing entries or removes the indices of missing entries.
            remove_missing_indices: Active when `allow_missing_indices=True`.
                If `False`, returns `None` in place of `seq_frame_index` entries that
                are not present in the dataset.
                If `True` removes missing indices from the returned indices.
            suppress_missing_index_warning:
                Active if `allow_missing_indices==True`. Suppressess a warning message
                in case an entry from `seq_frame_index` is missing in the dataset
                (expected in certain cases - e.g. when setting
                `self.remove_empty_masks=True`).

        Returns:
            dataset_idx: Indices of dataset entries corresponding to`seq_frame_index`.
        """
        _dataset_seq_frame_n_index = {
            seq: {
                # pyre-ignore[16]
                self.frame_annots[idx]["frame_annotation"].frame_number: idx
                for idx in seq_idx
            }
            # pyre-ignore[16]
            for seq, seq_idx in self._seq_to_idx.items()
        }

        def _get_dataset_idx(
            seq_name: str, frame_no: int, path: Optional[str] = None
        ) -> Optional[int]:
            idx_seq = _dataset_seq_frame_n_index.get(seq_name, None)
            idx = idx_seq.get(frame_no, None) if idx_seq is not None else None
            if idx is None:
                msg = (
                    f"sequence_name={seq_name} / frame_number={frame_no}"
                    " not in the dataset!"
                )
                if not allow_missing_indices:
                    raise IndexError(msg)
                if not suppress_missing_index_warning:
                    warnings.warn(msg)
                return idx
            if path is not None:
                # Check that the loaded frame path is consistent
                # with the one stored in self.frame_annots.
                assert os.path.normpath(
                    # pyre-ignore[16]
                    self.frame_annots[idx]["frame_annotation"].image.path
                ) == os.path.normpath(
                    path
                ), f"Inconsistent frame indices {seq_name, frame_no, path}."
            return idx

        dataset_idx = [
            [_get_dataset_idx(*b) for b in batch]  # pyre-ignore [6]
            for batch in seq_frame_index
        ]

        if allow_missing_indices and remove_missing_indices:
            # remove all None indices, and also batches with only None entries
            valid_dataset_idx = [
                [b for b in batch if b is not None] for batch in dataset_idx
            ]
            return [batch for batch in valid_dataset_idx if len(batch) > 0]

        return dataset_idx

    def subset_from_frame_index(
        self,
        frame_index: List[Union[Tuple[str, int], Tuple[str, int, str]]],
        allow_missing_indices: bool = True,
    ) -> "JsonIndexDataset":
        """
        Generate a dataset subset given the list of frames specified in `frame_index`.

        Args:
            frame_index: The list of frame indentifiers (as stored in the metadata)
                specified as `List[Tuple[sequence_name:str, frame_number:int]]`. Optionally,
                Image paths relative to the dataset_root can be stored specified as well:
                `List[Tuple[sequence_name:str, frame_number:int, image_path:str]]`,
                in the latter case, if imaga_path do not match the stored paths, an error
                is raised.
            allow_missing_indices: If `False`, throws an IndexError upon reaching the first
                entry from `frame_index` which is missing in the dataset.
                Otherwise, generates a subset consisting of frames entries that actually
                exist in the dataset.
        """
        # Get the indices into the frame annots.
        dataset_indices = self.seq_frame_index_to_dataset_index(
            [frame_index],
            allow_missing_indices=self.is_filtered() and allow_missing_indices,
        )[0]
        valid_dataset_indices = [i for i in dataset_indices if i is not None]

        # Deep copy the whole dataset except frame_annots, which are large so we
        # deep copy only the requested subset of frame_annots.
        memo = {id(self.frame_annots): None}  # pyre-ignore[16]
        dataset_new = copy.deepcopy(self, memo)
        dataset_new.frame_annots = copy.deepcopy(
            [self.frame_annots[i] for i in valid_dataset_indices]
        )

        # This will kill all unneeded sequence annotations.
        dataset_new._invalidate_indexes(filter_seq_annots=True)

        # Finally annotate the frame annotations with the name of the subset
        # stored in meta.
        for frame_annot in dataset_new.frame_annots:
            frame_annotation = frame_annot["frame_annotation"]
            if frame_annotation.meta is not None:
                frame_annot["subset"] = frame_annotation.meta.get("frame_type", None)

        # A sanity check - this will crash in case some entries from frame_index are missing
        # in dataset_new.
        valid_frame_index = [
            fi for fi, di in zip(frame_index, dataset_indices) if di is not None
        ]
        dataset_new.seq_frame_index_to_dataset_index(
            [valid_frame_index], allow_missing_indices=False
        )

        return dataset_new

    def __str__(self) -> str:
        # pyre-ignore[16]
        return f"JsonIndexDataset #frames={len(self.frame_annots)}"

    def __len__(self) -> int:
        # pyre-ignore[16]
        return len(self.frame_annots)

    def _get_frame_type(self, entry: FrameAnnotsEntry) -> Optional[str]:
        return entry["subset"]

    def get_all_train_cameras(self) -> CamerasBase:
        """
        Returns the cameras corresponding to all the known frames.
        """
        logger.info("Loading all train cameras.")
        cameras = []
        # pyre-ignore[16]
        for frame_idx, frame_annot in enumerate(tqdm(self.frame_annots)):
            frame_type = self._get_frame_type(frame_annot)
            if frame_type is None:
                raise ValueError("subsets not loaded")
            if is_known_frame_scalar(frame_type):
                cameras.append(self[frame_idx].camera)
        return join_cameras_as_batch(cameras)

    def __getitem__(self, index) -> FrameData:
        # pyre-ignore[16]
        if index >= len(self.frame_annots):
            raise IndexError(f"index {index} out of range {len(self.frame_annots)}")

        entry = self.frame_annots[index]["frame_annotation"]

        # pyre-ignore
        frame_data = self._frame_data_builder.build(
            entry,
            # pyre-ignore
            self.seq_annots[entry.sequence_name],
        )
        # Optional field
        frame_data.frame_type = self._get_frame_type(self.frame_annots[index])

        return frame_data

    def _load_frames(self) -> None:
        logger.info(f"Loading Co3D frames from {self.frame_annotations_file}.")
        local_file = self._local_path(self.frame_annotations_file)
        with gzip.open(local_file, "rt", encoding="utf8") as zipfile:
            frame_annots_list = types.load_dataclass(
                zipfile, List[self.frame_annotations_type]
            )
        if not frame_annots_list:
            raise ValueError("Empty dataset!")
        # pyre-ignore[16]
        self.frame_annots = [
            FrameAnnotsEntry(frame_annotation=a, subset=None) for a in frame_annots_list
        ]

    def _load_sequences(self) -> None:
        logger.info(f"Loading Co3D sequences from {self.sequence_annotations_file}.")
        local_file = self._local_path(self.sequence_annotations_file)
        with gzip.open(local_file, "rt", encoding="utf8") as zipfile:
            seq_annots = types.load_dataclass(zipfile, List[types.SequenceAnnotation])
        if not seq_annots:
            raise ValueError("Empty sequences file!")
        # pyre-ignore[16]
        self.seq_annots = {entry.sequence_name: entry for entry in seq_annots}

    def _load_subset_lists(self) -> None:
        logger.info(f"Loading Co3D subset lists from {self.subset_lists_file}.")
        if not self.subset_lists_file:
            return

        with open(self._local_path(self.subset_lists_file), "r") as f:
            subset_to_seq_frame = json.load(f)

        frame_path_to_subset = {
            path: subset
            for subset, frames in subset_to_seq_frame.items()
            for _, _, path in frames
        }
        # pyre-ignore[16]
        for frame in self.frame_annots:
            frame["subset"] = frame_path_to_subset.get(
                frame["frame_annotation"].image.path, None
            )
            if frame["subset"] is None:
                warnings.warn(
                    "Subset lists are given but don't include "
                    + frame["frame_annotation"].image.path
                )

    def _sort_frames(self) -> None:
        # Sort frames to have them grouped by sequence, ordered by timestamp
        # pyre-ignore[16]
        self.frame_annots = sorted(
            self.frame_annots,
            key=lambda f: (
                f["frame_annotation"].sequence_name,
                f["frame_annotation"].frame_timestamp or 0,
            ),
        )

    def _filter_db(self) -> None:
        if self.remove_empty_masks:
            logger.info("Removing images with empty masks.")
            # pyre-ignore[16]
            old_len = len(self.frame_annots)

            msg = "remove_empty_masks needs every MaskAnnotation.mass to be set."

            def positive_mass(frame_annot: types.FrameAnnotation) -> bool:
                mask = frame_annot.mask
                if mask is None:
                    return False
                if mask.mass is None:
                    raise ValueError(msg)
                return mask.mass > 1

            self.frame_annots = [
                frame
                for frame in self.frame_annots
                if positive_mass(frame["frame_annotation"])
            ]
            logger.info("... filtered %d -> %d" % (old_len, len(self.frame_annots)))

        # this has to be called after joining with categories!!
        subsets = self.subsets
        if subsets:
            if not self.subset_lists_file:
                raise ValueError(
                    "Subset filter is on but subset_lists_file was not given"
                )

            logger.info(f"Limiting Co3D dataset to the '{subsets}' subsets.")

            # truncate the list of subsets to the valid one
            self.frame_annots = [
                entry for entry in self.frame_annots if entry["subset"] in subsets
            ]
            if len(self.frame_annots) == 0:
                raise ValueError(f"There are no frames in the '{subsets}' subsets!")

            self._invalidate_indexes(filter_seq_annots=True)

        if len(self.limit_category_to) > 0:
            logger.info(f"Limiting dataset to categories: {self.limit_category_to}")
            # pyre-ignore[16]
            self.seq_annots = {
                name: entry
                for name, entry in self.seq_annots.items()
                if entry.category in self.limit_category_to
            }

        # sequence filters
        for prefix in ("pick", "exclude"):
            orig_len = len(self.seq_annots)
            attr = f"{prefix}_sequence"
            arr = getattr(self, attr)
            if len(arr) > 0:
                logger.info(f"{attr}: {str(arr)}")
                self.seq_annots = {
                    name: entry
                    for name, entry in self.seq_annots.items()
                    if (name in arr) == (prefix == "pick")
                }
                logger.info("... filtered %d -> %d" % (orig_len, len(self.seq_annots)))

        if self.limit_sequences_to > 0:
            self.seq_annots = dict(
                islice(self.seq_annots.items(), self.limit_sequences_to)
            )

        # retain only frames from retained sequences
        self.frame_annots = [
            f
            for f in self.frame_annots
            if f["frame_annotation"].sequence_name in self.seq_annots
        ]

        self._invalidate_indexes()

        if self.n_frames_per_sequence > 0:
            logger.info(f"Taking max {self.n_frames_per_sequence} per sequence.")
            keep_idx = []
            # pyre-ignore[16]
            for seq, seq_indices in self._seq_to_idx.items():
                # infer the seed from the sequence name, this is reproducible
                # and makes the selection differ for different sequences
                seed = _seq_name_to_seed(seq) + self.seed
                seq_idx_shuffled = random.Random(seed).sample(
                    sorted(seq_indices), len(seq_indices)
                )
                keep_idx.extend(seq_idx_shuffled[: self.n_frames_per_sequence])

            logger.info(
                "... filtered %d -> %d" % (len(self.frame_annots), len(keep_idx))
            )
            self.frame_annots = [self.frame_annots[i] for i in keep_idx]
            self._invalidate_indexes(filter_seq_annots=False)
            # sequences are not decimated, so self.seq_annots is valid

        if self.limit_to > 0 and self.limit_to < len(self.frame_annots):
            logger.info(
                "limit_to: filtered %d -> %d" % (len(self.frame_annots), self.limit_to)
            )
            self.frame_annots = self.frame_annots[: self.limit_to]
            self._invalidate_indexes(filter_seq_annots=True)

    def _invalidate_indexes(self, filter_seq_annots: bool = False) -> None:
        # update _seq_to_idx and filter seq_meta according to frame_annots change
        # if filter_seq_annots, also uldates seq_annots based on the changed _seq_to_idx
        self._invalidate_seq_to_idx()

        if filter_seq_annots:
            # pyre-ignore[16]
            self.seq_annots = {
                k: v
                for k, v in self.seq_annots.items()
                # pyre-ignore[16]
                if k in self._seq_to_idx
            }

    def _invalidate_seq_to_idx(self) -> None:
        seq_to_idx = defaultdict(list)
        # pyre-ignore[16]
        for idx, entry in enumerate(self.frame_annots):
            seq_to_idx[entry["frame_annotation"].sequence_name].append(idx)
        # pyre-ignore[16]
        self._seq_to_idx = seq_to_idx

    def _local_path(self, path: str) -> str:
        if self.path_manager is None:
            return path
        return self.path_manager.get_local_path(path)

    def get_frame_numbers_and_timestamps(
        self, idxs: Sequence[int], subset_filter: Optional[Sequence[str]] = None
    ) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for idx in idxs:
            if (
                subset_filter is not None
                # pyre-fixme[16]: `JsonIndexDataset` has no attribute `frame_annots`.
                and self.frame_annots[idx]["subset"] not in subset_filter
            ):
                continue

            frame_annotation = self.frame_annots[idx]["frame_annotation"]
            out.append(
                (frame_annotation.frame_number, frame_annotation.frame_timestamp)
            )
        return out

    def category_to_sequence_names(self) -> Dict[str, List[str]]:
        c2seq = defaultdict(list)
        # pyre-ignore
        for sequence_name, sa in self.seq_annots.items():
            c2seq[sa.category].append(sequence_name)
        return dict(c2seq)

    def get_eval_batches(self) -> Optional[List[List[int]]]:
        return self.eval_batches


def _seq_name_to_seed(seq_name) -> int:
    return int(hashlib.sha1(seq_name.encode("utf-8")).hexdigest(), 16)
