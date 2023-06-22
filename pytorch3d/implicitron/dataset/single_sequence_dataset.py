# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# This file defines a base class for dataset map providers which
# provide data for a single scene.

from dataclasses import field
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pytorch3d.implicitron.tools.config import (
    Configurable,
    expand_args_fields,
    run_auto_creation,
)
from pytorch3d.renderer import CamerasBase, join_cameras_as_batch, PerspectiveCameras

from .dataset_base import DatasetBase
from .dataset_map_provider import DatasetMap, DatasetMapProviderBase, PathManagerFactory
from .frame_data import FrameData
from .utils import DATASET_TYPE_KNOWN, DATASET_TYPE_UNKNOWN

_SINGLE_SEQUENCE_NAME: str = "one_sequence"


@expand_args_fields
class SingleSceneDataset(DatasetBase, Configurable):
    """
    A dataset from images from a single scene.
    """

    images: List[torch.Tensor] = field()
    fg_probabilities: Optional[List[torch.Tensor]] = field()
    poses: List[PerspectiveCameras] = field()
    object_name: str = field()
    frame_types: List[str] = field()
    eval_batches: Optional[List[List[int]]] = field()

    def sequence_names(self) -> Iterable[str]:
        return [_SINGLE_SEQUENCE_NAME]

    def __len__(self) -> int:
        return len(self.poses)

    def sequence_frames_in_order(
        self, seq_name: str, subset_filter: Optional[Sequence[str]] = None
    ) -> Iterator[Tuple[float, int, int]]:
        for i in range(len(self)):
            if subset_filter is None or self.frame_types[i] in subset_filter:
                yield 0.0, i, i

    def __getitem__(self, index) -> FrameData:
        if index >= len(self):
            raise IndexError(f"index {index} out of range {len(self)}")
        image = self.images[index]
        pose = self.poses[index]
        frame_type = self.frame_types[index]
        fg_probability = (
            None if self.fg_probabilities is None else self.fg_probabilities[index]
        )

        frame_data = FrameData(
            frame_number=index,
            sequence_name=_SINGLE_SEQUENCE_NAME,
            sequence_category=self.object_name,
            camera=pose,
            # pyre-ignore
            image_size_hw=torch.tensor(image.shape[1:], dtype=torch.long),
            image_rgb=image,
            fg_probability=fg_probability,
            frame_type=frame_type,
        )
        return frame_data

    def get_eval_batches(self) -> Optional[List[List[int]]]:
        return self.eval_batches


# pyre-fixme[13]: Uninitialized attribute
class SingleSceneDatasetMapProviderBase(DatasetMapProviderBase):
    """
    Base for provider of data for one scene from LLFF or blender datasets.

    Members:
        base_dir: directory holding the data for the scene.
        object_name: The name of the scene (e.g. "lego"). This is just used as a label.
            It will typically be equal to the name of the directory self.base_dir.
        path_manager_factory: Creates path manager which may be used for
            interpreting paths.
        n_known_frames_for_test: If set, training frames are included in the val
            and test datasets, and this many random training frames are added to
            each test batch. If not set, test batches each contain just a single
            testing frame.
    """

    base_dir: str
    object_name: str
    path_manager_factory: PathManagerFactory
    path_manager_factory_class_type: str = "PathManagerFactory"
    n_known_frames_for_test: Optional[int] = None

    def __post_init__(self) -> None:
        run_auto_creation(self)
        self._load_data()

    def _load_data(self) -> None:
        # This must be defined by each subclass,
        # and should set the following on self.
        # - poses: a list of length-1 camera objects
        # - images: [N, 3, H, W] tensor of rgb images - floats in [0,1]
        # - fg_probabilities: None or [N, 1, H, W] of floats in [0,1]
        # - splits: List[List[int]] of indices for train/val/test subsets.
        raise NotImplementedError()

    def _get_dataset(
        self, split_idx: int, frame_type: str, set_eval_batches: bool = False
    ) -> SingleSceneDataset:
        # pyre-ignore[16]
        split = self.i_split[split_idx]
        frame_types = [frame_type] * len(split)
        fg_probabilities = (
            None
            # pyre-ignore[16]
            if self.fg_probabilities is None
            else self.fg_probabilities[split]
        )
        eval_batches = [[i] for i in range(len(split))]
        if split_idx != 0 and self.n_known_frames_for_test is not None:
            train_split = self.i_split[0]
            if set_eval_batches:
                generator = np.random.default_rng(seed=0)
                for batch in eval_batches:
                    # using permutation so that changes to n_known_frames_for_test
                    # result in consistent batches.
                    to_add = generator.permutation(len(train_split))[
                        : self.n_known_frames_for_test
                    ]
                    batch.extend((to_add + len(split)).tolist())
            split = np.concatenate([split, train_split])
            frame_types.extend([DATASET_TYPE_KNOWN] * len(train_split))

        # pyre-ignore[28]
        return SingleSceneDataset(
            object_name=self.object_name,
            # pyre-ignore[16]
            images=self.images[split],
            fg_probabilities=fg_probabilities,
            # pyre-ignore[16]
            poses=[self.poses[i] for i in split],
            frame_types=frame_types,
            eval_batches=eval_batches if set_eval_batches else None,
        )

    def get_dataset_map(self) -> DatasetMap:
        return DatasetMap(
            train=self._get_dataset(0, DATASET_TYPE_KNOWN),
            val=self._get_dataset(1, DATASET_TYPE_UNKNOWN),
            test=self._get_dataset(2, DATASET_TYPE_UNKNOWN, True),
        )

    def get_all_train_cameras(self) -> Optional[CamerasBase]:
        # pyre-ignore[16]
        cameras = [self.poses[i] for i in self.i_split[0]]
        return join_cameras_as_batch(cameras)


def _interpret_blender_cameras(
    poses: torch.Tensor, focal: float
) -> List[PerspectiveCameras]:
    """
    Convert 4x4 matrices representing cameras in blender format
    to PyTorch3D format.

    Args:
        poses: N x 3 x 4 camera matrices
        focal: ndc space focal length
    """
    pose_target_cameras = []
    for pose_target in poses:
        pose_target = pose_target[:3, :4]
        mtx = torch.eye(4, dtype=pose_target.dtype)
        mtx[:3, :3] = pose_target[:3, :3].t()
        mtx[3, :3] = pose_target[:, 3]
        mtx = mtx.inverse()

        # flip the XZ coordinates.
        mtx[:, [0, 2]] *= -1.0

        Rpt3, Tpt3 = mtx[:, :3].split([3, 1], dim=0)

        focal_length_pt3 = torch.FloatTensor([[focal, focal]])
        principal_point_pt3 = torch.FloatTensor([[0.0, 0.0]])

        cameras = PerspectiveCameras(
            focal_length=focal_length_pt3,
            principal_point=principal_point_pt3,
            R=Rpt3[None],
            T=Tpt3,
        )
        pose_target_cameras.append(cameras)
    return pose_target_cameras
