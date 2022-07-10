# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    ClassVar,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import torch
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.structures.pointclouds import join_pointclouds_as_batch, Pointclouds


@dataclass
class FrameData(Mapping[str, Any]):
    """
    A type of the elements returned by indexing the dataset object.
    It can represent both individual frames and batches of thereof;
    in this documentation, the sizes of tensors refer to single frames;
    add the first batch dimension for the collation result.

    Args:
        frame_number: The number of the frame within its sequence.
            0-based continuous integers.
        sequence_name: The unique name of the frame's sequence.
        sequence_category: The object category of the sequence.
        frame_timestamp: The time elapsed since the start of a sequence in sec.
        image_size_hw: The size of the image in pixels; (height, width) tensor
                        of shape (2,).
        image_path: The qualified path to the loaded image (with dataset_root).
        image_rgb: A Tensor of shape `(3, H, W)` holding the RGB image
            of the frame; elements are floats in [0, 1].
        mask_crop: A binary mask of shape `(1, H, W)` denoting the valid image
            regions. Regions can be invalid (mask_crop[i,j]=0) in case they
            are a result of zero-padding of the image after cropping around
            the object bounding box; elements are floats in {0.0, 1.0}.
        depth_path: The qualified path to the frame's depth map.
        depth_map: A float Tensor of shape `(1, H, W)` holding the depth map
            of the frame; values correspond to distances from the camera;
            use `depth_mask` and `mask_crop` to filter for valid pixels.
        depth_mask: A binary mask of shape `(1, H, W)` denoting pixels of the
            depth map that are valid for evaluation, they have been checked for
            consistency across views; elements are floats in {0.0, 1.0}.
        mask_path: A qualified path to the foreground probability mask.
        fg_probability: A Tensor of `(1, H, W)` denoting the probability of the
            pixels belonging to the captured object; elements are floats
            in [0, 1].
        bbox_xywh: The bounding box tightly enclosing the foreground object in the
            format (x0, y0, width, height). The convention assumes that
            `x0+width` and `y0+height` includes the boundary of the box.
            I.e., to slice out the corresponding crop from an image tensor `I`
            we execute `crop = I[..., y0:y0+height, x0:x0+width]`
        crop_bbox_xywh: The bounding box denoting the boundaries of `image_rgb`
            in the original image coordinates in the format (x0, y0, width, height).
            The convention is the same as for `bbox_xywh`. `crop_bbox_xywh` differs
            from `bbox_xywh` due to padding (which can happen e.g. due to
            setting `JsonIndexDataset.box_crop_context > 0`)
        camera: A PyTorch3D camera object corresponding the frame's viewpoint,
            corrected for cropping if it happened.
        camera_quality_score: The score proportional to the confidence of the
            frame's camera estimation (the higher the more accurate).
        point_cloud_quality_score: The score proportional to the accuracy of the
            frame's sequence point cloud (the higher the more accurate).
        sequence_point_cloud_path: The path to the sequence's point cloud.
        sequence_point_cloud: A PyTorch3D Pointclouds object holding the
            point cloud corresponding to the frame's sequence. When the object
            represents a batch of frames, point clouds may be deduplicated;
            see `sequence_point_cloud_idx`.
        sequence_point_cloud_idx: Integer indices mapping frame indices to the
            corresponding point clouds in `sequence_point_cloud`; to get the
            corresponding point cloud to `image_rgb[i]`, use
            `sequence_point_cloud[sequence_point_cloud_idx[i]]`.
        frame_type: The type of the loaded frame specified in
            `subset_lists_file`, if provided.
        meta: A dict for storing additional frame information.
    """

    frame_number: Optional[torch.LongTensor]
    sequence_name: Union[str, List[str]]
    sequence_category: Union[str, List[str]]
    frame_timestamp: Optional[torch.Tensor] = None
    image_size_hw: Optional[torch.Tensor] = None
    image_path: Union[str, List[str], None] = None
    image_rgb: Optional[torch.Tensor] = None
    # masks out padding added due to cropping the square bit
    mask_crop: Optional[torch.Tensor] = None
    depth_path: Union[str, List[str], None] = None
    depth_map: Optional[torch.Tensor] = None
    depth_mask: Optional[torch.Tensor] = None
    mask_path: Union[str, List[str], None] = None
    fg_probability: Optional[torch.Tensor] = None
    bbox_xywh: Optional[torch.Tensor] = None
    crop_bbox_xywh: Optional[torch.Tensor] = None
    camera: Optional[PerspectiveCameras] = None
    camera_quality_score: Optional[torch.Tensor] = None
    point_cloud_quality_score: Optional[torch.Tensor] = None
    sequence_point_cloud_path: Union[str, List[str], None] = None
    sequence_point_cloud: Optional[Pointclouds] = None
    sequence_point_cloud_idx: Optional[torch.Tensor] = None
    frame_type: Union[str, List[str], None] = None  # known | unseen
    meta: dict = field(default_factory=lambda: {})

    def to(self, *args, **kwargs):
        new_params = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (torch.Tensor, Pointclouds, CamerasBase)):
                new_params[f.name] = value.to(*args, **kwargs)
            else:
                new_params[f.name] = value
        return type(self)(**new_params)

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    # the following functions make sure **frame_data can be passed to functions
    def __iter__(self):
        for f in fields(self):
            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return len(fields(self))

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """

        elem = batch[0]

        if isinstance(elem, cls):
            pointcloud_ids = [id(el.sequence_point_cloud) for el in batch]
            id_to_idx = defaultdict(list)
            for i, pc_id in enumerate(pointcloud_ids):
                id_to_idx[pc_id].append(i)

            sequence_point_cloud = []
            sequence_point_cloud_idx = -np.ones((len(batch),))
            for i, ind in enumerate(id_to_idx.values()):
                sequence_point_cloud_idx[ind] = i
                sequence_point_cloud.append(batch[ind[0]].sequence_point_cloud)
            assert (sequence_point_cloud_idx >= 0).all()

            override_fields = {
                "sequence_point_cloud": sequence_point_cloud,
                "sequence_point_cloud_idx": sequence_point_cloud_idx.tolist(),
            }
            # note that the pre-collate value of sequence_point_cloud_idx is unused

            collated = {}
            for f in fields(elem):
                list_values = override_fields.get(
                    f.name, [getattr(d, f.name) for d in batch]
                )
                collated[f.name] = (
                    cls.collate(list_values)
                    if all(list_value is not None for list_value in list_values)
                    else None
                )
            return cls(**collated)

        elif isinstance(elem, Pointclouds):
            return join_pointclouds_as_batch(batch)

        elif isinstance(elem, CamerasBase):
            # TODO: don't store K; enforce working in NDC space
            return join_cameras_as_batch(batch)
        else:
            return torch.utils.data._utils.collate.default_collate(batch)


class _GenericWorkaround:
    """
    OmegaConf.structured has a weirdness when you try to apply
    it to a dataclass whose first base class is a Generic which is not
    Dict. The issue is with a function called get_dict_key_value_types
    in omegaconf/_utils.py.
    For example this fails:

        @dataclass(eq=False)
        class D(torch.utils.data.Dataset[int]):
            a: int = 3

        OmegaConf.structured(D)

    We avoid the problem by adding this class as an extra base class.
    """

    pass


@dataclass(eq=False)
class DatasetBase(_GenericWorkaround, torch.utils.data.Dataset[FrameData]):
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
        self, idxs: Sequence[int]
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
            idx: frame index in self

        Returns:
            tuple of
                - frame index in video
                - timestamp of frame in video
        """
        raise ValueError("This dataset does not contain videos.")

    def get_eval_batches(self) -> Optional[List[List[int]]]:
        return None

    def sequence_names(self) -> Iterable[str]:
        """Returns an iterator over sequence names in the dataset."""
        # pyre-ignore[16]
        return self._seq_to_idx.keys()

    def sequence_frames_in_order(
        self, seq_name: str
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
        nos_timestamps = self.get_frame_numbers_and_timestamps(seq_frame_indices)

        yield from sorted(
            [
                (timestamp, frame_no, idx)
                for idx, (frame_no, timestamp) in zip(seq_frame_indices, nos_timestamps)
            ]
        )

    def sequence_indices_in_order(self, seq_name: str) -> Iterator[int]:
        """Same as `sequence_frames_in_order` but returns the iterator over
        only dataset indices.
        """
        for _, _, idx in self.sequence_frames_in_order(seq_name):
            yield idx

    # frame_data_type is the actual type of frames returned by the dataset.
    # Collation uses its classmethod `collate`
    frame_data_type: ClassVar[Type[FrameData]] = FrameData
