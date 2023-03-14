# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import warnings
from typing import List, Optional, Tuple

import numpy as np

import torch


DATASET_TYPE_TRAIN = "train"
DATASET_TYPE_TEST = "test"
DATASET_TYPE_KNOWN = "known"
DATASET_TYPE_UNKNOWN = "unseen"


def is_known_frame_scalar(frame_type: str) -> bool:
    """
    Given a single frame type corresponding to a single frame, return whether
    the frame is a known frame.
    """
    return frame_type.endswith(DATASET_TYPE_KNOWN)


def is_known_frame(
    frame_type: List[str], device: Optional[str] = None
) -> torch.BoolTensor:
    """
    Given a list `frame_type` of frame types in a batch, return a tensor
    of boolean flags expressing whether the corresponding frame is a known frame.
    """
    # pyre-fixme[7]: Expected `BoolTensor` but got `Tensor`.
    return torch.tensor(
        [is_known_frame_scalar(ft) for ft in frame_type],
        dtype=torch.bool,
        device=device,
    )


def is_train_frame(
    frame_type: List[str], device: Optional[str] = None
) -> torch.BoolTensor:
    """
    Given a list `frame_type` of frame types in a batch, return a tensor
    of boolean flags expressing whether the corresponding frame is a training frame.
    """
    # pyre-fixme[7]: Expected `BoolTensor` but got `Tensor`.
    return torch.tensor(
        [ft.startswith(DATASET_TYPE_TRAIN) for ft in frame_type],
        dtype=torch.bool,
        device=device,
    )


def _get_bbox_from_mask(
    mask, thr, decrease_quant: float = 0.05
) -> Tuple[int, int, int, int]:
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    if thr <= 0.0:
        warnings.warn(
            f"Empty masks_for_bbox (thr={thr}) => using full image.", stacklevel=1
        )

    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0


def _crop_around_box(tensor, bbox, impath: str = ""):
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox = _clamp_box_to_image_bounds_and_round(
        bbox,
        image_size_hw=tensor.shape[-2:],
    )
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"
    return tensor


def _clamp_box_to_image_bounds_and_round(
    bbox_xyxy: torch.Tensor,
    image_size_hw: Tuple[int, int],
) -> torch.LongTensor:
    bbox_xyxy = bbox_xyxy.clone()
    bbox_xyxy[[0, 2]] = torch.clamp(bbox_xyxy[[0, 2]], 0, image_size_hw[-1])
    bbox_xyxy[[1, 3]] = torch.clamp(bbox_xyxy[[1, 3]], 0, image_size_hw[-2])
    if not isinstance(bbox_xyxy, torch.LongTensor):
        bbox_xyxy = bbox_xyxy.round().long()
    return bbox_xyxy  # pyre-ignore [7]


def _bbox_xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    wh = xyxy[2:] - xyxy[:2]
    xywh = torch.cat([xyxy[:2], wh])
    return xywh


def _get_clamp_bbox(
    bbox: torch.Tensor,
    box_crop_context: float = 0.0,
    image_path: str = "",
) -> torch.Tensor:
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    bbox = bbox.clone()  # do not edit bbox in place

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.float()
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        raise ValueError(
            f"squashed image {image_path}!! The bounding box contains no pixels."
        )

    bbox[2:] = torch.clamp(bbox[2:], 2)  # set min height, width to 2 along both axes
    bbox_xyxy = _bbox_xywh_to_xyxy(bbox, clamp_size=2)

    return bbox_xyxy


def _rescale_bbox(bbox: torch.Tensor, orig_res, new_res) -> torch.Tensor:
    assert bbox is not None
    assert np.prod(orig_res) > 1e-8
    # average ratio of dimensions
    rel_size = (new_res[0] / orig_res[0] + new_res[1] / orig_res[1]) / 2.0
    return bbox * rel_size


def _bbox_xywh_to_xyxy(
    xywh: torch.Tensor, clamp_size: Optional[int] = None
) -> torch.Tensor:
    xyxy = xywh.clone()
    if clamp_size is not None:
        xyxy[2:] = torch.clamp(xyxy[2:], clamp_size)
    xyxy[2:] += xyxy[:2]
    return xyxy


def _get_1d_bounds(arr) -> Tuple[int, int]:
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1] + 1
