# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Optional

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
