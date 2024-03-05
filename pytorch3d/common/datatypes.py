# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Union

import torch


Device = Union[str, torch.device]


def make_device(device: Device) -> torch.device:
    """
    Makes an actual torch.device object from the device specified as
    either a string or torch.device object. If the device is `cuda` without
    a specific index, the index of the current device is assigned.

    Args:
        device: Device (as str or torch.device)

    Returns:
        A matching torch.device object
    """
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == "cuda" and device.index is None:
        # If cuda but with no index, then the current cuda device is indicated.
        # In that case, we fix to that device
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    return device


def get_device(x, device: Optional[Device] = None) -> torch.device:
    """
    Gets the device of the specified variable x if it is a tensor, or
    falls back to a default CPU device otherwise. Allows overriding by
    providing an explicit device.

    Args:
        x: a torch.Tensor to get the device from or another type
        device: Device (as str or torch.device) to fall back to

    Returns:
        A matching torch.device object
    """

    # User overrides device
    if device is not None:
        return make_device(device)

    # Set device based on input tensor
    if torch.is_tensor(x):
        return x.device

    # Default device is cpu
    return torch.device("cpu")
