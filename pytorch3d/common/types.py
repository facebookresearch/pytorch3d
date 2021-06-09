# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Optional, Union

import torch


Device = Union[str, torch.device]


def make_device(device: Device) -> torch.device:
    return torch.device(device) if isinstance(device, str) else device


def get_device(x, device: Optional[Device] = None) -> torch.device:
    # User overrides device
    if device is not None:
        return make_device(device)

    # Set device based on input tensor
    if torch.is_tensor(x):
        return x.device

    # Default device is cpu
    return torch.device("cpu")
