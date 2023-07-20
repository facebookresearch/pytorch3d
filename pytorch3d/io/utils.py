# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import pathlib
import warnings
from typing import cast, ContextManager, IO, Optional, Union

import numpy as np
import torch
from iopath.common.file_io import PathManager
from PIL import Image

from ..common.datatypes import Device


PathOrStr = Union[pathlib.Path, str]


def _open_file(f, path_manager: PathManager, mode: str = "r") -> ContextManager[IO]:
    if isinstance(f, str):
        # pyre-fixme[6]: For 2nd argument expected `Union[typing_extensions.Literal['...
        f = path_manager.open(f, mode)
        return contextlib.closing(f)
    elif isinstance(f, pathlib.Path):
        f = f.open(mode)
        return contextlib.closing(f)
    else:
        return contextlib.nullcontext(cast(IO, f))


def _make_tensor(
    data, cols: int, dtype: torch.dtype, device: Device = "cpu"
) -> torch.Tensor:
    """
    Return a 2D tensor with the specified cols and dtype filled with data,
    even when data is empty.
    """
    if not len(data):
        return torch.zeros((0, cols), dtype=dtype, device=device)

    return torch.tensor(data, dtype=dtype, device=device)


def _check_faces_indices(
    faces_indices: torch.Tensor, max_index: int, pad_value: Optional[int] = None
) -> torch.Tensor:
    if pad_value is None:
        mask = torch.ones(faces_indices.shape[:-1]).bool()  # Keep all faces
    else:
        mask = faces_indices.ne(pad_value).any(dim=-1)
    if torch.any(faces_indices[mask] >= max_index) or torch.any(
        faces_indices[mask] < 0
    ):
        warnings.warn("Faces have invalid indices")
    return faces_indices


def _read_image(file_name: str, path_manager: PathManager, format=None):
    """
    Read an image from a file using Pillow.
    Args:
        file_name: image file path.
        path_manager: PathManager for interpreting file_name.
        format: one of ["RGB", "BGR"]
    Returns:
        image: an image of shape (H, W, C).
    """
    if format not in ["RGB", "BGR"]:
        raise ValueError("format can only be one of [RGB, BGR]; got %s", format)
    with path_manager.open(file_name, "rb") as f:
        image = Image.open(f)
        if format is not None:
            # PIL only supports RGB. First convert to RGB and flip channels
            # below for BGR.
            image = image.convert("RGB")
        image = np.asarray(image).astype(np.float32)
        if format == "BGR":
            image = image[:, :, ::-1]
        return image
