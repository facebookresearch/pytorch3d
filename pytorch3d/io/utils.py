# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import contextlib
import pathlib
import warnings
from typing import IO, ContextManager, Optional

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image


def _open_file(f, mode="r") -> ContextManager[IO]:
    if isinstance(f, str):
        f = open(f, mode)
        return contextlib.closing(f)
    elif isinstance(f, pathlib.Path):
        f = f.open(mode)
        return contextlib.closing(f)
    else:
        return contextlib.nullcontext(f)


def _make_tensor(
    data, cols: int, dtype: torch.dtype, device: str = "cpu"
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
        # pyre-fixme[16]: `torch.ByteTensor` has no attribute `any`
        mask = faces_indices.ne(pad_value).any(dim=-1)
    if torch.any(faces_indices[mask] >= max_index) or torch.any(
        faces_indices[mask] < 0
    ):
        warnings.warn("Faces have invalid indices")
    return faces_indices


def _read_image(file_name: str, format=None):
    """
    Read an image from a file using Pillow.
    Args:
        file_name: image file path.
        format: one of ["RGB", "BGR"]
    Returns:
        image: an image of shape (H, W, C).
    """
    if format not in ["RGB", "BGR"]:
        raise ValueError("format can only be one of [RGB, BGR]; got %s", format)
    with PathManager.open(file_name, "rb") as f:
        # pyre-fixme[6]: Expected `Union[str, typing.BinaryIO]` for 1st param but
        #  got `Union[typing.IO[bytes], typing.IO[str]]`.
        image = Image.open(f)
        if format is not None:
            # PIL only supports RGB. First convert to RGB and flip channels
            # below for BGR.
            image = image.convert("RGB")
        image = np.asarray(image).astype(np.float32)
        if format == "BGR":
            image = image[:, :, ::-1]
        return image