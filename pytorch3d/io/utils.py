# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import contextlib
import pathlib
from typing import IO, ContextManager

import numpy as np
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
