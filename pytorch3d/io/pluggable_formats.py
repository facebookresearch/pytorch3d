# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import pathlib
from typing import Optional, Tuple

from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.io.utils import PathOrStr
from pytorch3d.structures import Meshes, Pointclouds


"""
This module has the base classes which must be extended to define
an interpreter for loading and saving data in a particular format.
These can be registered on an IO object so that they can be used in
its load_* and save_* functions.
"""


def endswith(path: PathOrStr, suffixes: Tuple[str, ...]) -> bool:
    """
    Returns whether the path ends with one of the given suffixes.
    If `path` is not actually a path, returns True. This is useful
    for allowing interpreters to bypass inappropriate paths, but
    always accepting streams.
    """
    if isinstance(path, pathlib.Path):
        return path.suffix.lower() in suffixes
    if isinstance(path, str):
        return path.lower().endswith(suffixes)
    return True


class MeshFormatInterpreter:
    """
    This is a base class for an interpreter which can read or write
    a mesh in a particular format.
    """

    def read(
        self,
        path: PathOrStr,
        include_textures: bool,
        device: Device,
        path_manager: PathManager,
        **kwargs,
    ) -> Optional[Meshes]:
        """
        Read the data from the specified file and return it as
        a Meshes object.

        Args:
            path: path to load.
            include_textures: whether to try to load texture information.
            device: torch.device to load data on to.
            path_manager: PathManager to interpret the path.

        Returns:
            None if self is not the appropriate object to interpret the given
                path.
            Otherwise, the read Meshes object.
        """
        raise NotImplementedError()

    def save(
        self,
        data: Meshes,
        path: PathOrStr,
        path_manager: PathManager,
        binary: Optional[bool],
        **kwargs,
    ) -> bool:
        """
        Save the given Meshes object to the given path.

        Args:
            data: mesh to save
            path: path to save to, which may be overwritten.
            path_manager: PathManager to interpret the path.
            binary: If there is a choice, whether to save in a binary format.

        Returns:
            False: if self is not the appropriate object to write to the given path.
            True: on success.
        """
        raise NotImplementedError()


class PointcloudFormatInterpreter:
    """
    This is a base class for an interpreter which can read or write
    a point cloud in a particular format.
    """

    def read(
        self, path: PathOrStr, device: Device, path_manager: PathManager, **kwargs
    ) -> Optional[Pointclouds]:
        """
        Read the data from the specified file and return it as
        a Pointclouds object.

        Args:
            path: path to load.
            device: torch.device to load data on to.
            path_manager: PathManager to interpret the path.

        Returns:
            None if self is not the appropriate object to interpret the given
                path.
            Otherwise, the read Pointclouds object.
        """
        raise NotImplementedError()

    def save(
        self,
        data: Pointclouds,
        path: PathOrStr,
        path_manager: PathManager,
        binary: Optional[bool],
        **kwargs,
    ) -> bool:
        """
        Save the given Pointclouds object to the given path.

        Args:
            data: point cloud object to save
            path: path to save to, which may be overwritten.
            path_manager: PathManager to interpret the path.
            binary: If there is a choice, whether to save in a binary format.

        Returns:
            False: if self is not the appropriate object to write to the given path.
            True: on success.
        """
        raise NotImplementedError()
