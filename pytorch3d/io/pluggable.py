# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections import deque
from pathlib import Path
from typing import Deque, Optional, Union

from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.structures import Meshes, Pointclouds

from .obj_io import MeshObjFormat
from .off_io import MeshOffFormat
from .pluggable_formats import MeshFormatInterpreter, PointcloudFormatInterpreter
from .ply_io import MeshPlyFormat, PointcloudPlyFormat


"""
This module has the master functions for loading and saving data.

The main usage is via the IO object, and its methods
`load_mesh`, `save_mesh`, `load_pointcloud` and `save_pointcloud`.

For example, to load a mesh you might do::

    from pytorch3d.io import IO

    mesh = IO().load_mesh("mymesh.obj")

and to save a point cloud you might do::

    pcl = Pointclouds(...)
    IO().save_pointcloud(pcl, "output_pointcloud.obj")

"""


class IO:
    """
    This class is the interface to flexible loading and saving of meshes and point clouds.

    In simple cases the user will just initialize an instance of this class as `IO()`
    and then use its load and save functions. The arguments of the initializer are not
    usually needed.

    The user can add their own formats for saving and loading by passing their own objects
    to the register_* functions.

    Args:
        include_default_formats: If False, the built-in file formats will not be available.
            Then only user-registered formats can be used.
        path_manager: Used to customize how paths given as strings are interpreted.
    """

    def __init__(
        self,
        include_default_formats: bool = True,
        path_manager: Optional[PathManager] = None,
    ) -> None:
        if path_manager is None:
            self.path_manager = PathManager()
        else:
            self.path_manager = path_manager

        self.mesh_interpreters: Deque[MeshFormatInterpreter] = deque()
        self.pointcloud_interpreters: Deque[PointcloudFormatInterpreter] = deque()

        if include_default_formats:
            self.register_default_formats()

    def register_default_formats(self) -> None:
        self.register_meshes_format(MeshObjFormat())
        self.register_meshes_format(MeshOffFormat())
        self.register_meshes_format(MeshPlyFormat())
        self.register_pointcloud_format(PointcloudPlyFormat())

    def register_meshes_format(self, interpreter: MeshFormatInterpreter) -> None:
        """
        Register a new interpreter for a new mesh file format.

        Args:
            interpreter: the new interpreter to use, which must be an instance
                of a class which inherits MeshFormatInterpreter.
        """
        if not isinstance(interpreter, MeshFormatInterpreter):
            raise ValueError("Invalid interpreter")
        self.mesh_interpreters.appendleft(interpreter)

    def register_pointcloud_format(
        self, interpreter: PointcloudFormatInterpreter
    ) -> None:
        """
        Register a new interpreter for a new point cloud file format.

        Args:
            interpreter: the new interpreter to use, which must be an instance
                of a class which inherits PointcloudFormatInterpreter.
        """
        if not isinstance(interpreter, PointcloudFormatInterpreter):
            raise ValueError("Invalid interpreter")
        self.pointcloud_interpreters.appendleft(interpreter)

    def load_mesh(
        self,
        path: Union[str, Path],
        include_textures: bool = True,
        device: Device = "cpu",
        **kwargs,
    ) -> Meshes:
        """
        Attempt to load a mesh from the given file, using a registered format.
        Materials are not returned. If you have a .obj file with materials
        you might want to load them with the load_obj function instead.

        Args:
            path: file to read
            include_textures: whether to try to load texture information
            device: device on which to leave the data.

        Returns:
            new Meshes object containing one mesh.
        """
        for mesh_interpreter in self.mesh_interpreters:
            mesh = mesh_interpreter.read(
                path,
                include_textures=include_textures,
                path_manager=self.path_manager,
                device=device,
                **kwargs,
            )
            if mesh is not None:
                return mesh

        raise ValueError(f"No mesh interpreter found to read {path}.")

    def save_mesh(
        self,
        data: Meshes,
        path: Union[str, Path],
        binary: Optional[bool] = None,
        include_textures: bool = True,
        **kwargs,
    ) -> None:
        """
        Attempt to save a mesh to the given file, using a registered format.

        Args:
            data: a 1-element Meshes
            path: file to write
            binary: If there is a choice, whether to save in a binary format.
            include_textures: If textures are present, whether to try to save
                                them.
        """
        if not isinstance(data, Meshes):
            raise ValueError("Meshes object expected.")

        if len(data) != 1:
            raise ValueError("Can only save a single mesh.")

        for mesh_interpreter in self.mesh_interpreters:
            success = mesh_interpreter.save(
                data, path, path_manager=self.path_manager, binary=binary, **kwargs
            )
            if success:
                return

        raise ValueError(f"No mesh interpreter found to write to {path}.")

    def load_pointcloud(
        self, path: Union[str, Path], device: Device = "cpu", **kwargs
    ) -> Pointclouds:
        """
        Attempt to load a point cloud from the given file, using a registered format.

        Args:
            path: file to read
            device: Device (as str or torch.device) on which to load the data.

        Returns:
            new Pointclouds object containing one mesh.
        """
        for pointcloud_interpreter in self.pointcloud_interpreters:
            pointcloud = pointcloud_interpreter.read(
                path, path_manager=self.path_manager, device=device, **kwargs
            )
            if pointcloud is not None:
                return pointcloud

        raise ValueError(f"No point cloud interpreter found to read {path}.")

    def save_pointcloud(
        self,
        data: Pointclouds,
        path: Union[str, Path],
        binary: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Attempt to save a point cloud to the given file, using a registered format.

        Args:
            data: a 1-element Pointclouds
            path: file to write
            binary: If there is a choice, whether to save in a binary format.
        """
        if not isinstance(data, Pointclouds):
            raise ValueError("Pointclouds object expected.")

        if len(data) != 1:
            raise ValueError("Can only save a single point cloud.")

        for pointcloud_interpreter in self.pointcloud_interpreters:
            success = pointcloud_interpreter.save(
                data, path, path_manager=self.path_manager, binary=binary, **kwargs
            )
            if success:
                return

        raise ValueError(f"No point cloud interpreter found to write to {path}.")
