# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.common.datatypes import Device, make_device
from pytorch3d.transforms import Scale, Transform3d

from . import utils as struct_utils


_Scalar = Union[int, float]
_Vector = Union[torch.Tensor, Tuple[_Scalar, ...], List[_Scalar]]
_ScalarOrVector = Union[_Scalar, _Vector]

_VoxelSize = _ScalarOrVector
_Translation = _Vector

_TensorBatch = Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]]
_ALL_CONTENT: slice = slice(0, None)


class Volumes:
    """
    This class provides functions for working with batches of volumetric grids
    of possibly varying spatial sizes.

    VOLUME DENSITIES

    The Volumes class can be either constructed from a 5D tensor of
    `densities` of size `batch x density_dim x depth x height x width` or
    from a list of differently-sized 4D tensors `[D_1, ..., D_batch]`,
    where each `D_i` is of size `[density_dim x depth_i x height_i x width_i]`.

    In case the `Volumes` object is initialized from the list of `densities`,
    the list of tensors is internally converted to a single 5D tensor by
    zero-padding the relevant dimensions. Both list and padded representations can be
    accessed with the `Volumes.densities()` or `Volumes.densities_list()` getters.
    The sizes of the individual volumes in the structure can be retrieved
    with the `Volumes.get_grid_sizes()` getter.

    The `Volumes` class is immutable. I.e. after generating a `Volumes` object,
    one cannot change its properties, such as `self._densities` or `self._features`
    anymore.


    VOLUME FEATURES

    While the `densities` field is intended to represent various measures of the
    "density" of the volume cells (opacity, signed/unsigned distances
    from the nearest surface, ...), one can additionally initialize the
    object with the `features` argument. `features` are either a 5D tensor
    of shape `batch x feature_dim x depth x height x width` or a list of
    of differently-sized 4D tensors `[F_1, ..., F_batch]`,
    where each `F_i` is of size `[feature_dim x depth_i x height_i x width_i]`.
    `features` are intended to describe other properties of volume cells,
    such as per-voxel 3D vectors of RGB colors that can be later used
    for rendering the volume.


    VOLUME COORDINATES

    Additionally, using the `VolumeLocator` class the `Volumes` class keeps track
    of the locations of the centers of the volume cells in the local volume
    coordinates as well as in the world coordinates.

        Local coordinates:
            - Represent the locations of the volume cells in the local coordinate
              frame of the volume.
            - The center of the voxel indexed with `[·, ·, 0, 0, 0]` in the volume
              has its 3D local coordinate set to `[-1, -1, -1]`, while the voxel
              at index `[·, ·, depth_i-1, height_i-1, width_i-1]` has its
              3D local coordinate set to `[1, 1, 1]`.
            - The first/second/third coordinate of each of the 3D per-voxel
              XYZ vector denotes the horizontal/vertical/depth-wise position
              respectively. I.e the order of the coordinate dimensions in the
              volume is reversed w.r.t. the order of the 3D coordinate vectors.
            - The intermediate coordinates between `[-1, -1, -1]` and `[1, 1, 1]`.
              are linearly interpolated over the spatial dimensions of the volume.
            - Note that the convention is the same as for the 5D version of the
              `torch.nn.functional.grid_sample` function called with
              the same value of `align_corners` argument.
            - Note that the local coordinate convention of `Volumes`
              (+X = left to right, +Y = top to bottom, +Z = away from the user)
              is *different* from the world coordinate convention of the
              renderer for `Meshes` or `Pointclouds`
              (+X = right to left, +Y = bottom to top, +Z = away from the user).

        World coordinates:
            - These define the locations of the centers of the volume cells
              in the world coordinates.
            - They are specified with the following mapping that converts
              points `x_local` in the local coordinates to points `x_world`
              in the world coordinates::

                    x_world = (
                        x_local * (volume_size - 1) * 0.5 * voxel_size
                    ) - volume_translation,

              here `voxel_size` specifies the size of each voxel of the volume,
              and `volume_translation` is the 3D offset of the central voxel of
              the volume w.r.t. the origin of the world coordinate frame.
              Both `voxel_size` and `volume_translation` are specified in
              the world coordinate units. `volume_size` is the spatial size of
              the volume in form of a 3D vector `[width, height, depth]`.
            - Given the above definition of `x_world`, one can derive the
              inverse mapping from `x_world` to `x_local` as follows::

                    x_local = (
                        (x_world + volume_translation) / (0.5 * voxel_size)
                    ) / (volume_size - 1)

            - For a trivial volume with `volume_translation==[0, 0, 0]`
              with `voxel_size=-1`, `x_world` would range
              from -(volume_size-1)/2` to `+(volume_size-1)/2`.

    Coordinate tensors that denote the locations of each of the volume cells in
    local / world coordinates (with shape `(depth x height x width x 3)`)
    can be retrieved by calling the `Volumes.get_coord_grid()` getter with the
    appropriate `world_coordinates` argument.

    Internally, the mapping between `x_local` and `x_world` is represented
    as a `Transform3d` object `Volumes.VolumeLocator._local_to_world_transform`.
    Users can access the relevant transformations with the
    `Volumes.get_world_to_local_coords_transform()` and
    `Volumes.get_local_to_world_coords_transform()`
    functions.

    Example coordinate conversion:
        - For a "trivial" volume with `voxel_size = 1.`,
          `volume_translation=[0., 0., 0.]`, and the spatial size of
          `DxHxW = 5x5x5`, the point `x_world = (-2, 0, 2)` gets mapped
          to `x_local=(-1, 0, 1)`.
        - For a "trivial" volume `v` with `voxel_size = 1.`,
          `volume_translation=[0., 0., 0.]`, the following holds:

                torch.nn.functional.grid_sample(
                    v.densities(),
                    v.get_coord_grid(world_coordinates=False),
                    align_corners=align_corners,
                ) == v.densities(),

            i.e. sampling the volume at trivial local coordinates
            (no scaling with `voxel_size`` or shift with `volume_translation`)
            results in the same volume.
    """

    def __init__(
        self,
        densities: _TensorBatch,
        features: Optional[_TensorBatch] = None,
        voxel_size: _VoxelSize = 1.0,
        volume_translation: _Translation = (0.0, 0.0, 0.0),
        align_corners: bool = True,
    ) -> None:
        """
        Args:
            **densities**: Batch of input feature volume occupancies of shape
                `(minibatch, density_dim, depth, height, width)`, or a list
                of 4D tensors `[D_1, ..., D_minibatch]` where each `D_i` has
                shape `(density_dim, depth_i, height_i, width_i)`.
                Typically, each voxel contains a non-negative number
                corresponding to its opaqueness.
            **features**: Batch of input feature volumes of shape:
                `(minibatch, feature_dim, depth, height, width)` or a list
                of 4D tensors `[F_1, ..., F_minibatch]` where each `F_i` has
                shape `(feature_dim, depth_i, height_i, width_i)`.
                The field is optional and can be set to `None` in case features are
                not required.
            **voxel_size**: Denotes the size of each volume voxel in world units.
                Has to be one of:
                a) A scalar (square voxels)
                b) 3-tuple or a 3-list of scalars
                c) a Tensor of shape (3,)
                d) a Tensor of shape (minibatch, 3)
                e) a Tensor of shape (minibatch, 1)
                f) a Tensor of shape (1,) (square voxels)
            **volume_translation**: Denotes the 3D translation of the center
                of the volume in world units. Has to be one of:
                a) 3-tuple or a 3-list of scalars
                b) a Tensor of shape (3,)
                c) a Tensor of shape (minibatch, 3)
                d) a Tensor of shape (1,) (square voxels)
            **align_corners**: If set (default), the coordinates of the corner voxels are
                exactly −1 or +1 in the local coordinate system. Otherwise, the coordinates
                correspond to the centers of the corner voxels. Cf. the namesake argument to
                `torch.nn.functional.grid_sample`.
        """

        # handle densities
        densities_, grid_sizes = self._convert_densities_features_to_tensor(
            densities, "densities"
        )

        # take device from densities
        self.device = densities_.device

        # assign to the internal buffers
        self._densities = densities_

        # assign a coordinate transformation member
        self.locator = VolumeLocator(
            batch_size=len(self),
            grid_sizes=grid_sizes,
            voxel_size=voxel_size,
            volume_translation=volume_translation,
            device=self.device,
            align_corners=align_corners,
        )

        # handle features
        self._features = None
        if features is not None:
            self._set_features(features)

    def _convert_densities_features_to_tensor(
        self, x: _TensorBatch, var_name: str
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Handle the `densities` or `features` arguments to the constructor.
        """
        if isinstance(x, (list, tuple)):
            x_tensor = struct_utils.list_to_padded(x)
            if any(x_.ndim != 4 for x_ in x):
                raise ValueError(
                    f"`{var_name}` has to be a list of 4-dim tensors of shape: "
                    f"({var_name}_dim, height, width, depth)"
                )
            if any(x_.shape[0] != x[0].shape[0] for x_ in x):
                raise ValueError(
                    f"Each entry in the list of `{var_name}` has to have the "
                    "same number of channels (first dimension in the tensor)."
                )
            x_shapes = torch.stack(
                [
                    torch.tensor(
                        list(x_.shape[1:]), dtype=torch.long, device=x_tensor.device
                    )
                    for x_ in x
                ],
                dim=0,
            )
        elif torch.is_tensor(x):
            if x.ndim != 5:
                raise ValueError(
                    f"`{var_name}` has to be a 5-dim tensor of shape: "
                    f"(minibatch, {var_name}_dim, height, width, depth)"
                )
            x_tensor = x
            x_shapes = torch.tensor(
                list(x.shape[2:]), dtype=torch.long, device=x.device
            )[None].repeat(x.shape[0], 1)
        else:
            raise ValueError(
                f"{var_name} must be either a list or a tensor with "
                f"shape (batch_size, {var_name}_dim, H, W, D)."
            )
        # pyre-ignore[7]
        return x_tensor, x_shapes

    def __len__(self) -> int:
        return self._densities.shape[0]

    def __getitem__(
        self,
        index: Union[
            int, List[int], Tuple[int], slice, torch.BoolTensor, torch.LongTensor
        ],
    ) -> "Volumes":
        """
        Args:
            index: Specifying the index of the volume to retrieve.
                Can be an int, slice, list of ints or a boolean or a long tensor.

        Returns:
            Volumes object with selected volumes. The tensors are not cloned.
        """
        if isinstance(index, int):
            index = torch.LongTensor([index])
        elif isinstance(index, (slice, list, tuple)):
            pass
        elif torch.is_tensor(index):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
        else:
            raise IndexError(index)

        new = self.__class__(
            # pyre-fixme[16]: `Optional` has no attribute `__getitem__`.
            features=self.features()[index] if self._features is not None else None,
            densities=self.densities()[index],
        )
        # dont forget to update grid_sizes!
        self.locator._copy_transform_and_sizes(new.locator, index=index)
        return new

    def features(self) -> Optional[torch.Tensor]:
        """
        Returns the features of the volume.

        Returns:
            **features**: The tensor of volume features.
        """
        return self._features

    def densities(self) -> torch.Tensor:
        """
        Returns the densities of the volume.

        Returns:
            **densities**: The tensor of volume densities.
        """
        return self._densities

    def densities_list(self) -> List[torch.Tensor]:
        """
        Get the list representation of the densities.

        Returns:
            list of tensors of densities of shape (dim_i, D_i, H_i, W_i).
        """
        return self._features_densities_list(self.densities())

    def features_list(self) -> List[torch.Tensor]:
        """
        Get the list representation of the features.

        Returns:
            list of tensors of features of shape (dim_i, D_i, H_i, W_i)
            or `None` for feature-less volumes.
        """
        features_ = self.features()
        if features_ is None:
            # No features provided so return None
            # pyre-fixme[7]: Expected `List[torch.Tensor]` but got `None`.
            return None
        return self._features_densities_list(features_)

    def get_align_corners(self) -> bool:
        """
        Return whether the corners of the voxels should be aligned with the
        image pixels.
        """
        return self.locator._align_corners

    def _features_densities_list(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Retrieve the list representation of features/densities.

        Args:
            x: self.features() or self.densities()

        Returns:
            list of tensors of features/densities of shape (dim_i, D_i, H_i, W_i).
        """
        x_dim = x.shape[1]
        pad_sizes = torch.nn.functional.pad(
            self.get_grid_sizes(), [1, 0], mode="constant", value=x_dim
        )
        x_list = struct_utils.padded_to_list(x, pad_sizes.tolist())
        return x_list

    def update_padded(
        self, new_densities: torch.Tensor, new_features: Optional[torch.Tensor] = None
    ) -> "Volumes":
        """
        Returns a Volumes structure with updated padded tensors and copies of
        the auxiliary tensors `self._local_to_world_transform`,
        `device` and `self._grid_sizes`. This function allows for an update of
        densities (and features) without having to explicitly
        convert it to the list representation for heterogeneous batches.

        Args:
            new_densities: FloatTensor of shape (N, dim_density, D, H, W)
            new_features: (optional) FloatTensor of shape (N, dim_feature, D, H, W)

        Returns:
            Volumes with updated features and densities
        """
        new = copy.copy(self)
        new._set_densities(new_densities)
        if new_features is None:
            new._features = None
        else:
            new._set_features(new_features)
        return new

    def _set_features(self, features: _TensorBatch) -> None:
        self._set_densities_features("features", features)

    def _set_densities(self, densities: _TensorBatch) -> None:
        self._set_densities_features("densities", densities)

    def _set_densities_features(self, var_name: str, x: _TensorBatch) -> None:
        x_tensor, grid_sizes = self._convert_densities_features_to_tensor(x, var_name)
        if x_tensor.device != self.device:
            raise ValueError(
                f"`{var_name}` have to be on the same device as `self.densities`."
            )
        if len(x_tensor.shape) != 5:
            raise ValueError(
                f"{var_name} has to be a 5-dim tensor of shape: "
                f"(minibatch, {var_name}_dim, height, width, depth)"
            )

        if not (
            (self.get_grid_sizes().shape == grid_sizes.shape)
            and torch.allclose(self.get_grid_sizes(), grid_sizes)
        ):
            raise ValueError(
                f"The size of every grid in `{var_name}` has to match the size of"
                "the corresponding `densities` grid."
            )
        setattr(self, "_" + var_name, x_tensor)

    def clone(self) -> "Volumes":
        """
        Deep copy of Volumes object. All internal tensors are cloned
        individually.

        Returns:
            new Volumes object.
        """
        return copy.deepcopy(self)

    def to(self, device: Device, copy: bool = False) -> "Volumes":
        """
        Match the functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
            device: Device (as str or torch.device) for the new tensor.
            copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
            Volumes object.
        """
        device_ = make_device(device)
        if not copy and self.device == device_:
            return self

        other = self.clone()
        if self.device == device_:
            return other

        other.device = device_
        other._densities = self._densities.to(device_)
        if self._features is not None:
            # pyre-fixme[16]: `Optional` has no attribute `to`.
            other._features = self.features().to(device_)
        self.locator._copy_transform_and_sizes(other.locator, device=device_)
        other.locator = other.locator.to(device, copy)
        return other

    def cpu(self) -> "Volumes":
        return self.to("cpu")

    def cuda(self) -> "Volumes":
        return self.to("cuda")

    def get_grid_sizes(self) -> torch.LongTensor:
        """
        Returns the sizes of individual volumetric grids in the structure.

        Returns:
            **grid_sizes**: Tensor of spatial sizes of each of the volumes
                of size (batchsize, 3), where i-th row holds (D_i, H_i, W_i).
        """
        return self.locator.get_grid_sizes()

    def get_local_to_world_coords_transform(self) -> Transform3d:
        """
        Return a Transform3d object that converts points in the
        the local coordinate frame of the volume to world coordinates.
        Local volume coordinates are scaled s.t. the coordinates along one
        side of the volume are in range [-1, 1].

        Returns:
            **local_to_world_transform**: A Transform3d object converting
                points from local coordinates to the world coordinates.
        """
        return self.locator.get_local_to_world_coords_transform()

    def get_world_to_local_coords_transform(self) -> Transform3d:
        """
        Return a Transform3d object that converts points in the
        world coordinates to the local coordinate frame of the volume.
        Local volume coordinates are scaled s.t. the coordinates along one
        side of the volume are in range [-1, 1].

        Returns:
            **world_to_local_transform**: A Transform3d object converting
                points from world coordinates to local coordinates.
        """
        return self.get_local_to_world_coords_transform().inverse()

    def world_to_local_coords(self, points_3d_world: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of 3D point coordinates `points_3d_world` of shape
        (minibatch, ..., dim) in the world coordinates to
        the local coordinate frame of the volume. Local volume
        coordinates are scaled s.t. the coordinates along one side of the volume
        are in range [-1, 1].

        Args:
            **points_3d_world**: A tensor of shape `(minibatch, ..., 3)`
                containing the 3D coordinates of a set of points that will
                be converted from the local volume coordinates (ranging
                within [-1, 1]) to the world coordinates
                defined by the `self.center` and `self.voxel_size` parameters.

        Returns:
            **points_3d_local**: `points_3d_world` converted to the local
                volume coordinates of shape `(minibatch, ..., 3)`.
        """
        return self.locator.world_to_local_coords(points_3d_world)

    def local_to_world_coords(self, points_3d_local: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of 3D point coordinates `points_3d_local` of shape
        (minibatch, ..., dim) in the local coordinate frame of the volume
        to the world coordinates.

        Args:
            **points_3d_local**: A tensor of shape `(minibatch, ..., 3)`
                containing the 3D coordinates of a set of points that will
                be converted from the local volume coordinates (ranging
                within [-1, 1]) to the world coordinates
                defined by the `self.center` and `self.voxel_size` parameters.

        Returns:
            **points_3d_world**: `points_3d_local` converted to the world
                coordinates of the volume of shape `(minibatch, ..., 3)`.
        """
        return self.locator.local_to_world_coords(points_3d_local)

    def get_coord_grid(self, world_coordinates: bool = True) -> torch.Tensor:
        """
        Return the 3D coordinate grid of the volumetric grid
        in local (`world_coordinates=False`) or world coordinates
        (`world_coordinates=True`).

        The grid records location of each center of the corresponding volume voxel.

        Local coordinates are scaled s.t. the values along one side of the
        volume are in range [-1, 1].

        Args:
            **world_coordinates**: if `True`, the method
                returns the grid in the world coordinates,
                otherwise, in local coordinates.

        Returns:
            **coordinate_grid**: The grid of coordinates of shape
                `(minibatch, depth, height, width, 3)`, where `minibatch`,
                `height`, `width` and `depth` are the batch size, height, width
                and depth of the volume `features` or `densities`.
        """
        return self.locator.get_coord_grid(world_coordinates)


class VolumeLocator:
    """
    The `VolumeLocator` class keeps track of the locations of the
    centers of the volume cells in the local volume coordinates as well as in
    the world coordinates for a voxel grid structure in 3D.

        Local coordinates:
            - Represent the locations of the volume cells in the local coordinate
              frame of the volume.
            - The center of the voxel indexed with `[·, ·, 0, 0, 0]` in the volume
              has its 3D local coordinate set to `[-1, -1, -1]`, while the voxel
              at index `[·, ·, depth_i-1, height_i-1, width_i-1]` has its
              3D local coordinate set to `[1, 1, 1]`.
            - The first/second/third coordinate of each of the 3D per-voxel
              XYZ vector denotes the horizontal/vertical/depth-wise position
              respectively. I.e the order of the coordinate dimensions in the
              volume is reversed w.r.t. the order of the 3D coordinate vectors.
            - The intermediate coordinates between `[-1, -1, -1]` and `[1, 1, 1]`.
              are linearly interpolated over the spatial dimensions of the volume.
            - Note that the convention is the same as for the 5D version of the
              `torch.nn.functional.grid_sample` function called with
              the same value of `align_corners` argument.
            - Note that the local coordinate convention of `VolumeLocator`
              (+X = left to right, +Y = top to bottom, +Z = away from the user)
              is *different* from the world coordinate convention of the
              renderer for `Meshes` or `Pointclouds`
              (+X = right to left, +Y = bottom to top, +Z = away from the user).

        World coordinates:
            - These define the locations of the centers of the volume cells
              in the world coordinates.
            - They are specified with the following mapping that converts
              points `x_local` in the local coordinates to points `x_world`
              in the world coordinates::

                    x_world = (
                        x_local * (volume_size - 1) * 0.5 * voxel_size
                    ) - volume_translation,

              here `voxel_size` specifies the size of each voxel of the volume,
              and `volume_translation` is the 3D offset of the central voxel of
              the volume w.r.t. the origin of the world coordinate frame.
              Both `voxel_size` and `volume_translation` are specified in
              the world coordinate units. `volume_size` is the spatial size of
              the volume in form of a 3D vector `[width, height, depth]`.
            - Given the above definition of `x_world`, one can derive the
              inverse mapping from `x_world` to `x_local` as follows::

                    x_local = (
                        (x_world + volume_translation) / (0.5 * voxel_size)
                    ) / (volume_size - 1)

            - For a trivial volume with `volume_translation==[0, 0, 0]`
              with `voxel_size=-1`, `x_world` would range
              from -(volume_size-1)/2` to `+(volume_size-1)/2`.

    Coordinate tensors that denote the locations of each of the volume cells in
    local / world coordinates (with shape `(depth x height x width x 3)`)
    can be retrieved by calling the `VolumeLocator.get_coord_grid()` getter with the
    appropriate `world_coordinates` argument.

    Internally, the mapping between `x_local` and `x_world` is represented
    as a `Transform3d` object `VolumeLocator._local_to_world_transform`.
    Users can access the relevant transformations with the
    `VolumeLocator.get_world_to_local_coords_transform()` and
    `VolumeLocator.get_local_to_world_coords_transform()`
    functions.

    Example coordinate conversion:
        - For a "trivial" volume with `voxel_size = 1.`,
          `volume_translation=[0., 0., 0.]`, and the spatial size of
          `DxHxW = 5x5x5`, the point `x_world = (-2, 0, 2)` gets mapped
          to `x_local=(-1, 0, 1)`.
        - For a "trivial" volume `v` with `voxel_size = 1.`,
          `volume_translation=[0., 0., 0.]`, the following holds::

                torch.nn.functional.grid_sample(
                    v.densities(),
                    v.get_coord_grid(world_coordinates=False),
                    align_corners=align_corners,
                ) == v.densities(),

            i.e. sampling the volume at trivial local coordinates
            (no scaling with `voxel_size`` or shift with `volume_translation`)
            results in the same volume.
    """

    def __init__(
        self,
        batch_size: int,
        grid_sizes: Union[
            torch.LongTensor, Tuple[int, int, int], List[torch.LongTensor]
        ],
        device: torch.device,
        voxel_size: _VoxelSize = 1.0,
        volume_translation: _Translation = (0.0, 0.0, 0.0),
        align_corners: bool = True,
    ):
        """
        **batch_size** : Batch size of the underlying grids
        **grid_sizes** : Represents the resolutions of different grids in the batch. Can be
                a) tuple of form (H, W, D)
                b) list/tuple of length batch_size of lists/tuples of form (H, W, D)
                c) torch.Tensor of shape (batch_size, H, W, D)
            H, W, D are height, width, depth respectively.  If `grid_sizes` is a tuple than
            all the  grids in the batch have the same resolution.
        **voxel_size**: Denotes the size of each volume voxel in world units.
            Has to be one of:
            a) A scalar (square voxels)
            b) 3-tuple or a 3-list of scalars
            c) a Tensor of shape (3,)
            d) a Tensor of shape (minibatch, 3)
            e) a Tensor of shape (minibatch, 1)
            f) a Tensor of shape (1,) (square voxels)
        **volume_translation**: Denotes the 3D translation of the center
            of the volume in world units. Has to be one of:
            a) 3-tuple or a 3-list of scalars
            b) a Tensor of shape (3,)
            c) a Tensor of shape (minibatch, 3)
            d) a Tensor of shape (1,) (square voxels)
        **align_corners**: If set (default), the coordinates of the corner voxels are
            exactly −1 or +1 in the local coordinate system. Otherwise, the coordinates
            correspond to the centers of the corner voxels. Cf. the namesake argument to
            `torch.nn.functional.grid_sample`.
        """
        self.device = device
        self._batch_size = batch_size
        self._grid_sizes = self._convert_grid_sizes2tensor(grid_sizes)
        self._resolution = tuple(torch.max(self._grid_sizes.cpu(), dim=0).values)
        self._align_corners = align_corners

        # set the local_to_world transform
        self._set_local_to_world_transform(
            voxel_size=voxel_size,
            volume_translation=volume_translation,
        )

    def _convert_grid_sizes2tensor(
        self, x: Union[torch.LongTensor, List[torch.LongTensor], Tuple[int, int, int]]
    ) -> torch.LongTensor:
        """
        Handle the grid_sizes argument to the constructor.
        """
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], (torch.LongTensor, list, tuple)):
                if self._batch_size != len(x):
                    raise ValueError("x should have a batch size of 'batch_size'")
                # pyre-ignore[6]
                if any(len(x_) != 3 for x_ in x):
                    raise ValueError(
                        "`grid_sizes` has to be a list of 3-dim tensors of shape: "
                        "(height, width, depth)"
                    )
                x_shapes = torch.stack(
                    [
                        torch.tensor(
                            # pyre-ignore[6]
                            list(x_),
                            dtype=torch.long,
                            device=self.device,
                        )
                        for x_ in x
                    ],
                    dim=0,
                )
            elif isinstance(x[0], int):
                x_shapes = torch.stack(
                    [
                        torch.tensor(list(x), dtype=torch.long, device=self.device)
                        for _ in range(self._batch_size)
                    ],
                    dim=0,
                )
            else:
                raise ValueError(
                    "`grid_sizes` can be a list/tuple of int or torch.Tensor not of "
                    + "{type(x[0])}."
                )

        elif torch.is_tensor(x):
            if x.ndim != 2:
                raise ValueError(
                    "`grid_sizes` has to be a 2-dim tensor of shape: (minibatch, 3)"
                )
            x_shapes = x.to(self.device)
        else:
            raise ValueError(
                "grid_sizes must be either a list of tensors with shape (H, W, D), tensor with"
                "shape (batch_size, H, W, D) or a tuple of (H, W, D)."
            )
        # pyre-ignore[7]
        return x_shapes

    def _voxel_size_translation_to_transform(
        self,
        voxel_size: torch.Tensor,
        volume_translation: torch.Tensor,
        batch_size: int,
    ) -> Transform3d:
        """
        Converts the `voxel_size` and `volume_translation` constructor arguments
        to the internal `Transform3d` object `local_to_world_transform`.
        """
        volume_size_zyx = self.get_grid_sizes().float()
        volume_size_xyz = volume_size_zyx[:, [2, 1, 0]]

        # x_local = (
        #       (x_world + volume_translation) / (0.5 * voxel_size)
        #   ) / (volume_size - 1)

        # x_world = (
        #       x_local * (volume_size - 1) * 0.5 * voxel_size
        #   ) - volume_translation

        local_to_world_transform = Scale(
            (volume_size_xyz - 1) * voxel_size * 0.5, device=self.device
        ).translate(-volume_translation)

        return local_to_world_transform

    def get_coord_grid(self, world_coordinates: bool = True) -> torch.Tensor:
        """
        Return the 3D coordinate grid of the volumetric grid
        in local (`world_coordinates=False`) or world coordinates
        (`world_coordinates=True`).

        The grid records location of each center of the corresponding volume voxel.

        Local coordinates are scaled s.t. the values along one side of the
        volume are in range [-1, 1].

        Args:
            **world_coordinates**: if `True`, the method
                returns the grid in the world coordinates,
                otherwise, in local coordinates.

        Returns:
            **coordinate_grid**: The grid of coordinates of shape
                `(minibatch, depth, height, width, 3)`, where `minibatch`,
                `height`, `width` and `depth` are the batch size, height, width
                and depth of the volume `features` or `densities`.
        """
        # TODO(dnovotny): Implement caching of the coordinate grid.
        return self._calculate_coordinate_grid(world_coordinates=world_coordinates)

    def _calculate_coordinate_grid(
        self, world_coordinates: bool = True
    ) -> torch.Tensor:
        """
        Calculate the 3D coordinate grid of the volumetric grid either
        in local (`world_coordinates=False`) or
        world coordinates (`world_coordinates=True`) .
        """

        ba, (de, he, wi) = self._batch_size, self._resolution
        grid_sizes = self.get_grid_sizes()

        # generate coordinate axes
        def corner_coord_adjustment(r):
            return 0.0 if self._align_corners else 1.0 / r

        vol_axes = [
            torch.linspace(
                -1.0 + corner_coord_adjustment(r),
                1.0 - corner_coord_adjustment(r),
                r,
                dtype=torch.float32,
                device=self.device,
            )
            for r in (de, he, wi)
        ]

        # generate per-coord meshgrids
        Z, Y, X = meshgrid_ij(vol_axes)

        # stack the coord grids ... this order matches the coordinate convention
        # of torch.nn.grid_sample
        vol_coords_local = torch.stack((X, Y, Z), dim=3)[None].repeat(ba, 1, 1, 1, 1)

        # get grid sizes relative to the maximal volume size
        grid_sizes_relative = (
            torch.tensor([[de, he, wi]], device=grid_sizes.device, dtype=torch.float32)
            - 1
        ) / (grid_sizes - 1).float()

        if (grid_sizes_relative != 1.0).any():
            # if any of the relative sizes != 1.0, adjust the grid
            grid_sizes_relative_reshape = grid_sizes_relative[:, [2, 1, 0]][
                :, None, None, None
            ]
            vol_coords_local *= grid_sizes_relative_reshape
            vol_coords_local += grid_sizes_relative_reshape - 1

        if world_coordinates:
            vol_coords = self.local_to_world_coords(vol_coords_local)
        else:
            vol_coords = vol_coords_local

        return vol_coords

    def get_local_to_world_coords_transform(self) -> Transform3d:
        """
        Return a Transform3d object that converts points in the
        the local coordinate frame of the volume to world coordinates.
        Local volume coordinates are scaled s.t. the coordinates along one
        side of the volume are in range [-1, 1].

        Returns:
            **local_to_world_transform**: A Transform3d object converting
                points from local coordinates to the world coordinates.
        """
        return self._local_to_world_transform

    def get_world_to_local_coords_transform(self) -> Transform3d:
        """
        Return a Transform3d object that converts points in the
        world coordinates to the local coordinate frame of the volume.
        Local volume coordinates are scaled s.t. the coordinates along one
        side of the volume are in range [-1, 1].

        Returns:
            **world_to_local_transform**: A Transform3d object converting
                points from world coordinates to local coordinates.
        """
        return self.get_local_to_world_coords_transform().inverse()

    def world_to_local_coords(self, points_3d_world: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of 3D point coordinates `points_3d_world` of shape
        (minibatch, ..., dim) in the world coordinates to
        the local coordinate frame of the volume. Local volume
        coordinates are scaled s.t. the coordinates along one side of the volume
        are in range [-1, 1].

        Args:
            **points_3d_world**: A tensor of shape `(minibatch, ..., 3)`
                containing the 3D coordinates of a set of points that will
                be converted from the local volume coordinates (ranging
                within [-1, 1]) to the world coordinates
                defined by the `self.center` and `self.voxel_size` parameters.

        Returns:
            **points_3d_local**: `points_3d_world` converted to the local
                volume coordinates of shape `(minibatch, ..., 3)`.
        """
        pts_shape = points_3d_world.shape
        return (
            self.get_world_to_local_coords_transform()
            .transform_points(points_3d_world.view(pts_shape[0], -1, 3))
            .view(pts_shape)
        )

    def local_to_world_coords(self, points_3d_local: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of 3D point coordinates `points_3d_local` of shape
        (minibatch, ..., dim) in the local coordinate frame of the volume
        to the world coordinates.

        Args:
            **points_3d_local**: A tensor of shape `(minibatch, ..., 3)`
                containing the 3D coordinates of a set of points that will
                be converted from the local volume coordinates (ranging
                within [-1, 1]) to the world coordinates
                defined by the `self.center` and `self.voxel_size` parameters.

        Returns:
            **points_3d_world**: `points_3d_local` converted to the world
                coordinates of the volume of shape `(minibatch, ..., 3)`.
        """
        pts_shape = points_3d_local.shape
        return (
            self.get_local_to_world_coords_transform()
            .transform_points(points_3d_local.view(pts_shape[0], -1, 3))
            .view(pts_shape)
        )

    def get_grid_sizes(self) -> torch.LongTensor:
        """
        Returns the sizes of individual volumetric grids in the structure.

        Returns:
            **grid_sizes**: Tensor of spatial sizes of each of the volumes
                of size (batchsize, 3), where i-th row holds (D_i, H_i, W_i).
        """
        return self._grid_sizes

    def _set_local_to_world_transform(
        self,
        voxel_size: _VoxelSize = 1.0,
        volume_translation: _Translation = (0.0, 0.0, 0.0),
    ):
        """
        Sets the internal representation of the transformation between the
        world and local volume coordinates by specifying
        `voxel_size` and `volume_translation`

        Args:
            **voxel_size**: Denotes the size of input voxels. Has to be one of:
                a) A scalar (square voxels)
                b) 3-tuple or a 3-list of scalars
                c) a Tensor of shape (3,)
                d) a Tensor of shape (minibatch, 3)
                e) a Tensor of shape (1,) (square voxels)
            **volume_translation**: Denotes the 3D translation of the center
                of the volume in world units. Has to be one of:
                a) 3-tuple or a 3-list of scalars
                b) a Tensor of shape (3,)
                c) a Tensor of shape (minibatch, 3)
                d) a Tensor of shape (1,) (square voxels)
        """
        # handle voxel size and center
        # here we force the tensors to lie on self.device
        voxel_size = self._handle_voxel_size(voxel_size, len(self))
        volume_translation = self._handle_volume_translation(
            volume_translation, len(self)
        )
        self._local_to_world_transform = self._voxel_size_translation_to_transform(
            voxel_size, volume_translation, len(self)
        )

    def _copy_transform_and_sizes(
        self,
        other: "VolumeLocator",
        device: Optional[torch.device] = None,
        index: Optional[
            Union[int, List[int], Tuple[int], slice, torch.Tensor]
        ] = _ALL_CONTENT,
    ) -> None:
        """
        Copies the local to world transform and grid sizes to other VolumeLocator object
        and moves it to specified device. Operates in place on other.

        Args:
            other: VolumeLocator object to which to copy
            device: torch.device on which to put the result, defatults to self.device
            index: Specifies which parts to copy.
                Can be an int, slice, list of ints or a boolean or a long tensor.
                Defaults to all items (`:`).
        """
        device = device if device is not None else self.device
        other._grid_sizes = self._grid_sizes[index].to(device)
        other._local_to_world_transform = self.get_local_to_world_coords_transform()[
            # pyre-fixme[6]: For 1st param expected `Union[List[int], int, slice,
            #  BoolTensor, LongTensor]` but got `Union[None, List[int], Tuple[int],
            #  int, slice, Tensor]`.
            index
        ].to(device)

    def _handle_voxel_size(
        self, voxel_size: _VoxelSize, batch_size: int
    ) -> torch.Tensor:
        """
        Handle the `voxel_size` argument to the `VolumeLocator` constructor.
        """
        err_msg = (
            "voxel_size has to be either a 3-tuple of scalars, or a scalar, or"
            " a torch.Tensor of shape (3,) or (1,) or (minibatch, 3) or (minibatch, 1)."
        )
        if isinstance(voxel_size, (float, int)):
            # convert a scalar to a 3-element tensor
            voxel_size = torch.full(
                (1, 3), voxel_size, device=self.device, dtype=torch.float32
            )
        elif isinstance(voxel_size, torch.Tensor):
            if voxel_size.numel() == 1:
                # convert a single-element tensor to a 3-element one
                voxel_size = voxel_size.view(-1).repeat(3)
            elif len(voxel_size.shape) == 2 and (
                voxel_size.shape[0] == batch_size and voxel_size.shape[1] == 1
            ):
                voxel_size = voxel_size.repeat(1, 3)
        return self._convert_volume_property_to_tensor(voxel_size, batch_size, err_msg)

    def _handle_volume_translation(
        self, translation: _Translation, batch_size: int
    ) -> torch.Tensor:
        """
        Handle the `volume_translation` argument to the `VolumeLocator` constructor.
        """
        err_msg = (
            "`volume_translation` has to be either a 3-tuple of scalars, or"
            " a Tensor of shape (1,3) or (minibatch, 3) or (3,)`."
        )
        return self._convert_volume_property_to_tensor(translation, batch_size, err_msg)

    def __len__(self) -> int:
        return self._batch_size

    def _convert_volume_property_to_tensor(
        self, x: _Vector, batch_size: int, err_msg: str
    ) -> torch.Tensor:
        """
        Handle the `volume_translation` or `voxel_size` argument to
        the VolumeLocator constructor.
        Return a tensor of shape (N, 3) where N is the batch_size.
        """
        if isinstance(x, (list, tuple)):
            if len(x) != 3:
                raise ValueError(err_msg)
            x = torch.tensor(x, device=self.device, dtype=torch.float32)[None]
            x = x.repeat((batch_size, 1))
        elif isinstance(x, torch.Tensor):
            ok = (
                (x.shape[0] == 1 and x.shape[1] == 3)
                or (x.shape[0] == 3 and len(x.shape) == 1)
                or (x.shape[0] == batch_size and x.shape[1] == 3)
            )
            if not ok:
                raise ValueError(err_msg)
            if x.device != self.device:
                x = x.to(self.device)
            if x.shape[0] == 3 and len(x.shape) == 1:
                x = x[None]
            if x.shape[0] == 1:
                x = x.repeat((batch_size, 1))
        else:
            raise ValueError(err_msg)

        return x

    def to(self, device: Device, copy: bool = False) -> "VolumeLocator":
        """
        Match the functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
            device: Device (as str or torch.device) for the new tensor.
            copy: Boolean indicator whether or not to clone self. Default False.

        Returns:
            VolumeLocator object.
        """
        device_ = make_device(device)
        if not copy and self.device == device_:
            return self

        other = self.clone()
        if self.device == device_:
            return other

        other.device = device_
        other._grid_sizes = self._grid_sizes.to(device_)
        other._local_to_world_transform = self.get_local_to_world_coords_transform().to(
            device
        )
        return other

    def clone(self) -> "VolumeLocator":
        """
        Deep copy of VoluVolumeLocatormes object. All internal tensors are cloned
        individually.

        Returns:
            new VolumeLocator object.
        """
        return copy.deepcopy(self)

    def cpu(self) -> "VolumeLocator":
        return self.to("cpu")

    def cuda(self) -> "VolumeLocator":
        return self.to("cuda")
