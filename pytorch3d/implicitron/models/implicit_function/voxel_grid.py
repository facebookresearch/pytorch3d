# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains classes that implement Voxel grids, both in their full resolution
as in the factorized form. There are two factorized forms implemented, Tensor rank decomposition
or CANDECOMP/PARAFAC (here CP) and Vector Matrix (here VM) factorization from the
TensoRF (https://arxiv.org/abs/2203.09517) paper.

In addition, the module VoxelGridModule implements a trainable instance of one of
these classes.

"""

import logging
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

from distutils.version import LooseVersion
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Tuple, Type

import torch
from omegaconf import DictConfig
from pytorch3d.implicitron.tools.config import (
    Configurable,
    registry,
    ReplaceableBase,
    run_auto_creation,
)
from pytorch3d.structures.volumes import VolumeLocator

from .utils import interpolate_line, interpolate_plane, interpolate_volume


logger = logging.getLogger(__name__)


@dataclass
class VoxelGridValuesBase:
    pass


class VoxelGridBase(ReplaceableBase, torch.nn.Module):
    """
    Base class for all the voxel grid variants whith added trilinear interpolation between
    voxels (for example if voxel (0.333, 1, 3) is queried that would return the result
    2/3*voxel[0, 1, 3] + 1/3*voxel[1, 1, 3])

    Internally voxel grids are indexed by (features, x, y, z). If queried the point is not
    inside the voxel grid the vector that will be returned is determined by padding.

    Members:
        align_corners: parameter used in torch.functional.grid_sample. For details go to
            https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html by
            default is True
        padding: padding mode for outside grid values 'zeros' | 'border' | 'reflection'.
            Default is 'zeros'
        mode: interpolation mode to calculate output values :
            'bilinear' | 'nearest' | 'bicubic' | 'trilinear'.
            Default: 'bilinear' Note: mode='bicubic' supports only FullResolutionVoxelGrid.
            When mode='bilinear' and the input is 5-D, the interpolation mode used internally
            will actually be trilinear.
        n_features: number of dimensions of base feature vector. Determines how many features
            the grid returns.
        resolution_changes: a dictionary, where keys are change epochs and values are
            3-tuples containing x, y, z grid sizes corresponding to each axis to each epoch
    """

    align_corners: bool = True
    padding: str = "zeros"
    mode: str = "bilinear"
    n_features: int = 1
    # return the line below once we drop OmegaConf 2.1 support
    # resolution_changes: Dict[int, List[int]] = field(
    resolution_changes: Dict[int, Any] = field(
        default_factory=lambda: {0: [128, 128, 128]}
    )

    def __post_init__(self):
        if 0 not in self.resolution_changes:
            raise ValueError("There has to be key `0` in `resolution_changes`.")

    def evaluate_world(
        self,
        points: torch.Tensor,
        grid_values: VoxelGridValuesBase,
        locator: VolumeLocator,
    ) -> torch.Tensor:
        """
        Evaluates the voxel grid at points in the world coordinate frame.
        The interpolation type is determined by the `mode` member.

        Arguments:
            points (torch.Tensor): tensor of points that you want to query
                of a form (n_grids, ..., 3)
            grid_values: an object of type Class.values_type which has tensors as
                members which have shapes derived from the get_shapes() method
            locator: a VolumeLocator object
        Returns:
            torch.Tensor: shape (n_grids, ..., n_features)
        """
        points_local = locator.world_to_local_coords(points)
        return self.evaluate_local(points_local, grid_values)

    def evaluate_local(
        self, points: torch.Tensor, grid_values: VoxelGridValuesBase
    ) -> torch.Tensor:
        """
        Evaluates the voxel grid at points in the local coordinate frame,
        The interpolation type is determined by the `mode` member.

        Arguments:
            points (torch.Tensor): tensor of points that you want to query
                of a form (n_grids, ..., 3), in a normalized form (coordinates are in [-1, 1])
            grid_values: an object of type VMFactorizedVoxelGrid.values_type which has tensors
                as members which have shapes derived from the get_shapes() method
        Returns:
            torch.Tensor: shape (n_grids, ..., n_features)
        """
        raise NotImplementedError()

    def get_shapes(self, epoch: int) -> Dict[str, Tuple]:
        """
        Using parameters from the __init__ method, this method returns the
        shapes of individual tensors needed to run the evaluate method.

        Args:
            epoch: If the shape varies during training, which training epoch's shape to return.
        Returns:
            a dictionary of needed shapes. To use the evaluate_local and evaluate_world methods
                replace the shapes in the dictionary with tensors of those shapes and add the
                first 'batch' dimension. If the required shape is (a, b) and you want to
                have g grids then the tensor that replaces the shape should have the
                shape (g, a, b).
        """
        raise NotImplementedError()

    def get_resolution(self, epoch: int) -> List[int]:
        """
        Returns the resolution which the grid should have at specific epoch

        Args:
            epoch which to use in the resolution calculation
        Returns:
            resolution at specific epoch
        """
        last_change = 0
        for change_epoch in self.resolution_changes:
            if change_epoch <= epoch:
                last_change = max(last_change, change_epoch)
        return self.resolution_changes[last_change]

    @staticmethod
    def get_output_dim(args: DictConfig) -> int:
        """
        Given all the arguments of the grid's __init__, returns output's last dimension length.

        In particular, if self.evaluate_world or self.evaluate_local
        are called with `points` of shape (n_grids, n_points, 3),
        their output will be of shape
        (n_grids, n_points, grid.get_output_dim()).

        Args:
            args: DictConfig which would be used to initialize the object
        Returns:
            output's last dimension length
        """
        return args["n_features"]

    def change_resolution(
        self,
        grid_values: VoxelGridValuesBase,
        *,
        epoch: Optional[int] = None,
        grid_values_with_wanted_resolution: Optional[VoxelGridValuesBase] = None,
        mode: str = "linear",
        align_corners: bool = True,
        antialias: bool = False,
    ) -> Tuple[VoxelGridValuesBase, bool]:
        """
        Changes resolution of tensors in `grid_values` to match the
        `grid_values_with_wanted_resolution` or resolution on wanted epoch.

        Args:
            epoch: current training epoch, used to see if the grid needs regridding
            grid_values: instance of self.values_type which contains
                the voxel grid which will be interpolated to create the new grid
            epoch: epoch which is used to get the resolution of the new
                `grid_values` using `self.resolution_changes`.
            grid_values_with_wanted_resolution: `VoxelGridValuesBase` to whose resolution
                to interpolate grid_values
            align_corners: as for torch.nn.functional.interpolate
            mode: as for torch.nn.functional.interpolate
                'nearest' | 'bicubic' | 'linear' | 'area' | 'nearest-exact'.
                Default: 'linear'
            antialias: as for torch.nn.functional.interpolate.
                 Using anti-alias option
                 together with align_corners=False and mode='bicubic', interpolation
                 result would match Pillow result for downsampling operation.
                 Supported mode: 'bicubic'
        Returns:
            tuple of
                - new voxel grid_values of desired resolution, of type self.values_type
                - True if regridding has happened.
        """

        if (epoch is None) == (grid_values_with_wanted_resolution is None):
            raise ValueError(
                "Exactly one of `epoch` or "
                "`grid_values_with_wanted_resolution` has to be defined."
            )

        if mode not in ("nearest", "bicubic", "linear", "area", "nearest-exact"):
            raise ValueError(
                "`mode` should be one of the following 'nearest'"
                + "| 'bicubic' | 'linear' | 'area' | 'nearest-exact'"
            )

        interpolate_has_antialias = LooseVersion(torch.__version__) >= "1.11"

        if antialias and not interpolate_has_antialias:
            warnings.warn("Antialiased interpolation requires PyTorch 1.11+; ignoring")

        interp_kwargs = {"antialias": antialias} if interpolate_has_antialias else {}

        def change_individual_resolution(tensor, wanted_resolution):
            if mode == "linear":
                n_dim = len(wanted_resolution)
                new_mode = ("linear", "bilinear", "trilinear")[n_dim - 1]
            else:
                new_mode = mode
            return torch.nn.functional.interpolate(
                input=tensor,
                size=wanted_resolution,
                mode=new_mode,
                align_corners=align_corners,
                recompute_scale_factor=False,
                **interp_kwargs,
            )

        if epoch is not None:
            if epoch not in self.resolution_changes:
                return grid_values, False

            wanted_shapes = self.get_shapes(epoch=epoch)
            params = {
                name: change_individual_resolution(
                    getattr(grid_values, name), shape[1:]
                )
                for name, shape in wanted_shapes.items()
            }
            res = self.get_resolution(epoch)
            logger.info(f"Changed grid resolutiuon at epoch {epoch} to {res}")
        else:
            params = {
                name: (
                    change_individual_resolution(
                        getattr(grid_values, name), tensor.shape[2:]
                    )
                    if tensor is not None
                    else None
                )
                for name, tensor in vars(grid_values_with_wanted_resolution).items()
            }

        return self.values_type(**params), True

    def get_resolution_change_epochs(self) -> Tuple[int, ...]:
        """
        Returns epochs at which this grid should change epochs.
        """
        return tuple(self.resolution_changes.keys())

    def get_align_corners(self) -> bool:
        """
        Returns True if voxel grid uses align_corners=True
        """
        return self.align_corners

    def crop_world(
        self,
        min_point_world: torch.Tensor,
        max_point_world: torch.Tensor,
        grid_values: VoxelGridValuesBase,
        volume_locator: VolumeLocator,
    ) -> VoxelGridValuesBase:
        """
        Crops the voxel grid based on minimum and maximum occupied point in
        world coordinates. After cropping all 8 corner points are preserved in
        the voxel grid. This is achieved by preserving all the voxels needed to
        calculate the point.

           +--------B
          /        /|
         /        / |
        +--------+  |  <==== Bounding box represented by points A and B:
        |        |  |           - B has x, y and z coordinates bigger or equal
        |        |  +              to all other points of the object
        |        | /            - A has x, y and z coordinates smaller or equal
        |        |/                to all other points of the object
        A--------+

        Args:
            min_point_world: torch.Tensor of shape (3,). Has x, y and z coordinates
                smaller or equal to all other occupied points. Point A from the
                picture above.
            max_point_world: torch.Tensor of shape (3,). Has x, y and z coordinates
                bigger or equal to all other occupied points. Point B from the
                picture above.
            grid_values: instance of self.values_type which contains
                the voxel grid which will be cropped to create the new grid
            volume_locator: VolumeLocator object used to convert world to local
                cordinates
        Returns:
            instance of self.values_type which has volume cropped to desired size.
        """
        min_point_local = volume_locator.world_to_local_coords(min_point_world[None])[0]
        max_point_local = volume_locator.world_to_local_coords(max_point_world[None])[0]
        return self.crop_local(min_point_local, max_point_local, grid_values)

    def crop_local(
        self,
        min_point_local: torch.Tensor,
        max_point_local: torch.Tensor,
        grid_values: VoxelGridValuesBase,
    ) -> VoxelGridValuesBase:
        """
        Crops the voxel grid based on minimum and maximum occupied point in local
        coordinates. After cropping both min and max point are preserved in the voxel
        grid. This is achieved by preserving all the voxels needed to calculate the point.

           +--------B
          /        /|
         /        / |
        +--------+  |  <==== Bounding box represented by points A and B:
        |        |  |           - B has x, y and z coordinates bigger or equal
        |        |  +              to all other points of the object
        |        | /            - A has x, y and z coordinates smaller or equal
        |        |/                to all other points of the object
        A--------+

        Args:
            min_point_local: torch.Tensor of shape (3,). Has x, y and z coordinates
                smaller or equal to all other occupied points. Point A from the
                picture above. All elements in [-1, 1].
            max_point_local: torch.Tensor of shape (3,). Has x, y and z coordinates
                bigger or equal to all other occupied points. Point B from the
                picture above. All elements in [-1, 1].
            grid_values: instance of self.values_type which contains
                the voxel grid which will be cropped to create the new grid
        Returns:
            instance of self.values_type which has volume cropped to desired size.
        """
        raise NotImplementedError()


@dataclass
class FullResolutionVoxelGridValues(VoxelGridValuesBase):
    voxel_grid: torch.Tensor


@registry.register
class FullResolutionVoxelGrid(VoxelGridBase):
    """
    Full resolution voxel grid equivalent to 4D tensor where shape is
    (features, width, height, depth) with linear interpolation between voxels.
    """

    # the type of grid_values argument needed to run evaluate_local()
    values_type: ClassVar[Type[VoxelGridValuesBase]] = FullResolutionVoxelGridValues

    # pyre-fixme[14]: `evaluate_local` overrides method defined in `VoxelGridBase`
    #  inconsistently.
    def evaluate_local(
        self, points: torch.Tensor, grid_values: FullResolutionVoxelGridValues
    ) -> torch.Tensor:
        """
        Evaluates the voxel grid at points in the local coordinate frame,
        The interpolation type is determined by the `mode` member.

        Arguments:
            points (torch.Tensor): tensor of points that you want to query
                of a form (..., 3), in a normalized form (coordinates are in [-1, 1])
            grid_values: an object of type values_type which has tensors as
                members which have shapes derived from the get_shapes() method
        Returns:
            torch.Tensor: shape (n_grids, ..., n_features)
        """
        # (n_grids, n_points_total, n_features) from (n_grids, ..., n_features)
        recorded_shape = points.shape
        points = points.view(points.shape[0], -1, points.shape[-1])
        interpolated = interpolate_volume(
            points,
            grid_values.voxel_grid,
            align_corners=self.align_corners,
            padding_mode=self.padding,
            mode=self.mode,
        )
        return interpolated.view(*recorded_shape[:-1], -1)

    def get_shapes(self, epoch: int) -> Dict[str, Tuple]:
        width, height, depth = self.get_resolution(epoch)
        return {"voxel_grid": (self.n_features, width, height, depth)}

    # pyre-ignore[14]
    def crop_local(
        self,
        min_point_local: torch.Tensor,
        max_point_local: torch.Tensor,
        grid_values: FullResolutionVoxelGridValues,
    ) -> FullResolutionVoxelGridValues:
        assert torch.all(min_point_local < max_point_local)
        min_point_local = torch.clamp(min_point_local, -1, 1)
        max_point_local = torch.clamp(max_point_local, -1, 1)
        _, _, width, height, depth = grid_values.voxel_grid.shape
        resolution = grid_values.voxel_grid.new_tensor([width, height, depth])
        min_point_local01 = (min_point_local + 1) / 2
        max_point_local01 = (max_point_local + 1) / 2

        if self.align_corners:
            minx, miny, minz = torch.floor(min_point_local01 * (resolution - 1)).long()
            maxx, maxy, maxz = torch.ceil(max_point_local01 * (resolution - 1)).long()
        else:
            minx, miny, minz = torch.floor(min_point_local01 * resolution - 0.5).long()
            maxx, maxy, maxz = torch.ceil(max_point_local01 * resolution - 0.5).long()

        return FullResolutionVoxelGridValues(
            voxel_grid=grid_values.voxel_grid[
                :, :, minx : maxx + 1, miny : maxy + 1, minz : maxz + 1
            ]
        )


@dataclass
class CPFactorizedVoxelGridValues(VoxelGridValuesBase):
    vector_components_x: torch.Tensor
    vector_components_y: torch.Tensor
    vector_components_z: torch.Tensor
    basis_matrix: Optional[torch.Tensor] = None


@registry.register
class CPFactorizedVoxelGrid(VoxelGridBase):
    """
    Canonical Polyadic (CP/CANDECOMP/PARAFAC) Factorization factorizes the 3d grid into three
    vectors (x, y, z). For n_components=n, the 3d grid is a sum of the two outer products
    (call it ⊗) of each vector type (x, y, z):

    3d_grid = x0 ⊗ y0 ⊗ z0 + x1 ⊗ y1 ⊗ z1 + ... + xn ⊗ yn ⊗ zn

    These tensors are passed in a object of CPFactorizedVoxelGridValues (here obj) as
    obj.vector_components_x, obj.vector_components_y, obj.vector_components_z. Their shapes are
    `(n_components, r)` where `r` is the relevant resolution.

    Each element of this sum has an extra dimension, which gets matrix-multiplied by an
    appropriate "basis matrix" of shape (n_grids, n_components, n_features). This multiplication
    brings us to the desired "n_features" dimensionality. If basis_matrix=False the elements
    of different components are summed together to create (n_grids, n_components, 1) tensor.
    With some notation abuse, ignoring the interpolation operation, simplifying and denoting
    n_features as F, n_components as C and n_grids as G:

    3d_grid = (x ⊗ y ⊗ z) @ basis # GWHDC x GCF -> GWHDF

    The basis feature vectors are passed as obj.basis_matrix.

    Members:
        n_components: number of vector triplets, higher number gives better approximation.
        basis_matrix: how to transform components. If matrix_reduction=True result
            matrix of shape (n_grids, n_points_total, n_components) is batch matrix multiplied
            by the basis_matrix of shape (n_grids, n_components, n_features). If
            matrix_reduction=False, the result tensor of (n_grids, n_points_total, n_components)
            is summed along the rows to get (n_grids, n_points_total, 1), which is then viewed
            to return to starting shape (n_grids, ..., 1).
    """

    # the type of grid_values argument needed to run evaluate_local()
    values_type: ClassVar[Type[VoxelGridValuesBase]] = CPFactorizedVoxelGridValues

    n_components: int = 24
    basis_matrix: bool = True

    # pyre-fixme[14]: `evaluate_local` overrides method defined in `VoxelGridBase`
    #  inconsistently.
    def evaluate_local(
        self, points: torch.Tensor, grid_values: CPFactorizedVoxelGridValues
    ) -> torch.Tensor:
        def factor(axis):
            i = {"x": 0, "y": 1, "z": 2}[axis]
            index = points[..., i, None]
            vector = getattr(grid_values, "vector_components_" + axis)
            return interpolate_line(
                index,
                vector,
                align_corners=self.align_corners,
                padding_mode=self.padding,
                mode=self.mode,
            )

        # (n_grids, n_points_total, n_features) from (n_grids, ..., n_features)
        recorded_shape = points.shape
        points = points.view(points.shape[0], -1, points.shape[-1])

        # collect points from all the vectors and multipy them out
        mult = factor("x") * factor("y") * factor("z")

        # reduce the result from
        # (n_grids, n_points_total, n_components) to (n_grids, n_points_total, n_features)
        if grid_values.basis_matrix is not None:
            # (n_grids, n_points_total, n_features) =
            # (n_grids, n_points_total, total_n_components) @
            # (n_grids, total_n_components, n_features)
            result = torch.bmm(mult, grid_values.basis_matrix)
        else:
            # (n_grids, n_points_total, 1) from (n_grids, n_points_total, n_features)
            result = mult.sum(axis=-1, keepdim=True)
        # (n_grids, ..., n_features)
        return result.view(*recorded_shape[:-1], -1)

    def get_shapes(self, epoch: int) -> Dict[str, Tuple[int, int]]:
        if self.basis_matrix is False and self.n_features != 1:
            raise ValueError("Cannot set basis_matrix=False and n_features to != 1")

        width, height, depth = self.get_resolution(epoch=epoch)
        shape_dict = {
            "vector_components_x": (self.n_components, width),
            "vector_components_y": (self.n_components, height),
            "vector_components_z": (self.n_components, depth),
        }
        if self.basis_matrix:
            shape_dict["basis_matrix"] = (self.n_components, self.n_features)
        return shape_dict

    # pyre-ignore[14]
    def crop_local(
        self,
        min_point_local: torch.Tensor,
        max_point_local: torch.Tensor,
        grid_values: CPFactorizedVoxelGridValues,
    ) -> CPFactorizedVoxelGridValues:
        assert torch.all(min_point_local < max_point_local)
        min_point_local = torch.clamp(min_point_local, -1, 1)
        max_point_local = torch.clamp(max_point_local, -1, 1)
        _, _, width = grid_values.vector_components_x.shape
        _, _, height = grid_values.vector_components_y.shape
        _, _, depth = grid_values.vector_components_z.shape
        resolution = grid_values.vector_components_x.new_tensor([width, height, depth])
        min_point_local01 = (min_point_local + 1) / 2
        max_point_local01 = (max_point_local + 1) / 2

        if self.align_corners:
            minx, miny, minz = torch.floor(min_point_local01 * (resolution - 1)).long()
            maxx, maxy, maxz = torch.ceil(max_point_local01 * (resolution - 1)).long()
        else:
            minx, miny, minz = torch.floor(min_point_local01 * resolution - 0.5).long()
            maxx, maxy, maxz = torch.ceil(max_point_local01 * resolution - 0.5).long()

        return CPFactorizedVoxelGridValues(
            vector_components_x=grid_values.vector_components_x[:, :, minx : maxx + 1],
            vector_components_y=grid_values.vector_components_y[:, :, miny : maxy + 1],
            vector_components_z=grid_values.vector_components_z[:, :, minz : maxz + 1],
            basis_matrix=grid_values.basis_matrix,
        )


@dataclass
class VMFactorizedVoxelGridValues(VoxelGridValuesBase):
    vector_components_x: torch.Tensor
    vector_components_y: torch.Tensor
    vector_components_z: torch.Tensor
    matrix_components_xy: torch.Tensor
    matrix_components_yz: torch.Tensor
    matrix_components_xz: torch.Tensor
    basis_matrix: Optional[torch.Tensor] = None


@registry.register
class VMFactorizedVoxelGrid(VoxelGridBase):
    """
    Implementation of Vector-Matrix Factorization of a tensor from
    https://arxiv.org/abs/2203.09517.

    Vector-Matrix Factorization factorizes the 3d grid into three matrices
    (xy, xz, yz) and three vectors (x, y, z). For n_components=1, the 3d grid
    is a sum of the outer products (call it ⊗) of each matrix with its
    complementary vector:

    3d_grid = xy ⊗ z + xz ⊗ y + yz ⊗ x.

    These tensors are passed in a VMFactorizedVoxelGridValues object (here obj)
    as obj.matrix_components_xy, obj.matrix_components_xy, obj.vector_components_y, etc.

    Their shapes are `(n_grids, n_components, r0, r1)` for matrix_components and
    (n_grids, n_components, r2)` for vector_componenets. Each of `r0, r1 and r2` coresponds
    to one resolution in (width, height and depth).

    Each element of this sum has an extra dimension, which gets matrix-multiplied by an
    appropriate "basis matrix" of shape (n_grids, n_components, n_features). This multiplication
    brings us to the desired "n_features" dimensionality. If basis_matrix=False the elements
    of different components are summed together to create (n_grids, n_components, 1) tensor.
    With some notation abuse, ignoring the interpolation operation, simplifying and denoting
    n_features as F, n_components as C (which can differ for each dimension) and n_grids as G:

    3d_grid = concat((xy ⊗ z), (xz ⊗ y).permute(0, 2, 1),
                (yz ⊗ x).permute(2, 0, 1)) @ basis_matrix # GWHDC x GCF -> GWHDF

    Members:
        n_components: total number of matrix vector pairs, this must be divisible by 3. Set
            this if you want to have equal representational power in all 3 directions. You
            must specify either n_components or distribution_of_components, you cannot
            specify both.
        distribution_of_components: if you do not want equal representational power in
            all 3 directions specify a tuple of numbers of matrix_vector pairs for each
            coordinate of a form (n_xy_planes, n_yz_planes, n_xz_planes). You must specify
            either n_components or distribution_of_components, you cannot specify both.
        basis_matrix: how to transform components. If matrix_reduction=True result
            matrix of shape (n_grids, n_points_total, n_components) is batch matrix multiplied
            by the basis_matrix of shape (n_grids, n_components, n_features). If
            matrix_reduction=False, the result tensor of (n_grids, n_points_total, n_components)
            is summed along the rows to get (n_grids, n_points_total, 1), which is then viewed
            to return to starting shape (n_grids, ..., 1).
    """

    # the type of grid_values argument needed to run evaluate_local()
    values_type: ClassVar[Type[VoxelGridValuesBase]] = VMFactorizedVoxelGridValues

    n_components: Optional[int] = None
    distribution_of_components: Optional[Tuple[int, int, int]] = None
    basis_matrix: bool = True

    # pyre-fixme[14]: `evaluate_local` overrides method defined in `VoxelGridBase`
    #  inconsistently.
    def evaluate_local(
        self, points: torch.Tensor, grid_values: VMFactorizedVoxelGridValues
    ) -> torch.Tensor:
        # (n_grids, n_points_total, n_features) from (n_grids, ..., n_features)
        recorded_shape = points.shape
        points = points.view(points.shape[0], -1, points.shape[-1])

        # collect points from matrices and vectors and multiply them
        a = interpolate_plane(
            points[..., :2],
            grid_values.matrix_components_xy,
            align_corners=self.align_corners,
            padding_mode=self.padding,
            mode=self.mode,
        ) * interpolate_line(
            points[..., 2:],
            grid_values.vector_components_z,
            align_corners=self.align_corners,
            padding_mode=self.padding,
            mode=self.mode,
        )
        b = interpolate_plane(
            points[..., [0, 2]],
            grid_values.matrix_components_xz,
            align_corners=self.align_corners,
            padding_mode=self.padding,
            mode=self.mode,
        ) * interpolate_line(
            points[..., 1:2],
            grid_values.vector_components_y,
            align_corners=self.align_corners,
            padding_mode=self.padding,
            mode=self.mode,
        )
        c = interpolate_plane(
            points[..., 1:],
            grid_values.matrix_components_yz,
            align_corners=self.align_corners,
            padding_mode=self.padding,
            mode=self.mode,
        ) * interpolate_line(
            points[..., :1],
            grid_values.vector_components_x,
            align_corners=self.align_corners,
            padding_mode=self.padding,
            mode=self.mode,
        )
        # pyre-ignore[28]
        feats = torch.cat((a, b, c), axis=-1)

        # reduce the result from
        # (n_grids, n_points, n_components) to (n_grids, n_points, n_features)
        if grid_values.basis_matrix is not None:
            # (n_grids, n_points, n_features) =
            # (n_grids, n_points, total_n_components) x
            #               (n_grids, total_n_components, n_features)
            result = torch.bmm(feats, grid_values.basis_matrix)
        else:
            # pyre-ignore[28]
            # (n_grids, n_points, 1) from (n_grids, n_points, n_features)
            result = feats.sum(axis=-1, keepdim=True)
        # (n_grids, ..., n_features)
        return result.view(*recorded_shape[:-1], -1)

    def get_shapes(self, epoch: int) -> Dict[str, Tuple]:
        if self.basis_matrix is False and self.n_features != 1:
            raise ValueError("Cannot set basis_matrix=False and n_features to != 1")
        if self.distribution_of_components is None and self.n_components is None:
            raise ValueError(
                "You need to provide n_components or distribution_of_components"
            )
        if (
            self.distribution_of_components is not None
            and self.n_components is not None
        ):
            raise ValueError(
                "You cannot define n_components and distribution_of_components"
            )
        # pyre-ignore[58]
        if self.distribution_of_components is None and self.n_components % 3 != 0:
            raise ValueError("n_components must be divisible by 3")
        if self.distribution_of_components is None:
            calculated_distribution_of_components = [
                # pyre-fixme[58]: `//` is not supported for operand types
                #  `Optional[int]` and `int`.
                self.n_components // 3
                for _ in range(3)
            ]
        else:
            calculated_distribution_of_components = self.distribution_of_components

        width, height, depth = self.get_resolution(epoch=epoch)
        shape_dict = {
            "vector_components_x": (
                calculated_distribution_of_components[1],
                width,
            ),
            "vector_components_y": (
                calculated_distribution_of_components[2],
                height,
            ),
            "vector_components_z": (
                calculated_distribution_of_components[0],
                depth,
            ),
            "matrix_components_xy": (
                calculated_distribution_of_components[0],
                width,
                height,
            ),
            "matrix_components_yz": (
                calculated_distribution_of_components[1],
                height,
                depth,
            ),
            "matrix_components_xz": (
                calculated_distribution_of_components[2],
                width,
                depth,
            ),
        }
        if self.basis_matrix:
            shape_dict["basis_matrix"] = (
                sum(calculated_distribution_of_components),
                self.n_features,
            )

        return shape_dict

    # pyre-ignore[14]
    def crop_local(
        self,
        min_point_local: torch.Tensor,
        max_point_local: torch.Tensor,
        grid_values: VMFactorizedVoxelGridValues,
    ) -> VMFactorizedVoxelGridValues:
        assert torch.all(min_point_local < max_point_local)
        min_point_local = torch.clamp(min_point_local, -1, 1)
        max_point_local = torch.clamp(max_point_local, -1, 1)
        _, _, width = grid_values.vector_components_x.shape
        _, _, height = grid_values.vector_components_y.shape
        _, _, depth = grid_values.vector_components_z.shape
        resolution = grid_values.vector_components_x.new_tensor([width, height, depth])
        min_point_local01 = (min_point_local + 1) / 2
        max_point_local01 = (max_point_local + 1) / 2

        if self.align_corners:
            minx, miny, minz = torch.floor(min_point_local01 * (resolution - 1)).long()
            maxx, maxy, maxz = torch.ceil(max_point_local01 * (resolution - 1)).long()
        else:
            minx, miny, minz = torch.floor(min_point_local01 * resolution - 0.5).long()
            maxx, maxy, maxz = torch.ceil(max_point_local01 * resolution - 0.5).long()

        return VMFactorizedVoxelGridValues(
            vector_components_x=grid_values.vector_components_x[:, :, minx : maxx + 1],
            vector_components_y=grid_values.vector_components_y[:, :, miny : maxy + 1],
            vector_components_z=grid_values.vector_components_z[:, :, minz : maxz + 1],
            matrix_components_xy=grid_values.matrix_components_xy[
                :, :, minx : maxx + 1, miny : maxy + 1
            ],
            matrix_components_yz=grid_values.matrix_components_yz[
                :, :, miny : maxy + 1, minz : maxz + 1
            ],
            matrix_components_xz=grid_values.matrix_components_xz[
                :, :, minx : maxx + 1, minz : maxz + 1
            ],
            basis_matrix=grid_values.basis_matrix,
        )


# pyre-fixme[13]: Attribute `voxel_grid` is never initialized.
class VoxelGridModule(Configurable, torch.nn.Module):
    """
    A wrapper torch.nn.Module for the VoxelGrid classes, which
    contains parameters that are needed to train the VoxelGrid classes.
    Can contain the parameters for the voxel grid as pytorch parameters
    or as registered buffers.

    Members:
        voxel_grid_class_type: The name of the class to use for voxel_grid,
            which must be available in the registry. Default FullResolutionVoxelGrid.
        voxel_grid: An instance of `VoxelGridBase`. This is the object which
            this class wraps.
        extents: 3-tuple of a form (width, height, depth), denotes the size of the grid
            in world units.
        translation: 3-tuple of float. The center of the volume in world units as (x, y, z).
        init_std: Parameters are initialized using the gaussian distribution
            with mean=init_mean and std=init_std. Default 0.1
        init_mean: Parameters are initialized using the gaussian distribution
            with mean=init_mean and std=init_std. Default 0.
        hold_voxel_grid_as_parameters: if True components of the underlying voxel grids
            will be saved as parameters and therefore be trainable. Default True.
        param_groups: dictionary where keys are names of individual parameters
            or module members and values are the parameter group where the
            parameter/member will be sorted to. "self" key is used to denote the
            parameter group at the module level. Possible keys, including the "self" key
            do not have to be defined. By default all parameters are put into "default"
            parameter group and have the learning rate defined in the optimizer,
            it can be overridden at the:
                - module level with “self” key, all the parameters and child
                    module's parameters will be put to that parameter group
                - member level, which is the same as if the `param_groups` in that
                    member has key=“self” and value equal to that parameter group.
                    This is useful if members do not have `param_groups`, for
                    example torch.nn.Linear.
                - parameter level, parameter with the same name as the key
                    will be put to that parameter group.
    """

    voxel_grid_class_type: str = "FullResolutionVoxelGrid"
    voxel_grid: VoxelGridBase

    extents: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    init_std: float = 0.1
    init_mean: float = 0

    hold_voxel_grid_as_parameters: bool = True
    param_groups: Dict[str, str] = field(default_factory=lambda: {})

    def __post_init__(self):
        run_auto_creation(self)
        n_grids = 1  # Voxel grid objects are batched. We need only a single grid.
        shapes = self.voxel_grid.get_shapes(epoch=0)
        params = {
            name: torch.normal(
                mean=torch.zeros((n_grids, *shape)) + self.init_mean,
                std=self.init_std,
            )
            for name, shape in shapes.items()
        }

        self.set_voxel_grid_parameters(self.voxel_grid.values_type(**params))
        self._register_load_state_dict_pre_hook(self._create_parameters_with_new_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates points in the world coordinate frame on the voxel_grid.

        Args:
            points (torch.Tensor): tensor of points that you want to query
                of a form (..., 3)
        Returns:
            torch.Tensor of shape (..., n_features)
        """
        locator = self._get_volume_locator()
        grid_values = self.voxel_grid.values_type(**self.params)
        # voxel grids operate with extra n_grids dimension, which we fix to one
        return self.voxel_grid.evaluate_world(points[None], grid_values, locator)[0]

    def set_voxel_grid_parameters(self, params: VoxelGridValuesBase) -> None:
        """
        Sets the parameters of the underlying voxel grid.

        Args:
            params: parameters of type `self.voxel_grid.values_type` which will
                replace current parameters
        """
        if self.hold_voxel_grid_as_parameters:
            self.params = torch.nn.ParameterDict(
                {
                    k: torch.nn.Parameter(val)
                    for k, val in vars(params).items()
                    if val is not None
                }
            )
        else:
            # Torch Module to hold parameters since they can only be registered
            # at object level.
            self.params = _RegistratedBufferDict(vars(params))

    @staticmethod
    def get_output_dim(args: DictConfig) -> int:
        """
        Utility to help predict the shape of the output of `forward`.

        Args:
            args: DictConfig which would be used to initialize the object
        Returns:
            int: the length of the last dimension of the output tensor
        """
        grid = registry.get(VoxelGridBase, args["voxel_grid_class_type"])
        return grid.get_output_dim(
            args["voxel_grid_" + args["voxel_grid_class_type"] + "_args"]
        )

    def subscribe_to_epochs(self) -> Tuple[Tuple[int, ...], Callable[[int], bool]]:
        """
        Method which expresses interest in subscribing to optimization epoch updates.

        Returns:
            tuple of epochs on which to call a callable and callable to be called on
                particular epoch. The callable returns True if parameter change has
                happened else False and it must be supplied with one argument, epoch.
        """
        return self.voxel_grid.get_resolution_change_epochs(), self._apply_epochs

    def _apply_epochs(self, epoch: int) -> bool:
        """
        Asks voxel_grid to change the resolution.
        This method is returned with subscribe_to_epochs and is the method that collects
        updates on training epochs, it is run on the training epochs that are requested.

        Args:
            epoch: current training epoch used for voxel grids to know to which
                resolution to change
        Returns:
            True if parameter change has happened else False.
        """
        grid_values = self.voxel_grid.values_type(**self.params)
        grid_values, change = self.voxel_grid.change_resolution(
            grid_values, epoch=epoch
        )
        if change:
            self.set_voxel_grid_parameters(grid_values)
        return change and self.hold_voxel_grid_as_parameters

    def _create_parameters_with_new_size(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        '''
        Automatically ran before loading the parameters with `load_state_dict()`.
        Creates new parameters with the sizes of the ones in the loaded state dict.
        This is necessary because the parameters are changing throughout training and
        at the time of construction `VoxelGridModule` does not know the size of
        parameters which will be loaded.

        Args:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            local_metadata (dict): a dict containing the metadata for this module.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=True``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=True``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        Returns:
            nothing
        """
        '''
        new_params = {}
        for name in self.params:
            key = prefix + "params." + name
            if key in state_dict:
                new_params[name] = torch.zeros_like(state_dict[key])
        self.set_voxel_grid_parameters(self.voxel_grid.values_type(**new_params))

    def get_device(self) -> torch.device:
        """
        Returns torch.device on which module parameters are located
        """
        return next(val for val in self.params.values() if val is not None).device

    def crop_self(self, min_point: torch.Tensor, max_point: torch.Tensor) -> None:
        """
        Crops self to only represent points between min_point and max_point (inclusive).

        Args:
            min_point: torch.Tensor of shape (3,). Has x, y and z coordinates
                smaller or equal to all other occupied points.
            max_point: torch.Tensor of shape (3,). Has x, y and z coordinates
                bigger or equal to all other occupied points.
        Returns:
            nothing
        """
        locator = self._get_volume_locator()
        #  torch.nn.modules.module.Module]` is not a function.
        old_grid_values = self.voxel_grid.values_type(**self.params)
        new_grid_values = self.voxel_grid.crop_world(
            min_point, max_point, old_grid_values, locator
        )
        grid_values, _ = self.voxel_grid.change_resolution(
            new_grid_values, grid_values_with_wanted_resolution=old_grid_values
        )
        self.params = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(val)
                for k, val in vars(grid_values).items()
                if val is not None
            }
        )
        # New center of voxel grid is the middle point between max and min points.
        self.translation = tuple((max_point + min_point) / 2)
        # new extents of voxel grid are distances between min and max points
        self.extents = tuple(max_point - min_point)

    def _get_volume_locator(self) -> VolumeLocator:
        """
        Returns VolumeLocator calculated from `extents` and `translation` members.
        """
        return VolumeLocator(
            batch_size=1,
            # The resolution of the voxel grid does not need to be known
            # to the locator object. It is easiest to fix the resolution of the locator.
            # In particular we fix it to (2,2,2) so that there is exactly one voxel of the
            # desired size. The locator object uses (z, y, x) convention for the grid_size,
            # and this module uses (x, y, z) convention so the order has to be reversed
            # (irrelevant in this case since they are all equal).
            # It is (2, 2, 2) because the VolumeLocator object behaves like
            # align_corners=True, which means that the points are in the corners of
            # the volume. So in the grid of (2, 2, 2) there is only one voxel.
            grid_sizes=(2, 2, 2),
            # The locator object uses (x, y, z) convention for the
            # voxel size and translation.
            voxel_size=tuple(self.extents),
            # volume_translation is defined in `VolumeLocator` as a vector from the origin
            # of local coordinate frame to origin of world coordinate frame, that is:
            # x_world = x_local * extents/2 - translation.
            # To get the reverse we need to negate it.
            volume_translation=tuple(-t for t in self.translation),
            device=self.get_device(),
        )

    def get_grid_points(self, epoch: int) -> torch.Tensor:
        """
        Returns a grid of points that represent centers of voxels of the
        underlying voxel grid in world coordinates at specific epoch.

        Args:
            epoch: underlying voxel grids change resolution depending on the
                epoch, this argument is used to determine the resolution
                of the voxel grid at that epoch.
        Returns:
            tensor of shape [xresolution, yresolution, zresolution, 3] where
                xresolution, yresolution, zresolution are resolutions of the
                underlying voxel grid
        """
        xresolution, yresolution, zresolution = self.voxel_grid.get_resolution(epoch)
        width, height, depth = self.extents
        if not self.voxel_grid.get_align_corners():
            width = (
                width * (xresolution - 1) / xresolution if xresolution > 1 else width
            )
            height = (
                height * (xresolution - 1) / xresolution if xresolution > 1 else height
            )
            depth = (
                depth * (xresolution - 1) / xresolution if xresolution > 1 else depth
            )
        xs = torch.linspace(
            -width / 2, width / 2, xresolution, device=self.get_device()
        )
        ys = torch.linspace(
            -height / 2, height / 2, yresolution, device=self.get_device()
        )
        zs = torch.linspace(
            -depth / 2, depth / 2, zresolution, device=self.get_device()
        )
        xmesh, ymesh, zmesh = torch.meshgrid(xs, ys, zs, indexing="ij")
        return torch.stack((xmesh, ymesh, zmesh), dim=3)


class _RegistratedBufferDict(torch.nn.Module, Mapping):
    """
    Mapping class and a torch.nn.Module that registeres its values
    with `self.register_buffer`. Can be indexed like a regular Python
    dictionary, but torch.Tensors it contains are properly registered, and will be visible
    by all Module methods. Supports only `torch.Tensor` as value and str as key.
    """

    def __init__(self, init_dict: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """
        Args:
            init_dict: dictionary which will be used to populate the object
        """
        super().__init__()
        self._keys = set()
        if init_dict is not None:
            for k, v in init_dict.items():
                self[k] = v

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter({k: self[k] for k in self._keys})

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    def __setitem__(self, key, value) -> None:
        self._keys.add(key)
        self.register_buffer(key, value)

    def __hash__(self) -> int:
        return hash(repr(self))
