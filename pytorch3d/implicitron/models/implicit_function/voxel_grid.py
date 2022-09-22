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

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple, Type

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
        resolution: 3-tuple containing x, y, z grid sizes corresponding to each axis.
    """

    align_corners: bool = True
    padding: str = "zeros"
    mode: str = "bilinear"
    n_features: int = 1
    resolution: Tuple[int, int, int] = (128, 128, 128)

    def __post_init__(self):
        super().__init__()

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

    def get_shapes(self) -> Dict[str, Tuple]:
        """
        Using parameters from the __init__ method, this method returns the
        shapes of individual tensors needed to run the evaluate method.

        Returns:
            a dictionary of needed shapes. To use the evaluate_local and evaluate_world methods
                replace the shapes in the dictionary with tensors of those shapes and add the
                first 'batch' dimension. If the required shape is (a, b) and you want to
                have g grids then the tensor that replaces the shape should have the
                shape (g, a, b).
        """
        raise NotImplementedError()

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

    def get_shapes(self) -> Dict[str, Tuple]:
        return {"voxel_grid": (self.n_features, *self.resolution)}


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
    brings us to the desired "n_features" dimensionality. If matrix_reduction=False the elements
    of different components are summed together to create (n_grids, n_components, 1) tensor.
    With some notation abuse, ignoring the interpolation operation, simplifying and denoting
    n_features as F, n_components as C and n_grids as G:

    3d_grid = (x ⊗ y ⊗ z) @ basis # GWHDC x GCF -> GWHDF

    The basis feature vectors are passed as obj.basis_matrix.

    Members:
        n_components: number of vector triplets, higher number gives better approximation.
        matrix_reduction: how to transform components. If matrix_reduction=True result
            matrix of shape (n_grids, n_points_total, n_components) is batch matrix multiplied
            by the basis_matrix of shape (n_grids, n_components, n_features). If
            matrix_reduction=False, the result tensor of (n_grids, n_points_total, n_components)
            is summed along the rows to get (n_grids, n_points_total, 1), which is then viewed
            to return to starting shape (n_grids, ..., 1).
    """

    # the type of grid_values argument needed to run evaluate_local()
    values_type: ClassVar[Type[VoxelGridValuesBase]] = CPFactorizedVoxelGridValues

    n_components: int = 24
    matrix_reduction: bool = True

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

    def get_shapes(self) -> Dict[str, Tuple[int, int]]:
        if self.matrix_reduction is False and self.n_features != 1:
            raise ValueError("Cannot set matrix_reduction=False and n_features to != 1")

        shape_dict = {
            "vector_components_x": (self.n_components, self.resolution[0]),
            "vector_components_y": (self.n_components, self.resolution[1]),
            "vector_components_z": (self.n_components, self.resolution[2]),
        }
        if self.matrix_reduction:
            shape_dict["basis_matrix"] = (self.n_components, self.n_features)
        return shape_dict


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
    brings us to the desired "n_features" dimensionality. If matrix_reduction=False the elements
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
        matrix_reduction: how to transform components. If matrix_reduction=True result
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
    matrix_reduction: bool = True

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

    def get_shapes(self) -> Dict[str, Tuple]:
        if self.matrix_reduction is False and self.n_features != 1:
            raise ValueError("Cannot set matrix_reduction=False and n_features to != 1")
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

        shape_dict = {
            "vector_components_x": (
                calculated_distribution_of_components[1],
                self.resolution[0],
            ),
            "vector_components_y": (
                calculated_distribution_of_components[2],
                self.resolution[1],
            ),
            "vector_components_z": (
                calculated_distribution_of_components[0],
                self.resolution[2],
            ),
            "matrix_components_xy": (
                calculated_distribution_of_components[0],
                self.resolution[0],
                self.resolution[1],
            ),
            "matrix_components_yz": (
                calculated_distribution_of_components[1],
                self.resolution[1],
                self.resolution[2],
            ),
            "matrix_components_xz": (
                calculated_distribution_of_components[2],
                self.resolution[0],
                self.resolution[2],
            ),
        }
        if self.matrix_reduction:
            shape_dict["basis_matrix"] = (
                sum(calculated_distribution_of_components),
                self.n_features,
            )

        return shape_dict


# pyre-fixme[13]: Attribute `voxel_grid` is never initialized.
class VoxelGridModule(Configurable, torch.nn.Module):
    """
    A wrapper torch.nn.Module for the VoxelGrid classes, which
    contains parameters that are needed to train the VoxelGrid classes.

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
    """

    voxel_grid_class_type: str = "FullResolutionVoxelGrid"
    voxel_grid: VoxelGridBase

    extents: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    init_std: float = 0.1
    init_mean: float = 0

    def __post_init__(self):
        super().__init__()
        run_auto_creation(self)
        n_grids = 1  # Voxel grid objects are batched. We need only a single grid.
        shapes = self.voxel_grid.get_shapes()
        params = {
            name: torch.normal(
                mean=torch.zeros((n_grids, *shape)) + self.init_mean,
                std=self.init_std,
            )
            for name, shape in shapes.items()
        }
        self.params = torch.nn.ParameterDict(params)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Evaluates points in the world coordinate frame on the voxel_grid.

        Args:
            points (torch.Tensor): tensor of points that you want to query
                of a form (..., 3)
        Returns:
            torch.Tensor of shape (..., n_features)
        """
        locator = VolumeLocator(
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
            volume_translation=tuple(self.translation),
            # pyre-ignore[29]
            device=next(self.params.values()).device,
        )
        # pyre-fixme[29]: `Union[torch._tensor.Tensor,
        #  torch.nn.modules.module.Module]` is not a function.
        grid_values = self.voxel_grid.values_type(**self.params)
        # voxel grids operate with extra n_grids dimension, which we fix to one
        return self.voxel_grid.evaluate_world(points[None], grid_values, locator)[0]

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
