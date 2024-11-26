# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import Optional, Tuple, TYPE_CHECKING

import torch
from pytorch3d import _C
from torch.autograd import Function
from torch.autograd.function import once_differentiable


if TYPE_CHECKING:
    from ..structures import Pointclouds, Volumes


class _points_to_volumes_function(Function):
    """
    For each point in a pointcloud, add point_weight to the
    corresponding volume density and point_weight times its features
    to the corresponding volume features.

    This function does not require any contiguity internally and therefore
    doesn't need to make copies of its inputs, which is useful when GPU memory
    is at a premium. (An implementation requiring contiguous inputs might be faster
    though). The volumes are modified in place.

    This function is differentiable with respect to
    points_features, volume_densities and volume_features.
    If splat is True then it is also differentiable with respect to
    points_3d.

    It may be useful to think about this function as a sort of opposite to
    torch.nn.functional.grid_sample with 5D inputs.

    Args:
        points_3d: Batch of 3D point cloud coordinates of shape
            `(minibatch, N, 3)` where N is the number of points
            in each point cloud. Coordinates have to be specified in the
            local volume coordinates (ranging in [-1, 1]).
        points_features: Features of shape `(minibatch, N, feature_dim)`
            corresponding to the points of the input point cloud `points_3d`.
        volume_features: Batch of input feature volumes
            of shape `(minibatch, feature_dim, D, H, W)`
        volume_densities: Batch of input feature volume densities
            of shape `(minibatch, 1, D, H, W)`. Each voxel should
            contain a non-negative number corresponding to its
            opaqueness (the higher, the less transparent).

        grid_sizes: `LongTensor` of shape (minibatch, 3) representing the
            spatial resolutions of each of the the non-flattened `volumes`
            tensors. Note that the following has to hold:
                `torch.prod(grid_sizes, dim=1)==N_voxels`.

        point_weight: A scalar controlling how much weight a single point has.

        mask: A binary mask of shape `(minibatch, N)` determining
            which 3D points are going to be converted to the resulting
            volume. Set to `None` if all points are valid.

        align_corners: as for grid_sample.

        splat: if true, trilinear interpolation. If false all the weight goes in
            the nearest voxel.

    Returns:
        volume_densities and volume_features, which have been modified in place.
    """

    @staticmethod
    # pyre-fixme[14]: `forward` overrides method defined in `Function` inconsistently.
    def forward(
        ctx,
        points_3d: torch.Tensor,
        points_features: torch.Tensor,
        volume_densities: torch.Tensor,
        volume_features: torch.Tensor,
        grid_sizes: torch.LongTensor,
        point_weight: float,
        mask: torch.Tensor,
        align_corners: bool,
        splat: bool,
    ):
        ctx.mark_dirty(volume_densities, volume_features)

        N, P, D = points_3d.shape
        if D != 3:
            raise ValueError("points_3d must be 3D")
        if points_3d.dtype != torch.float32:
            raise ValueError("points_3d must be float32")
        if points_features.dtype != torch.float32:
            raise ValueError("points_features must be float32")
        N1, P1, C = points_features.shape
        if N1 != N or P1 != P:
            raise ValueError("Bad points_features shape")
        if volume_densities.dtype != torch.float32:
            raise ValueError("volume_densities must be float32")
        N2, one, D, H, W = volume_densities.shape
        if N2 != N or one != 1:
            raise ValueError("Bad volume_densities shape")
        if volume_features.dtype != torch.float32:
            raise ValueError("volume_features must be float32")
        N3, C1, D1, H1, W1 = volume_features.shape
        if N3 != N or C1 != C or D1 != D or H1 != H or W1 != W:
            raise ValueError("Bad volume_features shape")
        if grid_sizes.dtype != torch.int64:
            raise ValueError("grid_sizes must be int64")
        N4, D1 = grid_sizes.shape
        if N4 != N or D1 != 3:
            raise ValueError("Bad grid_sizes.shape")
        if mask.dtype != torch.float32:
            raise ValueError("mask must be float32")
        N5, P2 = mask.shape
        if N5 != N or P2 != P:
            raise ValueError("Bad mask shape")

        # pyre-fixme[16]: Module `pytorch3d` has no attribute `_C`.
        _C.points_to_volumes_forward(
            points_3d,
            points_features,
            volume_densities,
            volume_features,
            grid_sizes,
            mask,
            point_weight,
            align_corners,
            splat,
        )
        if splat:
            ctx.save_for_backward(points_3d, points_features, grid_sizes, mask)
        else:
            ctx.save_for_backward(points_3d, grid_sizes, mask)
        ctx.point_weight = point_weight
        ctx.splat = splat
        ctx.align_corners = align_corners
        return volume_densities, volume_features

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_volume_densities, grad_volume_features):
        splat = ctx.splat
        N, C = grad_volume_features.shape[:2]
        if splat:
            points_3d, points_features, grid_sizes, mask = ctx.saved_tensors
            P = points_3d.shape[1]
            grad_points_3d = torch.zeros_like(points_3d)
        else:
            points_3d, grid_sizes, mask = ctx.saved_tensors
            P = points_3d.shape[1]
            ones = points_3d.new_zeros(1, 1, 1)
            # There is no gradient. Just need something to let its accessors exist.
            grad_points_3d = ones.expand_as(points_3d)
            # points_features not needed. Just need something to let its accessors exist.
            points_features = ones.expand(N, P, C)
        grad_points_features = points_3d.new_zeros(N, P, C)
        _C.points_to_volumes_backward(
            points_3d,
            points_features,
            grid_sizes,
            mask,
            ctx.point_weight,
            ctx.align_corners,
            splat,
            grad_volume_densities,
            grad_volume_features,
            grad_points_3d,
            grad_points_features,
        )

        return (
            (grad_points_3d if splat else None),
            grad_points_features,
            grad_volume_densities,
            grad_volume_features,
            None,
            None,
            None,
            None,
            None,
        )


_points_to_volumes = _points_to_volumes_function.apply


def add_pointclouds_to_volumes(
    pointclouds: "Pointclouds",
    initial_volumes: "Volumes",
    mode: str = "trilinear",
    min_weight: float = 1e-4,
    rescale_features: bool = True,
    _python: bool = False,
) -> "Volumes":
    """
    Add a batch of point clouds represented with a `Pointclouds` structure
    `pointclouds` to a batch of existing volumes represented with a
    `Volumes` structure `initial_volumes`.

    More specifically, the method casts a set of weighted votes (the weights are
    determined based on `mode="trilinear"|"nearest"`) into the pre-initialized
    `features` and `densities` fields of `initial_volumes`.

    The method returns an updated `Volumes` object that contains a copy
    of `initial_volumes` with its `features` and `densities` updated with the
    result of the pointcloud addition.

    Example::

        # init a random point cloud
        pointclouds = Pointclouds(
            points=torch.randn(4, 100, 3), features=torch.rand(4, 100, 5)
        )
        # init an empty volume centered around [0.5, 0.5, 0.5] in world coordinates
        # with a voxel size of 1.0.
        initial_volumes = Volumes(
            features = torch.zeros(4, 5, 25, 25, 25),
            densities = torch.zeros(4, 1, 25, 25, 25),
            volume_translation = [-0.5, -0.5, -0.5],
            voxel_size = 1.0,
        )
        # add the pointcloud to the 'initial_volumes' buffer using
        # trilinear splatting
        updated_volumes = add_pointclouds_to_volumes(
            pointclouds=pointclouds,
            initial_volumes=initial_volumes,
            mode="trilinear",
        )

    Args:
        pointclouds: Batch of 3D pointclouds represented with a `Pointclouds`
            structure. Note that `pointclouds.features` have to be defined.
        initial_volumes: Batch of initial `Volumes` with pre-initialized 1-dimensional
            densities which contain non-negative numbers corresponding to the
            opaqueness of each voxel (the higher, the less transparent).
        mode: The mode of the conversion of individual points into the volume.
            Set either to `nearest` or `trilinear`:
            `nearest`: Each 3D point is first rounded to the volumetric
                lattice. Each voxel is then labeled with the average
                over features that fall into the given voxel.
                The gradients of nearest neighbor conversion w.r.t. the
                3D locations of the points in `pointclouds` are *not* defined.
            `trilinear`: Each 3D point casts 8 weighted votes to the 8-neighborhood
                of its floating point coordinate. The weights are
                determined using a trilinear interpolation scheme.
                Trilinear splatting is fully differentiable w.r.t. all input arguments.
        min_weight: A scalar controlling the lowest possible total per-voxel
            weight used to normalize the features accumulated in a voxel.
            Only active for `mode==trilinear`.
        rescale_features: If False, output features are just the sum of input and
                            added points. If True, they are averaged. In both cases,
                            output densities are just summed without rescaling, so
                            you may need to rescale them afterwards.
        _python: Set to True to use a pure Python implementation, e.g. for test
            purposes, which requires more memory and may be slower.

    Returns:
        updated_volumes: Output `Volumes` structure containing the conversion result.
    """

    if len(initial_volumes) != len(pointclouds):
        raise ValueError(
            "'initial_volumes' and 'pointclouds' have to have the same batch size."
        )

    # obtain the features and densities
    pcl_feats = pointclouds.features_padded()
    pcl_3d = pointclouds.points_padded()

    if pcl_feats is None:
        raise ValueError("'pointclouds' have to have their 'features' defined.")

    # obtain the conversion mask
    n_per_pcl = pointclouds.num_points_per_cloud().type_as(pcl_feats)
    # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but got `Tensor`.
    mask = torch.arange(n_per_pcl.max(), dtype=pcl_feats.dtype, device=pcl_feats.device)
    mask = (mask[None, :] < n_per_pcl[:, None]).type_as(mask)

    # convert to the coord frame of the volume
    pcl_3d_local = initial_volumes.world_to_local_coords(pcl_3d)

    features_new, densities_new = add_points_features_to_volume_densities_features(
        points_3d=pcl_3d_local,
        points_features=pcl_feats,
        volume_features=initial_volumes.features(),
        volume_densities=initial_volumes.densities(),
        min_weight=min_weight,
        grid_sizes=initial_volumes.get_grid_sizes(),
        mask=mask,
        mode=mode,
        rescale_features=rescale_features,
        align_corners=initial_volumes.get_align_corners(),
        _python=_python,
    )

    return initial_volumes.update_padded(
        new_densities=densities_new, new_features=features_new
    )


def add_points_features_to_volume_densities_features(
    points_3d: torch.Tensor,
    points_features: torch.Tensor,
    volume_densities: torch.Tensor,
    volume_features: Optional[torch.Tensor],
    mode: str = "trilinear",
    min_weight: float = 1e-4,
    mask: Optional[torch.Tensor] = None,
    grid_sizes: Optional[torch.LongTensor] = None,
    rescale_features: bool = True,
    _python: bool = False,
    align_corners: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of point clouds represented with tensors of per-point
    3d coordinates and their features to a batch of volumes represented
    with tensors of densities and features.

    Args:
        points_3d: Batch of 3D point cloud coordinates of shape
            `(minibatch, N, 3)` where N is the number of points
            in each point cloud. Coordinates have to be specified in the
            local volume coordinates (ranging in [-1, 1]).
        points_features: Features of shape `(minibatch, N, feature_dim)` corresponding
            to the points of the input point clouds `pointcloud`.
        volume_densities: Batch of input feature volume densities of shape
            `(minibatch, 1, D, H, W)`. Each voxel should
            contain a non-negative number corresponding to its
            opaqueness (the higher, the less transparent).
        volume_features: Batch of input feature volumes of shape
            `(minibatch, feature_dim, D, H, W)`
            If set to `None`, the `volume_features` will be automatically
            instantiated with a correct size and filled with 0s.
        mode: The mode of the conversion of individual points into the volume.
            Set either to `nearest` or `trilinear`:
            `nearest`: Each 3D point is first rounded to the volumetric
                lattice. Each voxel is then labeled with the average
                over features that fall into the given voxel.
                The gradients of nearest neighbor rounding w.r.t. the
                input point locations `points_3d` are *not* defined.
            `trilinear`: Each 3D point casts 8 weighted votes to the 8-neighborhood
                of its floating point coordinate. The weights are
                determined using a trilinear interpolation scheme.
                Trilinear splatting is fully differentiable w.r.t. all input arguments.
        min_weight: A scalar controlling the lowest possible total per-voxel
            weight used to normalize the features accumulated in a voxel.
            Only active for `mode==trilinear`.
        mask: A binary mask of shape `(minibatch, N)` determining which 3D points
            are going to be converted to the resulting volume.
            Set to `None` if all points are valid.
        grid_sizes: `LongTensor` of shape (minibatch, 3) representing the
            spatial resolutions of each of the the non-flattened `volumes` tensors,
            or None to indicate the whole volume is used for every batch element.
        rescale_features: If False, output features are just the sum of input and
                            added points. If True, they are averaged. In both cases,
                            output densities are just summed without rescaling, so
                            you may need to rescale them afterwards.
        _python: Set to True to use a pure Python implementation.
        align_corners: as for grid_sample.
    Returns:
        volume_features: Output volume of shape `(minibatch, feature_dim, D, H, W)`
        volume_densities: Occupancy volume of shape `(minibatch, 1, D, H, W)`
            containing the total amount of votes cast to each of the voxels.
    """

    # number of points in the point cloud, its dim and batch size
    ba, n_points, feature_dim = points_features.shape
    ba_volume, density_dim = volume_densities.shape[:2]

    if density_dim != 1:
        raise ValueError("Only one-dimensional densities are allowed.")

    # init the volumetric grid sizes if uninitialized
    if grid_sizes is None:
        # grid sizes shape (minibatch, 3)
        grid_sizes = (
            torch.LongTensor(list(volume_densities.shape[2:]))
            .to(volume_densities.device)
            .expand(volume_densities.shape[0], 3)
        )

    if _python:
        return _add_points_features_to_volume_densities_features_python(
            points_3d=points_3d,
            points_features=points_features,
            volume_densities=volume_densities,
            volume_features=volume_features,
            mode=mode,
            min_weight=min_weight,
            mask=mask,
            # pyre-fixme[6]: For 8th param expected `LongTensor` but got `Tensor`.
            grid_sizes=grid_sizes,
        )

    if mode == "trilinear":
        splat = True
    elif mode == "nearest":
        splat = False
    else:
        raise ValueError('No such interpolation mode "%s"' % mode)

    if mask is None:
        mask = points_3d.new_ones(1).expand(points_3d.shape[:2])

    volume_densities, volume_features = _points_to_volumes(
        points_3d,
        points_features,
        volume_densities,
        volume_features,
        grid_sizes,
        1.0,  # point_weight
        mask,
        align_corners,  # align_corners
        splat,
    )

    if rescale_features:
        # divide each feature by the total weight of the votes
        if splat:
            volume_features = volume_features / volume_densities.clamp(min_weight)
        else:
            volume_features = volume_features / volume_densities.clamp(1.0)

    return volume_features, volume_densities


def _add_points_features_to_volume_densities_features_python(
    *,
    points_3d: torch.Tensor,
    points_features: torch.Tensor,
    volume_densities: torch.Tensor,
    volume_features: Optional[torch.Tensor],
    mode: str,
    min_weight: float,
    mask: Optional[torch.Tensor],
    grid_sizes: torch.LongTensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Python implementation for add_points_features_to_volume_densities_features.

    Returns:
        volume_features: Output volume of shape `(minibatch, feature_dim, D, H, W)`
        volume_densities: Occupancy volume of shape `(minibatch, 1, D, H, W)`
            containing the total amount of votes cast to each of the voxels.
    """
    ba, n_points, feature_dim = points_features.shape

    # flatten densities and features
    v_shape = volume_densities.shape[2:]
    volume_densities_flatten = volume_densities.view(ba, -1, 1)
    n_voxels = volume_densities_flatten.shape[1]

    if volume_features is None:
        # initialize features if not passed in
        volume_features_flatten = volume_densities.new_zeros(ba, feature_dim, n_voxels)
    else:
        # otherwise just flatten
        volume_features_flatten = volume_features.view(ba, feature_dim, n_voxels)

    if mode == "trilinear":  # do the splatting (trilinear interp)
        volume_features, volume_densities = _splat_points_to_volumes(
            points_3d,
            points_features,
            volume_densities_flatten,
            volume_features_flatten,
            grid_sizes,
            mask=mask,
            min_weight=min_weight,
        )
    elif mode == "nearest":  # nearest neighbor interp
        volume_features, volume_densities = _round_points_to_volumes(
            points_3d,
            points_features,
            volume_densities_flatten,
            volume_features_flatten,
            grid_sizes,
            mask=mask,
        )
    else:
        raise ValueError('No such interpolation mode "%s"' % mode)

    # reshape into the volume shape
    volume_features = volume_features.view(ba, feature_dim, *v_shape)
    volume_densities = volume_densities.view(ba, 1, *v_shape)
    return volume_features, volume_densities


def _check_points_to_volumes_inputs(
    points_3d: torch.Tensor,
    points_features: torch.Tensor,
    volume_densities: torch.Tensor,
    volume_features: torch.Tensor,
    grid_sizes: torch.LongTensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    max_grid_size = grid_sizes.max(dim=0).values
    if torch.prod(max_grid_size) > volume_densities.shape[1]:
        raise ValueError(
            "One of the grid sizes corresponds to a larger number"
            + " of elements than the number of elements in volume_densities."
        )

    _, n_voxels, density_dim = volume_densities.shape

    if density_dim != 1:
        raise ValueError("Only one-dimensional densities are allowed.")

    ba, n_points, feature_dim = points_features.shape

    if volume_features.shape[1] != feature_dim:
        raise ValueError(
            "volume_features have a different number of channels"
            + " than points_features."
        )

    if volume_features.shape[2] != n_voxels:
        raise ValueError(
            "volume_features have a different number of elements"
            + " than volume_densities."
        )


def _splat_points_to_volumes(
    points_3d: torch.Tensor,
    points_features: torch.Tensor,
    volume_densities: torch.Tensor,
    volume_features: torch.Tensor,
    grid_sizes: torch.LongTensor,
    min_weight: float = 1e-4,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of point clouds to a batch of volumes using trilinear
    splatting into a volume.

    Args:
        points_3d: Batch of 3D point cloud coordinates of shape
            `(minibatch, N, 3)` where N is the number of points
            in each point cloud. Coordinates have to be specified in the
            local volume coordinates (ranging in [-1, 1]).
        points_features: Features of shape `(minibatch, N, feature_dim)`
            corresponding to the points of the input point cloud `points_3d`.
        volume_features: Batch of input *flattened* feature volumes
            of shape `(minibatch, feature_dim, N_voxels)`
        volume_densities: Batch of input *flattened* feature volume densities
            of shape `(minibatch, N_voxels, 1)`. Each voxel should
            contain a non-negative number corresponding to its
            opaqueness (the higher, the less transparent).
        grid_sizes: `LongTensor` of shape (minibatch, 3) representing the
            spatial resolutions of each of the the non-flattened `volumes` tensors.
            Note that the following has to hold:
                `torch.prod(grid_sizes, dim=1)==N_voxels`
        min_weight: A scalar controlling the lowest possible total per-voxel
            weight used to normalize the features accumulated in a voxel.
        mask: A binary mask of shape `(minibatch, N)` determining which 3D points
            are going to be converted to the resulting volume.
            Set to `None` if all points are valid.
    Returns:
        volume_features: Output volume of shape `(minibatch, D, N_voxels)`.
        volume_densities: Occupancy volume of shape `(minibatch, 1, N_voxels)`
            containing the total amount of votes cast to each of the voxels.
    """

    _check_points_to_volumes_inputs(
        points_3d,
        points_features,
        volume_densities,
        volume_features,
        grid_sizes,
        mask=mask,
    )

    _, n_voxels, density_dim = volume_densities.shape
    ba, n_points, feature_dim = points_features.shape

    # minibatch x n_points x feature_dim -> minibatch x feature_dim x n_points
    points_features = points_features.permute(0, 2, 1).contiguous()

    # XYZ = the upper-left volume index of the 8-neighborhood of every point
    # grid_sizes is of the form (minibatch, depth-height-width)
    grid_sizes_xyz = grid_sizes[:, [2, 1, 0]]

    # Convert from points_3d in the range [-1, 1] to
    # indices in the volume grid in the range [0, grid_sizes_xyz-1]
    points_3d_indices = ((points_3d + 1) * 0.5) * (
        grid_sizes_xyz[:, None].type_as(points_3d) - 1
    )
    XYZ = points_3d_indices.floor().long()
    rXYZ = points_3d_indices - XYZ.type_as(points_3d)  # remainder of floor

    # split into separate coordinate vectors
    X, Y, Z = XYZ.split(1, dim=2)
    # rX = remainder after floor = 1-"the weight of each vote into
    #      the X coordinate of the 8-neighborhood"
    rX, rY, rZ = rXYZ.split(1, dim=2)

    # get random indices for the purpose of adding out-of-bounds values
    rand_idx = X.new_zeros(X.shape).random_(0, n_voxels)

    # iterate over the x, y, z indices of the 8-neighborhood (xdiff, ydiff, zdiff)
    for xdiff in (0, 1):
        X_ = X + xdiff
        wX = (1 - xdiff) + (2 * xdiff - 1) * rX
        for ydiff in (0, 1):
            Y_ = Y + ydiff
            wY = (1 - ydiff) + (2 * ydiff - 1) * rY
            for zdiff in (0, 1):
                Z_ = Z + zdiff
                wZ = (1 - zdiff) + (2 * zdiff - 1) * rZ

                # weight of each vote into the given cell of 8-neighborhood
                w = wX * wY * wZ

                # valid - binary indicators of votes that fall into the volume
                # pyre-fixme[16]: `int` has no attribute `long`.
                valid = (
                    (0 <= X_)
                    * (X_ < grid_sizes_xyz[:, None, 0:1])
                    * (0 <= Y_)
                    * (Y_ < grid_sizes_xyz[:, None, 1:2])
                    * (0 <= Z_)
                    * (Z_ < grid_sizes_xyz[:, None, 2:3])
                ).long()

                # linearized indices into the volume
                idx = (Z_ * grid_sizes[:, None, 1:2] + Y_) * grid_sizes[
                    :, None, 2:3
                ] + X_

                # out-of-bounds features added to a random voxel idx with weight=0.
                idx_valid = idx * valid + rand_idx * (1 - valid)
                w_valid = w * valid.type_as(w)
                if mask is not None:
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got `int`.
                    w_valid = w_valid * mask.type_as(w)[:, :, None]

                # scatter add casts the votes into the weight accumulator
                # and the feature accumulator
                # pyre-fixme[6]: For 3rd argument expected `Tensor` but got
                #  `Union[int, Tensor]`.
                volume_densities.scatter_add_(1, idx_valid, w_valid)

                # reshape idx_valid -> (minibatch, feature_dim, n_points)
                idx_valid = idx_valid.view(ba, 1, n_points).expand_as(points_features)
                # pyre-fixme[16]: Item `int` of `Union[int, Tensor]` has no
                #  attribute `view`.
                w_valid = w_valid.view(ba, 1, n_points)

                # volume_features of shape (minibatch, feature_dim, n_voxels)
                volume_features.scatter_add_(2, idx_valid, w_valid * points_features)

    # divide each feature by the total weight of the votes
    volume_features = volume_features / volume_densities.view(ba, 1, n_voxels).clamp(
        min_weight
    )

    return volume_features, volume_densities


def _round_points_to_volumes(
    points_3d: torch.Tensor,
    points_features: torch.Tensor,
    volume_densities: torch.Tensor,
    volume_features: torch.Tensor,
    grid_sizes: torch.LongTensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of point clouds to a batch of volumes using rounding to the
    nearest integer coordinate of the volume. Features that fall into the same
    voxel are averaged.

    Args:
        points_3d: Batch of 3D point cloud coordinates of shape
            `(minibatch, N, 3)` where N is the number of points
            in each point cloud. Coordinates have to be specified in the
            local volume coordinates (ranging in [-1, 1]).
        points_features: Features of shape `(minibatch, N, feature_dim)`
            corresponding to the points of the input point cloud `points_3d`.
        volume_features: Batch of input *flattened* feature volumes
            of shape `(minibatch, feature_dim, N_voxels)`
        volume_densities: Batch of input *flattened* feature volume densities
            of shape `(minibatch, 1, N_voxels)`. Each voxel should
            contain a non-negative number corresponding to its
            opaqueness (the higher, the less transparent).
        grid_sizes: `LongTensor` of shape (minibatch, 3) representing the
            spatial resolutions of each of the the non-flattened `volumes` tensors.
            Note that the following has to hold:
                `torch.prod(grid_sizes, dim=1)==N_voxels`
        mask: A binary mask of shape `(minibatch, N)` determining which 3D points
            are going to be converted to the resulting volume.
            Set to `None` if all points are valid.
    Returns:
        volume_features: Output volume of shape `(minibatch, D, N_voxels)`.
        volume_densities: Occupancy volume of shape `(minibatch, 1, N_voxels)`
            containing the total amount of votes cast to each of the voxels.
    """

    _check_points_to_volumes_inputs(
        points_3d,
        points_features,
        volume_densities,
        volume_features,
        grid_sizes,
        mask=mask,
    )

    _, n_voxels, density_dim = volume_densities.shape
    ba, n_points, feature_dim = points_features.shape

    # minibatch x n_points x feature_dim-> minibatch x feature_dim x n_points
    points_features = points_features.permute(0, 2, 1).contiguous()

    # round the coordinates to nearest integer
    # grid_sizes is of the form (minibatch, depth-height-width)
    grid_sizes_xyz = grid_sizes[:, [2, 1, 0]]
    XYZ = ((points_3d.detach() + 1) * 0.5) * (
        grid_sizes_xyz[:, None].type_as(points_3d) - 1
    )
    XYZ = torch.round(XYZ).long()

    # split into separate coordinate vectors
    X, Y, Z = XYZ.split(1, dim=2)

    # valid - binary indicators of votes that fall into the volume
    # pyre-fixme[9]: grid_sizes has type `LongTensor`; used as `Tensor`.
    grid_sizes = grid_sizes.type_as(XYZ)
    # pyre-fixme[16]: `int` has no attribute `long`.
    valid = (
        (0 <= X)
        * (X < grid_sizes_xyz[:, None, 0:1])
        * (0 <= Y)
        * (Y < grid_sizes_xyz[:, None, 1:2])
        * (0 <= Z)
        * (Z < grid_sizes_xyz[:, None, 2:3])
    ).long()
    if mask is not None:
        valid = valid * mask[:, :, None].long()

    # get random indices for the purpose of adding out-of-bounds values
    rand_idx = valid.new_zeros(X.shape).random_(0, n_voxels)

    # linearized indices into the volume
    idx = (Z * grid_sizes[:, None, 1:2] + Y) * grid_sizes[:, None, 2:3] + X

    # out-of-bounds features added to a random voxel idx with weight=0.
    idx_valid = idx * valid + rand_idx * (1 - valid)
    w_valid = valid.type_as(volume_features)

    # scatter add casts the votes into the weight accumulator
    # and the feature accumulator
    volume_densities.scatter_add_(1, idx_valid, w_valid)

    # reshape idx_valid -> (minibatch, feature_dim, n_points)
    idx_valid = idx_valid.view(ba, 1, n_points).expand_as(points_features)
    w_valid = w_valid.view(ba, 1, n_points)

    # volume_features of shape (minibatch, feature_dim, n_voxels)
    volume_features.scatter_add_(2, idx_valid, w_valid * points_features)

    # divide each feature by the total weight of the votes
    volume_features = volume_features / volume_densities.view(ba, 1, n_voxels).clamp(
        1.0
    )

    return volume_features, volume_densities
