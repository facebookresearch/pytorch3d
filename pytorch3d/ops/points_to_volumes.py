# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING, Optional, Tuple

import torch


if TYPE_CHECKING:
    from ..structures import Pointclouds, Volumes


def add_pointclouds_to_volumes(
    pointclouds: "Pointclouds",
    initial_volumes: "Volumes",
    mode: str = "trilinear",
    min_weight: float = 1e-4,
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

    Example:
        ```
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
        ```

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
        mask: A binary mask of shape `(minibatch, N)` determining which 3D points
            are going to be converted to the resulting volume.
            Set to `None` if all points are valid.
        min_weight: A scalar controlling the lowest possible total per-voxel
            weight used to normalize the features accumulated in a voxel.
            Only active for `mode==trilinear`.
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
        grid_sizes = torch.LongTensor(list(volume_densities.shape[2:])).to(
            volume_densities
        )

    # flatten densities and features
    v_shape = volume_densities.shape[2:]
    volume_densities_flatten = volume_densities.view(ba, -1, 1)
    n_voxels = volume_densities_flatten.shape[1]

    if volume_features is None:
        # initialize features if not passed in
        # pyre-fixme[16]: `Tensor` has no attribute `new_zeros`.
        volume_features_flatten = volume_densities.new_zeros(ba, feature_dim, n_voxels)
    else:
        # otherwise just flatten
        volume_features_flatten = volume_features.view(ba, feature_dim, n_voxels)

    if mode == "trilinear":  # do the splatting (trilinear interp)
        volume_features, volume_densities = splat_points_to_volumes(
            points_3d,
            points_features,
            volume_densities_flatten,
            volume_features_flatten,
            grid_sizes,
            mask=mask,
            min_weight=min_weight,
        )
    elif mode == "nearest":  # nearest neighbor interp
        volume_features, volume_densities = round_points_to_volumes(
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
):

    # pyre-fixme[16]: `Tuple` has no attribute `values`.
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


def splat_points_to_volumes(
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
    # pyre-fixme[16]: `Tensor` has no attribute `new_zeros`.
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
                    w_valid = w_valid * mask.type_as(w)[:, :, None]

                # scatter add casts the votes into the weight accumulator
                # and the feature accumulator
                # pyre-fixme[16]: `Tensor` has no attribute `scatter_add_`.
                volume_densities.scatter_add_(1, idx_valid, w_valid)

                # reshape idx_valid -> (minibatch, feature_dim, n_points)
                idx_valid = idx_valid.view(ba, 1, n_points).expand_as(points_features)
                w_valid = w_valid.view(ba, 1, n_points)

                # volume_features of shape (minibatch, feature_dim, n_voxels)
                volume_features.scatter_add_(2, idx_valid, w_valid * points_features)

    # divide each feature by the total weight of the votes
    volume_features = volume_features / volume_densities.view(ba, 1, n_voxels).clamp(
        min_weight
    )

    return volume_features, volume_densities


def round_points_to_volumes(
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

    # get random indices for the purpose of adding out-of-bounds values
    rand_idx = X.new_zeros(X.shape).random_(0, n_voxels)

    # valid - binary indicators of votes that fall into the volume
    grid_sizes = grid_sizes.type_as(XYZ)
    valid = (
        (0 <= X)
        * (X < grid_sizes_xyz[:, None, 0:1])
        * (0 <= Y)
        * (Y < grid_sizes_xyz[:, None, 1:2])
        * (0 <= Z)
        * (Z < grid_sizes_xyz[:, None, 2:3])
    ).long()

    # get random indices for the purpose of adding out-of-bounds values
    rand_idx = valid.new_zeros(X.shape).random_(0, n_voxels)

    # linearized indices into the volume
    idx = (Z * grid_sizes[:, None, 1:2] + Y) * grid_sizes[:, None, 2:3] + X

    # out-of-bounds features added to a random voxel idx with weight=0.
    idx_valid = idx * valid + rand_idx * (1 - valid)
    w_valid = valid.type_as(volume_features)

    # scatter add casts the votes into the weight accumulator
    # and the feature accumulator
    # pyre-fixme[16]: `Tensor` has no attribute `scatter_add_`.
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
