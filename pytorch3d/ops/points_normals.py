# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, TYPE_CHECKING, Union

import torch
from pytorch3d.common.workaround import symeig3x3

from .utils import convert_pointclouds_to_tensor, get_point_covariances


if TYPE_CHECKING:
    from ..structures import Pointclouds


def estimate_pointcloud_normals(
    pointclouds: Union[torch.Tensor, "Pointclouds"],
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
    *,
    use_symeig_workaround: bool = True,
) -> torch.Tensor:
    """
    Estimates the normals of a batch of `pointclouds`.

    The function uses `estimate_pointcloud_local_coord_frames` to estimate
    the normals. Please refer to that function for more detailed information.

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neighboring points.
      **use_symeig_workaround**: If `True`, uses a custom eigenvalue
        calculation.

    Returns:
      **normals**: A tensor of normals for each input point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """

    curvatures, local_coord_frames = estimate_pointcloud_local_coord_frames(
        pointclouds,
        neighborhood_size=neighborhood_size,
        disambiguate_directions=disambiguate_directions,
        use_symeig_workaround=use_symeig_workaround,
    )

    # the normals correspond to the first vector of each local coord frame
    normals = local_coord_frames[:, :, :, 0]

    return normals


def estimate_pointcloud_local_coord_frames(
    pointclouds: Union[torch.Tensor, "Pointclouds"],
    neighborhood_size: int = 50,
    disambiguate_directions: bool = True,
    *,
    use_symeig_workaround: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimates the principal directions of curvature (which includes normals)
    of a batch of `pointclouds`.

    The algorithm first finds `neighborhood_size` nearest neighbors for each
    point of the point clouds, followed by obtaining principal vectors of
    covariance matrices of each of the point neighborhoods.
    The main principal vector corresponds to the normals, while the
    other 2 are the direction of the highest curvature and the 2nd highest
    curvature.

    Note that each principal direction is given up to a sign. Hence,
    the function implements `disambiguate_directions` switch that allows
    to ensure consistency of the sign of neighboring normals. The implementation
    follows the sign disabiguation from SHOT descriptors [1].

    The algorithm also returns the curvature values themselves.
    These are the eigenvalues of the estimated covariance matrices
    of each point neighborhood.

    Args:
      **pointclouds**: Batch of 3-dimensional points of shape
        `(minibatch, num_point, 3)` or a `Pointclouds` object.
      **neighborhood_size**: The size of the neighborhood used to estimate the
        geometry around each point.
      **disambiguate_directions**: If `True`, uses the algorithm from [1] to
        ensure sign consistency of the normals of neighboring points.
      **use_symeig_workaround**: If `True`, uses a custom eigenvalue
        calculation.

    Returns:
      **curvatures**: The three principal curvatures of each point
        of shape `(minibatch, num_point, 3)`.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.
      **local_coord_frames**: The three principal directions of the curvature
        around each point of shape `(minibatch, num_point, 3, 3)`.
        The principal directions are stored in columns of the output.
        E.g. `local_coord_frames[i, j, :, 0]` is the normal of
        `j`-th point in the `i`-th pointcloud.
        If `pointclouds` are of `Pointclouds` class, returns a padded tensor.

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """

    points_padded, num_points = convert_pointclouds_to_tensor(pointclouds)

    ba, N, dim = points_padded.shape
    if dim != 3:
        raise ValueError(
            "The pointclouds argument has to be of shape (minibatch, N, 3)"
        )

    if (num_points <= neighborhood_size).any():
        raise ValueError(
            "The neighborhood_size argument has to be"
            + " >= size of each of the point clouds."
        )

    # undo global mean for stability
    # TODO: replace with tutil.wmean once landed
    pcl_mean = points_padded.sum(1) / num_points[:, None]
    points_centered = points_padded - pcl_mean[:, None, :]

    # get the per-point covariance and nearest neighbors used to compute it
    cov, knns = get_point_covariances(points_centered, num_points, neighborhood_size)

    # get the local coord frames as principal directions of
    # the per-point covariance
    # this is done with torch.symeig / torch.linalg.eigh, which returns the
    # eigenvectors (=principal directions) in an ascending order of their
    # corresponding eigenvalues, and the smallest eigenvalue's eigenvector
    # corresponds to the normal direction; or with a custom equivalent.
    if use_symeig_workaround:
        curvatures, local_coord_frames = symeig3x3(cov, eigenvectors=True)
    else:
        curvatures, local_coord_frames = torch.linalg.eigh(cov)

    # disambiguate the directions of individual principal vectors
    if disambiguate_directions:
        # disambiguate normal
        n = _disambiguate_vector_directions(
            points_centered, knns, local_coord_frames[:, :, :, 0]
        )
        # disambiguate the main curvature
        z = _disambiguate_vector_directions(
            points_centered, knns, local_coord_frames[:, :, :, 2]
        )
        # the secondary curvature is just a cross between n and z
        y = torch.cross(n, z, dim=2)
        # cat to form the set of principal directions
        local_coord_frames = torch.stack((n, y, z), dim=3)

    return curvatures, local_coord_frames


def _disambiguate_vector_directions(pcl, knns, vecs: torch.Tensor) -> torch.Tensor:
    """
    Disambiguates normal directions according to [1].

    References:
      [1] Tombari, Salti, Di Stefano: Unique Signatures of Histograms for
      Local Surface Description, ECCV 2010.
    """
    # parse out K from the shape of knns
    K = knns.shape[2]
    # the difference between the mean of each neighborhood and
    # each element of the neighborhood
    df = knns - pcl[:, :, None]
    # projection of the difference on the principal direction
    proj = (vecs[:, :, None] * df).sum(3)
    # check how many projections are positive
    n_pos = (proj > 0).type_as(knns).sum(2, keepdim=True)
    # flip the principal directions where number of positive correlations
    flip = (n_pos < (0.5 * K)).type_as(knns)
    vecs = (1.0 - 2.0 * flip) * vecs
    return vecs
