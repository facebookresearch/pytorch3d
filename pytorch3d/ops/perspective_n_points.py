# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file contains Efficient PnP algorithm for Perspective-n-Points problem.
It finds a camera position (defined by rotation `R` and translation `T`) that
minimizes re-projection error between the given 3D points `x` and
the corresponding uncalibrated 2D points `y`.
"""

import warnings
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from pytorch3d.ops import points_alignment, utils as oputil


class EpnpSolution(NamedTuple):
    x_cam: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    err_2d: torch.Tensor
    err_3d: torch.Tensor


def _define_control_points(x, weight, storage_opts=None):
    """
    Returns control points that define barycentric coordinates
    Args:
        x: Batch of 3-dimensional points of shape `(minibatch, num_points, 3)`.
        weight: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
        storage_opts: dict of keyword arguments to the tensor constructor.
    """
    storage_opts = storage_opts or {}
    x_mean = oputil.wmean(x, weight)
    c_world = F.pad(torch.eye(3, **storage_opts), (0, 0, 0, 1), value=0.0).expand_as(
        x[:, :4, :]
    )
    return c_world + x_mean


def _compute_alphas(x, c_world):
    """
    Computes barycentric coordinates of x in the frame c_world.
    Args:
        x: Batch of 3-dimensional points of shape `(minibatch, num_points, 3)`.
        c_world: control points in world coordinates.
    """
    x = F.pad(x, (0, 1), value=1.0)
    c = F.pad(c_world, (0, 1), value=1.0)
    return torch.matmul(x, torch.inverse(c))  # B x N x 4


def _build_M(y, alphas, weight):
    """Returns the matrix defining the reprojection equations.
    Args:
        y: projected points in camera coordinates of size B x N x 2
        alphas: barycentric coordinates of size B x N x 4
        weight: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
    """
    bs, n, _ = y.size()

    # prepend t with the column of v's
    def prepad(t, v):
        return F.pad(t, (1, 0), value=v)

    if weight is not None:
        # weight the alphas in order to get a correctly weighted version of M
        alphas = alphas * weight[:, :, None]

    # outer left-multiply by alphas
    def lm_alphas(t):
        return torch.matmul(alphas[..., None], t).reshape(bs, n, 12)

    M = torch.cat(
        (
            lm_alphas(
                prepad(prepad(-y[:, :, 0, None, None], 0.0), 1.0)
            ),  # u constraints
            lm_alphas(
                prepad(prepad(-y[:, :, 1, None, None], 1.0), 0.0)
            ),  # v constraints
        ),
        dim=-1,
    ).reshape(bs, -1, 12)

    return M


def _null_space(m, kernel_dim):
    """Finds the null space (kernel) basis of the matrix
    Args:
        m: the batch of input matrices, B x N x 12
        kernel_dim: number of dimensions to approximate the kernel
    Returns:
        * a batch of null space basis vectors
            of size B x 4 x 3 x kernel_dim
        * a batch of spectral values where near-0s correspond to actual
            kernel vectors, of size B x kernel_dim
    """
    mTm = torch.bmm(m.transpose(1, 2), m)
    s, v = torch.linalg.eigh(mTm)
    return v[:, :, :kernel_dim].reshape(-1, 4, 3, kernel_dim), s[:, :kernel_dim]


def _reproj_error(y_hat, y, weight, eps=1e-9):
    """Projects estimated 3D points and computes the reprojection error
    Args:
        y_hat: a batch of predicted 2D points in homogeneous coordinates
        y: a batch of ground-truth 2D points
        weight: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
    Returns:
        Optionally weighted RMSE of difference between y and y_hat.
    """
    y_hat = y_hat / torch.clamp(y_hat[..., 2:], eps)
    dist = ((y - y_hat[..., :2]) ** 2).sum(dim=-1, keepdim=True) ** 0.5
    return oputil.wmean(dist, weight)[:, 0, 0]


def _algebraic_error(x_w_rotated, x_cam, weight):
    """Computes the residual of Umeyama in 3D.
    Args:
        x_w_rotated: The given 3D points rotated with the predicted camera.
        x_cam: the lifted 2D points y
        weight: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
    Returns:
        Optionally weighted MSE of difference between x_w_rotated and x_cam.
    """
    dist = ((x_w_rotated - x_cam) ** 2).sum(dim=-1, keepdim=True)
    return oputil.wmean(dist, weight)[:, 0, 0]


def _compute_norm_sign_scaling_factor(c_cam, alphas, x_world, y, weight, eps=1e-9):
    """Given a solution, adjusts the scale and flip
    Args:
        c_cam: control points in camera coordinates
        alphas: barycentric coordinates of the points
        x_world: Batch of 3-dimensional points of shape `(minibatch, num_points, 3)`.
        y: Batch of 2-dimensional points of shape `(minibatch, num_points, 2)`.
        weights: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
        eps: epsilon to threshold negative `z` values
    """
    # position of reference points in camera coordinates
    x_cam = torch.matmul(alphas, c_cam)

    x_cam = x_cam * (1.0 - 2.0 * (oputil.wmean(x_cam[..., 2:], weight) < 0).float())
    if torch.any(x_cam[..., 2:] < -eps):
        neg_rate = oputil.wmean((x_cam[..., 2:] < 0).float(), weight, dim=(0, 1)).item()
        warnings.warn("\nEPnP: %2.2f%% points have z<0." % (neg_rate * 100.0))

    R, T, s = points_alignment.corresponding_points_alignment(
        x_world, x_cam, weight, estimate_scale=True
    )
    s = s.clamp(eps)
    x_cam = x_cam / s[:, None, None]
    T = T / s[:, None]
    x_w_rotated = torch.matmul(x_world, R) + T[:, None, :]
    err_2d = _reproj_error(x_w_rotated, y, weight)
    err_3d = _algebraic_error(x_w_rotated, x_cam, weight)

    return EpnpSolution(x_cam, R, T, err_2d, err_3d)


def _gen_pairs(input, dim=-2, reducer=lambda a, b: ((a - b) ** 2).sum(dim=-1)):
    """Generates all pairs of different rows and then applies the reducer
    Args:
        input: a tensor
        dim: a dimension to generate pairs across
        reducer: a function of generated pair of rows to apply (beyond just concat)
    Returns:
        for default args, for A x B x C input, will output A x (B choose 2)
    """
    n = input.size()[dim]
    range = torch.arange(n)
    idx = torch.combinations(range).to(input).long()
    left = input.index_select(dim, idx[:, 0])
    right = input.index_select(dim, idx[:, 1])
    return reducer(left, right)


def _kernel_vec_distances(v):
    """Computes the coefficients for linearization of the quadratic system
        to match all pairwise distances between 4 control points (dim=1).
        The last dimension corresponds to the coefficients for quadratic terms
        Bij = Bi * Bj, where Bi and Bj correspond to kernel vectors.
    Arg:
        v: tensor of B x 4 x 3 x D, where D is dim(kernel), usually 4
    Returns:
        a tensor of B x 6 x [(D choose 2) + D];
        for D=4, the last dim means [B11 B22 B33 B44 B12 B13 B14 B23 B24 B34].
    """
    dv = _gen_pairs(v, dim=-3, reducer=lambda a, b: a - b)  # B x 6 x 3 x D

    # we should take dot-product of all (i,j), i < j, with coeff 2
    rows_2ij = 2.0 * _gen_pairs(dv, dim=-1, reducer=lambda a, b: (a * b).sum(dim=-2))
    # this should produce B x 6 x (D choose 2) tensor

    # we should take dot-product of all (i,i)
    rows_ii = (dv**2).sum(dim=-2)
    # this should produce B x 6 x D tensor

    return torch.cat((rows_ii, rows_2ij), dim=-1)


def _solve_lstsq_subcols(rhs, lhs, lhs_col_idx):
    """Solves an over-determined linear system for selected LHS columns.
        A batched version of `torch.lstsq`.
    Args:
        rhs: right-hand side vectors
        lhs: left-hand side matrices
        lhs_col_idx: a slice of columns in lhs
    Returns:
        a least-squares solution for lhs * X = rhs
    """
    lhs = lhs.index_select(-1, torch.tensor(lhs_col_idx, device=lhs.device).long())
    return torch.matmul(torch.pinverse(lhs), rhs[:, :, None])


def _binary_sign(t):
    return (t >= 0).to(t) * 2.0 - 1.0


def _find_null_space_coords_1(kernel_dsts, cw_dst, eps=1e-9):
    """Solves case 1 from the paper [1]; solve for 4 coefficients:
       [B11 B22 B33 B44 B12 B13 B14 B23 B24 B34]
         ^               ^   ^   ^
    Args:
        kernel_dsts: distances between kernel vectors
        cw_dst: distances between control points
    Returns:
        coefficients to weight kernel vectors
    [1] Moreno-Noguer, F., Lepetit, V., & Fua, P. (2009).
    EPnP: An Accurate O(n) solution to the PnP problem.
    International Journal of Computer Vision.
    https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/
    """
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 5, 6])

    beta = beta * _binary_sign(beta[:, :1, :])
    return beta / torch.clamp(beta[:, :1, :] ** 0.5, eps)


def _find_null_space_coords_2(kernel_dsts, cw_dst):
    """Solves case 2 from the paper; solve for 3 coefficients:
        [B11 B22 B33 B44 B12 B13 B14 B23 B24 B34]
          ^   ^           ^
    Args:
        kernel_dsts: distances between kernel vectors
        cw_dst: distances between control points
    Returns:
        coefficients to weight kernel vectors
    [1] Moreno-Noguer, F., Lepetit, V., & Fua, P. (2009).
    EPnP: An Accurate O(n) solution to the PnP problem.
    International Journal of Computer Vision.
    https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/
    """
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 1])

    coord_0 = (beta[:, :1, :].abs() ** 0.5) * _binary_sign(beta[:, 1:2, :])
    coord_1 = (beta[:, 2:3, :].abs() ** 0.5) * (
        (beta[:, :1, :] >= 0) == (beta[:, 2:3, :] >= 0)
    ).float()

    return torch.cat((coord_0, coord_1, torch.zeros_like(beta[:, :2, :])), dim=1)


def _find_null_space_coords_3(kernel_dsts, cw_dst, eps=1e-9):
    """Solves case 3 from the paper; solve for 5 coefficients:
        [B11 B22 B33 B44 B12 B13 B14 B23 B24 B34]
          ^   ^           ^   ^       ^
    Args:
        kernel_dsts: distances between kernel vectors
        cw_dst: distances between control points
    Returns:
        coefficients to weight kernel vectors
    [1] Moreno-Noguer, F., Lepetit, V., & Fua, P. (2009).
    EPnP: An Accurate O(n) solution to the PnP problem.
    International Journal of Computer Vision.
    https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/
    """
    beta = _solve_lstsq_subcols(cw_dst, kernel_dsts, [0, 4, 1, 5, 7])

    coord_0 = (beta[:, :1, :].abs() ** 0.5) * _binary_sign(beta[:, 1:2, :])
    coord_1 = (beta[:, 2:3, :].abs() ** 0.5) * (
        (beta[:, :1, :] >= 0) == (beta[:, 2:3, :] >= 0)
    ).float()
    coord_2 = beta[:, 3:4, :] / torch.clamp(coord_0[:, :1, :], eps)

    return torch.cat(
        (coord_0, coord_1, coord_2, torch.zeros_like(beta[:, :1, :])), dim=1
    )


def efficient_pnp(
    x: torch.Tensor,
    y: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    skip_quadratic_eq: bool = False,
) -> EpnpSolution:
    """
    Implements Efficient PnP algorithm [1] for Perspective-n-Points problem:
    finds a camera position (defined by rotation `R` and translation `T`) that
    minimizes re-projection error between the given 3D points `x` and
    the corresponding uncalibrated 2D points `y`, i.e. solves

    `y[i] = Proj(x[i] R[i] + T[i])`

    in the least-squares sense, where `i` are indices within the batch, and
    `Proj` is the perspective projection operator: `Proj([x y z]) = [x/z y/z]`.
    In the noise-less case, 4 points are enough to find the solution as long
    as they are not co-planar.

    Args:
        x: Batch of 3-dimensional points of shape `(minibatch, num_points, 3)`.
        y: Batch of 2-dimensional points of shape `(minibatch, num_points, 2)`.
        weights: Batch of non-negative weights of
            shape `(minibatch, num_point)`. `None` means equal weights.
        skip_quadratic_eq: If True, assumes the solution space for the
            linear system is one-dimensional, i.e. takes the scaled eigenvector
            that corresponds to the smallest eigenvalue as a solution.
            If False, finds the candidate coordinates in the potentially
            4D null space by approximately solving the systems of quadratic
            equations. The best candidate is chosen by examining the 2D
            re-projection error. While this option finds a better solution,
            especially when the number of points is small or perspective
            distortions are low (the points are far away), it may be more
            difficult to back-propagate through.

    Returns:
        `EpnpSolution` namedtuple containing elements:
        **x_cam**: Batch of transformed points `x` that is used to find
            the camera parameters, of shape `(minibatch, num_points, 3)`.
            In the general (noisy) case, they are not exactly equal to
            `x[i] R[i] + T[i]` but are some affine transform of `x[i]`s.
        **R**: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        **T**: Batch of translation vectors of shape `(minibatch, 3)`.
        **err_2d**: Batch of mean 2D re-projection errors of shape
            `(minibatch,)`. Specifically, if `yhat` is the re-projection for
            the `i`-th batch element, it returns `sum_j norm(yhat_j - y_j)`
            where `j` iterates over points and `norm` denotes the L2 norm.
        **err_3d**: Batch of mean algebraic errors of shape `(minibatch,)`.
            Specifically, those are squared distances between `x_world` and
            estimated points on the rays defined by `y`.

    [1] Moreno-Noguer, F., Lepetit, V., & Fua, P. (2009).
    EPnP: An Accurate O(n) solution to the PnP problem.
    International Journal of Computer Vision.
    https://www.epfl.ch/labs/cvlab/software/multi-view-stereo/epnp/
    """
    # define control points in a world coordinate system (centered on the 3d
    # points centroid); 4 x 3
    # TODO: more stable when initialised with the center and eigenvectors!
    c_world = _define_control_points(
        x.detach(), weights, storage_opts={"dtype": x.dtype, "device": x.device}
    )

    # find the linear combination of the control points to represent the 3d points
    alphas = _compute_alphas(x, c_world)

    M = _build_M(y, alphas, weights)

    # Compute kernel M
    kernel, spectrum = _null_space(M, 4)

    c_world_distances = _gen_pairs(c_world)
    kernel_dsts = _kernel_vec_distances(kernel)

    betas = (
        []
        if skip_quadratic_eq
        else [
            fnsc(kernel_dsts, c_world_distances)
            for fnsc in [
                _find_null_space_coords_1,
                _find_null_space_coords_2,
                _find_null_space_coords_3,
            ]
        ]
    )

    c_cam_variants = [kernel] + [
        torch.matmul(kernel, beta[:, None, :, :]) for beta in betas
    ]

    solutions = [
        _compute_norm_sign_scaling_factor(c_cam[..., 0], alphas, x, y, weights)
        for c_cam in c_cam_variants
    ]

    sol_zipped = EpnpSolution(*(torch.stack(list(col)) for col in zip(*solutions)))
    best = torch.argmin(sol_zipped.err_2d, dim=0)

    def gather1d(source, idx):
        # reduces the dim=1 by picking the slices in a 1D tensor idx
        # in other words, it is batched index_select.
        return source.gather(
            0,
            idx.reshape(1, -1, *([1] * (len(source.shape) - 2))).expand_as(source[:1]),
        )[0]

    return EpnpSolution(*[gather1d(sol_col, best) for sol_col in sol_zipped])
