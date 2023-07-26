# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

import torch

from .. import ops


if TYPE_CHECKING:
    from pytorch3d.renderer.cameras import CamerasBase


def corresponding_cameras_alignment(
    cameras_src: "CamerasBase",
    cameras_tgt: "CamerasBase",
    estimate_scale: bool = True,
    mode: str = "extrinsics",
    eps: float = 1e-9,
) -> "CamerasBase":  # pragma: no cover
    """
    .. warning::
        The `corresponding_cameras_alignment` API is experimental
        and subject to change!

    Estimates a single similarity transformation between two sets of cameras
    `cameras_src` and `cameras_tgt` and returns an aligned version of
    `cameras_src`.

    Given source cameras [(R_1, T_1), (R_2, T_2), ..., (R_N, T_N)] and
    target cameras [(R_1', T_1'), (R_2', T_2'), ..., (R_N', T_N')],
    where (R_i, T_i) is a 2-tuple of the camera rotation and translation matrix
    respectively, the algorithm finds a global rotation, translation and scale
    (R_A, T_A, s_A) which aligns all source cameras with the target cameras
    such that the following holds:

        Under the change of coordinates using a similarity transform
        (R_A, T_A, s_A) a 3D point X' is mapped to X with: ::

            X = (X' R_A + T_A) / s_A

        Then, for all cameras `i`, we assume that the following holds: ::

            X R_i + T_i = s' (X' R_i' + T_i'),

        i.e. an adjusted point X' is mapped by a camera (R_i', T_i')
        to the same point as imaged from camera (R_i, T_i) after resolving
        the scale ambiguity with a global scalar factor s'.

        Substituting for X above gives rise to the following: ::

            (X' R_A + T_A) / s_A R_i + T_i = s' (X' R_i' + T_i')       // · s_A
            (X' R_A + T_A) R_i + T_i s_A = (s' s_A) (X' R_i' + T_i')
            s' := 1 / s_A  # without loss of generality
            (X' R_A + T_A) R_i + T_i s_A = X' R_i' + T_i'
            X' R_A R_i + T_A R_i + T_i s_A = X' R_i' + T_i'
               ^^^^^^^   ^^^^^^^^^^^^^^^^^
               ~= R_i'        ~= T_i'

        i.e. after estimating R_A, T_A, s_A, the aligned source cameras have
        extrinsics: ::

            cameras_src_align = (R_A R_i, T_A R_i + T_i s_A) ~= (R_i', T_i')

    We support two ways `R_A, T_A, s_A` can be estimated:
        1) `mode=='centers'`
            Estimates the similarity alignment between camera centers using
            Umeyama's algorithm (see `pytorch3d.ops.corresponding_points_alignment`
            for details) and transforms camera extrinsics accordingly.

        2) `mode=='extrinsics'`
            Defines the alignment problem as a system
            of the following equations: ::

                for all i:
                [ R_A   0 ] x [ R_i         0 ] = [ R_i' 0 ]
                [ T_A^T 1 ]   [ (s_A T_i^T) 1 ]   [ T_i' 1 ]

            `R_A, T_A` and `s_A` are then obtained by solving the
            system in the least squares sense.

    The estimated camera transformation is a true similarity transform, i.e.
    it cannot be a reflection.

    Args:
        cameras_src: `N` cameras to be aligned.
        cameras_tgt: `N` target cameras.
        estimate_scale: Controls whether the alignment transform is rigid
            (`estimate_scale=False`), or a similarity (`estimate_scale=True`).
            `s_A` is set to `1` if `estimate_scale==False`.
        mode: Controls the alignment algorithm.
            Can be one either `'centers'` or `'extrinsics'`. Please refer to the
            description above for details.
        eps: A scalar for clamping to avoid dividing by zero.
            Active when `estimate_scale==True`.

    Returns:
        cameras_src_aligned: `cameras_src` after applying the alignment transform.
    """

    if cameras_src.R.shape[0] != cameras_tgt.R.shape[0]:
        raise ValueError(
            "cameras_src and cameras_tgt have to contain the same number of cameras!"
        )

    if mode == "centers":
        align_fun = _align_camera_centers
    elif mode == "extrinsics":
        align_fun = _align_camera_extrinsics
    else:
        raise ValueError("mode has to be one of (centers, extrinsics)")

    align_t_R, align_t_T, align_t_s = align_fun(
        cameras_src, cameras_tgt, estimate_scale=estimate_scale, eps=eps
    )

    # create a new cameras object and set the R and T accordingly
    cameras_src_aligned = cameras_src.clone()
    cameras_src_aligned.R = torch.bmm(align_t_R.expand_as(cameras_src.R), cameras_src.R)
    cameras_src_aligned.T = (
        torch.bmm(
            align_t_T[:, None].repeat(cameras_src.R.shape[0], 1, 1),
            cameras_src.R,
        )[:, 0]
        + cameras_src.T * align_t_s
    )

    return cameras_src_aligned


def _align_camera_centers(
    cameras_src: "CamerasBase",
    cameras_tgt: "CamerasBase",
    estimate_scale: bool = True,
    eps: float = 1e-9,
):  # pragma: no cover
    """
    Use Umeyama's algorithm to align the camera centers.
    """
    centers_src = cameras_src.get_camera_center()
    centers_tgt = cameras_tgt.get_camera_center()
    align_t = ops.corresponding_points_alignment(
        centers_src[None],
        centers_tgt[None],
        estimate_scale=estimate_scale,
        allow_reflection=False,
        eps=eps,
    )
    # the camera transform is the inverse of the estimated transform between centers
    align_t_R = align_t.R.permute(0, 2, 1)
    align_t_T = -(torch.bmm(align_t.T[:, None], align_t_R))[:, 0]
    align_t_s = align_t.s[0]

    return align_t_R, align_t_T, align_t_s


def _align_camera_extrinsics(
    cameras_src: "CamerasBase",
    cameras_tgt: "CamerasBase",
    estimate_scale: bool = True,
    eps: float = 1e-9,
):  # pragma: no cover
    """
    Get the global rotation R_A with svd of cov(RR^T):
        ```
        R_A R_i = R_i' for all i
        R_A [R_1 R_2 ... R_N] = [R_1' R_2' ... R_N']
        U, _, V = svd([R_1 R_2 ... R_N]^T [R_1' R_2' ... R_N'])
        R_A = (U V^T)^T
        ```
    """
    RRcov = torch.bmm(cameras_src.R, cameras_tgt.R.transpose(2, 1)).mean(0)
    U, _, V = torch.svd(RRcov)
    align_t_R = V @ U.t()

    """
    The translation + scale `T_A` and `s_A` is computed by finding
    a translation and scaling that aligns two tensors `A, B`
    defined as follows:
        ```
        T_A R_i + s_A T_i   = T_i'        ;  for all i    // · R_i^T
        s_A T_i R_i^T + T_A = T_i' R_i^T  ;  for all i
            ^^^^^^^^^         ^^^^^^^^^^
                A_i                B_i

        A_i := T_i R_i^T
        A = [A_1 A_2 ... A_N]
        B_i := T_i' R_i^T
        B = [B_1 B_2 ... B_N]
        ```
    The scale s_A can be retrieved by matching the correlations of
    the points sets A and B:
        ```
        s_A = (A-mean(A))*(B-mean(B)).sum() / ((A-mean(A))**2).sum()
        ```
    The translation `T_A` is then defined as:
        ```
        T_A = mean(B) - mean(A) * s_A
        ```
    """
    A = torch.bmm(cameras_src.R, cameras_src.T[:, :, None])[:, :, 0]
    B = torch.bmm(cameras_src.R, cameras_tgt.T[:, :, None])[:, :, 0]
    Amu = A.mean(0, keepdim=True)
    Bmu = B.mean(0, keepdim=True)
    if estimate_scale and A.shape[0] > 1:
        # get the scaling component by matching covariances
        # of centered A and centered B
        Ac = A - Amu
        Bc = B - Bmu
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        align_t_s = (Ac * Bc).mean() / (Ac**2).mean().clamp(eps)
    else:
        # set the scale to identity
        align_t_s = 1.0
    # get the translation as the difference between the means of A and B
    align_t_T = Bmu - align_t_s * Amu

    return align_t_R, align_t_T, align_t_s
