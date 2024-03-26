# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import warnings
from dataclasses import dataclass
from math import pi
from typing import Optional

import torch


def get_rotation_to_best_fit_xy(
    points: torch.Tensor, centroid: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Returns a rotation R such that `points @ R` has a best fit plane
    parallel to the xy plane

    Args:
        points: (*, N, 3) tensor of points in 3D
        centroid: (*, 1, 3), (3,) or scalar: their centroid

    Returns:
        (*, 3, 3) tensor rotation matrix
    """
    if centroid is None:
        centroid = points.mean(dim=-2, keepdim=True)

    points_centered = points - centroid
    _, evec = torch.linalg.eigh(points_centered.transpose(-1, -2) @ points_centered)
    # in general, evec can form either right- or left-handed basis,
    # but we need the former to have a proper rotation (not reflection)
    return torch.cat(
        (evec[..., 1:], torch.cross(evec[..., 1], evec[..., 2])[..., None]), dim=-1
    )


def _signed_area(path: torch.Tensor) -> torch.Tensor:
    """
    Calculates the signed area / Lévy area of a 2D path. If the path is closed,
    i.e. ends where it starts, this is the integral of the winding number over
    the whole plane. If not, consider a closed path made by adding a straight
    line from the end to the start; the signed area is the integral of the
    winding number (also over the plane) with respect to that closed path.

    If this number is positive, it indicates in some sense that the path
    turns anticlockwise more than clockwise, and vice versa.

    Args:
        path: N x 2 tensor of points.

    Returns:
        signed area, shape ()
    """
    # This calculation is a sum of areas of triangles of the form
    # (path[0], path[i], path[i+1]), where each triangle is half a
    # parallelogram.
    x, y = (path[1:] - path[:1]).unbind(1)
    return (y[1:] * x[:-1] - x[1:] * y[:-1]).sum() * 0.5


@dataclass(frozen=True)
class Circle2D:
    """
    Contains details of a circle in a plane.
    Members
        center: tensor shape (2,)
        radius: tensor shape ()
        generated_points: points around the circle, shape (n_points, 2)
    """

    center: torch.Tensor
    radius: torch.Tensor
    generated_points: torch.Tensor


def fit_circle_in_2d(
    points2d, *, n_points: int = 0, angles: Optional[torch.Tensor] = None
) -> Circle2D:
    """
    Simple best fitting of a circle to 2D points. In particular, the circle which
    minimizes the sum of the squares of the squared-distances to the circle.

    Finds (a,b) and r to minimize the sum of squares (over the x,y pairs) of
        r**2 - [(x-a)**2+(y-b)**2]
    i.e.
        (2*a)*x + (2*b)*y + (r**2 - a**2 - b**2)*1 - (x**2 + y**2)

    In addition, generates points along the circle. If angles is None (default)
    then n_points around the circle equally spaced are given. These begin at the
    point closest to the first input point. They continue in the direction which
    seems to match the movement of points in points2d, as judged by its
    signed area. If `angles` are provided, then n_points is ignored, and points
    along the circle at the given angles are returned, with the starting point
    and direction as before.

    (Note that `generated_points` is affected by the order of the points in
    points2d, but the other outputs are not.)

    Args:
        points2d: N x 2 tensor of 2D points
        n_points: number of points to generate on the circle, if angles not given
        angles: optional angles in radians of points to generate.

    Returns:
        Circle2D object
    """
    design = torch.cat([points2d, torch.ones_like(points2d[:, :1])], dim=1)
    rhs = (points2d**2).sum(1)
    n_provided = points2d.shape[0]
    if n_provided < 3:
        raise ValueError(f"{n_provided} points are not enough to determine a circle")
    solution = torch.linalg.lstsq(design, rhs[:, None]).solution
    center = solution[:2, 0] / 2
    radius = torch.sqrt(solution[2, 0] + (center**2).sum())
    if n_points > 0:
        if angles is not None:
            warnings.warn("n_points ignored because angles provided")
        else:
            angles = torch.linspace(0, 2 * pi, n_points, device=points2d.device)

    if angles is not None:
        initial_direction_xy = (points2d[0] - center).unbind()
        initial_angle = torch.atan2(initial_direction_xy[1], initial_direction_xy[0])
        with torch.no_grad():
            anticlockwise = _signed_area(points2d) > 0
        if anticlockwise:
            use_angles = initial_angle + angles
        else:
            use_angles = initial_angle - angles
        generated_points = center[None] + radius * torch.stack(
            [torch.cos(use_angles), torch.sin(use_angles)], dim=-1
        )
    else:
        generated_points = points2d.new_zeros(0, 2)
    return Circle2D(center=center, radius=radius, generated_points=generated_points)


@dataclass(frozen=True)
class Circle3D:
    """
    Contains details of a circle in 3D.
    Members
        center: tensor shape (3,)
        radius: tensor shape ()
        normal: tensor shape (3,)
        generated_points: points around the circle, shape (n_points, 3)
    """

    center: torch.Tensor
    radius: torch.Tensor
    normal: torch.Tensor
    generated_points: torch.Tensor


def fit_circle_in_3d(
    points,
    *,
    n_points: int = 0,
    angles: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    up: Optional[torch.Tensor] = None,
) -> Circle3D:
    """
    Simple best fit circle to 3D points. Uses circle_2d in the
    least-squares best fit plane.

    In addition, generates points along the circle. If angles is None (default)
    then n_points around the circle equally spaced are given. These begin at the
    point closest to the first input point. They continue in the direction which
    seems to be match the movement of points. If angles is provided, then n_points
    is ignored, and points along the circle at the given angles are returned,
    with the starting point and direction as before.

    Further, an offset can be given to add to the generated points; this is
    interpreted in a rotated coordinate system where (0, 0, 1) is normal to the
    circle, specifically the normal which is approximately in the direction of a
    given `up` vector. The remaining rotation is disambiguated in an unspecified
    but deterministic way.

    (Note that `generated_points` is affected by the order of the points in
    points, but the other outputs are not.)

    Args:
        points2d: N x 3 tensor of 3D points
        n_points: number of points to generate on the circle
        angles: optional angles in radians of points to generate.
        offset: optional tensor (3,), a displacement expressed in a "canonical"
                coordinate system to add to the generated points.
        up: optional tensor (3,), a vector which helps define the
            "canonical" coordinate system for interpretting `offset`.
            Required if offset is used.


    Returns:
        Circle3D object
    """
    centroid = points.mean(0)
    r = get_rotation_to_best_fit_xy(points, centroid)
    normal = r[:, 2]
    rotated_points = (points - centroid) @ r
    result_2d = fit_circle_in_2d(
        rotated_points[:, :2], n_points=n_points, angles=angles
    )
    center_3d = result_2d.center @ r[:, :2].t() + centroid
    n_generated_points = result_2d.generated_points.shape[0]
    if n_generated_points > 0:
        generated_points_in_plane = torch.cat(
            [
                result_2d.generated_points,
                torch.zeros_like(result_2d.generated_points[:, :1]),
            ],
            dim=1,
        )
        if offset is not None:
            if up is None:
                raise ValueError("Missing `up` input for interpreting offset")
            with torch.no_grad():
                swap = torch.dot(up, normal) < 0
            if swap:
                # We need some rotation which takes +z to -z. Here's one.
                generated_points_in_plane += offset * offset.new_tensor([1, -1, -1])
            else:
                generated_points_in_plane += offset

        generated_points = generated_points_in_plane @ r.t() + centroid
    else:
        generated_points = points.new_zeros(0, 3)

    return Circle3D(
        radius=result_2d.radius,
        center=center_3d,
        normal=normal,
        generated_points=generated_points,
    )
