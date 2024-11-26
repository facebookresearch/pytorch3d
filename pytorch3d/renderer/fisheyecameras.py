# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import math
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import _R, _T, CamerasBase

_focal_length = torch.tensor(((1.0,),))
_principal_point = torch.tensor(((0.0, 0.0),))
_radial_params = torch.tensor(((0.0, 0.0, 0.0, 0.0, 0.0, 0.0),))
_tangential_params = torch.tensor(((0.0, 0.0),))
_thin_prism_params = torch.tensor(((0.0, 0.0, 0.0, 0.0),))


class FishEyeCameras(CamerasBase):
    """
    A class which extends Pinhole camera by considering radial, tangential and
    thin-prism distortion. For the fisheye camera model, k1, k2, ..., k_n_radial are
    polynomial coefficents to model radial distortions. Two common types of radial
    distortions are barrel and pincusion radial distortions.

    a = x / z, b = y / z, r = (a*a+b*b)^(1/2)
    th = atan(r)
    [x_r]  = (th+ k0 * th^3 + k1* th^5 + ...) [a/r]
    [y_r]                                     [b/r]                    [1]


    The tangential distortion parameters are p1 and p2. The primary cause is
    due to the lens assembly not being centered over and parallel to the image plane.
    tangentialDistortion = [(2 x_r^2 + rd^2)*p_0 + 2*x_r*y_r*p_1]
                           [(2 y_r^2 + rd^2)*p_1 + 2*x_r*y_r*p_0]      [2]
    where rd^2 = x_r^2 + y_r^2

    The thin-prism distortion is modeled with s1, s2, s3, s4 coefficients
    thinPrismDistortion = [s0 * rd^2 + s1 rd^4]
                          [s2 * rd^2 + s3 rd^4]                        [3]

    The projection
    proj = diag(f, f) * uvDistorted + [cu; cv]
    uvDistorted = [x_r]  + tangentialDistortion  + thinPrismDistortion [4]
                  [y_r]
    f is the focal length and cu, cv are principal points in x, y axis.

    """

    _FIELDS = (
        "focal_length",
        "principal_point",
        "R",
        "T",
        "radial_params",
        "tangential_params",
        "thin_prism_params",
        "world_coordinates",
        "use_radial",
        "use_tangential",
        "use_tin_prism",
        "device",
        "image_size",
    )

    def __init__(
        self,
        focal_length=_focal_length,
        principal_point=_principal_point,
        radial_params=_radial_params,
        tangential_params=_tangential_params,
        thin_prism_params=_thin_prism_params,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        world_coordinates: bool = False,
        use_radial: bool = True,
        use_tangential: bool = True,
        use_thin_prism: bool = True,
        device: Device = "cpu",
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> None:
        """

        Args:
            focal_ength: Focal length of the camera in world units.
                A tensor of shape (N, 1) for square pixels,
                where N is number of transforms.
            principal_point: xy coordinates of the center of
                the principal point of the camera in pixels.
                A tensor of shape (N, 2).
            radial_params: parameters for radial distortions.
                A tensor of shape (N, num_radial).
            tangential_params:parameters for tangential distortions.
                A tensor of shape (N, 2).
            thin_prism_params: parameters for thin-prism distortions.
                A tensor of shape (N, 4).
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            world_coordinates: if True, project from world coordinates; otherwise from camera
                coordinates
            use_radial: radial_distortion, default to True
            use_tangential: tangential distortion, default to True
            use_thin_prism: thin prism distortion, default to True
            device: torch.device or string
            image_size: (height, width) of image size.
                A tensor of shape (N, 2) or a list/tuple. Required for screen cameras.

        """

        kwargs = {"image_size": image_size} if image_size is not None else {}
        super().__init__(
            device=device,
            R=R,
            T=T,
            **kwargs,  # pyre-ignore
        )
        if image_size is not None:
            if (self.image_size < 1).any():  # pyre-ignore
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        self.device = device
        self.focal = focal_length.to(self.device)
        self.principal_point = principal_point.to(self.device)
        self.radial_params = radial_params.to(self.device)
        self.tangential_params = tangential_params.to(self.device)
        self.thin_prism_params = thin_prism_params.to(self.device)
        self.R = R
        self.T = T
        self.world_coordinates = world_coordinates
        self.use_radial = use_radial
        self.use_tangential = use_tangential
        self.use_thin_prism = use_thin_prism
        self.epsilon = 1e-10
        self.num_distortion_iters = 50

        self.R = self.R.to(self.device)
        self.T = self.T.to(self.device)
        self.num_radial = radial_params.shape[-1]

    def _project_points_batch(
        self,
        focal,
        principal_point,
        radial_params,
        tangential_params,
        thin_prism_params,
        points,
    ) -> torch.Tensor:
        """
        Takes in points in the local reference frame of the camera and projects it
        onto the image plan. Since this is a symmetric model, points with negative z are
        projected to the positive sphere. i.e project(1,1,-1) == project(-1,-1,1)

        Args:
            focal: (1)
            principal_point: (2)
            radial_params: (num_radial)
            tangential_params: (2)
            thin_prism_params: (4)
            points in the camera coordinate frame: (..., 3). E.g., (P, 3) (1, P, 3)
                or (M, P, 3) where P is the number of points

        Returns:
            projected_points in the image plane: (..., 3). E.g., (P, 3) or
                (1, P, 3) or (M, P, 3)

        """
        assert points.shape[-1] == 3, "points shape incorrect"
        ab = points[..., :2] / points[..., 2:]
        uv_distorted = ab

        r = ab.norm(dim=-1)
        th = r.atan()
        theta_sq = th * th

        # compute radial distortions, eq 1
        t = theta_sq
        theta_pow = torch.stack([t, t**2, t**3, t**4, t**5, t**6], dim=-1)
        th_radial = 1 + torch.sum(theta_pow * radial_params, dim=-1)

        # compute th/r, using the limit for small values
        th_divr = th / r
        boolean_mask = abs(r) < self.epsilon
        th_divr[boolean_mask] = 1.0

        # the distorted coordinates -- except for focal length and principal point
        # start with the radial term
        coeff = th_radial * th_divr
        xr_yr = coeff[..., None] * ab
        xr_yr_squared_norm = torch.pow(xr_yr, 2).sum(dim=-1, keepdim=True)

        if self.use_radial:
            uv_distorted = xr_yr

        # compute tangential distortions, eq 2
        if self.use_tangential:
            temp = 2 * torch.sum(
                xr_yr * tangential_params,
                dim=-1,
            )
            uv_distorted = uv_distorted + (
                temp[..., None] * xr_yr + xr_yr_squared_norm * tangential_params
            )

        # compute thin-prism distortions, eq 3
        sh = uv_distorted.shape[:-1]
        if self.use_thin_prism:
            radial_powers = torch.cat(
                [xr_yr_squared_norm, xr_yr_squared_norm * xr_yr_squared_norm], dim=-1
            )
            uv_distorted[..., 0] = uv_distorted[..., 0] + torch.sum(
                thin_prism_params[..., 0:2] * radial_powers,
                dim=-1,
            )
            uv_distorted[..., 1] = uv_distorted[..., 1] + torch.sum(
                thin_prism_params[..., 2:4] * radial_powers,
                dim=-1,
            )
        # return value: distorted points on the uv plane, eq 4
        projected_points = focal * uv_distorted + principal_point
        return torch.cat(
            [projected_points, torch.ones(list(sh) + [1], device=self.device)], dim=-1
        )

    def check_input(self, points: torch.Tensor, batch_size: int):
        """
        Check if the shapes are broadcastable between points and transforms.
        Accept points of shape (P, 3) or (1, P, 3) or (M, P, 3). The batch_size
        for transforms should be 1 when points take (M, P, 3). The batch_size
        can be 1 or N when points take shape (P, 3).

        Args:
            points: tensor of shape (P, 3) or (1, P, 3) or (M, P, 3)
            batch_size: number of transforms

        Returns:
            Boolean value if the input shapes are compatible.
        """
        if points.ndim > 3:
            return False
        if points.ndim == 3:
            M, P, K = points.shape
            if K != 3:
                return False
            if M > 1 and batch_size > 1:
                return False
        return True

    def transform_points(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Transform input points from camera space to image space.
        Args:
            points: tensor of (..., 3). E.g., (P, 3) or (1, P, 3), (M, P, 3)
            eps: tiny number to avoid zero divsion

        Returns:
            torch.Tensor
            when points take shape (P, 3) or (1, P, 3), output is (N, P, 3)
            when points take shape (M, P, 3), output is (M, P, 3)
            where N is the number of transforms, P number of points
        """
        # project from world space to camera space
        if self.world_coordinates:
            world_to_view_transform = self.get_world_to_view_transform(
                R=self.R, T=self.T
            )
            points = world_to_view_transform.transform_points(
                points.to(self.device), eps=eps
            )
        else:
            points = points.to(self.device)

        # project from camera space to image space
        N = len(self.radial_params)
        if not self.check_input(points, N):
            msg = (
                "Expected points of (P, 3) with batch_size 1 or N, or shape (M, P, 3) \
            with batch_size 1; got points of shape %r and batch_size %r"
            )
            raise ValueError(msg % (points.shape, N))

        if N == 1:
            return self._project_points_batch(
                self.focal[0],
                self.principal_point[0],
                self.radial_params[0],
                self.tangential_params[0],
                self.thin_prism_params[0],
                points,
            )
        else:
            outputs = []
            for i in range(N):
                outputs.append(
                    self._project_points_batch(
                        self.focal[i],
                        self.principal_point[i],
                        self.radial_params[i],
                        self.tangential_params[i],
                        self.thin_prism_params[i],
                        points,
                    )
                )
            outputs = torch.stack(outputs, dim=0)
        return outputs.squeeze()

    def _unproject_points_batch(
        self,
        focal,
        principal_point,
        radial_params,
        tangential_params,
        thin_prism_params,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            focal: (1)
            principal_point: (2)
            radial_params: (num_radial)
            tangential_params: (2)
            thin_prism_params: (4)
            xy: (..., 2)

        Returns:
            point3d_est: (..., 3)
        """
        sh = list(xy.shape[:-1])
        assert xy.shape[-1] == 2, "xy_depth shape incorrect"
        uv_distorted = (xy - principal_point) / focal

        # get xr_yr from uvDistorted
        xr_yr = self._compute_xr_yr_from_uv_distorted(
            tangential_params, thin_prism_params, uv_distorted
        )
        xr_yrNorm = torch.norm(xr_yr, dim=-1)

        # find theta
        theta = self._get_theta_from_norm_xr_yr(radial_params, xr_yrNorm)
        # get the point coordinates:
        point3d_est = theta.new_ones(*sh, 3)
        point3d_est[..., :2] = theta.tan()[..., None] / xr_yrNorm[..., None] * xr_yr
        return point3d_est

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Takes in 3-point ``uv_depth`` in the image plane of the camera and unprojects it
        into the reference frame of the camera.
        This function is the inverse of ``transform_points``. In particular it holds that

        X = unproject(project(X))
            and
        x = project(unproject(s*x))

        Args:
            xy_depth: points in the image plane of shape (..., 3). E.g.,
                (P, 3) or (1, P, 3) or (M, P, 3)
            world_coordinates: if the output is in world_coordinate, if False, convert to
            camera coordinate
            scaled_depth_input: False

        Returns:
            unprojected_points in the camera frame with z = 1
            when points take shape (P, 3) or (1, P, 3), output is (N, P, 3)
            when points take shape (M, P, 3), output is (M, P, 3)
            where N is the number of transforms, P number of point
        """
        xy_depth = xy_depth.to(self.device)
        N = len(self.radial_params)
        if N == 1:
            return self._unproject_points_batch(
                self.focal[0],
                self.principal_point[0],
                self.radial_params[0],
                self.tangential_params[0],
                self.thin_prism_params[0],
                xy_depth[..., 0:2],
            )
        else:
            outputs = []
            for i in range(N):
                outputs.append(
                    self._unproject_points_batch(
                        self.focal[i],
                        self.principal_point[i],
                        self.radial_params[i],
                        self.tangential_params[i],
                        self.thin_prism_params[i],
                        xy_depth[..., 0:2],
                    )
                )
            outputs = torch.stack(outputs, dim=0)
        return outputs.squeeze()

    def _compute_xr_yr_from_uv_distorted(
        self, tangential_params, thin_prism_params, uv_distorted: torch.Tensor
    ) -> torch.Tensor:
        """
        Helper function to compute the vector [x_r; y_r] from uvDistorted

        Args:
            tangential_params: (2)
            thin_prism_params: (4)
            uv_distorted: (..., 2), E.g., (P, 2), (1, P, 2), (M, P, 2)

        Returns:
            xr_yr: (..., 2)
        """
        # early exit if we're not using any tangential/ thin prism distortions
        if not self.use_tangential and not self.use_thin_prism:
            return uv_distorted

        xr_yr = uv_distorted
        # do Newton iterations to find xr_yr
        for _ in range(self.num_distortion_iters):
            # compute the estimated uvDistorted
            uv_distorted_est = xr_yr.clone()
            xr_yr_squared_norm = torch.pow(xr_yr, 2).sum(dim=-1, keepdim=True)

            if self.use_tangential:
                temp = 2.0 * torch.sum(
                    xr_yr * tangential_params[..., 0:2],
                    dim=-1,
                    keepdim=True,
                )
                uv_distorted_est = uv_distorted_est + (
                    temp * xr_yr + xr_yr_squared_norm * tangential_params[..., 0:2]
                )

            if self.use_thin_prism:
                radial_powers = torch.cat(
                    [xr_yr_squared_norm, xr_yr_squared_norm * xr_yr_squared_norm],
                    dim=-1,
                )
                uv_distorted_est[..., 0] = uv_distorted_est[..., 0] + torch.sum(
                    thin_prism_params[..., 0:2] * radial_powers,
                    dim=-1,
                )
                uv_distorted_est[..., 1] = uv_distorted_est[..., 1] + torch.sum(
                    thin_prism_params[..., 2:4] * radial_powers,
                    dim=-1,
                )

            # compute the derivative of uvDistorted wrt xr_yr
            duv_distorted_dxryr = self._compute_duv_distorted_dxryr(
                tangential_params, thin_prism_params, xr_yr, xr_yr_squared_norm[..., 0]
            )
            # compute correction:
            # note: the matrix duvDistorted_dxryr will be close to identity (for reasonable
            # values of tangential/thin prism distortions)
            correction = torch.linalg.solve(
                duv_distorted_dxryr, (uv_distorted - uv_distorted_est)[..., None]
            )
            xr_yr = xr_yr + correction[..., 0]
        return xr_yr

    def _get_theta_from_norm_xr_yr(
        self, radial_params, th_radial_desired
    ) -> torch.Tensor:
        """
        Helper function to compute the angle theta from the norm of the vector [x_r; y_r]

        Args:
            radial_params: k1, k2, ..., k_num_radial, (num_radial)
            th_radial_desired: desired angle of shape (...), E.g., (P), (1, P), (M, P)

        Returns:
            th: angle theta (in radians) of shape (...), E.g., (P), (1, P), (M, P)
        """
        sh = list(th_radial_desired.shape)
        th = th_radial_desired
        c = torch.tensor(
            [2.0 * i + 3 for i in range(self.num_radial)], device=self.device
        )
        for _ in range(self.num_distortion_iters):
            theta_sq = th * th
            th_radial = 1.0
            dthD_dth = 1.0

            # compute the theta polynomial and its derivative wrt theta
            t = theta_sq
            theta_pow = torch.stack([t, t**2, t**3, t**4, t**5, t**6], dim=-1)
            th_radial = th_radial + torch.sum(theta_pow * radial_params, dim=-1)

            dthD_dth = dthD_dth + torch.sum(c * radial_params * theta_pow, dim=-1)
            th_radial = th_radial * th

            # compute the correction
            step = torch.zeros(*sh, device=self.device)
            # make sure don't divide by zero
            nonzero_mask = dthD_dth.abs() > self.epsilon
            step = step + nonzero_mask * (th_radial_desired - th_radial) / dthD_dth
            # if derivative is close to zero, apply small correction in the appropriate
            # direction to avoid numerical explosions
            close_to_zero_mask = dthD_dth.abs() <= self.epsilon
            dir_mask = (th_radial_desired - th_radial) * dthD_dth > 0.0
            boolean_mask = close_to_zero_mask & dir_mask
            step = step + 10.0 * self.epsilon * boolean_mask
            step = step - 10 * self.epsilon * (~nonzero_mask & ~boolean_mask)

            # apply correction
            th = th + step
            # revert to within 180 degrees FOV to avoid numerical overflow
            idw = th.abs() >= math.pi / 2.0
            th[idw] = 0.999 * math.pi / 2.0
        return th

    def _compute_duv_distorted_dxryr(
        self, tangential_params, thin_prism_params, xr_yr, xr_yr_squareNorm
    ) -> torch.Tensor:
        """
        Helper function, computes the Jacobian of uvDistorted wrt the vector [x_r;y_r]

        Args:
            tangential_params: (2)
            thin_prism_params: (4)
            xr_yr: (P, 2)
            xr_yr_squareNorm: (...), E.g., (P), (1, P), (M, P)

        Returns:
            duv_distorted_dxryr: (..., 2, 2) Jacobian
        """
        sh = list(xr_yr.shape[:-1])
        duv_distorted_dxryr = torch.empty((*sh, 2, 2), device=self.device)
        if self.use_tangential:
            duv_distorted_dxryr[..., 0, 0] = (
                1.0
                + 6.0 * xr_yr[..., 0] * tangential_params[..., 0]
                + 2.0 * xr_yr[..., 1] * tangential_params[..., 1]
            )
            offdiag = 2.0 * (
                xr_yr[..., 0] * tangential_params[..., 1]
                + xr_yr[..., 1] * tangential_params[..., 0]
            )
            duv_distorted_dxryr[..., 0, 1] = offdiag
            duv_distorted_dxryr[..., 1, 0] = offdiag
            duv_distorted_dxryr[..., 1, 1] = (
                1.0
                + 6.0 * xr_yr[..., 1] * tangential_params[..., 1]
                + 2.0 * xr_yr[..., 0] * tangential_params[..., 0]
            )
        else:
            duv_distorted_dxryr = torch.eye(2).repeat(*sh, 1, 1)

        if self.use_thin_prism:
            temp1 = 2.0 * (
                thin_prism_params[..., 0]
                + 2.0 * thin_prism_params[..., 1] * xr_yr_squareNorm[...]
            )
            duv_distorted_dxryr[..., 0, 0] = (
                duv_distorted_dxryr[..., 0, 0] + xr_yr[..., 0] * temp1
            )
            duv_distorted_dxryr[..., 0, 1] = (
                duv_distorted_dxryr[..., 0, 1] + xr_yr[..., 1] * temp1
            )

            temp2 = 2.0 * (
                thin_prism_params[..., 2]
                + 2.0 * thin_prism_params[..., 3] * xr_yr_squareNorm[...]
            )
            duv_distorted_dxryr[..., 1, 0] = (
                duv_distorted_dxryr[..., 1, 0] + xr_yr[..., 0] * temp2
            )
            duv_distorted_dxryr[..., 1, 1] = (
                duv_distorted_dxryr[..., 1, 1] + xr_yr[..., 1] * temp2
            )
        return duv_distorted_dxryr

    def in_ndc(self):
        return True

    def is_perspective(self):
        return False
