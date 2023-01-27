# @lint-ignore-every LICENSELINT
# Adapted from https://github.com/lioryariv/idr
# Copyright (c) 2020 Lior Yariv

from typing import Any, Callable, Tuple

import torch
import torch.nn as nn
from pytorch3d.implicitron.tools.config import Configurable


class RayTracing(Configurable, nn.Module):
    """
    Finds the intersection points of rays with the implicit surface defined
    by a signed distance function (SDF). The algorithm follows the pipeline:
    1. Initialise start and end points on rays by the intersections with
        the circumscribing sphere.
    2. Run sphere tracing from both ends.
    3. Divide the untraced segments of non-convergent rays into uniform
        intervals and find the one with the sign transition.
    4. Run the secant method to estimate the point of the sign transition.

    Args:
        object_bounding_sphere: The radius of the initial sphere circumscribing
            the object.
        sdf_threshold: Absolute SDF value small enough for the sphere tracer
            to consider it a surface.
        line_search_step: Length of the backward correction on sphere tracing
            iterations.
        line_step_iters: Number of backward correction iterations.
        sphere_tracing_iters: Maximum number of sphere tracing iterations
            (the actual number of iterations may be smaller if all ray
            intersections are found).
        n_steps: Number of intervals sampled for unconvergent rays.
        n_secant_steps: Number of iterations in the secant algorithm.
    """

    object_bounding_sphere: float = 1.0
    sdf_threshold: float = 5.0e-5
    line_search_step: float = 0.5
    line_step_iters: int = 1
    sphere_tracing_iters: int = 10
    n_steps: int = 100
    n_secant_steps: int = 8

    def forward(
        self,
        sdf: Callable[[torch.Tensor], torch.Tensor],
        cam_loc: torch.Tensor,
        object_mask: torch.BoolTensor,
        ray_directions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sdf: A callable that takes a (N, 3) tensor of points and returns
                a tensor of (N,) SDF values.
            cam_loc: A tensor of (B, N, 3) ray origins.
            object_mask: A (N, 3) tensor of indicators whether a sampled pixel
                corresponds to the rendered object or background.
            ray_directions: A tensor of (B, N, 3) ray directions.

        Returns:
            curr_start_points: A tensor of (B*N, 3) found intersection points
                with the implicit surface.
            network_object_mask: A tensor of (B*N,) indicators denoting whether
                intersections were found.
            acc_start_dis: A tensor of (B*N,) distances from the ray origins
                to intersrection points.
        """
        batch_size, num_pixels, _ = ray_directions.shape
        device = cam_loc.device

        sphere_intersections, mask_intersect = _get_sphere_intersection(
            cam_loc, ray_directions, r=self.object_bounding_sphere
        )

        (
            curr_start_points,
            unfinished_mask_start,
            acc_start_dis,
            acc_end_dis,
            min_dis,
            max_dis,
        ) = self.sphere_tracing(
            batch_size,
            num_pixels,
            sdf,
            cam_loc,
            ray_directions,
            mask_intersect,
            sphere_intersections,
        )

        network_object_mask = acc_start_dis < acc_end_dis

        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        sampler_net_obj_mask = torch.zeros_like(
            sampler_mask, dtype=torch.bool, device=device
        )
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2), device=device)
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[
                sampler_mask
            ]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]

            sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(
                sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask
            )

            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        if not self.training:
            return curr_start_points, network_object_mask, acc_start_dis

        # in case we are training, we are updating curr_start_points and acc_start_dis for

        ray_directions = ray_directions.reshape(-1, 3)
        mask_intersect = mask_intersect.reshape(-1)
        # pyre-fixme[9]: object_mask has type `BoolTensor`; used as `Tensor`.
        object_mask = object_mask.reshape(-1)

        in_mask = ~network_object_mask & object_mask & ~sampler_mask
        out_mask = ~object_mask & ~sampler_mask

        mask_left_out = (in_mask | out_mask) & ~mask_intersect
        if (
            mask_left_out.sum() > 0
        ):  # project the origin to the not intersect points on the sphere
            cam_left_out = cam_loc.reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            acc_start_dis[mask_left_out] = -torch.bmm(
                rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)
            ).squeeze()
            curr_start_points[mask_left_out] = (
                cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out
            )

        mask = (in_mask | out_mask) & mask_intersect

        if mask.sum() > 0:
            min_dis[network_object_mask & out_mask] = acc_start_dis[
                network_object_mask & out_mask
            ]

            min_mask_points, min_mask_dist = self.minimal_sdf_points(
                sdf, cam_loc, ray_directions, mask, min_dis, max_dis
            )

            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist

        return curr_start_points, network_object_mask, acc_start_dis

    def sphere_tracing(
        self,
        batch_size: int,
        num_pixels: int,
        sdf: Callable[[torch.Tensor], torch.Tensor],
        cam_loc: torch.Tensor,
        ray_directions: torch.Tensor,
        mask_intersect: torch.Tensor,
        sphere_intersections: torch.Tensor,
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        Run sphere tracing algorithm for max iterations
        from both sides of unit sphere intersection

        Args:
            batch_size:
            num_pixels:
            sdf:
            cam_loc:
            ray_directions:
            mask_intersect:
            sphere_intersections:

        Returns:
            curr_start_points:
            unfinished_mask_start:
            acc_start_dis:
            acc_end_dis:
            min_dis:
            max_dis:
        """

        device = cam_loc.device
        sphere_intersections_points = (
            cam_loc[..., None, :]
            + sphere_intersections[..., None] * ray_directions[..., None, :]
        )
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # Initialize start current points
        curr_start_points = torch.zeros(batch_size * num_pixels, 3, device=device)
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[
            :, :, 0, :
        ].reshape(-1, 3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels, device=device)
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1, 2)[
            unfinished_mask_start, 0
        ]

        # Initialize end current points
        curr_end_points = torch.zeros(batch_size * num_pixels, 3, device=device)
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[
            :, :, 1, :
        ].reshape(-1, 3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(batch_size * num_pixels, device=device)
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1, 2)[
            unfinished_mask_end, 1
        ]

        # Initialise min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        # TODO: sdf should also pass info about batches

        next_sdf_start = torch.zeros_like(acc_start_dis)
        next_sdf_start[unfinished_mask_start] = sdf(
            curr_start_points[unfinished_mask_start]
        )

        next_sdf_end = torch.zeros_like(acc_end_dis)
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis)
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[
                unfinished_mask_start
            ]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis)
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (
                curr_sdf_start > self.sdf_threshold
            )
            unfinished_mask_end = unfinished_mask_end & (
                curr_sdf_end > self.sdf_threshold
            )

            if (
                unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0
            ) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = (
                cam_loc
                + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions
            ).reshape(-1, 3)
            curr_end_points = (
                cam_loc
                + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions
            ).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis)
            next_sdf_start[unfinished_mask_start] = sdf(
                curr_start_points[unfinished_mask_start]
            )

            next_sdf_end = torch.zeros_like(acc_end_dis)
            next_sdf_end[unfinished_mask_end] = sdf(
                curr_end_points[unfinished_mask_end]
            )

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (
                not_projected_start.sum() > 0 or not_projected_end.sum() > 0
            ) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= (
                    (1 - self.line_search_step) / (2**not_proj_iters)
                ) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (
                    cam_loc
                    + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions
                ).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += (
                    (1 - self.line_search_step) / (2**not_proj_iters)
                ) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (
                    cam_loc
                    + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions
                ).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(
                    curr_start_points[not_projected_start]
                )
                next_sdf_end[not_projected_end] = sdf(
                    curr_end_points[not_projected_end]
                )

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (
                acc_start_dis < acc_end_dis
            )
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return (
            curr_start_points,
            unfinished_mask_start,
            acc_start_dis,
            acc_end_dis,
            min_dis,
            max_dis,
        )

    def ray_sampler(
        self,
        sdf: Callable[[torch.Tensor], torch.Tensor],
        cam_loc: torch.Tensor,
        object_mask: torch.Tensor,
        ray_directions: torch.Tensor,
        sampler_min_max: torch.Tensor,
        sampler_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample the ray in a given range and run secant on rays which have sign transition.

        Args:
            sdf:
            cam_loc:
            object_mask:
            ray_directions:
            sampler_min_max:
            sampler_mask:

        Returns:

        """

        batch_size, num_pixels, _ = ray_directions.shape
        device = cam_loc.device
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3, device=device)
        sampler_dists = torch.zeros(n_total_pxl, device=device)

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps, device=device).view(
            1, 1, -1
        )

        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (
            sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]
        ).unsqueeze(-1)
        points = (
            cam_loc[..., None, :]
            + pts_intervals[..., None] * ray_directions[..., None, :]
        )

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(
            self.n_steps, 0, -1, device=device, dtype=torch.float32
        ).reshape(1, self.n_steps)
        # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[
            torch.arange(points.shape[0]), sampler_pts_ind, :
        ]
        sampler_dists[mask_intersect_idx] = pts_intervals[
            torch.arange(pts_intervals.shape[0]), sampler_pts_ind
        ]

        true_surface_pts = object_mask.reshape(-1)[sampler_mask]
        net_surface_pts = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][
                # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]`
                #  but got `Tensor`.
                torch.arange(n_p_out),
                out_pts_idx,
                :,
            ]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[
                p_out_mask,
                :
                # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]` but
                #  got `Tensor`.
            ][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = (
            net_surface_pts & true_surface_pts if self.training else net_surface_pts
        )
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[
                torch.arange(pts_intervals.shape[0]), sampler_pts_ind
            ][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][
                secant_pts
            ]
            z_low = pts_intervals[secant_pts][
                # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]`
                #  but got `Tensor`.
                torch.arange(n_secant_pts),
                sampler_pts_ind[secant_pts] - 1,
            ]
            sdf_low = sdf_val[secant_pts][
                # pyre-fixme[6]: For 1st param expected `Union[bool, float, int]`
                #  but got `Tensor`.
                torch.arange(n_secant_pts),
                sampler_pts_ind[secant_pts] - 1,
            ]
            cam_loc_secant = cam_loc.reshape(-1, 3)[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_directions.reshape((-1, 3))[
                mask_intersect_idx[secant_pts]
            ]
            z_pred_secant = self.secant(
                sdf_low,
                sdf_high,
                z_low,
                z_high,
                cam_loc_secant,
                ray_directions_secant,
                # pyre-fixme[6]: For 7th param expected `Module` but got `(Tensor)
                #  -> Tensor`.
                sdf,
            )

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = (
                cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            )
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(
        self,
        sdf_low: torch.Tensor,
        sdf_high: torch.Tensor,
        z_low: torch.Tensor,
        z_high: torch.Tensor,
        cam_loc: torch.Tensor,
        ray_directions: torch.Tensor,
        sdf: nn.Module,
    ) -> torch.Tensor:
        """
        Runs the secant method for interval [z_low, z_high] for n_secant_steps
        """

        z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for _ in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = -sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low

        return z_pred

    def minimal_sdf_points(
        self,
        sdf: Callable[[torch.Tensor], torch.Tensor],
        cam_loc: torch.Tensor,
        ray_directions: torch.Tensor,
        mask: torch.Tensor,
        min_dis: torch.Tensor,
        max_dis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find points with minimal SDF value on rays for P_out pixels
        """

        n_mask_points = mask.sum()

        n = self.n_steps
        steps = torch.empty(n, device=cam_loc.device).uniform_(0.0, 1.0)
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = (
            # pyre-fixme[6]: For 1st param expected `int` but got `Tensor`.
            steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis)
            + mask_min_dis
        )

        mask_points = cam_loc.reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(
            -1
        ) * mask_rays.unsqueeze(1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[
            # pyre-fixme[6]: For 2nd param expected `Union[bool, float, int]` but
            #  got `Tensor`.
            torch.arange(0, n_mask_points),
            min_idx,
        ]
        # pyre-fixme[6]: For 2nd param expected `Union[bool, float, int]` but got
        #  `Tensor`.
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist


# TODO: support variable origins
def _get_sphere_intersection(
    cam_loc: torch.Tensor, ray_directions: torch.Tensor, r: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Input: n_images x 3 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays

    n_imgs, n_pix, _ = ray_directions.shape
    device = cam_loc.device

    # cam_loc = cam_loc.unsqueeze(-1)
    # ray_cam_dot = torch.bmm(ray_directions, cam_loc).squeeze()
    ray_cam_dot = (ray_directions * cam_loc).sum(-1)  # n_images x n_rays
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    under_sqrt = ray_cam_dot**2 - (cam_loc.norm(2, dim=-1) ** 2 - r**2)

    under_sqrt = under_sqrt.reshape(-1)
    mask_intersect = under_sqrt > 0

    sphere_intersections = torch.zeros(n_imgs * n_pix, 2, device=device)
    sphere_intersections[mask_intersect] = torch.sqrt(
        under_sqrt[mask_intersect]
    ).unsqueeze(-1) * torch.tensor([-1.0, 1.0], device=device)
    sphere_intersections[mask_intersect] -= ray_cam_dot.reshape(-1)[
        mask_intersect
    ].unsqueeze(-1)

    sphere_intersections = sphere_intersections.reshape(n_imgs, n_pix, 2)
    sphere_intersections = sphere_intersections.clamp_min(0.0)
    mask_intersect = mask_intersect.reshape(n_imgs, n_pix)

    return sphere_intersections, mask_intersect
