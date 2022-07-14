# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.ops import perspective_n_points
from pytorch3d.transforms import rotation_conversions

from .common_testing import TestCaseMixin


def reproj_error(x_world, y, R, T, weight=None):
    # applies the affine transform, projects, and computes the reprojection error
    y_hat = torch.matmul(x_world, R) + T[:, None, :]
    y_hat = y_hat / y_hat[..., 2:]
    if weight is None:
        weight = y.new_ones((1, 1))
    return (((weight[:, :, None] * (y - y_hat[..., :2])) ** 2).sum(dim=-1) ** 0.5).mean(
        dim=-1
    )


class TestPerspectiveNPoints(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    @classmethod
    def _generate_epnp_test_from_2d(cls, y):
        """
        Instantiate random x_world, x_cam, R, T given a set of input
        2D projections y.
        """
        batch_size = y.shape[0]
        x_cam = torch.cat((y, torch.rand_like(y[:, :, :1]) * 2.0 + 3.5), dim=2)
        x_cam[:, :, :2] *= x_cam[:, :, 2:]  # unproject
        R = rotation_conversions.random_rotations(batch_size).to(y)
        T = torch.randn_like(R[:, :1, :])
        T[:, :, 2] = (T[:, :, 2] + 3.0).clamp(2.0)
        x_world = torch.matmul(x_cam - T, R.transpose(1, 2))
        return x_cam, x_world, R, T

    def _run_and_print(self, x_world, y, R, T, print_stats, skip_q, check_output=False):
        sol = perspective_n_points.efficient_pnp(
            x_world, y.expand_as(x_world[:, :, :2]), skip_quadratic_eq=skip_q
        )

        err_2d = reproj_error(x_world, y, sol.R, sol.T)
        R_est_quat = rotation_conversions.matrix_to_quaternion(sol.R)
        R_quat = rotation_conversions.matrix_to_quaternion(R)

        num_pts = x_world.shape[-2]
        if check_output:
            assert_msg = (
                f"test_perspective_n_points assertion failure for "
                f"n_points={num_pts}, "
                f"skip_quadratic={skip_q}, "
                f"no noise."
            )

            self.assertClose(err_2d, sol.err_2d, msg=assert_msg)
            self.assertTrue((err_2d < 1e-3).all(), msg=assert_msg)

            def norm_fn(t):
                return t.norm(dim=-1)

            self.assertNormsClose(
                T, sol.T[:, None, :], rtol=4e-3, norm_fn=norm_fn, msg=assert_msg
            )
            self.assertNormsClose(
                R_quat, R_est_quat, rtol=3e-3, norm_fn=norm_fn, msg=assert_msg
            )

        if print_stats:
            torch.set_printoptions(precision=5, sci_mode=False)
            for err_2d, err_3d, R_gt, T_gt in zip(
                sol.err_2d,
                sol.err_3d,
                torch.cat((sol.R, R), dim=-1),
                torch.stack((sol.T, T[:, 0, :]), dim=-1),
            ):
                print("2D Error: %1.4f" % err_2d.item())
                print("3D Error: %1.4f" % err_3d.item())
                print("R_hat | R_gt\n", R_gt)
                print("T_hat | T_gt\n", T_gt)

    def _testcase_from_2d(
        self, y, print_stats, benchmark, skip_q=False, skip_check_thresh=5
    ):
        """
        In case num_pts < 6, EPnP gets unstable, so we check it doesn't crash
        """
        x_cam, x_world, R, T = TestPerspectiveNPoints._generate_epnp_test_from_2d(
            y[None].repeat(16, 1, 1)
        )

        if print_stats:
            print("Run without noise")

        if benchmark:  # return curried call
            torch.cuda.synchronize()

            def result():
                self._run_and_print(x_world, y, R, T, False, skip_q)
                torch.cuda.synchronize()

            return result

        self._run_and_print(
            x_world,
            y,
            R,
            T,
            print_stats,
            skip_q,
            check_output=True if y.shape[1] > skip_check_thresh else False,
        )

        # in the noisy case, there are no guarantees, so we check it doesn't crash
        if print_stats:
            print("Run with noise")
        x_world += torch.randn_like(x_world) * 0.1
        self._run_and_print(x_world, y, R, T, print_stats, skip_q)

    def case_with_gaussian_points(
        self, batch_size=10, num_pts=20, print_stats=False, benchmark=True, skip_q=False
    ):
        return self._testcase_from_2d(
            torch.randn((num_pts, 2)).cuda() / 3.0,
            print_stats=print_stats,
            benchmark=benchmark,
            skip_q=skip_q,
        )

    def test_perspective_n_points(self, print_stats=False):
        if print_stats:
            print("RUN ON A DENSE GRID")
        u = torch.linspace(-1.0, 1.0, 20)
        v = torch.linspace(-1.0, 1.0, 15)
        for skip_q in [False, True]:
            self._testcase_from_2d(
                torch.cartesian_prod(u, v).cuda(), print_stats, False, skip_q
            )

        for num_pts in range(6, 3, -1):
            for skip_q in [False, True]:
                if print_stats:
                    print(f"RUN ON {num_pts} points; skip_quadratic: {skip_q}")

                self.case_with_gaussian_points(
                    num_pts=num_pts,
                    print_stats=print_stats,
                    benchmark=False,
                    skip_q=skip_q,
                )

    def test_weighted_perspective_n_points(self, batch_size=16, num_pts=200):
        # instantiate random x_world and y
        y = torch.randn((batch_size, num_pts, 2)).cuda() / 3.0
        x_cam, x_world, R, T = TestPerspectiveNPoints._generate_epnp_test_from_2d(y)

        # randomly drop 50% of the rows
        weights = (torch.rand_like(x_world[:, :, 0]) > 0.5).float()

        # make sure we retain at least 6 points for each case
        weights[:, :6] = 1.0

        # fill ignored y with trash to ensure that we get different
        # solution in case the weighting is wrong
        y = y + (1 - weights[:, :, None]) * 100.0

        def norm_fn(t):
            return t.norm(dim=-1)

        for skip_quadratic_eq in (True, False):
            # get the solution for the 0/1 weighted case
            sol = perspective_n_points.efficient_pnp(
                x_world, y, skip_quadratic_eq=skip_quadratic_eq, weights=weights
            )
            sol_R_quat = rotation_conversions.matrix_to_quaternion(sol.R)
            sol_T = sol.T

            # check that running only on points with non-zero weights ends in the
            # same place as running the 0/1 weighted version
            for i in range(batch_size):
                ok = weights[i] > 0
                x_world_ok = x_world[i, ok][None]
                y_ok = y[i, ok][None]
                sol_ok = perspective_n_points.efficient_pnp(
                    x_world_ok, y_ok, skip_quadratic_eq=False
                )
                R_est_quat_ok = rotation_conversions.matrix_to_quaternion(sol_ok.R)

                self.assertNormsClose(sol_T[i], sol_ok.T[0], rtol=3e-3, norm_fn=norm_fn)
                self.assertNormsClose(
                    sol_R_quat[i], R_est_quat_ok[0], rtol=3e-4, norm_fn=norm_fn
                )
