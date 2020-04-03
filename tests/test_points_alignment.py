#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import numpy as np
import unittest
import torch

from common_testing import TestCaseMixin

from pytorch3d.ops import points_alignment
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.transforms import rotation_conversions


def _apply_pcl_transformation(X, R, T, s=None):
    """
    Apply a batch of similarity/rigid transformations, parametrized with
    rotation `R`, translation `T` and scale `s`, to an input batch of
    point clouds `X`.
    """
    if isinstance(X, Pointclouds):
        num_points = X.num_points_per_cloud()
        X_t = X.points_padded()
    else:
        X_t = X

    if s is not None:
        X_t = s[:, None, None] * X_t

    X_t = torch.bmm(X_t, R) + T[:, None, :]

    if isinstance(X, Pointclouds):
        X_list = [x[:n_p] for x, n_p in zip(X_t, num_points)]
        X_t = Pointclouds(X_list)

    return X_t


class TestCorrespondingPointsAlignment(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)
        np.random.seed(42)

    @staticmethod
    def random_rotation(batch_size, dim, device=None):
        """
        Generates a batch of random `dim`-dimensional rotation matrices.
        """
        if dim == 3:
            R = rotation_conversions.random_rotations(batch_size, device=device)
        else:
            # generate random rotation matrices with orthogonalization of
            # random normal square matrices, followed by a transformation
            # that ensures determinant(R)==1
            H = torch.randn(
                batch_size, dim, dim, dtype=torch.float32, device=device
            )
            U, _, V = torch.svd(H)
            E = torch.eye(dim, dtype=torch.float32, device=device)[None].repeat(
                batch_size, 1, 1
            )
            E[:, -1, -1] = torch.det(torch.bmm(U, V.transpose(2, 1)))
            R = torch.bmm(torch.bmm(U, E), V.transpose(2, 1))
            assert torch.allclose(
                torch.det(R), R.new_ones(batch_size), atol=1e-4
            )

        return R

    @staticmethod
    def init_point_cloud(
        batch_size=10,
        n_points=1000,
        dim=3,
        device=None,
        use_pointclouds=False,
        random_pcl_size=True,
    ):
        """
        Generate a batch of normally distributed point clouds.
        """
        if use_pointclouds:
            assert dim == 3, "Pointclouds support only 3-dim points."
            # generate a `batch_size` point clouds with number of points
            # between 4 and `n_points`
            if random_pcl_size:
                n_points_per_batch = torch.randint(
                    low=4,
                    high=n_points,
                    size=(batch_size,),
                    device=device,
                    dtype=torch.int64,
                )
                X_list = [
                    torch.randn(
                        int(n_pt), dim, device=device, dtype=torch.float32
                    )
                    for n_pt in n_points_per_batch
                ]
                X = Pointclouds(X_list)
            else:
                X = torch.randn(
                    batch_size,
                    n_points,
                    dim,
                    device=device,
                    dtype=torch.float32,
                )
                X = Pointclouds(list(X))
        else:
            X = torch.randn(
                batch_size, n_points, dim, device=device, dtype=torch.float32
            )
        return X

    @staticmethod
    def generate_pcl_transformation(
        batch_size=10, scale=False, reflect=False, dim=3, device=None
    ):
        """
        Generate a batch of random rigid/similarity transformations.
        """
        R = TestCorrespondingPointsAlignment.random_rotation(
            batch_size, dim, device=device
        )
        T = torch.randn(batch_size, dim, dtype=torch.float32, device=device)
        if scale:
            s = torch.rand(batch_size, dtype=torch.float32, device=device) + 0.1
        else:
            s = torch.ones(batch_size, dtype=torch.float32, device=device)

        return R, T, s

    @staticmethod
    def generate_random_reflection(batch_size=10, dim=3, device=None):
        """
        Generate a batch of reflection matrices of shape (batch_size, dim, dim),
        where M_i is an identity matrix with one random entry on the
        diagonal equal to -1.
        """
        # randomly select one of the dimensions to reflect for each
        # element in the batch
        dim_to_reflect = torch.randint(
            low=0,
            high=dim,
            size=(batch_size,),
            device=device,
            dtype=torch.int64,
        )

        # convert dim_to_reflect to a batch of reflection matrices M
        M = torch.diag_embed(
            (
                dim_to_reflect[:, None]
                != torch.arange(dim, device=device, dtype=torch.float32)
            ).float()
            * 2
            - 1,
            dim1=1,
            dim2=2,
        )

        return M

    @staticmethod
    def corresponding_points_alignment(
        batch_size=10,
        n_points=100,
        dim=3,
        use_pointclouds=False,
        estimate_scale=False,
        allow_reflection=False,
        reflect=False,
        random_weights=False,
    ):

        device = torch.device("cuda:0")

        # initialize a ground truth point cloud
        X = TestCorrespondingPointsAlignment.init_point_cloud(
            batch_size=batch_size,
            n_points=n_points,
            dim=dim,
            device=device,
            use_pointclouds=use_pointclouds,
            random_pcl_size=True,
        )

        # generate the true transformation
        R, T, s = TestCorrespondingPointsAlignment.generate_pcl_transformation(
            batch_size=batch_size,
            scale=estimate_scale,
            reflect=reflect,
            dim=dim,
            device=device,
        )

        # apply the generated transformation to the generated
        # point cloud X
        X_t = _apply_pcl_transformation(X, R, T, s=s)

        weights = None
        if random_weights:
            template = X.points_padded() if use_pointclouds else X
            weights = torch.rand_like(template[:, :, 0])
            weights = weights / weights.sum(dim=1, keepdim=True)
            # zero out some weights as zero weights are a common use case
            # this guarantees there are no zero weight
            weights *= (weights * template.size()[1] > 0.3).to(weights)
            if use_pointclouds:  # convert to List[Tensor]
                weights = [
                    w[:npts]
                    for w, npts in zip(weights, X.num_points_per_cloud())
                ]

        torch.cuda.synchronize()

        def run_corresponding_points_alignment():
            points_alignment.corresponding_points_alignment(
                X,
                X_t,
                weights,
                allow_reflection=allow_reflection,
                estimate_scale=estimate_scale,
            )
            torch.cuda.synchronize()

        return run_corresponding_points_alignment

    def test_corresponding_points_alignment(self, batch_size=10):
        """
        Tests whether we can estimate a rigid/similarity motion between
        a randomly initialized point cloud and its randomly transformed version.

        The tests are done for all possible combinations
        of the following boolean flags:
            - estimate_scale ... Estimate also a scaling component of
                                 the transformation.
            - reflect ... The ground truth orthonormal part of the generated
                         transformation is a reflection (det==-1).
            - allow_reflection ... If True, the orthonormal matrix of the
                                  estimated transformation is allowed to be
                                  a reflection (det==-1).
            - use_pointclouds ... If True, passes the Pointclouds objects
                                  to corresponding_points_alignment.
        """

        # run this for several different point cloud sizes
        for n_points in (100, 3, 2, 1):
            # run this for several different dimensionalities
            for dim in range(2, 10):
                # switches whether we should use the Pointclouds inputs
                use_point_clouds_cases = (
                    (True, False) if dim == 3 and n_points > 3 else (False,)
                )
                for random_weights in (False, True,):
                    for use_pointclouds in use_point_clouds_cases:
                        for estimate_scale in (False, True):
                            for reflect in (False, True):
                                for allow_reflection in (False, True):
                                    self._test_single_corresponding_points_alignment(
                                        batch_size=10,
                                        n_points=n_points,
                                        dim=dim,
                                        use_pointclouds=use_pointclouds,
                                        estimate_scale=estimate_scale,
                                        reflect=reflect,
                                        allow_reflection=allow_reflection,
                                        random_weights=random_weights,
                                    )

    def _test_single_corresponding_points_alignment(
        self,
        batch_size=10,
        n_points=100,
        dim=3,
        use_pointclouds=False,
        estimate_scale=False,
        reflect=False,
        allow_reflection=False,
        random_weights=False,
    ):
        """
        Executes a single test for `corresponding_points_alignment` for a
        specific setting of the inputs / outputs.
        """

        device = torch.device("cuda:0")

        # initialize the a ground truth point cloud
        X = TestCorrespondingPointsAlignment.init_point_cloud(
            batch_size=batch_size,
            n_points=n_points,
            dim=dim,
            device=device,
            use_pointclouds=use_pointclouds,
            random_pcl_size=True,
        )

        # generate the true transformation
        R, T, s = TestCorrespondingPointsAlignment.generate_pcl_transformation(
            batch_size=batch_size,
            scale=estimate_scale,
            reflect=reflect,
            dim=dim,
            device=device,
        )

        if reflect:
            # generate random reflection M and apply to the rotations
            M = TestCorrespondingPointsAlignment.generate_random_reflection(
                batch_size=batch_size, dim=dim, device=device
            )
            R = torch.bmm(M, R)

        weights = None
        if random_weights:
            template = X.points_padded() if use_pointclouds else X
            weights = torch.rand_like(template[:, :, 0])
            weights = weights / weights.sum(dim=1, keepdim=True)
            # zero out some weights as zero weights are a common use case
            # this guarantees there are no zero weight
            weights *= (weights * template.size()[1] > 0.3).to(weights)
            if use_pointclouds:  # convert to List[Tensor]
                weights = [
                    w[:npts]
                    for w, npts in zip(weights, X.num_points_per_cloud())
                ]

        # apply the generated transformation to the generated
        # point cloud X
        X_t = _apply_pcl_transformation(X, R, T, s=s)

        # run the CorrespondingPointsAlignment algorithm
        R_est, T_est, s_est = points_alignment.corresponding_points_alignment(
            X,
            X_t,
            weights,
            allow_reflection=allow_reflection,
            estimate_scale=estimate_scale,
        )

        assert_error_message = (
            f"Corresponding_points_alignment assertion failure for "
            f"n_points={n_points}, "
            f"dim={dim}, "
            f"use_pointclouds={use_pointclouds}, "
            f"estimate_scale={estimate_scale}, "
            f"reflect={reflect}, "
            f"allow_reflection={allow_reflection},"
            f"random_weights={random_weights}."
        )

        # if we test the weighted case, check that weights help with noise
        if random_weights and not use_pointclouds and n_points >= (dim + 10):
            # add noise to 20% points with smallest weight
            X_noisy = X_t.clone()
            _, mink_idx = torch.topk(-weights, int(n_points * 0.2), dim=1)
            mink_idx = mink_idx[:, :, None].expand(-1, -1, X_t.shape[-1])
            X_noisy.scatter_add_(
                1, mink_idx, 0.3 * torch.randn_like(mink_idx, dtype=X_t.dtype)
            )

            def align_and_get_mse(weights_):
                R_n, T_n, s_n = points_alignment.corresponding_points_alignment(
                    X_noisy,
                    X_t,
                    weights_,
                    allow_reflection=allow_reflection,
                    estimate_scale=estimate_scale,
                )

                X_t_est = _apply_pcl_transformation(X_noisy, R_n, T_n, s=s_n)

                return (
                    ((X_t_est - X_t) * weights[..., None]) ** 2
                ).sum(dim=(1, 2)) / weights.sum(dim=-1)

            # check that using weights leads to lower weighted_MSE(X_noisy, X_t)
            self.assertTrue(
                torch.all(align_and_get_mse(weights) <= align_and_get_mse(None))
            )

        if reflect and not allow_reflection:
            # check that all rotations have det=1
            self._assert_all_close(
                torch.det(R_est),
                R_est.new_ones(batch_size),
                assert_error_message,
            )

        else:
            # mask out inputs with too few non-degenerate points for assertions
            w = (
                torch.ones_like(R_est[:, 0, 0])
                if weights is None or n_points >= dim + 10
                else (weights > 0.0).all(dim=1).to(R_est)
            )
            # check that the estimated tranformation is the same
            # as the ground truth
            if n_points >= (dim + 1):
                # the checks on transforms apply only when
                # the problem setup is unambiguous
                msg = assert_error_message
                self._assert_all_close(R_est, R, msg, w[:, None, None], atol=1e-5)
                self._assert_all_close(T_est, T, msg, w[:, None])
                self._assert_all_close(s_est, s, msg, w)

                # check that the orthonormal part of the
                # transformation has a correct determinant (+1/-1)
                desired_det = R_est.new_ones(batch_size)
                if reflect:
                    desired_det *= -1.0
                self._assert_all_close(torch.det(R_est), desired_det, msg, w)

            # check that the transformed point cloud
            # X matches X_t
            X_t_est = _apply_pcl_transformation(X, R_est, T_est, s=s_est)
            self._assert_all_close(
                X_t, X_t_est, assert_error_message, w[:, None, None], atol=1e-5
            )

    def _assert_all_close(self, a_, b_, err_message, weights=None, atol=1e-6):
        if isinstance(a_, Pointclouds):
            a_ = a_.points_packed()
        if isinstance(b_, Pointclouds):
            b_ = b_.points_packed()
        if weights is None:
            self.assertClose(a_, b_, atol=atol, msg=err_message)
        else:
            self.assertClose(
                a_ * weights, b_ * weights, atol=atol, msg=err_message
            )
