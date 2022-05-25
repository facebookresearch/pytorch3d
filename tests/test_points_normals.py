# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple, Union

import torch
from pytorch3d.ops import (
    estimate_pointcloud_local_coord_frames,
    estimate_pointcloud_normals,
)
from pytorch3d.structures.pointclouds import Pointclouds

from .common_testing import TestCaseMixin


DEBUG = False


class TestPCLNormals(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

    @staticmethod
    def init_spherical_pcl(
        batch_size=3, num_points=3000, device=None, use_pointclouds=False
    ) -> Tuple[Union[torch.Tensor, Pointclouds], torch.Tensor]:
        # random spherical point cloud
        pcl = torch.randn(
            (batch_size, num_points, 3), device=device, dtype=torch.float32
        )
        pcl = torch.nn.functional.normalize(pcl, dim=2)

        # GT normals are the same as
        # the location of each point on the 0-centered sphere
        normals = pcl.clone()

        # scale and offset the sphere randomly
        pcl *= torch.rand(batch_size, 1, 1).type_as(pcl) + 1.0
        pcl += torch.randn(batch_size, 1, 3).type_as(pcl)

        if use_pointclouds:
            num_points = torch.randint(
                size=(batch_size,), low=int(num_points * 0.7), high=num_points
            )
            pcl, normals = [
                [x[:np] for x, np in zip(X, num_points)] for X in (pcl, normals)
            ]
            pcl = Pointclouds(pcl, normals=normals)

        return pcl, normals

    def test_pcl_normals(self, batch_size=3, num_points=300, neighborhood_size=50):
        """
        Tests the normal estimation on a spherical point cloud, where
        we know the ground truth normals.
        """
        device = torch.device("cuda:0")
        # run several times for different random point clouds
        for run_idx in range(3):
            # either use tensors or Pointclouds as input
            for use_pointclouds in (True, False):
                # get a spherical point cloud
                pcl, normals_gt = TestPCLNormals.init_spherical_pcl(
                    num_points=num_points,
                    batch_size=batch_size,
                    device=device,
                    use_pointclouds=use_pointclouds,
                )
                if use_pointclouds:
                    normals_gt = pcl.normals_padded()
                    num_pcl_points = pcl.num_points_per_cloud()
                else:
                    num_pcl_points = [pcl.shape[1]] * batch_size

                # check for both disambiguation options
                for disambiguate_directions in (True, False):
                    (
                        curvatures,
                        local_coord_frames,
                    ) = estimate_pointcloud_local_coord_frames(
                        pcl,
                        neighborhood_size=neighborhood_size,
                        disambiguate_directions=disambiguate_directions,
                    )

                    # estimate the normals
                    normals = estimate_pointcloud_normals(
                        pcl,
                        neighborhood_size=neighborhood_size,
                        disambiguate_directions=disambiguate_directions,
                    )

                    # TODO: temporarily disabled
                    if use_pointclouds:
                        # test that the class method gives the same output
                        normals_pcl = pcl.estimate_normals(
                            neighborhood_size=neighborhood_size,
                            disambiguate_directions=disambiguate_directions,
                            assign_to_self=True,
                        )
                        normals_from_pcl = pcl.normals_padded()
                        for nrm, nrm_from_pcl, nrm_pcl, np in zip(
                            normals, normals_from_pcl, normals_pcl, num_pcl_points
                        ):
                            self.assertClose(nrm[:np], nrm_pcl[:np], atol=1e-5)
                            self.assertClose(nrm[:np], nrm_from_pcl[:np], atol=1e-5)

                    # check that local coord frames give the same normal
                    # as normals
                    for nrm, lcoord, np in zip(
                        normals, local_coord_frames, num_pcl_points
                    ):
                        self.assertClose(nrm[:np], lcoord[:np, :, 0], atol=1e-5)

                    # dotp between normals and normals_gt
                    normal_parallel = (normals_gt * normals).sum(2)

                    # check that normals are on average
                    # parallel to the expected ones
                    for normp, np in zip(normal_parallel, num_pcl_points):
                        abs_parallel = normp[:np].abs()
                        avg_parallel = abs_parallel.mean()
                        std_parallel = abs_parallel.std()
                        self.assertClose(
                            avg_parallel, torch.ones_like(avg_parallel), atol=1e-2
                        )
                        self.assertClose(
                            std_parallel, torch.zeros_like(std_parallel), atol=1e-2
                        )

                    if disambiguate_directions:
                        # check that 95% of normal dot products
                        # have the same sign
                        for normp, np in zip(normal_parallel, num_pcl_points):
                            n_pos = (normp[:np] > 0).sum()
                            self.assertTrue((n_pos > np * 0.95) or (n_pos < np * 0.05))

                    if DEBUG and run_idx == 0 and not use_pointclouds:
                        import os

                        from pytorch3d.io.ply_io import save_ply

                        # export to .ply
                        outdir = "/tmp/pt3d_pcl_normals_test/"
                        os.makedirs(outdir, exist_ok=True)
                        plyfile = os.path.join(
                            outdir, f"pcl_disamb={disambiguate_directions}.ply"
                        )
                        print(f"Storing point cloud with normals to {plyfile}.")
                        pcl_idx = 0
                        save_ply(
                            plyfile,
                            pcl[pcl_idx].cpu(),
                            faces=None,
                            verts_normals=normals[pcl_idx].cpu(),
                        )
