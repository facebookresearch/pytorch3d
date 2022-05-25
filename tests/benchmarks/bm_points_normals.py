# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
from fvcore.common.benchmark import benchmark
from pytorch3d.ops import estimate_pointcloud_normals
from tests.test_points_normals import TestPCLNormals


def to_bm(num_points, use_symeig_workaround):
    device = torch.device("cuda:0")
    points_padded, _normals = TestPCLNormals.init_spherical_pcl(
        num_points=num_points, device=device, use_pointclouds=False
    )
    torch.cuda.synchronize()

    def run():
        estimate_pointcloud_normals(
            points_padded, use_symeig_workaround=use_symeig_workaround
        )
        torch.cuda.synchronize()

    return run


def bm_points_normals() -> None:
    case_grid = {
        "use_symeig_workaround": [True, False],
        "num_points": [3000, 6000],
    }
    test_cases = itertools.product(*case_grid.values())
    kwargs_list = [dict(zip(case_grid.keys(), case)) for case in test_cases]
    benchmark(
        to_bm,
        "normals",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_points_normals()
