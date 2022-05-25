# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_point_mesh_distance import TestPointMeshDistance


def bm_point_mesh_distance() -> None:

    backend = ["cuda:0"]

    kwargs_list = []
    batch_size = [4, 8, 16]
    num_verts = [100, 1000]
    num_faces = [300, 3000]
    num_points = [5000, 10000]
    test_cases = product(batch_size, num_verts, num_faces, num_points, backend)
    for case in test_cases:
        n, v, f, p, b = case
        kwargs_list.append({"N": n, "V": v, "F": f, "P": p, "device": b})

    benchmark(
        TestPointMeshDistance.point_mesh_edge,
        "POINT_MESH_EDGE",
        kwargs_list,
        warmup_iters=1,
    )

    benchmark(
        TestPointMeshDistance.point_mesh_face,
        "POINT_MESH_FACE",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_point_mesh_distance()
