# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product

from fvcore.common.benchmark import benchmark
from test_mesh_edge_loss import TestMeshEdgeLoss


def bm_mesh_edge_loss() -> None:
    kwargs_list = []
    num_meshes = [1, 16, 32]
    max_v = [100, 10000]
    max_f = [300, 30000]
    test_cases = product(num_meshes, max_v, max_f)
    for case in test_cases:
        n, v, f = case
        kwargs_list.append({"num_meshes": n, "max_v": v, "max_f": f})
    benchmark(
        TestMeshEdgeLoss.mesh_edge_loss, "MESH_EDGE_LOSS", kwargs_list, warmup_iters=1
    )
