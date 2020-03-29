# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from itertools import product

from fvcore.common.benchmark import benchmark
from test_subdivide_meshes import TestSubdivideMeshes


def bm_subdivide() -> None:
    kwargs_list = []
    num_meshes = [1, 16, 32]
    same_topo = [True, False]
    test_cases = product(num_meshes, same_topo)
    for case in test_cases:
        n, s = case
        kwargs_list.append({"num_meshes": n, "same_topo": s})
    benchmark(
        TestSubdivideMeshes.subdivide_meshes_with_init,
        "SUBDIVIDE",
        kwargs_list,
        warmup_iters=1,
    )
