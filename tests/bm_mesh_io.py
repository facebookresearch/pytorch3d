# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from fvcore.common.benchmark import benchmark

from test_obj_io import TestMeshObjIO
from test_ply_io import TestMeshPlyIO


def bm_save_load() -> None:
    kwargs_list = [
        {"V": 100, "F": 300},
        {"V": 1000, "F": 3000},
        {"V": 10000, "F": 30000},
    ]
    benchmark(
        TestMeshObjIO.load_obj_with_init,
        "LOAD_OBJ",
        kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshObjIO.save_obj_with_init,
        "SAVE_OBJ",
        kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshPlyIO.load_ply_bm, "LOAD_PLY", kwargs_list, warmup_iters=1
    )
    benchmark(
        TestMeshPlyIO.save_ply_bm, "SAVE_PLY", kwargs_list, warmup_iters=1
    )
