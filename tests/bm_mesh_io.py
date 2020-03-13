# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from fvcore.common.benchmark import benchmark

from test_obj_io import TestMeshObjIO
from test_ply_io import TestMeshPlyIO


def bm_save_load() -> None:
    simple_kwargs_list = [
        {"V": 100, "F": 200},
        {"V": 1000, "F": 2000},
        {"V": 10000, "F": 20000},
    ]
    benchmark(
        TestMeshObjIO.bm_load_simple_obj_with_init,
        "LOAD_SIMPLE_OBJ",
        simple_kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshObjIO.bm_save_simple_obj_with_init,
        "SAVE_SIMPLE_OBJ",
        simple_kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshPlyIO.bm_load_simple_ply_with_init,
        "LOAD_SIMPLE_PLY",
        simple_kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshPlyIO.bm_save_simple_ply_with_init,
        "SAVE_SIMPLE_PLY",
        simple_kwargs_list,
        warmup_iters=1,
    )

    complex_kwargs_list = [{"N": 8}, {"N": 32}, {"N": 128}]
    benchmark(
        TestMeshObjIO.bm_load_complex_obj,
        "LOAD_COMPLEX_OBJ",
        complex_kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshObjIO.bm_save_complex_obj,
        "SAVE_COMPLEX_OBJ",
        complex_kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshPlyIO.bm_load_complex_ply,
        "LOAD_COMPLEX_PLY",
        complex_kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshPlyIO.bm_save_complex_ply,
        "SAVE_COMPLEX_PLY",
        complex_kwargs_list,
        warmup_iters=1,
    )
