# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_io_obj import TestMeshObjIO
from tests.test_io_ply import TestMeshPlyIO


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

    # Texture loading benchmarks
    kwargs_list = [{"R": 2}, {"R": 4}, {"R": 10}, {"R": 15}, {"R": 20}]
    benchmark(
        TestMeshObjIO.bm_load_texture_atlas,
        "PYTORCH3D_TEXTURE_ATLAS",
        kwargs_list,
        warmup_iters=1,
    )

    kwargs_list = []
    S = [64, 256, 1024]
    F = [100, 1000, 10000]
    R = [5, 10, 20]
    test_cases = product(S, F, R)

    for case in test_cases:
        s, f, r = case
        kwargs_list.append({"S": s, "F": f, "R": r})

    benchmark(
        TestMeshObjIO.bm_bilinear_sampling_vectorized,
        "BILINEAR_VECTORIZED",
        kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshObjIO.bm_bilinear_sampling_grid_sample,
        "BILINEAR_GRID_SAMPLE",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_save_load()
