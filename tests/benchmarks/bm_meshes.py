# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from tests.test_meshes import TestMeshes


def bm_compute_packed_padded_meshes() -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    kwargs_list = []
    num_meshes = [32, 128]
    max_v = [100, 1000, 10000]
    max_f = [300, 3000, 30000]
    test_cases = product(num_meshes, max_v, max_f, devices)
    for case in test_cases:
        n, v, f, d = case
        kwargs_list.append({"num_meshes": n, "max_v": v, "max_f": f, "device": d})
    benchmark(
        TestMeshes.compute_packed_with_init,
        "COMPUTE_PACKED",
        kwargs_list,
        warmup_iters=1,
    )
    benchmark(
        TestMeshes.compute_padded_with_init,
        "COMPUTE_PADDED",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_compute_packed_padded_meshes()
