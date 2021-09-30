# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from fvcore.common.benchmark import benchmark
from test_iou_box3d import TestIoU3D


def bm_iou_box3d() -> None:
    N = [1, 4, 8, 16]
    num_samples = [2000, 5000, 10000, 20000]

    kwargs_list = []
    test_cases = product(N, N)
    for case in test_cases:
        n, m = case
        kwargs_list.append({"N": n, "M": m, "device": "cuda:0"})

    benchmark(TestIoU3D.iou_naive, "3D_IOU_NAIVE", kwargs_list, warmup_iters=1)

    [k.update({"device": "cpu"}) for k in kwargs_list]
    benchmark(TestIoU3D.iou, "3D_IOU", kwargs_list, warmup_iters=1)

    kwargs_list = []
    test_cases = product([1, 4], [1, 4], num_samples)
    for case in test_cases:
        n, m, s = case
        kwargs_list.append({"N": n, "M": m, "num_samples": s})
    benchmark(TestIoU3D.iou_sampling, "3D_IOU_SAMPLING", kwargs_list, warmup_iters=1)


if __name__ == "__main__":
    bm_iou_box3d()
