# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_iou_box3d import TestIoU3D


def bm_iou_box3d() -> None:
    # Realistic use cases
    N = [30, 100]
    M = [5, 10, 100]
    kwargs_list = []
    test_cases = product(N, M)
    for case in test_cases:
        n, m = case
        kwargs_list.append({"N": n, "M": m, "device": "cuda:0"})
    benchmark(TestIoU3D.iou, "3D_IOU", kwargs_list, warmup_iters=1)

    # Comparison of C++/CUDA
    kwargs_list = []
    N = [1, 4, 8, 16]
    devices = ["cpu", "cuda:0"]
    test_cases = product(N, N, devices)
    for case in test_cases:
        n, m, d = case
        kwargs_list.append({"N": n, "M": m, "device": d})
    benchmark(TestIoU3D.iou, "3D_IOU", kwargs_list, warmup_iters=1)

    # Naive PyTorch
    N = [1, 4]
    kwargs_list = []
    test_cases = product(N, N)
    for case in test_cases:
        n, m = case
        kwargs_list.append({"N": n, "M": m, "device": "cuda:0"})
    benchmark(TestIoU3D.iou_naive, "3D_IOU_NAIVE", kwargs_list, warmup_iters=1)

    # Sampling based method
    num_samples = [2000, 5000]
    kwargs_list = []
    test_cases = product(N, N, num_samples)
    for case in test_cases:
        n, m, s = case
        kwargs_list.append({"N": n, "M": m, "num_samples": s, "device": "cuda:0"})
    benchmark(TestIoU3D.iou_sampling, "3D_IOU_SAMPLING", kwargs_list, warmup_iters=1)


if __name__ == "__main__":
    bm_iou_box3d()
