# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_knn import TestKNN


def bm_knn() -> None:
    backends = ["cpu", "cuda:0"]

    kwargs_list = []
    Ns = [32]
    P1s = [256]
    P2s = [128, 512]
    Ds = [3]
    Ks = [24]
    test_cases = product(Ns, P1s, P2s, Ds, Ks, backends)
    for case in test_cases:
        N, P1, P2, D, K, b = case
        kwargs_list.append({"N": N, "P1": P1, "P2": P2, "D": D, "K": K, "device": b})

    benchmark(TestKNN.knn_square, "KNN_SQUARE", kwargs_list, warmup_iters=1)

    benchmark(TestKNN.knn_ragged, "KNN_RAGGED", kwargs_list, warmup_iters=1)


if __name__ == "__main__":
    bm_knn()
