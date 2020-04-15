# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from itertools import product

from fvcore.common.benchmark import benchmark
from test_knn import TestKNN


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
