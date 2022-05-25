# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_ball_query import TestBallQuery


def bm_ball_query() -> None:

    backends = ["cpu", "cuda:0"]

    kwargs_list = []
    Ns = [32]
    P1s = [256]
    P2s = [128, 512]
    Ds = [3, 10]
    Ks = [3, 24, 100]
    Rs = [0.1, 0.2, 5]
    test_cases = product(Ns, P1s, P2s, Ds, Ks, Rs, backends)
    for case in test_cases:
        N, P1, P2, D, K, R, b = case
        kwargs_list.append(
            {"N": N, "P1": P1, "P2": P2, "D": D, "K": K, "radius": R, "device": b}
        )

    benchmark(
        TestBallQuery.ball_query_square, "BALLQUERY_SQUARE", kwargs_list, warmup_iters=1
    )
    benchmark(
        TestBallQuery.ball_query_ragged, "BALLQUERY_RAGGED", kwargs_list, warmup_iters=1
    )


if __name__ == "__main__":
    bm_ball_query()
