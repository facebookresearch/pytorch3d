# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from pytorch3d.ops.ball_query import ball_query


def ball_query_square(
    N: int, P1: int, P2: int, D: int, K: int, radius: float, device: str
):
    device = torch.device(device)
    pts1 = torch.rand(N, P1, D, device=device)
    pts2 = torch.rand(N, P2, D, device=device)
    torch.cuda.synchronize()

    def output():
        ball_query(pts1, pts2, K=K, radius=radius, skip_points_outside_cube=True)
        torch.cuda.synchronize()

    return output


def bm_ball_query() -> None:
    backends = ["cpu", "cuda:0"]

    kwargs_list = []
    Ns = [32]
    P1s = [256]
    P2s = [2**p for p in range(9, 20, 2)]
    Ds = [3, 10]
    Ks = [500]
    Rs = [0.01, 0.1]
    test_cases = product(Ns, P1s, P2s, Ds, Ks, Rs, backends)
    for case in test_cases:
        N, P1, P2, D, K, R, b = case
        kwargs_list.append(
            {"N": N, "P1": P1, "P2": P2, "D": D, "K": K, "radius": R, "device": b}
        )
    benchmark(
        ball_query_square,
        "BALLQUERY_SQUARE",
        kwargs_list,
        num_iters=30,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_ball_query()
