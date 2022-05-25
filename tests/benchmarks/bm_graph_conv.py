# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from tests.test_graph_conv import TestGraphConv


def bm_graph_conv() -> None:
    backends = ["cpu"]
    if torch.cuda.is_available():
        backends.append("cuda")

    kwargs_list = []
    gconv_dim = [128, 256]
    num_meshes = [32, 64]
    num_verts = [100]
    num_faces = [1000]
    directed = [False, True]
    test_cases = product(
        gconv_dim, num_meshes, num_verts, num_faces, directed, backends
    )
    for case in test_cases:
        g, n, v, f, d, b = case
        kwargs_list.append(
            {
                "gconv_dim": g,
                "num_meshes": n,
                "num_verts": v,
                "num_faces": f,
                "directed": d,
                "backend": b,
            }
        )
    benchmark(
        TestGraphConv.graph_conv_forward_backward,
        "GRAPH CONV",
        kwargs_list,
        warmup_iters=1,
    )


if __name__ == "__main__":
    bm_graph_conv()
