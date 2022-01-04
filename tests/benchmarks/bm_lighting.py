# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from pytorch3d.renderer.lighting import diffuse, specular


def _bm_diffuse_cuda_with_init(N, S, K):
    device = torch.device("cuda")
    normals = torch.randn(N, S, S, K, 3, device=device)
    color = torch.randn(1, 3, device=device)
    direction = torch.randn(N, S, S, K, 3, device=device)
    args = (normals, color, direction)
    torch.cuda.synchronize()
    return lambda: diffuse(*args)


def _bm_specular_cuda_with_init(N, S, K):
    device = torch.device("cuda")
    points = torch.randn(N, S, S, K, 3, device=device)
    normals = torch.randn(N, S, S, K, 3, device=device)
    direction = torch.randn(N, S, S, K, 3, device=device)
    color = torch.randn(1, 3, device=device)
    camera_position = torch.randn(N, 3, device=device)
    shininess = torch.randn(N, device=device)
    args = (points, normals, direction, color, camera_position, shininess)
    torch.cuda.synchronize()
    return lambda: specular(*args)


def bm_lighting() -> None:
    # For now only benchmark lighting on GPU
    if not torch.cuda.is_available():
        return

    kwargs_list = []
    Ns = [1, 8]
    Ss = [128, 256]
    Ks = [1, 10, 80]
    test_cases = product(Ns, Ss, Ks)
    for case in test_cases:
        N, S, K = case
        kwargs_list.append({"N": N, "S": S, "K": K})
    benchmark(_bm_diffuse_cuda_with_init, "DIFFUSE", kwargs_list, warmup_iters=3)
    benchmark(_bm_specular_cuda_with_init, "SPECULAR", kwargs_list, warmup_iters=3)


if __name__ == "__main__":
    bm_lighting()
