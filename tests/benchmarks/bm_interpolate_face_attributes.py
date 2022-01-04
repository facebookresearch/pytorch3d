# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

import torch
from fvcore.common.benchmark import benchmark
from pytorch3d.ops.interp_face_attrs import (
    interpolate_face_attributes,
    interpolate_face_attributes_python,
)


def _generate_data(N, S, K, F, D, device, requires_grad=False):
    pix_to_face = torch.randint(-10, F, (N, S, S, K), device=device)
    barycentric_coords = torch.randn(
        N, S, S, K, 3, device=device, requires_grad=requires_grad
    )
    face_attrs = torch.randn(F, 3, D, device=device, requires_grad=requires_grad)
    grad_pix_attrs = torch.randn(N, S, S, K, D, device=device)
    return pix_to_face, barycentric_coords, face_attrs, grad_pix_attrs


def _bm_forward(N, S, F, K, D, impl):
    # The runtime depends on the values of pix_to_face. So for proper
    # benchmarking we should probably take the average of multiple
    # values of pix to face. But this doesn't easily fit into fvcore
    # benchmarking, so instead we'll just set a manual seed to make sure
    # that different impls will use the same data.
    torch.manual_seed(0)
    device = torch.device("cuda")
    data = _generate_data(N, S, K, F, D, device, requires_grad=False)
    args = data[:3]
    torch.cuda.synchronize()
    if impl == "cuda":
        fun = interpolate_face_attributes
    elif impl == "python":
        fun = interpolate_face_attributes_python
    return lambda: fun(*args)


def _bm_forward_backward(N, S, F, K, D, impl):
    torch.manual_seed(0)
    device = torch.device("cuda")
    data = _generate_data(N, S, K, F, D, device, requires_grad=True)
    args, grad = data[:3], data[3]
    torch.cuda.synchronize()
    if impl == "cuda":
        fun = interpolate_face_attributes
    elif impl == "python":
        fun = interpolate_face_attributes_python

    def run():
        out = fun(*args)
        out.backward(gradient=grad)

    return run


def bm_interpolate_face_attribues() -> None:
    # For now only benchmark on GPU
    if not torch.cuda.is_available():
        return

    Ns = [1, 4]
    Ss = [128]
    Ks = [1, 10, 40]
    Fs = [5000]
    Ds = [1, 3, 16]
    impls = ["python", "cuda"]
    test_cases = product(Ns, Ss, Ks, Fs, Ds, impls)
    kwargs_list = []
    for case in test_cases:
        N, S, K, F, D, impl = case
        kwargs_list.append({"N": N, "S": S, "K": K, "F": F, "D": D, "impl": impl})
    benchmark(_bm_forward, "FORWARD", kwargs_list, warmup_iters=3)
    benchmark(_bm_forward_backward, "FORWARD+BACKWARD", kwargs_list, warmup_iters=3)


if __name__ == "__main__":
    bm_interpolate_face_attribues()
