# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from itertools import product
from typing import Any, Callable

import torch
from common_testing import get_random_cuda_device
from fvcore.common.benchmark import benchmark
from pytorch3d.common.workaround import symeig3x3
from tests.test_symeig3x3 import TestSymEig3x3


torch.set_num_threads(1)

CUDA_DEVICE = get_random_cuda_device()


def create_traced_func(func, device, batch_size):
    traced_func = torch.jit.trace(
        func, (TestSymEig3x3.create_random_sym3x3(device, batch_size),)
    )

    return traced_func


FUNC_NAME_TO_FUNC = {
    "sym3x3eig": (lambda inputs: symeig3x3(inputs, eigenvectors=True)),
    "sym3x3eig_traced_cuda": create_traced_func(
        (lambda inputs: symeig3x3(inputs, eigenvectors=True)), CUDA_DEVICE, 1024
    ),
    "torch_symeig": (lambda inputs: torch.symeig(inputs, eigenvectors=True)),
    "torch_linalg_eigh": (lambda inputs: torch.linalg.eigh(inputs)),
    "torch_pca_lowrank": (
        lambda inputs: torch.pca_lowrank(inputs, center=False, niter=1)
    ),
    "sym3x3eig_no_vecs": (lambda inputs: symeig3x3(inputs, eigenvectors=False)),
    "torch_symeig_no_vecs": (lambda inputs: torch.symeig(inputs, eigenvectors=False)),
    "torch_linalg_eigvalsh_no_vecs": (lambda inputs: torch.linalg.eigvalsh(inputs)),
}


def test_symeig3x3(func_name, batch_size, device) -> Callable[[], Any]:
    func = FUNC_NAME_TO_FUNC[func_name]
    inputs = TestSymEig3x3.create_random_sym3x3(device, batch_size)
    torch.cuda.synchronize()

    def symeig3x3():
        func(inputs)
        torch.cuda.synchronize()

    return symeig3x3


def bm_symeig3x3() -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append(CUDA_DEVICE)

    kwargs_list = []
    func_names = FUNC_NAME_TO_FUNC.keys()
    batch_sizes = [16, 128, 1024, 8192, 65536, 1048576]

    for func_name, batch_size, device in product(func_names, batch_sizes, devices):
        # Run CUDA-only implementations only on GPU
        if "cuda" in func_name and not device.startswith("cuda"):
            continue

        # Torch built-ins are quite slow on larger batches
        if "torch" in func_name and batch_size > 8192:
            continue

        # Avoid running CPU implementations on larger batches as well
        if device == "cpu" and batch_size > 8192:
            continue

        kwargs_list.append(
            {"func_name": func_name, "batch_size": batch_size, "device": device}
        )

    benchmark(
        test_symeig3x3,
        "SYMEIG3X3",
        kwargs_list,
        warmup_iters=3,
    )


if __name__ == "__main__":
    bm_symeig3x3()
