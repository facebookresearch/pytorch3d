# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from itertools import product

import torch
from fvcore.common.benchmark import benchmark

from pytorch3d import _C
from pytorch3d.ops.knn import _knn_points_idx_naive


def bm_knn() -> None:
    """ Entry point for the benchmark """
    benchmark_knn_cpu()
    benchmark_knn_cuda_vs_naive()
    benchmark_knn_cuda_versions()


def benchmark_knn_cuda_versions() -> None:
    # Compare our different KNN implementations,
    # and also compare against our existing 1-NN
    Ns = [1, 2]
    Ps = [4096, 16384]
    Ds = [3]
    Ks = [1, 4, 16, 64]
    versions = [0, 1, 2, 3]
    knn_kwargs, nn_kwargs = [], []
    for N, P, D, K, version in product(Ns, Ps, Ds, Ks, versions):
        if version == 2 and K > 32:
            continue
        if version == 3 and K > 4:
            continue
        knn_kwargs.append({'N': N, 'D': D, 'P': P, 'K': K, 'v': version})
    for N, P, D in product(Ns, Ps, Ds):
        nn_kwargs.append({'N': N, 'D': D, 'P': P})
    benchmark(
        knn_cuda_with_init,
        'KNN_CUDA_VERSIONS',
        knn_kwargs,
        warmup_iters=1,
    )
    benchmark(
        nn_cuda_with_init,
        'NN_CUDA',
        nn_kwargs,
        warmup_iters=1,
    )


def benchmark_knn_cuda_vs_naive() -> None:
    # Compare against naive pytorch version of KNN
    Ns = [1, 2, 4]
    Ps = [1024, 4096, 16384, 65536]
    Ds = [3]
    Ks = [1, 2, 4, 8, 16]
    knn_kwargs, naive_kwargs = [], []
    for N, P, D, K in product(Ns, Ps, Ds, Ks):
        knn_kwargs.append({'N': N, 'D': D, 'P': P, 'K': K})
        if P <= 4096:
            naive_kwargs.append({'N': N, 'D': D, 'P': P, 'K': K})
    benchmark(
        knn_python_cuda_with_init,
        'KNN_CUDA_PYTHON',
        naive_kwargs,
        warmup_iters=1,
    )
    benchmark(
        knn_cuda_with_init,
        'KNN_CUDA',
        knn_kwargs,
        warmup_iters=1,
    )


def benchmark_knn_cpu() -> None:
    Ns = [1, 2]
    Ps = [256, 512]
    Ds = [3]
    Ks = [1, 2, 4]
    knn_kwargs, nn_kwargs = [], []
    for N, P, D, K in product(Ns, Ps, Ds, Ks):
        knn_kwargs.append({'N': N, 'D': D, 'P': P, 'K': K})
    for N, P, D in product(Ns, Ps, Ds):
        nn_kwargs.append({'N': N, 'D': D, 'P': P})
    benchmark(
        knn_python_cpu_with_init,
        'KNN_CPU_PYTHON',
        knn_kwargs,
        warmup_iters=1,
    )
    benchmark(
        knn_cpu_with_init,
        'KNN_CPU_CPP',
        knn_kwargs,
        warmup_iters=1,
    )
    benchmark(
        nn_cpu_with_init,
        'NN_CPU_CPP',
        nn_kwargs,
        warmup_iters=1,
    )


def knn_cuda_with_init(N, D, P, K, v=-1):
    device = torch.device('cuda:0')
    x = torch.randn(N, P, D, device=device)
    y = torch.randn(N, P, D, device=device)
    torch.cuda.synchronize()

    def knn():
        _C.knn_points_idx(x, y, K, v)
        torch.cuda.synchronize()

    return knn


def knn_cpu_with_init(N, D, P, K):
    device = torch.device('cpu')
    x = torch.randn(N, P, D, device=device)
    y = torch.randn(N, P, D, device=device)

    def knn():
        _C.knn_points_idx(x, y, K, 0)

    return knn


def knn_python_cuda_with_init(N, D, P, K):
    device = torch.device('cuda')
    x = torch.randn(N, P, D, device=device)
    y = torch.randn(N, P, D, device=device)
    torch.cuda.synchronize()

    def knn():
        _knn_points_idx_naive(x, y, K)
        torch.cuda.synchronize()

    return knn


def knn_python_cpu_with_init(N, D, P, K):
    device = torch.device('cpu')
    x = torch.randn(N, P, D, device=device)
    y = torch.randn(N, P, D, device=device)

    def knn():
        _knn_points_idx_naive(x, y, K)

    return knn


def nn_cuda_with_init(N, D, P):
    device = torch.device('cuda')
    x = torch.randn(N, P, D, device=device)
    y = torch.randn(N, P, D, device=device)
    torch.cuda.synchronize()

    def knn():
        _C.nn_points_idx(x, y)
        torch.cuda.synchronize()

    return knn


def nn_cpu_with_init(N, D, P):
    device = torch.device('cpu')
    x = torch.randn(N, P, D, device=device)
    y = torch.randn(N, P, D, device=device)

    def knn():
        _C.nn_points_idx(x, y)

    return knn
