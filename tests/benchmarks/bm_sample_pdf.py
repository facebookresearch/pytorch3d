# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from itertools import product

from fvcore.common.benchmark import benchmark
from tests.test_sample_pdf import TestSamplePDF


def bm_sample_pdf() -> None:
    backends = ["python_cuda", "cuda", "python_cpu", "cpu"]

    kwargs_list = []
    sample_counts = [64]
    batch_sizes = [1024, 10240]
    bin_counts = [62, 600]
    test_cases = product(backends, sample_counts, batch_sizes, bin_counts)
    for case in test_cases:
        backend, n_samples, batch_size, n_bins = case
        kwargs_list.append(
            {
                "backend": backend,
                "n_samples": n_samples,
                "batch_size": batch_size,
                "n_bins": n_bins,
            }
        )

    benchmark(TestSamplePDF.bm_fn, "SAMPLE_PDF", kwargs_list, warmup_iters=1)


if __name__ == "__main__":
    bm_sample_pdf()
