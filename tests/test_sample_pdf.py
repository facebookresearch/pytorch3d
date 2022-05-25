# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import torch
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf, sample_pdf_python

from .common_testing import TestCaseMixin


class TestSamplePDF(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_single_bin(self):
        bins = torch.arange(2).expand(5, 2) + 17
        weights = torch.ones(5, 1)
        output = sample_pdf_python(bins, weights, 100, True)
        calc = torch.linspace(17, 18, 100).expand(5, -1)
        self.assertClose(output, calc)

    def test_simple_det(self):
        for n_bins, n_samples, batch in product(
            [7, 20], [2, 7, 31, 32, 33], [(), (1, 4), (31,), (32,), (33,)]
        ):
            weights = torch.rand(size=(batch + (n_bins,)))
            bins = torch.cumsum(torch.rand(size=(batch + (n_bins + 1,))), dim=-1)
            python = sample_pdf_python(bins, weights, n_samples, det=True)

            cpp = sample_pdf(bins, weights, n_samples, det=True)
            self.assertClose(cpp, python, atol=2e-3)

            nthreads = torch.get_num_threads()
            torch.set_num_threads(1)
            cpp_singlethread = sample_pdf(bins, weights, n_samples, det=True)
            self.assertClose(cpp_singlethread, python, atol=2e-3)
            torch.set_num_threads(nthreads)

            device = torch.device("cuda:0")
            cuda = sample_pdf(
                bins.to(device), weights.to(device), n_samples, det=True
            ).cpu()

            self.assertClose(cuda, python, atol=2e-3)

    def test_rand_cpu(self):
        n_bins, n_samples, batch_size = 11, 17, 9
        weights = torch.rand(size=(batch_size, n_bins))
        bins = torch.cumsum(torch.rand(size=(batch_size, n_bins + 1)), dim=-1)
        torch.manual_seed(1)
        python = sample_pdf_python(bins, weights, n_samples)
        torch.manual_seed(1)
        cpp = sample_pdf(bins, weights, n_samples)

        self.assertClose(cpp, python, atol=2e-3)

    def test_rand_nogap(self):
        # Case where random is actually deterministic
        weights = torch.FloatTensor([0, 10, 0])
        bins = torch.FloatTensor([0, 10, 10, 25])
        n_samples = 8
        predicted = torch.full((n_samples,), 10.0)
        python = sample_pdf_python(bins, weights, n_samples)
        self.assertClose(python, predicted)
        cpp = sample_pdf(bins, weights, n_samples)
        self.assertClose(cpp, predicted)

        device = torch.device("cuda:0")
        cuda = sample_pdf(bins.to(device), weights.to(device), n_samples).cpu()
        self.assertClose(cuda, predicted)

    @staticmethod
    def bm_fn(*, backend: str, n_samples, batch_size, n_bins):
        f = sample_pdf_python if "python" in backend else sample_pdf
        weights = torch.rand(size=(batch_size, n_bins))
        bins = torch.cumsum(torch.rand(size=(batch_size, n_bins + 1)), dim=-1)

        if "cuda" in backend:
            weights = weights.cuda()
            bins = bins.cuda()

        torch.cuda.synchronize()

        def output():
            f(bins, weights, n_samples)
            torch.cuda.synchronize()

        return output
