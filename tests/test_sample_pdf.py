# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from common_testing import TestCaseMixin
from pytorch3d.renderer.implicit.sample_pdf import sample_pdf_python


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

    @staticmethod
    def bm_fn(*, backend: str, n_samples, batch_size, n_bins):
        f = sample_pdf_python
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
