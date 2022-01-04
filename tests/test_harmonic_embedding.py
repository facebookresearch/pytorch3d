# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from common_testing import TestCaseMixin
from pytorch3d.renderer.implicit import HarmonicEmbedding


class TestHarmonicEmbedding(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    def test_correct_output_dim(self):
        embed_fun = HarmonicEmbedding(n_harmonic_functions=2, append_input=False)
        # input_dims * (2 * n_harmonic_functions + int(append_input))
        output_dim = 3 * (2 * 2 + int(False))
        self.assertEqual(
            output_dim,
            embed_fun.get_output_dim_static(
                input_dims=3, n_harmonic_functions=2, append_input=False
            ),
        )
        self.assertEqual(output_dim, embed_fun.get_output_dim())

    def test_correct_frequency_range(self):
        embed_fun_log = HarmonicEmbedding(n_harmonic_functions=3)
        embed_fun_lin = HarmonicEmbedding(n_harmonic_functions=3, logspace=False)
        self.assertClose(embed_fun_log._frequencies, torch.FloatTensor((1.0, 2.0, 4.0)))
        self.assertClose(embed_fun_lin._frequencies, torch.FloatTensor((1.0, 2.5, 4.0)))

    def test_correct_embed_out(self):
        embed_fun = HarmonicEmbedding(n_harmonic_functions=2, append_input=False)
        x = torch.randn((1, 5))
        D = 5 * 4
        embed_out = embed_fun(x)
        self.assertEqual(embed_out.shape, (1, D))
        # Sum the squares of the respective frequencies
        sum_squares = embed_out[0, : D // 2] ** 2 + embed_out[0, D // 2 :] ** 2
        self.assertClose(sum_squares, torch.ones((D // 2)))
        embed_fun = HarmonicEmbedding(n_harmonic_functions=2, append_input=True)
        embed_out = embed_fun(x)
        self.assertClose(embed_out.shape, torch.tensor((1, 5 * 5)))
        # Last plane in output is the input
        self.assertClose(embed_out[..., -5:], x)
