# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.renderer.implicit import HarmonicEmbedding
from torch.distributions import MultivariateNormal

from .common_testing import TestCaseMixin


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
        n_harmonic_functions = 2
        x = torch.randn((1, 5))
        D = 5 * n_harmonic_functions * 2  # sin + cos

        embed_fun = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=False
        )
        embed_out = embed_fun(x)

        self.assertEqual(embed_out.shape, (1, D))
        # Sum the squares of the respective frequencies
        # cos^2(x) + sin^2(x) = 1
        sum_squares = embed_out[0, : D // 2] ** 2 + embed_out[0, D // 2 :] ** 2
        self.assertClose(sum_squares, torch.ones((D // 2)))

        # Test append input
        embed_fun = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=True
        )
        embed_out_appended_input = embed_fun(x)
        self.assertClose(
            embed_out_appended_input.shape, torch.tensor((1, D + x.shape[-1]))
        )
        # Last plane in output is the input
        self.assertClose(embed_out_appended_input[..., -x.shape[-1] :], x)
        self.assertClose(embed_out_appended_input[..., : -x.shape[-1]], embed_out)

    def test_correct_embed_out_with_diag_cov(self):
        n_harmonic_functions = 2
        x = torch.randn((1, 3))
        diag_cov = torch.randn((1, 3))
        D = 3 * n_harmonic_functions * 2  # sin + cos

        embed_fun = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=False
        )
        embed_out = embed_fun(x, diag_cov=diag_cov)

        self.assertEqual(embed_out.shape, (1, D))

        # Compute the scaling factor introduce in MipNerf
        scale_factor = (
            -0.5 * diag_cov[..., None] * torch.pow(embed_fun._frequencies[None, :], 2)
        )
        scale_factor = torch.exp(scale_factor).reshape(1, -1).tile((1, 2))
        # If we remove this scaling factor, we should go back to the
        # classical harmonic embedding:
        # Sum the squares of the respective frequencies
        # cos^2(x) + sin^2(x) = 1
        embed_out_without_cov = embed_out / scale_factor
        sum_squares = (
            embed_out_without_cov[0, : D // 2] ** 2
            + embed_out_without_cov[0, D // 2 :] ** 2
        )
        self.assertClose(sum_squares, torch.ones((D // 2)))

        # Test append input
        embed_fun = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, append_input=True
        )
        embed_out_appended_input = embed_fun(x, diag_cov=diag_cov)
        self.assertClose(
            embed_out_appended_input.shape, torch.tensor((1, D + x.shape[-1]))
        )
        # Last plane in output is the input
        self.assertClose(embed_out_appended_input[..., -x.shape[-1] :], x)
        self.assertClose(embed_out_appended_input[..., : -x.shape[-1]], embed_out)

    def test_correct_behavior_between_ipe_and_its_estimation_from_harmonic_embedding(
        self,
    ):
        """
        Check that the HarmonicEmbedding with integrated_position_encoding (IPE) set to
        True is coherent with the HarmonicEmbedding.

        What is the idea behind this test?

        We wish to produce an IPE that is the expectation
        of our lifted multivariate gaussian, modulated by the sine and cosine of
        the coordinates. These expectation has a closed-form
        (see equations 11, 12, 13, 14 of [1]).

        We sample N elements from the multivariate gaussian defined by its mean and covariance
        and compute the HarmonicEmbedding. The expected value of those embeddings should be
        equal to our IPE.

        Inspired from:
        https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/mip_test.py#L359

        References:
            [1] `MIP-NeRF <https://arxiv.org/abs/2103.13415>`_.
        """
        num_dims = 3
        n_harmonic_functions = 6
        mean = torch.randn(num_dims)
        diag_cov = torch.rand(num_dims)

        he_fun = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions, logspace=True, append_input=False
        )
        ipe_fun = HarmonicEmbedding(
            n_harmonic_functions=n_harmonic_functions,
            append_input=False,
        )

        embedding_ipe = ipe_fun(mean, diag_cov=diag_cov)

        rand_mvn = MultivariateNormal(mean, torch.eye(num_dims) * diag_cov)

        # Providing a large enough number of samples
        # we should obtain an estimation close to our IPE
        num_samples = 100000
        embedding_he = he_fun(rand_mvn.sample_n(num_samples))
        self.assertClose(embedding_he.mean(0), embedding_ipe, rtol=1e-2, atol=1e-2)
