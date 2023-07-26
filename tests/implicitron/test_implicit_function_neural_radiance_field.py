# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.implicitron.models.implicit_function.base import ImplicitronRayBundle
from pytorch3d.implicitron.models.implicit_function.neural_radiance_field import (
    NeuralRadianceFieldImplicitFunction,
)


class TestNeuralRadianceFieldImplicitFunction(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_forward_with_integrated_positionial_embedding(self):
        shape = [2, 4, 4]
        ray_bundle = ImplicitronRayBundle(
            origins=torch.randn(*shape, 3),
            directions=torch.randn(*shape, 3),
            bins=torch.randn(*shape, 6 + 1),
            lengths=torch.randn(*shape, 6),
            pixel_radii_2d=torch.randn(*shape, 1),
            xys=None,
        )
        model = NeuralRadianceFieldImplicitFunction(
            n_hidden_neurons_dir=32, use_integrated_positional_encoding=True
        )
        raw_densities, ray_colors, _ = model(ray_bundle=ray_bundle)

        self.assertEqual(raw_densities.shape, (*shape, ray_bundle.lengths.shape[-1], 1))
        self.assertEqual(ray_colors.shape, (*shape, ray_bundle.lengths.shape[-1], 3))

    def test_forward_with_integrated_positionial_embedding_raise_exception(self):
        shape = [2, 4, 4]
        ray_bundle = ImplicitronRayBundle(
            origins=torch.randn(*shape, 3),
            directions=torch.randn(*shape, 3),
            bins=None,
            lengths=torch.randn(*shape, 6),
            pixel_radii_2d=torch.randn(*shape, 1),
            xys=None,
        )
        model = NeuralRadianceFieldImplicitFunction(
            n_hidden_neurons_dir=32, use_integrated_positional_encoding=True
        )
        with self.assertRaises(ValueError):
            _ = model(ray_bundle=ray_bundle)

    def test_forward(self):
        shape = [2, 4, 4]
        ray_bundle = ImplicitronRayBundle(
            origins=torch.randn(*shape, 3),
            directions=torch.randn(*shape, 3),
            lengths=torch.randn(*shape, 6),
            pixel_radii_2d=torch.randn(*shape, 1),
            xys=None,
        )
        model = NeuralRadianceFieldImplicitFunction(n_hidden_neurons_dir=32)
        raw_densities, ray_colors, _ = model(ray_bundle=ray_bundle)
        self.assertEqual(raw_densities.shape, (*shape, 6, 1))
        self.assertEqual(ray_colors.shape, (*shape, 6, 3))
