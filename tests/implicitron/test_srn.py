# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import torch
from pytorch3d.implicitron.models.implicit_function.scene_representation_networks import (
    SRNHyperNetImplicitFunction,
    SRNImplicitFunction,
    SRNPixelGenerator,
)
from pytorch3d.implicitron.models.renderer.base import ImplicitFunctionWrapper
from pytorch3d.implicitron.tools.config import get_default_args
from pytorch3d.renderer import RayBundle


if os.environ.get("FB_TEST", False):
    from common_testing import TestCaseMixin
else:
    from tests.common_testing import TestCaseMixin

_BATCH_SIZE: int = 3
_N_RAYS: int = 100
_N_POINTS_ON_RAY: int = 10


class TestSRN(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        get_default_args(SRNHyperNetImplicitFunction)
        get_default_args(SRNImplicitFunction)

    def test_pixel_generator(self):
        SRNPixelGenerator()

    def _get_bundle(self, *, device) -> RayBundle:
        origins = torch.rand(_BATCH_SIZE, _N_RAYS, 3, device=device)
        directions = torch.rand(_BATCH_SIZE, _N_RAYS, 3, device=device)
        lengths = torch.rand(_BATCH_SIZE, _N_RAYS, _N_POINTS_ON_RAY, device=device)
        bundle = RayBundle(
            lengths=lengths, origins=origins, directions=directions, xys=None
        )
        return bundle

    def test_srn_implicit_function(self):
        implicit_function = SRNImplicitFunction()
        device = torch.device("cpu")
        bundle = self._get_bundle(device=device)
        rays_densities, rays_colors = implicit_function(bundle)
        out_features = implicit_function.raymarch_function.out_features
        self.assertEqual(
            rays_densities.shape,
            (_BATCH_SIZE, _N_RAYS, _N_POINTS_ON_RAY, out_features),
        )
        self.assertIsNone(rays_colors)

    def test_srn_hypernet_implicit_function(self):
        # TODO investigate: If latent_dim_hypernet=0, why does this crash and dump core?
        latent_dim_hypernet = 39
        hypernet_args = {"latent_dim_hypernet": latent_dim_hypernet}
        device = torch.device("cuda:0")
        implicit_function = SRNHyperNetImplicitFunction(hypernet_args=hypernet_args)
        implicit_function.to(device)
        global_code = torch.rand(_BATCH_SIZE, latent_dim_hypernet, device=device)
        bundle = self._get_bundle(device=device)
        rays_densities, rays_colors = implicit_function(bundle, global_code=global_code)
        out_features = implicit_function.hypernet.out_features
        self.assertEqual(
            rays_densities.shape,
            (_BATCH_SIZE, _N_RAYS, _N_POINTS_ON_RAY, out_features),
        )
        self.assertIsNone(rays_colors)

    def test_srn_hypernet_implicit_function_optim(self):
        # Test optimization loop, requiring that the cache is properly
        # cleared in new_args_bound
        latent_dim_hypernet = 39
        hyper_args = {"latent_dim_hypernet": latent_dim_hypernet}
        device = torch.device("cuda:0")
        global_code = torch.rand(_BATCH_SIZE, latent_dim_hypernet, device=device)
        bundle = self._get_bundle(device=device)

        implicit_function = SRNHyperNetImplicitFunction(hypernet_args=hyper_args)
        implicit_function2 = SRNHyperNetImplicitFunction(hypernet_args=hyper_args)
        implicit_function.to(device)
        implicit_function2.to(device)

        wrapper = ImplicitFunctionWrapper(implicit_function)
        optimizer = torch.optim.Adam(implicit_function.parameters())
        for _step in range(3):
            optimizer.zero_grad()
            wrapper.bind_args(global_code=global_code)
            rays_densities, _rays_colors = wrapper(bundle)
            wrapper.unbind_args()
            loss = rays_densities.sum()
            loss.backward()
            optimizer.step()

        wrapper2 = ImplicitFunctionWrapper(implicit_function)
        optimizer2 = torch.optim.Adam(implicit_function2.parameters())
        implicit_function2.load_state_dict(implicit_function.state_dict())
        optimizer2.load_state_dict(optimizer.state_dict())
        for _step in range(3):
            optimizer2.zero_grad()
            wrapper2.bind_args(global_code=global_code)
            rays_densities, _rays_colors = wrapper2(bundle)
            wrapper2.unbind_args()
            loss = rays_densities.sum()
            loss.backward()
            optimizer2.step()
