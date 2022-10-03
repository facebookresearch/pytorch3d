# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from pytorch3d.implicitron.models.generic_model import GenericModel
from pytorch3d.implicitron.models.implicit_function.scene_representation_networks import (
    SRNHyperNetImplicitFunction,
    SRNImplicitFunction,
    SRNPixelGenerator,
)
from pytorch3d.implicitron.models.renderer.ray_sampler import ImplicitronRayBundle
from pytorch3d.implicitron.tools.config import get_default_args
from pytorch3d.renderer import PerspectiveCameras

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

    def _get_bundle(self, *, device) -> ImplicitronRayBundle:
        origins = torch.rand(_BATCH_SIZE, _N_RAYS, 3, device=device)
        directions = torch.rand(_BATCH_SIZE, _N_RAYS, 3, device=device)
        lengths = torch.rand(_BATCH_SIZE, _N_RAYS, _N_POINTS_ON_RAY, device=device)
        bundle = ImplicitronRayBundle(
            lengths=lengths,
            origins=origins,
            directions=directions,
            xys=None,
            camera_ids=None,
            camera_counts=None,
        )
        return bundle

    def test_srn_implicit_function(self):
        implicit_function = SRNImplicitFunction()
        device = torch.device("cpu")
        bundle = self._get_bundle(device=device)
        rays_densities, rays_colors = implicit_function(ray_bundle=bundle)
        out_features = implicit_function.raymarch_function.out_features
        self.assertEqual(
            rays_densities.shape,
            (_BATCH_SIZE, _N_RAYS, _N_POINTS_ON_RAY, out_features),
        )
        self.assertIsNone(rays_colors)

    def test_srn_hypernet_implicit_function(self):
        # TODO investigate: If latent_dim_hypernet=0, why does this crash and dump core?
        latent_dim_hypernet = 39
        device = torch.device("cuda:0")
        implicit_function = SRNHyperNetImplicitFunction(
            latent_dim_hypernet=latent_dim_hypernet
        )
        implicit_function.to(device)
        global_code = torch.rand(_BATCH_SIZE, latent_dim_hypernet, device=device)
        bundle = self._get_bundle(device=device)
        rays_densities, rays_colors = implicit_function(
            ray_bundle=bundle, global_code=global_code
        )
        out_features = implicit_function.hypernet.out_features
        self.assertEqual(
            rays_densities.shape,
            (_BATCH_SIZE, _N_RAYS, _N_POINTS_ON_RAY, out_features),
        )
        self.assertIsNone(rays_colors)

    @torch.no_grad()
    def test_lstm(self):
        args = get_default_args(GenericModel)
        args.render_image_height = 80
        args.render_image_width = 80
        args.implicit_function_class_type = "SRNImplicitFunction"
        args.renderer_class_type = "LSTMRenderer"
        args.raysampler_class_type = "NearFarRaySampler"
        args.raysampler_NearFarRaySampler_args.n_pts_per_ray_training = 1
        args.raysampler_NearFarRaySampler_args.n_pts_per_ray_evaluation = 1
        args.renderer_LSTMRenderer_args.bg_color = [0.4, 0.4, 0.2]
        gm = GenericModel(**args)

        camera = PerspectiveCameras()
        image = gm.forward(
            camera=camera,
            image_rgb=None,
            fg_probability=None,
            sequence_name="",
            mask_crop=None,
            depth_map=None,
        )["images_render"]
        self.assertEqual(image.shape, (1, 3, 80, 80))
        self.assertGreater(image.max(), 0.8)

        # Force everything to be background
        pixel_generator = gm._implicit_functions[0]._fn.pixel_generator
        pixel_generator._density_layer.weight.zero_()
        pixel_generator._density_layer.bias.fill_(-1.0e6)

        image = gm.forward(
            camera=camera,
            image_rgb=None,
            fg_probability=None,
            sequence_name="",
            mask_crop=None,
            depth_map=None,
        )["images_render"]
        self.assertConstant(image[:, :2], 0.4)
        self.assertConstant(image[:, 2], 0.2)
