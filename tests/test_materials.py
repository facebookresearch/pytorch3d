# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest

import torch
from common_testing import TestCaseMixin
from pytorch3d.renderer.materials import Materials


class TestMaterials(TestCaseMixin, unittest.TestCase):
    def test_init(self):
        """
        Initialize Materials class with the default values.
        """
        device = torch.device("cuda:0")
        mat = Materials(device=device)
        self.assertTrue(torch.is_tensor(mat.ambient_color))
        self.assertTrue(torch.is_tensor(mat.diffuse_color))
        self.assertTrue(torch.is_tensor(mat.specular_color))
        self.assertTrue(torch.is_tensor(mat.shininess))
        self.assertTrue(mat.ambient_color.device == device)
        self.assertTrue(mat.diffuse_color.device == device)
        self.assertTrue(mat.specular_color.device == device)
        self.assertTrue(mat.shininess.device == device)
        self.assertTrue(mat.ambient_color.shape == (1, 3))
        self.assertTrue(mat.diffuse_color.shape == (1, 3))
        self.assertTrue(mat.specular_color.shape == (1, 3))
        self.assertTrue(mat.shininess.shape == (1,))

    def test_materials_clone_to(self):
        device = torch.device("cuda:0")
        cpu = torch.device("cpu")
        mat = Materials()
        new_mat = mat.clone().to(device)
        self.assertTrue(mat.ambient_color.device == cpu)
        self.assertTrue(mat.diffuse_color.device == cpu)
        self.assertTrue(mat.specular_color.device == cpu)
        self.assertTrue(mat.shininess.device == cpu)
        self.assertTrue(new_mat.ambient_color.device == device)
        self.assertTrue(new_mat.diffuse_color.device == device)
        self.assertTrue(new_mat.specular_color.device == device)
        self.assertTrue(new_mat.shininess.device == device)
        self.assertSeparate(new_mat.ambient_color, mat.ambient_color)
        self.assertSeparate(new_mat.diffuse_color, mat.diffuse_color)
        self.assertSeparate(new_mat.specular_color, mat.specular_color)
        self.assertSeparate(new_mat.shininess, mat.shininess)

    def test_initialize_materials_broadcast(self):
        materials = Materials(
            ambient_color=torch.randn(10, 3),
            diffuse_color=torch.randn(1, 3),
            specular_color=torch.randn(1, 3),
            shininess=torch.randn(1),
        )
        self.assertTrue(materials.ambient_color.shape == (10, 3))
        self.assertTrue(materials.diffuse_color.shape == (10, 3))
        self.assertTrue(materials.specular_color.shape == (10, 3))
        self.assertTrue(materials.shininess.shape == (10,))

    def test_initialize_materials_broadcast_fail(self):
        """
        Batch dims have to be the same or 1.
        """
        with self.assertRaises(ValueError):
            Materials(
                ambient_color=torch.randn(10, 3), diffuse_color=torch.randn(15, 3)
            )

    def test_initialize_materials_dimensions_fail(self):
        """
        Color should have shape (N, 3) or (1, 3), Shininess should have shape
        (1), (1, 1), (N) or (N, 1)
        """
        with self.assertRaises(ValueError):
            Materials(ambient_color=torch.randn(10, 4))

        with self.assertRaises(ValueError):
            Materials(shininess=torch.randn(10, 2))

    def test_initialize_materials_mixed_inputs(self):
        mat = Materials(ambient_color=torch.randn(1, 3), diffuse_color=((1, 1, 1),))
        self.assertTrue(mat.ambient_color.shape == (1, 3))
        self.assertTrue(mat.diffuse_color.shape == (1, 3))

    def test_initialize_materials_mixed_inputs_broadcast(self):
        mat = Materials(ambient_color=torch.randn(10, 3), diffuse_color=((1, 1, 1),))
        self.assertTrue(mat.ambient_color.shape == (10, 3))
        self.assertTrue(mat.diffuse_color.shape == (10, 3))
        self.assertTrue(mat.specular_color.shape == (10, 3))
        self.assertTrue(mat.shininess.shape == (10,))
