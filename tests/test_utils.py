#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch

from pytorch3d.renderer.utils import TensorProperties

from common_testing import TestCaseMixin


# Example class for testing
class TensorPropertiesTestClass(TensorProperties):
    def __init__(self, x=None, y=None, device="cpu"):
        super().__init__(device=device, x=x, y=y)

    def clone(self):
        other = TensorPropertiesTestClass()
        return super().clone(other)


class TestTensorProperties(TestCaseMixin, unittest.TestCase):
    def test_init(self):
        example = TensorPropertiesTestClass(x=10.0, y=(100.0, 200.0))
        # Check kwargs set as attributes + converted to tensors
        self.assertTrue(torch.is_tensor(example.x))
        self.assertTrue(torch.is_tensor(example.y))
        # Check broadcasting
        self.assertTrue(example.x.shape == (2,))
        self.assertTrue(example.y.shape == (2,))
        self.assertTrue(len(example) == 2)

    def test_to(self):
        # Check to method
        example = TensorPropertiesTestClass(x=10.0, y=(100.0, 200.0))
        device = torch.device("cuda:0")
        new_example = example.to(device=device)
        self.assertTrue(new_example.device == device)

    def test_clone(self):
        # Check clone method
        example = TensorPropertiesTestClass(x=10.0, y=(100.0, 200.0))
        new_example = example.clone()
        self.assertSeparate(example.x, new_example.x)
        self.assertSeparate(example.y, new_example.y)

    def test_get_set(self):
        # Test getitem returns an accessor which can be used to modify
        # attributes at a particular index
        example = TensorPropertiesTestClass(x=10.0, y=(100.0, 200.0, 300.0))

        # update y1
        example[1].y = 5.0
        self.assertTrue(example.y[1] == 5.0)

        # Get item and get value
        ex0 = example[0]
        self.assertTrue(ex0.y == 100.0)

    def test_empty_input(self):
        example = TensorPropertiesTestClass(x=(), y=())
        self.assertTrue(len(example) == 0)
        self.assertTrue(example.isempty())
