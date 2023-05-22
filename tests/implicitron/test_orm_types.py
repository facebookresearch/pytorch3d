# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from pytorch3d.implicitron.dataset.orm_types import ArrayTypeFactory, TupleTypeFactory


class TestOrmTypes(unittest.TestCase):
    def test_tuple_serialization_none(self):
        ttype = TupleTypeFactory()()
        output = ttype.process_bind_param(None, None)
        self.assertIsNone(output)
        output = ttype.process_result_value(output, None)
        self.assertIsNone(output)

    def test_tuple_serialization_1d(self):
        for input_tuple in [(1, 2, 3), (4.5, 6.7)]:
            ttype = TupleTypeFactory(type(input_tuple[0]), (len(input_tuple),))()
            output = ttype.process_bind_param(input_tuple, None)
            input_hat = ttype.process_result_value(output, None)
            self.assertEqual(type(input_hat[0]), type(input_tuple[0]))
            np.testing.assert_almost_equal(input_hat, input_tuple, decimal=6)

    def test_tuple_serialization_2d(self):
        input_tuple = ((1.0, 2.0, 3.0), (4.5, 5.5, 6.6))
        ttype = TupleTypeFactory(type(input_tuple[0][0]), (2, 3))()
        output = ttype.process_bind_param(input_tuple, None)
        input_hat = ttype.process_result_value(output, None)
        self.assertEqual(type(input_hat[0][0]), type(input_tuple[0][0]))
        # we use float32 to serialise
        np.testing.assert_almost_equal(input_hat, input_tuple, decimal=6)

    def test_array_serialization_none(self):
        ttype = ArrayTypeFactory((3, 3))()
        output = ttype.process_bind_param(None, None)
        self.assertIsNone(output)
        output = ttype.process_result_value(output, None)
        self.assertIsNone(output)

    def test_array_serialization(self):
        for input_list in [[1, 2, 3], [[4.5, 6.7], [8.9, 10.0]]]:
            input_array = np.array(input_list)

            # first, dynamic-size array
            ttype = ArrayTypeFactory()()
            output = ttype.process_bind_param(input_array, None)
            input_hat = ttype.process_result_value(output, None)
            self.assertEqual(input_hat.dtype, np.float32)
            np.testing.assert_almost_equal(input_hat, input_array, decimal=6)

            # second, fixed-size array
            ttype = ArrayTypeFactory(tuple(input_array.shape))()
            output = ttype.process_bind_param(input_array, None)
            input_hat = ttype.process_result_value(output, None)
            self.assertEqual(input_hat.dtype, np.float32)
            np.testing.assert_almost_equal(input_hat, input_array, decimal=6)
