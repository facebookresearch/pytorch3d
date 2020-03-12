# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import numpy as np
import unittest
import torch


class TestCaseMixin(unittest.TestCase):
    def assertSeparate(self, tensor1, tensor2) -> None:
        """
        Verify that tensor1 and tensor2 have their data in distinct locations.
        """
        self.assertNotEqual(
            tensor1.storage().data_ptr(), tensor2.storage().data_ptr()
        )

    def assertNotSeparate(self, tensor1, tensor2) -> None:
        """
        Verify that tensor1 and tensor2 have their data in the same locations.
        """
        self.assertEqual(
            tensor1.storage().data_ptr(), tensor2.storage().data_ptr()
        )

    def assertAllSeparate(self, tensor_list) -> None:
        """
        Verify that all tensors in tensor_list have their data in
        distinct locations.
        """
        ptrs = [i.storage().data_ptr() for i in tensor_list]
        self.assertCountEqual(ptrs, set(ptrs))

    def assertClose(
        self,
        input,
        other,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False
    ) -> None:
        """
        Verify that two tensors or arrays are the same shape and close.
        Args:
            input, other: two tensors or two arrays.
            rtol, atol, equal_nan: as for torch.allclose.
        Note:
            Optional arguments here are all keyword-only, to avoid confusion
            with msg arguments on other assert functions.
        """

        self.assertEqual(np.shape(input), np.shape(other))

        if torch.is_tensor(input):
            close = torch.allclose(
                input, other, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        else:
            close = np.allclose(
                input, other, rtol=rtol, atol=atol, equal_nan=equal_nan
            )
        self.assertTrue(close)
