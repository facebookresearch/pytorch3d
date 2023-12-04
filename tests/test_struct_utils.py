# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from pytorch3d.structures import utils as struct_utils

from .common_testing import TestCaseMixin


class TestStructUtils(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(43)

    def _check_list_to_padded_slices(self, x, x_padded, ndim):
        N = len(x)
        for i in range(N):
            slices = [i]
            for dim in range(ndim):
                if x[i].nelement() == 0 and x[i].ndim == 1:
                    slice_ = slice(0, 0, 1)
                else:
                    slice_ = slice(0, x[i].shape[dim], 1)
                slices.append(slice_)
            if x[i].nelement() == 0 and x[i].ndim == 1:
                x_correct = x[i].new_zeros(*[[0] * ndim])
            else:
                x_correct = x[i]
            self.assertClose(x_padded[slices], x_correct)

    def test_list_to_padded(self):
        device = torch.device("cuda:0")
        N = 5
        K = 20
        for ndim in [1, 2, 3, 4]:
            x = []
            for _ in range(N):
                dims = torch.randint(K, size=(ndim,)).tolist()
                x.append(torch.rand(dims, device=device))

            # set 0th element to an empty 1D tensor
            x[0] = torch.tensor([], dtype=x[0].dtype, device=device)

            # set 1st element to an empty tensor with correct number of dims
            x[1] = x[1].new_zeros(*[[0] * ndim])

            pad_size = [K] * ndim
            x_padded = struct_utils.list_to_padded(
                x, pad_size=pad_size, pad_value=0.0, equisized=False
            )

            for dim in range(ndim):
                self.assertEqual(x_padded.shape[dim + 1], K)

            self._check_list_to_padded_slices(x, x_padded, ndim)

            # check for no pad size (defaults to max dimension)
            x_padded = struct_utils.list_to_padded(x, pad_value=0.0, equisized=False)
            max_sizes = (
                max(
                    (0 if (y.nelement() == 0 and y.ndim == 1) else y.shape[dim])
                    for y in x
                )
                for dim in range(ndim)
            )
            for dim, max_size in enumerate(max_sizes):
                self.assertEqual(x_padded.shape[dim + 1], max_size)

            self._check_list_to_padded_slices(x, x_padded, ndim)

            # check for equisized
            x = [torch.rand((K, *([10] * (ndim - 1))), device=device) for _ in range(N)]
            x_padded = struct_utils.list_to_padded(x, equisized=True)
            self.assertClose(x_padded, torch.stack(x, 0))

        # catch ValueError for invalid dimensions
        pad_size = [K] * (ndim + 1)
        with self.assertRaisesRegex(ValueError, "Pad size must"):
            struct_utils.list_to_padded(
                x, pad_size=pad_size, pad_value=0.0, equisized=False
            )

        # invalid input tensor dimensions
        x = []
        ndim = 3
        for _ in range(N):
            dims = torch.randint(K, size=(ndim,)).tolist()
            x.append(torch.rand(dims, device=device))
        pad_size = [K] * 2
        with self.assertRaisesRegex(ValueError, "Pad size must"):
            x_padded = struct_utils.list_to_padded(
                x, pad_size=pad_size, pad_value=0.0, equisized=False
            )

    def test_padded_to_list(self):
        device = torch.device("cuda:0")
        N = 5
        K = 20
        ndim = 2

        for ndim in (2, 3, 4):

            dims = [K] * ndim
            x = torch.rand([N] + dims, device=device)

            x_list = struct_utils.padded_to_list(x)
            for i in range(N):
                self.assertClose(x_list[i], x[i])

            split_size = torch.randint(1, K, size=(N, ndim)).unbind(0)
            x_list = struct_utils.padded_to_list(x, split_size)
            for i in range(N):
                slices = [i]
                for dim in range(ndim):
                    slices.append(slice(0, split_size[i][dim], 1))
                self.assertClose(x_list[i], x[slices])

            # split size is a list of ints
            split_size = [int(z) for z in torch.randint(1, K, size=(N,)).unbind(0)]
            x_list = struct_utils.padded_to_list(x, split_size)
            for i in range(N):
                self.assertClose(x_list[i], x[i][: split_size[i]])

    def test_padded_to_packed(self):
        device = torch.device("cuda:0")
        N = 5
        K = 20
        ndim = 2
        dims = [K] * ndim
        x = torch.rand([N] + dims, device=device)

        # Case 1: no split_size or pad_value provided
        # Check output is just the flattened input.
        x_packed = struct_utils.padded_to_packed(x)
        self.assertTrue(x_packed.shape == (x.shape[0] * x.shape[1], x.shape[2]))
        self.assertClose(x_packed, x.reshape(-1, K))

        # Case 2: pad_value is provided.
        # Check each section of the packed tensor matches the
        # corresponding unpadded elements of the padded tensor.
        # Check that only rows where all the values are padded
        # are removed in the conversion to packed.
        pad_value = -1
        x_list = []
        split_size = []
        for _ in range(N):
            dim = torch.randint(K, size=(1,)).item()
            # Add some random values in the input which are the same as the pad_value.
            # These should not be filtered out.
            x_list.append(
                torch.randint(low=pad_value, high=10, size=(dim, K), device=device)
            )
            split_size.append(dim)
        x_padded = struct_utils.list_to_padded(x_list, pad_value=pad_value)
        x_packed = struct_utils.padded_to_packed(x_padded, pad_value=pad_value)
        curr = 0
        for i in range(N):
            self.assertClose(x_packed[curr : curr + split_size[i], ...], x_list[i])
            self.assertClose(torch.cat(x_list), x_packed)
            curr += split_size[i]

        # Case 3: split_size is provided.
        # Check each section of the packed tensor matches the corresponding
        # unpadded elements.
        x_packed = struct_utils.padded_to_packed(x_padded, split_size=split_size)
        curr = 0
        for i in range(N):
            self.assertClose(x_packed[curr : curr + split_size[i], ...], x_list[i])
            self.assertClose(torch.cat(x_list), x_packed)
            curr += split_size[i]

        # Case 4: split_size of the wrong shape is provided.
        # Raise an error.
        split_size = torch.randint(1, K, size=(2 * N,)).view(N, 2).unbind(0)
        with self.assertRaisesRegex(ValueError, "1-dimensional"):
            x_packed = struct_utils.padded_to_packed(x_padded, split_size=split_size)

        split_size = torch.randint(1, K, size=(2 * N,)).view(N * 2).tolist()
        with self.assertRaisesRegex(
            ValueError, "same length as inputs first dimension"
        ):
            x_packed = struct_utils.padded_to_packed(x_padded, split_size=split_size)

        # Case 5: both pad_value and split_size are provided.
        # Raise an error.
        with self.assertRaisesRegex(ValueError, "Only one of"):
            x_packed = struct_utils.padded_to_packed(
                x_padded, split_size=split_size, pad_value=-1
            )

        # Case 6: Input has more than 3 dims.
        # Raise an error.
        x = torch.rand((N, K, K, K, K), device=device)
        split_size = torch.randint(1, K, size=(N,)).tolist()
        with self.assertRaisesRegex(ValueError, "Supports only"):
            struct_utils.padded_to_packed(x, split_size=split_size)

    def test_list_to_packed(self):
        device = torch.device("cuda:0")
        N = 5
        K = 20
        x, x_dims = [], []
        dim2 = torch.randint(K, size=(1,)).item()
        for _ in range(N):
            dim1 = torch.randint(K, size=(1,)).item()
            x_dims.append(dim1)
            x.append(torch.rand([dim1, dim2], device=device))

        out = struct_utils.list_to_packed(x)
        x_packed = out[0]
        num_items = out[1]
        item_packed_first_idx = out[2]
        item_packed_to_list_idx = out[3]

        cur = 0
        for i in range(N):
            self.assertTrue(num_items[i] == x_dims[i])
            self.assertTrue(item_packed_first_idx[i] == cur)
            self.assertTrue(item_packed_to_list_idx[cur : cur + x_dims[i]].eq(i).all())
            self.assertClose(x_packed[cur : cur + x_dims[i]], x[i])
            cur += x_dims[i]
