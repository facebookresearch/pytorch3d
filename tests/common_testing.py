# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import unittest
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import torch
from PIL import Image


def load_rgb_image(filename: str, data_dir: Union[str, Path]):
    filepath = data_dir / filename
    with Image.open(filepath) as raw_image:
        image = torch.from_numpy(np.array(raw_image) / 255.0)
    image = image.to(dtype=torch.float32)
    return image[..., :3]


TensorOrArray = Union[torch.Tensor, np.ndarray]


def get_random_cuda_device() -> str:
    """
    Function to get a random GPU device from the
    available devices. This is useful for testing
    that custom cuda kernels can support inputs on
    any device without having to set the device explicitly.
    """
    num_devices = torch.cuda.device_count()
    device_id = (
        torch.randint(high=num_devices, size=(1,)).item() if num_devices > 1 else 0
    )
    return "cuda:%d" % device_id


class TestCaseMixin(unittest.TestCase):
    def assertSeparate(self, tensor1, tensor2) -> None:
        """
        Verify that tensor1 and tensor2 have their data in distinct locations.
        """
        self.assertNotEqual(tensor1.storage().data_ptr(), tensor2.storage().data_ptr())

    def assertNotSeparate(self, tensor1, tensor2) -> None:
        """
        Verify that tensor1 and tensor2 have their data in the same locations.
        """
        self.assertEqual(tensor1.storage().data_ptr(), tensor2.storage().data_ptr())

    def assertAllSeparate(self, tensor_list) -> None:
        """
        Verify that all tensors in tensor_list have their data in
        distinct locations.
        """
        ptrs = [i.storage().data_ptr() for i in tensor_list]
        self.assertCountEqual(ptrs, set(ptrs))

    def assertNormsClose(
        self,
        input: TensorOrArray,
        other: TensorOrArray,
        norm_fn: Callable[[TensorOrArray], TensorOrArray],
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        msg: Optional[str] = None,
    ) -> None:
        """
        Verifies that two tensors or arrays have the same shape and are close
            given absolute and relative tolerance; raises AssertionError otherwise.
            A custom norm function is computed before comparison. If no such pre-
            processing needed, pass `torch.abs` or, equivalently, call `assertClose`.
        Args:
            input, other: two tensors or two arrays.
            norm_fn: The function evaluates
                `all(norm_fn(input - other) <= atol + rtol * norm_fn(other))`.
                norm_fn is a tensor -> tensor function; the output has:
                    * all entries non-negative,
                    * shape defined by the input shape only.
            rtol, atol, equal_nan: as for torch.allclose.
            msg: message in case the assertion is violated.
        Note:
            Optional arguments here are all keyword-only, to avoid confusion
            with msg arguments on other assert functions.
        """

        self.assertEqual(np.shape(input), np.shape(other))

        diff = norm_fn(input - other)
        other_ = norm_fn(other)

        # We want to generalise allclose(input, output), which is essentially
        #  all(diff <= atol + rtol * other)
        # but with a sophisticated handling non-finite values.
        # We work that around by calling allclose() with the following arguments:
        # allclose(diff + other_, other_). This computes what we want because
        #  all(|diff + other_ - other_| <= atol + rtol * |other_|) ==
        #    all(|norm_fn(input - other)| <= atol + rtol * |norm_fn(other)|) ==
        #    all(norm_fn(input - other) <= atol + rtol * norm_fn(other)).

        backend = torch if torch.is_tensor(input) else np
        close = backend.allclose(
            diff + other_, other_, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

        self.assertTrue(close, msg)

    def assertClose(
        self,
        input: TensorOrArray,
        other: TensorOrArray,
        *,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
        msg: Optional[str] = None,
    ) -> None:
        """
        Verifies that two tensors or arrays have the same shape and are close
            given absolute and relative tolerance, i.e. checks
            `all(|input - other| <= atol + rtol * |other|)`;
            raises AssertionError otherwise.
        Args:
            input, other: two tensors or two arrays.
            rtol, atol, equal_nan: as for torch.allclose.
            msg: message in case the assertion is violated.
        Note:
            Optional arguments here are all keyword-only, to avoid confusion
            with msg arguments on other assert functions.
        """

        self.assertEqual(np.shape(input), np.shape(other))

        backend = torch if torch.is_tensor(input) else np
        close = backend.allclose(
            input, other, rtol=rtol, atol=atol, equal_nan=equal_nan
        )

        if not close and msg is None:
            diff = backend.abs(input - other) + 0.0
            ratio = diff / backend.abs(other)
            try_relative = (diff <= atol) | (backend.isfinite(ratio) & (ratio > 0))
            if try_relative.all():
                if backend == np:
                    # Avoid a weirdness with zero dimensional arrays.
                    ratio = np.array(ratio)
                ratio[diff <= atol] = 0
                extra = f" Max relative diff {ratio.max()}"
            else:
                extra = ""
            shape = tuple(input.shape)
            max_diff = diff.max()
            self.fail(f"Not close. Max diff {max_diff}.{extra} Shape {shape}.")

        self.assertTrue(close, msg)
