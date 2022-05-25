# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from pytorch3d.common.workaround import symeig3x3
from pytorch3d.transforms.rotation_conversions import random_rotations

from .common_testing import get_random_cuda_device, TestCaseMixin


class TestSymEig3x3(TestCaseMixin, unittest.TestCase):
    TEST_BATCH_SIZE = 1024

    @staticmethod
    def create_random_sym3x3(device, n):
        random_3x3 = torch.randn((n, 3, 3), device=device)
        random_3x3_T = torch.transpose(random_3x3, 1, 2)
        random_sym_3x3 = (random_3x3 * random_3x3_T).contiguous()

        return random_sym_3x3

    @staticmethod
    def create_diag_sym3x3(device, n, noise=0.0):
        # Create purly diagonal matrices
        random_diag_3x3 = torch.randn((n, 3), device=device).diag_embed()

        # Make them 'almost' diagonal
        random_diag_3x3 += noise * TestSymEig3x3.create_random_sym3x3(device, n)

        return random_diag_3x3

    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(42)

        self._gpu = get_random_cuda_device()
        self._cpu = torch.device("cpu")

    def test_is_eigen_gpu(self):
        test_input = self.create_random_sym3x3(self._gpu, n=self.TEST_BATCH_SIZE)

        self._test_is_eigen(test_input)

    def test_is_eigen_cpu(self):
        test_input = self.create_random_sym3x3(self._cpu, n=self.TEST_BATCH_SIZE)

        self._test_is_eigen(test_input)

    def _test_is_eigen(self, test_input, atol=1e-04, rtol=1e-02):
        """
        Verify that values and vectors produced are really eigenvalues and eigenvectors
        and can restore the original input matrix with good precision
        """
        eigenvalues, eigenvectors = symeig3x3(test_input, eigenvectors=True)

        self.assertClose(
            test_input,
            eigenvectors @ eigenvalues.diag_embed() @ eigenvectors.transpose(-2, -1),
            atol=atol,
            rtol=rtol,
        )

    def test_eigenvectors_are_orthonormal_gpu(self):
        test_input = self.create_random_sym3x3(self._gpu, n=self.TEST_BATCH_SIZE)

        self._test_eigenvectors_are_orthonormal(test_input)

    def test_eigenvectors_are_orthonormal_cpu(self):
        test_input = self.create_random_sym3x3(self._cpu, n=self.TEST_BATCH_SIZE)

        self._test_eigenvectors_are_orthonormal(test_input)

    def _test_eigenvectors_are_orthonormal(self, test_input):
        """
        Verify that eigenvectors are an orthonormal set
        """
        eigenvalues, eigenvectors = symeig3x3(test_input, eigenvectors=True)

        batched_eye = torch.zeros_like(test_input)
        batched_eye[..., :, :] = torch.eye(3, device=batched_eye.device)

        self.assertClose(
            batched_eye, eigenvectors @ eigenvectors.transpose(-2, -1), atol=1e-06
        )

    def test_is_not_nan_or_inf_gpu(self):
        test_input = self.create_random_sym3x3(self._gpu, n=self.TEST_BATCH_SIZE)

        self._test_is_not_nan_or_inf(test_input)

    def test_is_not_nan_or_inf_cpu(self):
        test_input = self.create_random_sym3x3(self._cpu, n=self.TEST_BATCH_SIZE)

        self._test_is_not_nan_or_inf(test_input)

    def _test_is_not_nan_or_inf(self, test_input):
        eigenvalues, eigenvectors = symeig3x3(test_input, eigenvectors=True)

        self.assertTrue(torch.isfinite(eigenvalues).all())
        self.assertTrue(torch.isfinite(eigenvectors).all())

    def test_degenerate_inputs_gpu(self):
        self._test_degenerate_inputs(self._gpu)

    def test_degenerate_inputs_cpu(self):
        self._test_degenerate_inputs(self._cpu)

    def _test_degenerate_inputs(self, device):
        """
        Test degenerate case when input matrices are diagonal or near-diagonal
        """

        # Purely diagonal case
        test_input = self.create_diag_sym3x3(device, self.TEST_BATCH_SIZE)

        self._test_is_not_nan_or_inf(test_input)
        self._test_is_eigen(test_input)
        self._test_eigenvectors_are_orthonormal(test_input)

        # Almost-diagonal case
        test_input = self.create_diag_sym3x3(device, self.TEST_BATCH_SIZE, noise=1e-4)

        self._test_is_not_nan_or_inf(test_input)
        self._test_is_eigen(test_input)
        self._test_eigenvectors_are_orthonormal(test_input)

    def test_gradients_cpu(self):
        self._test_gradients(self._cpu)

    def test_gradients_gpu(self):
        self._test_gradients(self._gpu)

    def _test_gradients(self, device):
        """
        Tests if gradients pass though without any problems (infs, nans etc) and
        also performs gradcheck (compares numerical and analytical gradients)
        """
        test_random_input = self.create_random_sym3x3(device, n=16)
        test_diag_input = self.create_diag_sym3x3(device, n=16)
        test_almost_diag_input = self.create_diag_sym3x3(device, n=16, noise=1e-4)

        test_input = torch.cat(
            (test_random_input, test_diag_input, test_almost_diag_input)
        )
        test_input.requires_grad = True

        with torch.autograd.detect_anomaly():
            eigenvalues, eigenvectors = symeig3x3(test_input, eigenvectors=True)

            loss = eigenvalues.mean() + eigenvectors.mean()
            loss.backward()

        test_random_input.requires_grad = True
        # Inputs are converted to double to increase the precision of gradcheck.
        torch.autograd.gradcheck(
            symeig3x3, test_random_input.double(), eps=1e-6, atol=1e-2, rtol=1e-2
        )

    def _test_eigenvalues_and_eigenvectors(
        self, test_eigenvectors, test_eigenvalues, atol=1e-04, rtol=1e-04
    ):
        test_input = (
            test_eigenvectors.transpose(-2, -1)
            @ test_eigenvalues.diag_embed()
            @ test_eigenvectors
        )

        test_eigenvalues_sorted, _ = torch.sort(test_eigenvalues, dim=-1)

        eigenvalues, eigenvectors = symeig3x3(test_input, eigenvectors=True)

        self.assertClose(
            test_eigenvalues_sorted,
            eigenvalues,
            atol=atol,
            rtol=rtol,
        )

        self._test_is_not_nan_or_inf(test_input)
        self._test_is_eigen(test_input, atol=atol, rtol=rtol)
        self._test_eigenvectors_are_orthonormal(test_input)

    def test_degenerate_eigenvalues_gpu(self):
        self._test_degenerate_eigenvalues(self._gpu)

    def test_degenerate_eigenvalues_cpu(self):
        self._test_degenerate_eigenvalues(self._cpu)

    def _test_degenerate_eigenvalues(self, device):
        """
        Test degenerate eigenvalues like zero-valued and with 2-/3-multiplicity
        """
        # Error tolerances for degenerate values are increased as things might become
        #  numerically unstable
        deg_atol = 1e-3
        deg_rtol = 1.0

        # Construct random orthonormal sets
        test_eigenvecs = random_rotations(n=self.TEST_BATCH_SIZE, device=device)

        # Construct random eigenvalues
        test_eigenvals = torch.randn(
            (self.TEST_BATCH_SIZE, 3), device=test_eigenvecs.device
        )
        self._test_eigenvalues_and_eigenvectors(
            test_eigenvecs, test_eigenvals, atol=deg_atol, rtol=deg_rtol
        )

        # First eigenvalue is always 0.0 here: [0.0 X Y]
        test_eigenvals_with_zero = test_eigenvals.clone()
        test_eigenvals_with_zero[..., 0] = 0.0
        self._test_eigenvalues_and_eigenvectors(
            test_eigenvecs, test_eigenvals_with_zero, atol=deg_atol, rtol=deg_rtol
        )

        # First two eigenvalues are always the same here: [X X Y]
        test_eigenvals_with_multiplicity2 = test_eigenvals.clone()
        test_eigenvals_with_multiplicity2[..., 1] = test_eigenvals_with_multiplicity2[
            ..., 0
        ]
        self._test_eigenvalues_and_eigenvectors(
            test_eigenvecs,
            test_eigenvals_with_multiplicity2,
            atol=deg_atol,
            rtol=deg_rtol,
        )

        # All three eigenvalues are the same here: [X X X]
        test_eigenvals_with_multiplicity3 = test_eigenvals_with_multiplicity2.clone()
        test_eigenvals_with_multiplicity3[..., 2] = test_eigenvals_with_multiplicity2[
            ..., 0
        ]
        self._test_eigenvalues_and_eigenvectors(
            test_eigenvecs,
            test_eigenvals_with_multiplicity3,
            atol=deg_atol,
            rtol=deg_rtol,
        )

    def test_more_dimensions(self):
        """
        Tests if function supports arbitrary leading dimensions
        """
        repeat = 4

        test_input = self.create_random_sym3x3(self._cpu, n=16)
        test_input_4d = test_input[None, ...].expand((repeat,) + test_input.shape)

        eigenvalues, eigenvectors = symeig3x3(test_input, eigenvectors=True)
        eigenvalues_4d, eigenvectors_4d = symeig3x3(test_input_4d, eigenvectors=True)

        eigenvalues_4d_gt = eigenvalues[None, ...].expand((repeat,) + eigenvalues.shape)
        eigenvectors_4d_gt = eigenvectors[None, ...].expand(
            (repeat,) + eigenvectors.shape
        )

        self.assertClose(eigenvalues_4d_gt, eigenvalues_4d)
        self.assertClose(eigenvectors_4d_gt, eigenvectors_4d)
