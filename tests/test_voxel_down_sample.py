# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import math
import unittest

import torch
from pytorch3d.ops.voxel_down_sample import voxel_down_sample
from pytorch3d.structures import Pointclouds


def _reference(points, voxel_size, values=None):
    """Independent, small CPU reference with Python integer tuple keys."""
    groups = {}
    for i, point in enumerate(points.tolist()):
        key = tuple(math.floor(x / voxel_size) for x in point)
        groups.setdefault(key, []).append(i)
    data = points if values is None else values
    if not groups:
        return data.clone()
    return torch.stack([data[groups[key]].mean(0) for key in sorted(groups)])


class TestVoxelDownSample(unittest.TestCase):
    def test_reference(self):
        generator = torch.Generator().manual_seed(12)
        for dtype, voxel_size in itertools.product(
            (torch.float32, torch.float64), (0.125, 0.75, 3.0)
        ):
            with self.subTest(dtype=dtype, voxel_size=voxel_size):
                points = [
                    torch.randn(n, 3, dtype=dtype, generator=generator)
                    for n in (67, 0, 31)
                ]
                features = [
                    torch.randn(n, 5, dtype=dtype, generator=generator)
                    for n in (67, 0, 31)
                ]
                normals = [
                    torch.randn(n, 3, dtype=dtype, generator=generator)
                    for n in (67, 0, 31)
                ]
                clouds = Pointclouds(points, normals=normals, features=features)
                output = voxel_down_sample(clouds, voxel_size)
                for i in range(3):
                    for actual, original in (
                        (output.points_list()[i], points[i]),
                        (output.features_list()[i], features[i]),
                        (output.normals_list()[i], normals[i]),
                    ):
                        torch.testing.assert_close(
                            actual, _reference(points[i], voxel_size, original)
                        )
                        self.assertEqual(actual.dtype, original.dtype)
                        self.assertEqual(actual.device, original.device)
                self.assertEqual(len(output), len(clouds))

    def test_negative_coordinates_and_boundaries(self):
        points = torch.tensor(
            [
                [-1.125, 0, 0],
                [-1, 0, 0],
                [-0.875, 0, 0],
                [-0.125, 0, 0],
                [0, 0, 0],
                [0.125, 0, 0],
                [1, 0, 0],
            ],
            dtype=torch.float64,
        )
        output = voxel_down_sample(Pointclouds([points]), 1.0)
        expected = points.new_tensor(
            [[-1.125, 0, 0], [-2.0 / 3, 0, 0], [0.0625, 0, 0], [1, 0, 0]]
        )
        torch.testing.assert_close(output.points_list()[0], expected)

    def test_batch_separation_and_attributes(self):
        points = torch.tensor([[0.25, 0, 0], [0.75, 0, 0]])
        features = [torch.ones(2, 4), torch.full((2, 4), 9.0)]
        normals = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0]])
        output = voxel_down_sample(
            Pointclouds([points, points], [normals, normals], features), 1.0
        )
        self.assertEqual(output.num_points_per_cloud().tolist(), [1, 1])
        torch.testing.assert_close(output.features_list()[0], torch.ones(1, 4))
        torch.testing.assert_close(output.features_list()[1], torch.full((1, 4), 9.0))
        torch.testing.assert_close(output.normals_packed(), torch.zeros(2, 3))

    def test_empty_and_optional_attributes(self):
        for clouds in (
            Pointclouds([]),
            Pointclouds([torch.empty(0, 3, dtype=torch.float64)]),
            Pointclouds([torch.empty(0, 3), torch.empty(0, 3)]),
        ):
            output = voxel_down_sample(clouds, 1.0)
            self.assertEqual(len(output), len(clouds))
            torch.testing.assert_close(
                output.num_points_per_cloud(), clouds.num_points_per_cloud()
            )
            for actual, expected in zip(output.points_list(), clouds.points_list()):
                torch.testing.assert_close(actual, expected)
            self.assertIsNone(output.features_list())
            self.assertIsNone(output.normals_list())
        output = voxel_down_sample(Pointclouds(torch.zeros(2, 3, 3)), 1.0)
        self.assertEqual(output.num_points_per_cloud().tolist(), [1, 1])
        self.assertIsNone(output.features_list())
        self.assertIsNone(output.normals_list())

    def test_empty_attribute_schema_and_mixed_dtypes(self):
        points = [torch.empty(0, 3, dtype=torch.float64)]
        features = [torch.empty(0, 4, dtype=torch.float32)]
        normals = [torch.empty(0, 3, dtype=torch.float64)]
        output = voxel_down_sample(Pointclouds(points, normals, features), 1.0)
        torch.testing.assert_close(output.features_list()[0], features[0])
        torch.testing.assert_close(output.normals_list()[0], normals[0])
        points = [torch.tensor([[0.25, 0, 0], [0.75, 0, 0]], dtype=torch.float64)]
        features = [torch.ones(2, 4, dtype=torch.float32)]
        output = voxel_down_sample(Pointclouds(points, features=features), 1.0)
        self.assertEqual(output.points_list()[0].dtype, torch.float64)
        self.assertEqual(output.features_list()[0].dtype, torch.float32)

    def test_heterogeneous_padding_is_excluded(self):
        clouds = Pointclouds([torch.full((1, 3), 2.25), torch.full((5, 3), 2.75)])
        # Materialize both representations. Padded zeros are not occupied voxels.
        self.assertEqual(clouds.points_padded().shape, (2, 5, 3))
        output = voxel_down_sample(clouds, 1.0)
        self.assertEqual(output.num_points_per_cloud().tolist(), [1, 1])

    def test_noncontiguous_and_input_unchanged(self):
        points = torch.arange(36, dtype=torch.float64).reshape(6, 6)[:, ::2]
        features = points[:, :2]
        self.assertFalse(points.is_contiguous())
        original = points.clone()
        output = voxel_down_sample(Pointclouds([points], features=[features]), 10.0)
        torch.testing.assert_close(output.points_list()[0], _reference(points, 10.0))
        torch.testing.assert_close(points, original)

    def test_order_and_grid_origin(self):
        points = torch.tensor([[1.75, 0, 0], [-0.25, 0, 0], [1.25, 0, 0]])
        expected = voxel_down_sample(Pointclouds([points]), 1.0).points_list()[0]
        permuted = voxel_down_sample(Pointclouds([points[[2, 0, 1]]]), 1.0)
        torch.testing.assert_close(permuted.points_list()[0], expected)
        translated = voxel_down_sample(Pointclouds([points + 4]), 1.0)
        torch.testing.assert_close(translated.points_list()[0], expected + 4)
        distant = torch.cat([points, torch.tensor([[1000.0, 0, 0]])])
        output = voxel_down_sample(Pointclouds([distant]), 1.0).points_list()[0]
        torch.testing.assert_close(output[:-1], expected)

    def test_sparse_large_extent_and_finite_mean(self):
        # A dense bounding grid would be impossible; only two voxels are occupied.
        points = torch.tensor([[-1e8, -1e8, -1e8], [1e8, 1e8, 1e8]])
        output = voxel_down_sample(Pointclouds([points]), 0.125)
        torch.testing.assert_close(output.points_list()[0], points)
        points = torch.full((8, 3), 1e38)
        output = voxel_down_sample(Pointclouds([points]), 1e38)
        torch.testing.assert_close(output.points_list()[0], points[:1])

    def test_extreme_means_and_gradients(self):
        for dtype, count in itertools.product(
            (torch.float32, torch.float64), (3, 10, 31)
        ):
            limits = torch.finfo(dtype)
            subnormal = torch.nextafter(
                torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype)
            ).item()
            for value in (limits.max, subnormal, -subnormal):
                with self.subTest(dtype=dtype, count=count, value=value):
                    # Also exercise attributes whose magnitudes differ from the
                    # positions: each voxel/channel needs its own scaling.
                    points = torch.full(
                        (count, 3), value, dtype=dtype, requires_grad=True
                    )
                    features = torch.full(
                        (count, 2), value, dtype=dtype, requires_grad=True
                    )
                    output = voxel_down_sample(
                        Pointclouds([points], features=[features]), max(abs(value), 1.0)
                    )
                    actual = output.points_list()[0]
                    torch.testing.assert_close(actual, points[:1], rtol=0, atol=0)
                    torch.testing.assert_close(
                        output.features_list()[0], features[:1], rtol=0, atol=0
                    )
                    for result, source in (
                        (actual, points),
                        (output.features_list()[0], features),
                    ):
                        gradient = torch.autograd.grad(
                            result, source, torch.ones_like(result)
                        )[0]
                        torch.testing.assert_close(
                            gradient, torch.full_like(source, 1 / count)
                        )

    def test_invalid_voxel_size(self):
        clouds = Pointclouds([torch.zeros(1, 3)])
        for size in (0, -1, float("inf"), float("nan")):
            with self.subTest(size=size), self.assertRaises(ValueError):
                voxel_down_sample(clouds, size)
        for size in (True, "1", [1], torch.tensor(1.0)):
            with self.subTest(size=size), self.assertRaises(TypeError):
                voxel_down_sample(clouds, size)

    def test_invalid_input(self):
        with self.assertRaises(TypeError):
            voxel_down_sample(torch.zeros(1, 4, 3), 1.0)
        for dtype in (torch.int64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                voxel_down_sample(Pointclouds([torch.zeros(1, 3, dtype=dtype)]), 1.0)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                voxel_down_sample(Pointclouds([torch.full((1, 3), invalid)]), 1.0)

    def test_invalid_attributes(self):
        for name in ("features", "normals"):
            for values in (
                torch.ones(2, 3, dtype=torch.int64),
                torch.ones(2, 3, dtype=torch.float16),
                torch.full((2, 3), float("inf")),
                torch.full((2, 3), float("nan")),
            ):
                with self.subTest(name=name, values=values), self.assertRaises(
                    ValueError
                ):
                    voxel_down_sample(
                        Pointclouds([torch.zeros(2, 3)], **{name: [values]}), 1.0
                    )

    def test_quantization_overflow(self):
        for points, size in (
            (torch.tensor([[1.0, 0, 0], [2.0, 0, 0]]), 1e-300),
            (torch.tensor([[float(2**63), 0, 0]], dtype=torch.float64), 1.0),
            (torch.tensor([[-float(2**64), 0, 0]], dtype=torch.float64), 1.0),
        ):
            with self.subTest(points=points, size=size), self.assertRaises(ValueError):
                voxel_down_sample(Pointclouds([points]), size)
        zero = torch.zeros(1, 3)
        torch.testing.assert_close(
            voxel_down_sample(Pointclouds([zero]), 1e-300).points_list()[0], zero
        )

    def test_gradients(self):
        points = torch.tensor(
            [[0.25, 0.25, 0.25], [0.75, 0.5, 0.5], [1.25, 0.25, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        features = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
        normals = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)

        def evaluate(p, f, n):
            output = voxel_down_sample(Pointclouds([p], [n], [f]), 1.0)
            return (
                output.points_list()[0],
                output.features_list()[0],
                output.normals_list()[0],
            )

        self.assertTrue(torch.autograd.gradcheck(evaluate, (points, features, normals)))
        for output, source in zip(
            evaluate(points, features, normals), (points, features, normals)
        ):
            grad = torch.autograd.grad(output.sum(), source)[0]
            expected = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float64)[:, None]
            torch.testing.assert_close(grad, expected.expand_as(source))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_correctness_and_gradients(self):
        generator = torch.Generator().manual_seed(14)
        points = torch.randn(128, 3, generator=generator, dtype=torch.float64)
        features = torch.randn(128, 4, generator=generator, dtype=torch.float64)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                cpu_p = points.to(dtype).requires_grad_()
                cpu_f = features.to(dtype).requires_grad_()
                gpu_p = cpu_p.detach().cuda().requires_grad_()
                gpu_f = cpu_f.detach().cuda().requires_grad_()
                outputs = []
                for p, f in ((cpu_p, cpu_f), (gpu_p, gpu_f)):
                    output = voxel_down_sample(
                        Pointclouds([p, p[:0]], features=[f, f[:0]]), 0.75
                    )
                    loss = output.points_packed().square().sum()
                    loss = loss + output.features_packed().square().sum()
                    gradients = torch.autograd.grad(loss, (p, f))
                    outputs.append((output, gradients))
                cpu, gpu = outputs
                torch.testing.assert_close(
                    gpu[0].points_packed().cpu(), cpu[0].points_packed()
                )
                torch.testing.assert_close(
                    gpu[0].features_packed().cpu(), cpu[0].features_packed()
                )
                for actual, expected in zip(gpu[1], cpu[1]):
                    torch.testing.assert_close(actual.cpu(), expected)

    def test_deterministic_mode(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            for device in devices:
                points = torch.tensor([[0.25, 0, 0], [0.75, 0, 0]], device=device)
                output = voxel_down_sample(Pointclouds([points]), 1.0)
                torch.testing.assert_close(
                    output.points_list()[0], points.mean(0, keepdim=True)
                )
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn)
