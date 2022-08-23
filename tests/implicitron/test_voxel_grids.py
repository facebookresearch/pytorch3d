# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from typing import Optional, Tuple

import torch

from pytorch3d.implicitron.models.implicit_function.utils import (
    interpolate_line,
    interpolate_plane,
    interpolate_volume,
)
from pytorch3d.implicitron.models.implicit_function.voxel_grid import (
    CPFactorizedVoxelGrid,
    FullResolutionVoxelGrid,
    VMFactorizedVoxelGrid,
)

from pytorch3d.implicitron.tools.config import expand_args_fields
from tests.common_testing import TestCaseMixin


class TestVoxelGrids(TestCaseMixin, unittest.TestCase):
    """
    Tests Voxel grids, tests them by setting all elements to zero (after retrieving
    they should also return zero) and by setting all of the elements to one and
    getting the result. Also tests the interpolation by 'manually' interpolating
    one by one sample and comparing with the batched implementation.
    """

    def test_my_code(self):
        return

    def get_random_normalized_points(
        self, n_grids, n_points, dimension=3
    ) -> torch.Tensor:
        # create random query points
        return torch.rand(n_grids, n_points, dimension) * 2 - 1

    def _test_query_with_constant_init_cp(
        self,
        n_grids: int,
        n_features: int,
        n_components: int,
        resolution: Tuple[int],
        value: float = 1,
        n_points: int = 1,
    ) -> None:
        # set everything to 'value' and do query for elementsthe result should
        # be of shape (n_grids, n_points, n_features) and be filled with n_components
        # * value
        grid = CPFactorizedVoxelGrid(
            resolution=resolution,
            n_components=n_components,
            n_features=n_features,
        )
        shapes = grid.get_shapes()

        params = grid.values_type(
            **{k: torch.ones(n_grids, *shapes[k]) * value for k in shapes}
        )

        assert torch.allclose(
            grid.evaluate_local(
                self.get_random_normalized_points(n_grids, n_points), params
            ),
            torch.ones(n_grids, n_points, n_features) * n_components * value,
        )

    def _test_query_with_constant_init_vm(
        self,
        n_grids: int,
        n_features: int,
        resolution: Tuple[int],
        n_components: Optional[int] = None,
        distribution: Optional[Tuple[int]] = None,
        value: float = 1,
        n_points: int = 1,
    ) -> None:
        # set everything to 'value' and do query for elements
        grid = VMFactorizedVoxelGrid(
            n_features=n_features,
            resolution=resolution,
            n_components=n_components,
            distribution_of_components=distribution,
        )
        shapes = grid.get_shapes()
        params = grid.values_type(
            **{k: torch.ones(n_grids, *shapes[k]) * value for k in shapes}
        )

        expected_element = (
            n_components * value if distribution is None else sum(distribution) * value
        )
        assert torch.allclose(
            grid.evaluate_local(
                self.get_random_normalized_points(n_grids, n_points), params
            ),
            torch.ones(n_grids, n_points, n_features) * expected_element,
        )

    def _test_query_with_constant_init_full(
        self,
        n_grids: int,
        n_features: int,
        resolution: Tuple[int],
        value: int = 1,
        n_points: int = 1,
    ) -> None:
        # set everything to 'value' and do query for elements
        grid = FullResolutionVoxelGrid(n_features=n_features, resolution=resolution)
        shapes = grid.get_shapes()
        params = grid.values_type(
            **{k: torch.ones(n_grids, *shapes[k]) * value for k in shapes}
        )

        expected_element = value
        assert torch.allclose(
            grid.evaluate_local(
                self.get_random_normalized_points(n_grids, n_points), params
            ),
            torch.ones(n_grids, n_points, n_features) * expected_element,
        )

    def test_query_with_constant_init(self):
        with self.subTest("Full"):
            self._test_query_with_constant_init_full(
                n_grids=5, n_features=6, resolution=(3, 4, 5), n_points=3
            )
        with self.subTest("Full with 1 in dimensions"):
            self._test_query_with_constant_init_full(
                n_grids=5, n_features=1, resolution=(33, 41, 1), n_points=4
            )
        with self.subTest("CP"):
            self._test_query_with_constant_init_cp(
                n_grids=5,
                n_features=6,
                n_components=7,
                resolution=(3, 4, 5),
                n_points=2,
            )
        with self.subTest("CP with 1 in dimensions"):
            self._test_query_with_constant_init_cp(
                n_grids=2,
                n_features=1,
                n_components=3,
                resolution=(3, 1, 1),
                n_points=4,
            )
        with self.subTest("VM with symetric distribution"):
            self._test_query_with_constant_init_vm(
                n_grids=6,
                n_features=9,
                resolution=(2, 12, 2),
                n_components=12,
                n_points=3,
            )
        with self.subTest("VM with distribution"):
            self._test_query_with_constant_init_vm(
                n_grids=5,
                n_features=1,
                resolution=(5, 9, 7),
                distribution=(33, 41, 1),
                n_points=7,
            )

    def test_query_with_zero_init(self):
        with self.subTest("Query testing with zero init CPFactorizedVoxelGrid"):
            self._test_query_with_constant_init_cp(
                n_grids=5,
                n_features=6,
                n_components=7,
                resolution=(3, 2, 5),
                n_points=3,
                value=0,
            )
        with self.subTest("Query testing with zero init VMFactorizedVoxelGrid"):
            self._test_query_with_constant_init_vm(
                n_grids=2,
                n_features=9,
                resolution=(2, 11, 3),
                n_components=3,
                n_points=3,
                value=0,
            )
        with self.subTest("Query testing with zero init FullResolutionVoxelGrid"):
            self._test_query_with_constant_init_full(
                n_grids=4, n_features=2, resolution=(3, 3, 5), n_points=3, value=0
            )

    def setUp(self):
        torch.manual_seed(42)
        expand_args_fields(FullResolutionVoxelGrid)
        expand_args_fields(CPFactorizedVoxelGrid)
        expand_args_fields(VMFactorizedVoxelGrid)

    def _interpolate_1D(
        self, points: torch.Tensor, vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        interpolate vector from points, which are (batch, 1) and individual point is in [-1, 1]
        """
        result = []
        _, _, width = vectors.shape
        # transform from [-1, 1] to [0, width-1]
        points = (points + 1) / 2 * (width - 1)
        for vector, row in zip(vectors, points):
            newrow = []
            for x in row:
                xf, xc = int(torch.floor(x)), int(torch.ceil(x))
                itemf, itemc = vector[:, xf], vector[:, xc]
                tmp = itemf * (xc - x) + itemc * (x - xf)
                newrow.append(tmp[None, None, :])
            result.append(torch.cat(newrow, dim=1))
        return torch.cat(result)

    def _interpolate_2D(
        self, points: torch.Tensor, matrices: torch.Tensor
    ) -> torch.Tensor:
        """
        interpolate matrix from points, which are (batch, 2) and individual point is in [-1, 1]
        """
        result = []
        n_grids, _, width, height = matrices.shape
        points = (points + 1) / 2 * (torch.tensor([[[width, height]]]) - 1)
        for matrix, row in zip(matrices, points):
            newrow = []
            for x, y in row:
                xf, xc = int(torch.floor(x)), int(torch.ceil(x))
                yf, yc = int(torch.floor(y)), int(torch.ceil(y))
                itemff, itemfc = matrix[:, xf, yf], matrix[:, xf, yc]
                itemcf, itemcc = matrix[:, xc, yf], matrix[:, xc, yc]
                itemf = itemff * (xc - x) + itemcf * (x - xf)
                itemc = itemfc * (xc - x) + itemcc * (x - xf)
                tmp = itemf * (yc - y) + itemc * (y - yf)
                newrow.append(tmp[None, None, :])
            result.append(torch.cat(newrow, dim=1))
        return torch.cat(result)

    def _interpolate_3D(
        self, points: torch.Tensor, tensors: torch.Tensor
    ) -> torch.Tensor:
        """
        interpolate tensors from points, which are (batch, 3) and individual point is in [-1, 1]
        """
        result = []
        _, _, width, height, depth = tensors.shape
        batch_normalized_points = (
            (points + 1) / 2 * (torch.tensor([[[width, height, depth]]]) - 1)
        )
        batch_points = points

        for tensor, points, normalized_points in zip(
            tensors, batch_points, batch_normalized_points
        ):
            newrow = []
            for (x, y, z), (_, _, nz) in zip(points, normalized_points):
                zf, zc = int(torch.floor(nz)), int(torch.ceil(nz))
                itemf = self._interpolate_2D(
                    points=torch.tensor([[[x, y]]]), matrices=tensor[None, :, :, :, zf]
                )
                itemc = self._interpolate_2D(
                    points=torch.tensor([[[x, y]]]), matrices=tensor[None, :, :, :, zc]
                )
                tmp = self._interpolate_1D(
                    points=torch.tensor([[[z]]]),
                    vectors=torch.cat((itemf, itemc), dim=1).permute(0, 2, 1),
                )
                newrow.append(tmp)
            result.append(torch.cat(newrow, dim=1))
        return torch.cat(result)

    def test_interpolation(self):

        with self.subTest("1D interpolation"):
            points = self.get_random_normalized_points(
                n_grids=4, n_points=5, dimension=1
            )
            vector = torch.randn(size=(4, 3, 2))
            assert torch.allclose(
                self._interpolate_1D(points, vector),
                interpolate_line(
                    points,
                    vector,
                    align_corners=True,
                    padding_mode="zeros",
                    mode="bilinear",
                ),
            )
        with self.subTest("2D interpolation"):
            points = self.get_random_normalized_points(
                n_grids=4, n_points=5, dimension=2
            )
            matrix = torch.randn(size=(4, 2, 3, 5))
            assert torch.allclose(
                self._interpolate_2D(points, matrix),
                interpolate_plane(
                    points,
                    matrix,
                    align_corners=True,
                    padding_mode="zeros",
                    mode="bilinear",
                ),
            )

        with self.subTest("3D interpolation"):
            points = self.get_random_normalized_points(
                n_grids=4, n_points=5, dimension=3
            )
            tensor = torch.randn(size=(4, 5, 2, 7, 2))
            assert torch.allclose(
                self._interpolate_3D(points, tensor),
                interpolate_volume(
                    points,
                    tensor,
                    align_corners=True,
                    padding_mode="zeros",
                    mode="bilinear",
                ),
            )

    def test_floating_point_query(self):
        """
        test querying the voxel grids on some float positions
        """
        with self.subTest("FullResolution"):
            grid = FullResolutionVoxelGrid(n_features=1, resolution=(1, 1, 1))
            params = grid.values_type(**grid.get_shapes())
            params.voxel_grid = torch.tensor(
                [
                    [
                        [[[1, 3], [5, 7]], [[9, 11], [13, 15]]],
                        [[[2, 4], [6, 8]], [[10, 12], [14, 16]]],
                    ],
                    [
                        [[[17, 18], [19, 20]], [[21, 22], [23, 24]]],
                        [[[25, 26], [27, 28]], [[29, 30], [31, 32]]],
                    ],
                ],
                dtype=torch.float,
            )
            points = (
                torch.tensor(
                    [
                        [
                            [1, 0, 1],
                            [0.5, 1, 1],
                            [1 / 3, 1 / 3, 2 / 3],
                        ],
                        [
                            [0, 1, 1],
                            [0, 0.5, 1],
                            [1 / 4, 1 / 4, 3 / 4],
                        ],
                    ]
                )
                / torch.tensor([[1.0, 1, 1]])
                * 2
                - 1
            )
            expected_result = torch.tensor(
                [
                    [[11, 12], [11, 12], [6.333333, 7.3333333]],
                    [[20, 28], [19, 27], [19.25, 27.25]],
                ]
            )

            assert torch.allclose(
                grid.evaluate_local(points, params),
                expected_result,
                rtol=0.00001,
            ), grid.evaluate_local(points, params)
        with self.subTest("CP"):
            grid = CPFactorizedVoxelGrid(
                n_features=1, resolution=(1, 1, 1), n_components=3
            )
            params = grid.values_type(**grid.get_shapes())
            params.vector_components_x = torch.tensor(
                [
                    [[1, 2], [10.5, 20.5]],
                    [[10, 20], [2, 4]],
                ]
            )
            params.vector_components_y = torch.tensor(
                [
                    [[3, 4, 5], [30.5, 40.5, 50.5]],
                    [[30, 40, 50], [1, 3, 5]],
                ]
            )
            params.vector_components_z = torch.tensor(
                [
                    [[6, 7, 8, 9], [60.5, 70.5, 80.5, 90.5]],
                    [[60, 70, 80, 90], [6, 7, 8, 9]],
                ]
            )
            params.basis_matrix = torch.tensor(
                [
                    [[2.0], [2.0]],
                    [[1.0], [2.0]],
                ]
            )
            points = (
                torch.tensor(
                    [
                        [
                            [0, 2, 2],
                            [1, 2, 0.25],
                            [0.5, 0.5, 1],
                            [1 / 3, 2 / 3, 2 + 1 / 3],
                        ],
                        [
                            [1, 0, 1],
                            [0.5, 2, 2],
                            [1, 0.5, 0.5],
                            [1 / 4, 3 / 4, 2 + 1 / 4],
                        ],
                    ]
                )
                / torch.tensor([[[1.0, 2, 3]]])
                * 2
                - 1
            )
            expected_result_matrix = torch.tensor(
                [
                    [[85450.25], [130566.5], [77658.75], [86285.422]],
                    [[42056], [60240], [45604], [38775]],
                ]
            )
            expected_result_sum = torch.tensor(
                [
                    [[42725.125], [65283.25], [38829.375], [43142.711]],
                    [[42028], [60120], [45552], [38723.4375]],
                ]
            )
            with self.subTest("CP with basis_matrix reduction"):
                assert torch.allclose(
                    grid.evaluate_local(points, params),
                    expected_result_matrix,
                    rtol=0.00001,
                )
            del params.basis_matrix
            with self.subTest("CP with sum reduction"):
                assert torch.allclose(
                    grid.evaluate_local(points, params),
                    expected_result_sum,
                    rtol=0.00001,
                )

        with self.subTest("VM"):
            grid = VMFactorizedVoxelGrid(
                n_features=1, resolution=(1, 1, 1), n_components=3
            )
            params = VMFactorizedVoxelGrid.values_type(**grid.get_shapes())
            params.matrix_components_xy = torch.tensor(
                [
                    [[[1, 2], [3, 4]], [[19, 20], [21, 22.0]]],
                    [[[35, 36], [37, 38]], [[39, 40], [41, 42]]],
                ]
            )
            params.matrix_components_xz = torch.tensor(
                [
                    [[[7, 8], [9, 10]], [[25, 26], [27, 28.0]]],
                    [[[43, 44], [45, 46]], [[47, 48], [49, 50]]],
                ]
            )
            params.matrix_components_yz = torch.tensor(
                [
                    [[[13, 14], [15, 16]], [[31, 32], [33, 34.0]]],
                    [[[51, 52], [53, 54]], [[55, 56], [57, 58.0]]],
                ]
            )

            params.vector_components_z = torch.tensor(
                [
                    [[5, 6], [23, 24.0]],
                    [[59, 60], [61, 62]],
                ]
            )
            params.vector_components_y = torch.tensor(
                [
                    [[11, 12], [29, 30.0]],
                    [[63, 64], [65, 66]],
                ]
            )
            params.vector_components_x = torch.tensor(
                [
                    [[17, 18], [35, 36.0]],
                    [[67, 68], [69, 70.0]],
                ]
            )

            params.basis_matrix = torch.tensor(
                [
                    [2, 2, 2, 2, 2, 2.0],
                    [1, 2, 1, 2, 1, 2.0],
                ]
            )[:, :, None]
            points = (
                torch.tensor(
                    [
                        [
                            [1, 0, 1],
                            [0.5, 1, 1],
                            [1 / 3, 1 / 3, 2 / 3],
                        ],
                        [
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 1, 0],
                        ],
                    ]
                )
                / torch.tensor([[[1.0, 1, 1]]])
                * 2
                - 1
            )
            expected_result_matrix = torch.tensor(
                [
                    [[5696], [5854], [5484.888]],
                    [[27377], [26649], [27377]],
                ]
            )
            expected_result_sum = torch.tensor(
                [
                    [[2848], [2927], [2742.444]],
                    [[17902], [17420], [17902]],
                ]
            )
            with self.subTest("VM with basis_matrix reduction"):
                assert torch.allclose(
                    grid.evaluate_local(points, params),
                    expected_result_matrix,
                    rtol=0.00001,
                )
            del params.basis_matrix
            with self.subTest("VM with sum reduction"):
                assert torch.allclose(
                    grid.evaluate_local(points, params),
                    expected_result_sum,
                    rtol=0.0001,
                ), grid.evaluate_local(points, params)

    def test_forward_with_small_init_std(self):
        """
        Test does the grid return small values if it is initialized with small
        mean and small standard deviation.
        """

        def test(cls, **kwargs):
            with self.subTest(cls.__name__):
                n_grids = 3
                grid = cls(**kwargs)
                shapes = grid.get_shapes()
                params = cls.values_type(
                    **{
                        k: torch.normal(mean=torch.zeros(n_grids, *shape), std=0.0001)
                        for k, shape in shapes.items()
                    }
                )
                points = self.get_random_normalized_points(n_grids=n_grids, n_points=3)
                max_expected_result = torch.zeros((len(points), 10)) + 1e-2
                assert torch.all(
                    grid.evaluate_local(points, params) < max_expected_result
                )

        test(
            FullResolutionVoxelGrid,
            resolution=(4, 6, 9),
            n_features=10,
        )
        test(
            CPFactorizedVoxelGrid,
            resolution=(4, 6, 9),
            n_features=10,
            n_components=3,
        )
        test(
            VMFactorizedVoxelGrid,
            resolution=(4, 6, 9),
            n_features=10,
            n_components=3,
        )
