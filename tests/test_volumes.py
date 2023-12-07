# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import itertools
import random
import unittest

import numpy as np
import torch
from pytorch3d.structures.volumes import VolumeLocator, Volumes
from pytorch3d.transforms import Scale

from .common_testing import TestCaseMixin


class TestVolumes(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

    @staticmethod
    def _random_volume_list(
        num_volumes, min_size, max_size, num_channels, device, rand_sizes=None
    ):
        """
        Init a list of `num_volumes` random tensors of size [num_channels, *rand_size].
        If `rand_sizes` is None, rand_size is a 3D long vector sampled
        from [min_size, max_size]. Otherwise, rand_size should be a list
        [rand_size_1, rand_size_2, ..., rand_size_num_volumes] where each
        `rand_size_i` denotes the size of the corresponding `i`-th tensor.
        """
        if rand_sizes is None:
            rand_sizes = [
                [random.randint(min_size, vs) for vs in max_size]
                for _ in range(num_volumes)
            ]

        volume_list = [
            torch.randn(
                size=[num_channels, *rand_size], device=device, dtype=torch.float32
            )
            for rand_size in rand_sizes
        ]

        return volume_list, rand_sizes

    def _check_indexed_volumes(self, v, selected, indices):
        for selectedIdx, index in indices:
            self.assertClose(selected.densities()[selectedIdx], v.densities()[index])
            self.assertClose(
                v.locator._local_to_world_transform.get_matrix()[index],
                selected.locator._local_to_world_transform.get_matrix()[selectedIdx],
            )
            if selected.features() is not None:
                self.assertClose(selected.features()[selectedIdx], v.features()[index])

    def test_get_item(
        self,
        num_volumes=5,
        num_channels=4,
        volume_size=(10, 13, 8),
        dtype=torch.float32,
    ):

        device = torch.device("cuda:0")

        # make sure we have at least 3 volumes to prevent indexing crash
        num_volumes = max(num_volumes, 3)

        features = torch.randn(
            size=[num_volumes, num_channels, *volume_size],
            device=device,
            dtype=torch.float32,
        )
        densities = torch.randn(
            size=[num_volumes, 1, *volume_size], device=device, dtype=torch.float32
        )

        features_list, rand_sizes = TestVolumes._random_volume_list(
            num_volumes, 3, volume_size, num_channels, device
        )
        densities_list, _ = TestVolumes._random_volume_list(
            num_volumes, 3, volume_size, 1, device, rand_sizes=rand_sizes
        )

        volume_translation = -torch.randn(num_volumes, 3).type_as(features)
        voxel_size = torch.rand(num_volumes, 1).type_as(features) + 0.5

        for features_, densities_ in zip(
            (None, features, features_list), (densities, densities, densities_list)
        ):

            # init the volume structure
            v = Volumes(
                features=features_,
                densities=densities_,
                volume_translation=volume_translation,
                voxel_size=voxel_size,
            )

            # int index
            index = 1
            v_selected = v[index]
            self.assertEqual(len(v_selected), 1)
            self._check_indexed_volumes(v, v_selected, [(0, 1)])

            # list index
            index = [1, 2]
            v_selected = v[index]
            self.assertEqual(len(v_selected), len(index))
            self._check_indexed_volumes(v, v_selected, enumerate(index))

            # slice index
            index = slice(0, 2, 1)
            v_selected = v[0:2]
            self.assertEqual(len(v_selected), 2)
            self._check_indexed_volumes(v, v_selected, [(0, 0), (1, 1)])

            # bool tensor
            index = (torch.rand(num_volumes) > 0.5).to(device)
            index[:2] = True  # make sure smth is selected
            v_selected = v[index]
            self.assertEqual(len(v_selected), index.sum())
            self._check_indexed_volumes(
                v,
                v_selected,
                zip(
                    torch.arange(index.sum()),
                    torch.nonzero(index, as_tuple=False).squeeze(),
                ),
            )

            # int tensor
            index = torch.tensor([1, 2], dtype=torch.int64, device=device)
            v_selected = v[index]
            self.assertEqual(len(v_selected), index.numel())
            self._check_indexed_volumes(v, v_selected, enumerate(index.tolist()))

            # invalid index
            index = torch.tensor([1, 0, 1], dtype=torch.float32, device=device)
            with self.assertRaises(IndexError):
                v_selected = v[index]
            index = 1.2  # floating point index
            with self.assertRaises(IndexError):
                v_selected = v[index]

    def test_locator_init(self, batch_size=9, resolution=(3, 5, 7)):
        with self.subTest("VolumeLocator init with all sizes equal"):
            grid_sizes = [resolution for _ in range(batch_size)]
            locator_tuple = VolumeLocator(
                batch_size=batch_size, grid_sizes=resolution, device=torch.device("cpu")
            )
            locator_list = VolumeLocator(
                batch_size=batch_size, grid_sizes=grid_sizes, device=torch.device("cpu")
            )
            locator_tensor = VolumeLocator(
                batch_size=batch_size,
                grid_sizes=torch.tensor(grid_sizes),
                device=torch.device("cpu"),
            )
            expected_grid_sizes = torch.tensor(grid_sizes)
            expected_resolution = resolution
            assert torch.allclose(expected_grid_sizes, locator_tuple._grid_sizes)
            assert torch.allclose(expected_grid_sizes, locator_list._grid_sizes)
            assert torch.allclose(expected_grid_sizes, locator_tensor._grid_sizes)
            self.assertEqual(expected_resolution, locator_tuple._resolution)
            self.assertEqual(expected_resolution, locator_list._resolution)
            self.assertEqual(expected_resolution, locator_tensor._resolution)

        with self.subTest("VolumeLocator with different sizes in different grids"):
            grid_sizes_list = [
                torch.randint(low=1, high=42, size=(3,)) for _ in range(batch_size)
            ]
            grid_sizes_tensor = torch.cat([el[None] for el in grid_sizes_list])
            locator_list = VolumeLocator(
                batch_size=batch_size,
                grid_sizes=grid_sizes_list,
                device=torch.device("cpu"),
            )
            locator_tensor = VolumeLocator(
                batch_size=batch_size,
                grid_sizes=grid_sizes_tensor,
                device=torch.device("cpu"),
            )
            expected_grid_sizes = grid_sizes_tensor
            expected_resolution = tuple(torch.max(expected_grid_sizes, dim=0).values)
            assert torch.allclose(expected_grid_sizes, locator_list._grid_sizes)
            assert torch.allclose(expected_grid_sizes, locator_tensor._grid_sizes)
            self.assertEqual(expected_resolution, locator_list._resolution)
            self.assertEqual(expected_resolution, locator_tensor._resolution)

    def test_coord_transforms(self, num_volumes=3, num_channels=4, dtype=torch.float32):
        """
        Test the correctness of the conversion between the internal
        Transform3D Volumes.VolumeLocator._local_to_world_transform and the initialization
        from the translation and voxel_size.
        """

        device = torch.device("cuda:0")

        # try for 10 sets of different random sizes/centers/voxel_sizes
        for _ in range(10):

            size = torch.randint(high=10, size=(3,), low=3).tolist()

            densities = torch.randn(
                size=[num_volumes, num_channels, *size],
                device=device,
                dtype=torch.float32,
            )

            # init the transformation params
            volume_translation = torch.randn(num_volumes, 3)
            voxel_size = torch.rand(num_volumes, 3) * 3.0 + 0.5

            # get the corresponding Transform3d object
            local_offset = torch.tensor(list(size), dtype=torch.float32, device=device)[
                [2, 1, 0]
            ][None].repeat(num_volumes, 1)
            local_to_world_transform = (
                Scale(0.5 * local_offset - 0.5, device=device)
                .scale(voxel_size)
                .translate(-volume_translation)
            )

            # init the volume structures with the scale and translation,
            # then get the coord grid in world coords
            v_trans_vs = Volumes(
                densities=densities,
                voxel_size=voxel_size,
                volume_translation=volume_translation,
            )
            grid_rot_trans_vs = v_trans_vs.get_coord_grid(world_coordinates=True)

            # map the default local coords to the world coords
            # with local_to_world_transform
            v_default = Volumes(densities=densities)
            grid_default_local = v_default.get_coord_grid(world_coordinates=False)
            grid_default_world = local_to_world_transform.transform_points(
                grid_default_local.view(num_volumes, -1, 3)
            ).view(num_volumes, *size, 3)

            # check that both grids are the same
            self.assertClose(grid_rot_trans_vs, grid_default_world, atol=1e-5)

            # check that the transformations are the same
            self.assertClose(
                v_trans_vs.get_local_to_world_coords_transform().get_matrix(),
                local_to_world_transform.get_matrix(),
                atol=1e-5,
            )

    def test_coord_grid_convention(
        self, num_volumes=3, num_channels=4, dtype=torch.float32
    ):
        """
        Check that for a trivial volume with spatial size DxHxW=5x7x5:
        1) xyz_world=(0, 0, 0) lands right in the middle of the volume
        with xyz_local=(0, 0, 0).
        2) xyz_world=(-2, 3, 2) results in xyz_local=(-1, 1, -1).
        3) The centeral voxel of the volume coordinate grid
        has coords x_world=(0, 0, 0) and x_local=(0, 0, 0)
        4) grid_sampler(world_coordinate_grid, local_coordinate_grid)
        is the same as world_coordinate_grid itself. I.e. the local coordinate
        grid matches the grid_sampler coordinate convention.
        """

        device = torch.device("cuda:0")

        densities = torch.randn(
            size=[num_volumes, num_channels, 5, 7, 5],
            device=device,
            dtype=torch.float32,
        )
        v_trivial = Volumes(densities=densities)

        # check the case with x_world=(0,0,0)
        pts_world = torch.zeros(num_volumes, 1, 3, device=device, dtype=torch.float32)
        pts_local = v_trivial.world_to_local_coords(pts_world)
        pts_local_expected = torch.zeros_like(pts_local)
        self.assertClose(pts_local, pts_local_expected)

        # check the case with x_world=(-2, 3, -2)
        pts_world = torch.tensor([-2, 3, -2], device=device, dtype=torch.float32)[
            None, None
        ].repeat(num_volumes, 1, 1)
        pts_local = v_trivial.world_to_local_coords(pts_world)
        pts_local_expected = torch.tensor(
            [-1, 1, -1], device=device, dtype=torch.float32
        )[None, None].repeat(num_volumes, 1, 1)
        self.assertClose(pts_local, pts_local_expected)

        # check that the central voxel has coords x_world=(0, 0, 0) and x_local(0, 0, 0)
        grid_world = v_trivial.get_coord_grid(world_coordinates=True)
        grid_local = v_trivial.get_coord_grid(world_coordinates=False)
        for grid in (grid_world, grid_local):
            x0 = grid[0, :, :, 2, 0]
            y0 = grid[0, :, 3, :, 1]
            z0 = grid[0, 2, :, :, 2]
            for coord_line in (x0, y0, z0):
                self.assertClose(coord_line, torch.zeros_like(coord_line), atol=1e-7)

        # resample grid_world using grid_sampler with local coords
        # -> make sure the resampled version is the same as original
        grid_world_resampled = torch.nn.functional.grid_sample(
            grid_world.permute(0, 4, 1, 2, 3), grid_local, align_corners=True
        ).permute(0, 2, 3, 4, 1)
        self.assertClose(grid_world_resampled, grid_world, atol=1e-7)

        for align_corners in [True, False]:
            v_trivial = Volumes(densities=densities, align_corners=align_corners)

            # check the case with x_world=(0,0,0)
            pts_world = torch.zeros(
                num_volumes, 1, 3, device=device, dtype=torch.float32
            )
            pts_local = v_trivial.world_to_local_coords(pts_world)
            pts_local_expected = torch.zeros_like(pts_local)
            self.assertClose(pts_local, pts_local_expected)

            # check the case with x_world=(-2, 3, -2)
            pts_world_tuple = [-2, 3, -2]
            pts_world = torch.tensor(
                pts_world_tuple, device=device, dtype=torch.float32
            )[None, None].repeat(num_volumes, 1, 1)
            pts_local = v_trivial.world_to_local_coords(pts_world)
            pts_local_expected = torch.tensor(
                [-1, 1, -1], device=device, dtype=torch.float32
            )[None, None].repeat(num_volumes, 1, 1)
            self.assertClose(pts_local, pts_local_expected)

            # # check that the central voxel has coords x_world=(0, 0, 0) and x_local(0, 0, 0)
            grid_world = v_trivial.get_coord_grid(world_coordinates=True)
            grid_local = v_trivial.get_coord_grid(world_coordinates=False)
            for grid in (grid_world, grid_local):
                x0 = grid[0, :, :, 2, 0]
                y0 = grid[0, :, 3, :, 1]
                z0 = grid[0, 2, :, :, 2]
                for coord_line in (x0, y0, z0):
                    self.assertClose(
                        coord_line, torch.zeros_like(coord_line), atol=1e-7
                    )

            # resample grid_world using grid_sampler with local coords
            # -> make sure the resampled version is the same as original
            grid_world_resampled = torch.nn.functional.grid_sample(
                grid_world.permute(0, 4, 1, 2, 3),
                grid_local,
                align_corners=align_corners,
            ).permute(0, 2, 3, 4, 1)
            self.assertClose(grid_world_resampled, grid_world, atol=1e-7)

    def test_coord_grid_convention_heterogeneous(
        self, num_channels=4, dtype=torch.float32
    ):
        """
        Check that for a list of 2 trivial volumes with
        spatial sizes DxHxW=(5x7x5, 3x5x5):
        1) xyz_world=(0, 0, 0) lands right in the middle of the volume
        with xyz_local=(0, 0, 0).
        2) xyz_world=((-2, 3, -2), (-2, -2,  1)) results
        in xyz_local=((-1, 1, -1), (-1, -1, 1)).
        3) The centeral voxel of the volume coordinate grid
        has coords x_world=(0, 0, 0) and x_local=(0, 0, 0)
        4) grid_sampler(world_coordinate_grid, local_coordinate_grid)
        is the same as world_coordinate_grid itself. I.e. the local coordinate
        grid matches the grid_sampler coordinate convention.
        """

        device = torch.device("cuda:0")

        sizes = [(5, 7, 5), (3, 5, 5)]

        densities_list = [
            torch.randn(size=[num_channels, *size], device=device, dtype=torch.float32)
            for size in sizes
        ]

        # init the volume
        v_trivial = Volumes(densities=densities_list)

        # check the border point locations
        pts_world = torch.tensor(
            [[-2.0, 3.0, -2.0], [-2.0, -2.0, 1.0]], device=device, dtype=torch.float32
        )[:, None]
        pts_local = v_trivial.world_to_local_coords(pts_world)
        pts_local_expected = torch.tensor(
            [[-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]], device=device, dtype=torch.float32
        )[:, None]
        self.assertClose(pts_local, pts_local_expected)

        # check that the central voxel has coords x_world=(0, 0, 0) and x_local(0, 0, 0)
        grid_world = v_trivial.get_coord_grid(world_coordinates=True)
        grid_local = v_trivial.get_coord_grid(world_coordinates=False)
        for grid in (grid_world, grid_local):
            x0 = grid[0, :, :, 2, 0]
            y0 = grid[0, :, 3, :, 1]
            z0 = grid[0, 2, :, :, 2]
            for coord_line in (x0, y0, z0):
                self.assertClose(coord_line, torch.zeros_like(coord_line), atol=1e-7)
            x0 = grid[1, :, :, 2, 0]
            y0 = grid[1, :, 2, :, 1]
            z0 = grid[1, 1, :, :, 2]
            for coord_line in (x0, y0, z0):
                self.assertClose(coord_line, torch.zeros_like(coord_line), atol=1e-7)

        # resample grid_world using grid_sampler with local coords
        # -> make sure the resampled version is the same as original
        for grid_world_, grid_local_, size in zip(grid_world, grid_local, sizes):
            grid_world_crop = grid_world_[: size[0], : size[1], : size[2], :][None]
            grid_local_crop = grid_local_[: size[0], : size[1], : size[2], :][None]
            grid_world_crop_resampled = torch.nn.functional.grid_sample(
                grid_world_crop.permute(0, 4, 1, 2, 3),
                grid_local_crop,
                align_corners=True,
            ).permute(0, 2, 3, 4, 1)
            self.assertClose(grid_world_crop_resampled, grid_world_crop, atol=1e-7)

    def test_coord_grid_transforms(
        self, num_volumes=3, num_channels=4, dtype=torch.float32
    ):
        """
        Test whether conversion between local-world coordinates of the
        volume returns correct results.
        """

        device = torch.device("cuda:0")

        # try for 10 sets of different random sizes/centers/voxel_sizes
        for _ in range(10):

            size = torch.randint(high=10, size=(3,), low=3).tolist()

            center = torch.randn(num_volumes, 3, dtype=torch.float32, device=device)
            voxel_size = torch.rand(1, dtype=torch.float32, device=device) * 5.0 + 0.5

            for densities in (
                torch.randn(
                    size=[num_volumes, num_channels, *size],
                    device=device,
                    dtype=torch.float32,
                ),
                TestVolumes._random_volume_list(
                    num_volumes, 3, size, num_channels, device, rand_sizes=None
                )[0],
            ):

                # init the volume structure
                v = Volumes(
                    densities=densities,
                    voxel_size=voxel_size,
                    volume_translation=-center,
                )

                # get local coord grid
                grid_local = v.get_coord_grid(world_coordinates=False)

                # convert from world to local to world
                grid_world = v.get_coord_grid(world_coordinates=True)
                grid_local_2 = v.world_to_local_coords(grid_world)
                grid_world_2 = v.local_to_world_coords(grid_local_2)

                # assertions on shape and values of grid_world and grid_local
                self.assertClose(grid_world, grid_world_2, atol=1e-5)
                self.assertClose(grid_local, grid_local_2, atol=1e-5)

                # check that the individual slices of the location grid have
                # constant values along expected dimensions
                for plane_dim in (1, 2, 3):
                    for grid_plane in grid_world.split(1, dim=plane_dim):
                        grid_coord_dim = {1: 2, 2: 1, 3: 0}[plane_dim]
                        grid_coord_plane = grid_plane.squeeze()[..., grid_coord_dim]
                        # check that all elements of grid_coord_plane are
                        # the same for each batch element
                        self.assertClose(
                            grid_coord_plane.reshape(num_volumes, -1).max(dim=1).values,
                            grid_coord_plane.reshape(num_volumes, -1).min(dim=1).values,
                        )

    def test_clone(
        self, num_volumes=3, num_channels=4, size=(6, 8, 10), dtype=torch.float32
    ):
        """
        Test cloning of a `Volumes` object
        """

        device = torch.device("cuda:0")

        features = torch.randn(
            size=[num_volumes, num_channels, *size], device=device, dtype=torch.float32
        )
        densities = torch.rand(
            size=[num_volumes, 1, *size], device=device, dtype=torch.float32
        )

        for has_features in (True, False):
            v = Volumes(
                densities=densities, features=features if has_features else None
            )
            vnew = v.clone()
            vnew._densities.data[0, 0, 0, 0, 0] += 1.0
            self.assertNotAlmostEqual(
                float(
                    (vnew.densities()[0, 0, 0, 0, 0] - v.densities()[0, 0, 0, 0, 0])
                    .abs()
                    .max()
                ),
                0.0,
            )

            if has_features:
                vnew._features.data[0, 0, 0, 0, 0] += 1.0
                self.assertNotAlmostEqual(
                    float(
                        (vnew.features()[0, 0, 0, 0, 0] - v.features()[0, 0, 0, 0, 0])
                        .abs()
                        .max()
                    ),
                    0.0,
                )

    def _check_vars_on_device(self, v, desired_device):
        for var_name, var in vars(v).items():
            if var_name != "device":
                if var is not None:
                    self.assertTrue(
                        var.device.type == desired_device.type,
                        (var_name, var.device, desired_device),
                    )
            else:
                self.assertTrue(var.type == desired_device.type)

    def test_to(
        self, num_volumes=3, num_channels=4, size=(6, 8, 10), dtype=torch.float32
    ):
        """
        Test the moving of the volumes from/to gpu and cpu
        """

        features = torch.randn(
            size=[num_volumes, num_channels, *size], dtype=torch.float32
        )
        densities = torch.rand(size=[num_volumes, 1, *size], dtype=dtype)
        volumes = Volumes(densities=densities, features=features)
        locator = VolumeLocator(
            batch_size=5, grid_sizes=(3, 5, 7), device=volumes.device
        )

        for name, obj in (("VolumeLocator", locator), ("Volumes", volumes)):
            with self.subTest(f"Moving {name} from/to gpu and cpu"):
                # Test support for str and torch.device
                cpu_device = torch.device("cpu")

                converted_obj = obj.to("cpu")
                self.assertEqual(cpu_device, converted_obj.device)
                self.assertEqual(cpu_device, obj.device)
                self.assertIs(obj, converted_obj)

                converted_obj = obj.to(cpu_device)
                self.assertEqual(cpu_device, converted_obj.device)
                self.assertEqual(cpu_device, obj.device)
                self.assertIs(obj, converted_obj)

                cuda_device = torch.device("cuda:0")

                converted_obj = obj.to("cuda:0")
                self.assertEqual(cuda_device, converted_obj.device)
                self.assertEqual(cpu_device, obj.device)
                self.assertIsNot(obj, converted_obj)

                converted_obj = obj.to(cuda_device)
                self.assertEqual(cuda_device, converted_obj.device)
                self.assertEqual(cpu_device, obj.device)
                self.assertIsNot(obj, converted_obj)

        with self.subTest("Test device placement of internal tensors of Volumes"):
            features = features.to(cuda_device)
            densities = features.to(cuda_device)

            for features_ in (features, None):
                volumes = Volumes(densities=densities, features=features_)

                cpu_volumes = volumes.cpu()
                cuda_volumes = cpu_volumes.cuda()
                cuda_volumes2 = cuda_volumes.cuda()
                cpu_volumes2 = cuda_volumes2.cpu()

                for volumes1, volumes2 in itertools.combinations(
                    (volumes, cpu_volumes, cpu_volumes2, cuda_volumes, cuda_volumes2), 2
                ):
                    if volumes1 is cuda_volumes and volumes2 is cuda_volumes2:
                        # checks that we do not copy if the devices stay the same
                        assert_fun = self.assertIs
                    else:
                        assert_fun = self.assertSeparate
                    assert_fun(volumes1._densities, volumes2._densities)
                    if features_ is not None:
                        assert_fun(volumes1._features, volumes2._features)
                    for volumes_ in (volumes1, volumes2):
                        if volumes_ in (cpu_volumes, cpu_volumes2):
                            self._check_vars_on_device(volumes_, cpu_device)
                        else:
                            self._check_vars_on_device(volumes_, cuda_device)

        with self.subTest("Test device placement of internal tensors of VolumeLocator"):
            for device1, device2 in itertools.combinations(
                (torch.device("cpu"), torch.device("cuda:0")), 2
            ):
                locator = locator.to(device1)
                locator = locator.to(device2)
                self.assertEqual(locator._grid_sizes.device, device2)
                self.assertEqual(locator._local_to_world_transform.device, device2)

    def _check_padded(self, x_pad, x_list, grid_sizes):
        """
        Check that padded tensors x_pad are the same as x_list tensors.
        """
        num_volumes = len(x_list)
        for i in range(num_volumes):
            self.assertClose(
                x_pad[i][:, : grid_sizes[i][0], : grid_sizes[i][1], : grid_sizes[i][2]],
                x_list[i],
            )

    def test_feature_density_setters(self):
        """
        Tests getters and setters for padded/list representations.
        """

        device = torch.device("cuda:0")
        diff_device = torch.device("cpu")

        num_volumes = 30
        num_channels = 4
        K = 20

        densities = []
        features = []
        grid_sizes = []
        diff_grid_sizes = []

        for _ in range(num_volumes):
            grid_size = torch.randint(K - 1, size=(3,)).long() + 1
            densities.append(
                torch.rand((1, *grid_size), device=device, dtype=torch.float32)
            )
            features.append(
                torch.rand(
                    (num_channels, *grid_size), device=device, dtype=torch.float32
                )
            )
            grid_sizes.append(grid_size)

            diff_grid_size = (
                copy.deepcopy(grid_size) + torch.randint(2, size=(3,)).long() + 1
            )
            diff_grid_sizes.append(diff_grid_size)
        grid_sizes = torch.stack(grid_sizes).to(device)
        diff_grid_sizes = torch.stack(diff_grid_sizes).to(device)

        volumes = Volumes(densities=densities, features=features)
        self.assertClose(volumes.get_grid_sizes(), grid_sizes)

        # test the getters
        features_padded = volumes.features()
        densities_padded = volumes.densities()
        features_list = volumes.features_list()
        densities_list = volumes.densities_list()
        for x_pad, x_list in zip(
            (densities_padded, features_padded, densities_padded, features_padded),
            (densities_list, features_list, densities, features),
        ):
            self._check_padded(x_pad, x_list, grid_sizes)

        # test feature setters
        features_new = [
            torch.rand((num_channels, *grid_size), device=device, dtype=torch.float32)
            for grid_size in grid_sizes
        ]
        volumes._set_features(features_new)
        features_new_list = volumes.features_list()
        features_new_padded = volumes.features()
        for x_pad, x_list in zip(
            (features_new_padded, features_new_padded),
            (features_new, features_new_list),
        ):
            self._check_padded(x_pad, x_list, grid_sizes)

        # wrong features to update
        bad_features_new = [
            [
                torch.rand(
                    (num_channels, *grid_size), device=diff_device, dtype=torch.float32
                )
                for grid_size in diff_grid_sizes
            ],
            torch.rand(
                (num_volumes, num_channels, K + 1, K, K),
                device=device,
                dtype=torch.float32,
            ),
            None,
        ]
        for bad_features_new_ in bad_features_new:
            with self.assertRaises(ValueError):
                volumes._set_densities(bad_features_new_)

        # test density setters
        densities_new = [
            torch.rand((1, *grid_size), device=device, dtype=torch.float32)
            for grid_size in grid_sizes
        ]
        volumes._set_densities(densities_new)
        densities_new_list = volumes.densities_list()
        densities_new_padded = volumes.densities()
        for x_pad, x_list in zip(
            (densities_new_padded, densities_new_padded),
            (densities_new, densities_new_list),
        ):
            self._check_padded(x_pad, x_list, grid_sizes)

        # wrong densities to update
        bad_densities_new = [
            [
                torch.rand((1, *grid_size), device=diff_device, dtype=torch.float32)
                for grid_size in diff_grid_sizes
            ],
            torch.rand(
                (num_volumes, 1, K + 1, K, K), device=device, dtype=torch.float32
            ),
            None,
        ]
        for bad_densities_new_ in bad_densities_new:
            with self.assertRaises(ValueError):
                volumes._set_densities(bad_densities_new_)

        # test update_padded
        volumes = Volumes(densities=densities, features=features)
        volumes_updated = volumes.update_padded(
            densities_new, new_features=features_new
        )
        densities_new_list = volumes_updated.densities_list()
        densities_new_padded = volumes_updated.densities()
        features_new_list = volumes_updated.features_list()
        features_new_padded = volumes_updated.features()
        for x_pad, x_list in zip(
            (
                densities_new_padded,
                densities_new_padded,
                features_new_padded,
                features_new_padded,
            ),
            (densities_new, densities_new_list, features_new, features_new_list),
        ):
            self._check_padded(x_pad, x_list, grid_sizes)
        self.assertIs(volumes.get_grid_sizes(), volumes_updated.get_grid_sizes())
        self.assertIs(
            volumes.get_local_to_world_coords_transform(),
            volumes_updated.get_local_to_world_coords_transform(),
        )
        self.assertIs(volumes.device, volumes_updated.device)

    def test_constructor_for_padded_lists(self):
        """
        Tests constructor for padded/list representations.
        """

        device = torch.device("cuda:0")
        diff_device = torch.device("cpu")

        num_volumes = 3
        num_channels = 4
        size = (6, 8, 10)
        diff_size = (6, 8, 11)

        # good ways to define densities
        ok_densities = [
            torch.randn(
                size=[num_volumes, 1, *size], device=device, dtype=torch.float32
            ).unbind(0),
            torch.randn(
                size=[num_volumes, 1, *size], device=device, dtype=torch.float32
            ),
        ]

        # bad ways to define features
        bad_features = [
            torch.randn(
                size=[num_volumes + 1, num_channels, *size],
                device=device,
                dtype=torch.float32,
            ).unbind(
                0
            ),  # list with diff batch size
            torch.randn(
                size=[num_volumes + 1, num_channels, *size],
                device=device,
                dtype=torch.float32,
            ),  # diff batch size
            torch.randn(
                size=[num_volumes, num_channels, *diff_size],
                device=device,
                dtype=torch.float32,
            ).unbind(
                0
            ),  # list with different size
            torch.randn(
                size=[num_volumes, num_channels, *diff_size],
                device=device,
                dtype=torch.float32,
            ),  # different size
            torch.randn(
                size=[num_volumes, num_channels, *size],
                device=diff_device,
                dtype=torch.float32,
            ),  # different device
            torch.randn(
                size=[num_volumes, num_channels, *size],
                device=diff_device,
                dtype=torch.float32,
            ).unbind(
                0
            ),  # list with different device
        ]

        # good ways to define features
        ok_features = [
            torch.randn(
                size=[num_volumes, num_channels, *size],
                device=device,
                dtype=torch.float32,
            ).unbind(
                0
            ),  # list of features of correct size
            torch.randn(
                size=[num_volumes, num_channels, *size],
                device=device,
                dtype=torch.float32,
            ),
        ]

        for densities in ok_densities:
            for features in bad_features:
                self.assertRaises(
                    ValueError, Volumes, densities=densities, features=features
                )
            for features in ok_features:
                Volumes(densities=densities, features=features)

    def test_constructor(
        self, num_volumes=3, num_channels=4, size=(6, 8, 10), dtype=torch.float32
    ):
        """
        Test different ways of calling the `Volumes` constructor
        """

        device = torch.device("cuda:0")

        # all ways to define features
        features = [
            torch.randn(
                size=[num_volumes, num_channels, *size],
                device=device,
                dtype=torch.float32,
            ),  # padded tensor
            torch.randn(
                size=[num_volumes, num_channels, *size],
                device=device,
                dtype=torch.float32,
            ).unbind(
                0
            ),  # list of features
            None,  # no features
        ]

        # bad ways to define features
        bad_features = [
            torch.randn(
                size=[num_volumes, num_channels, 2, *size],
                device=device,
                dtype=torch.float32,
            ),  # 6 dims
            torch.randn(
                size=[num_volumes, *size], device=device, dtype=torch.float32
            ),  # 4 dims
            torch.randn(
                size=[num_volumes, *size], device=device, dtype=torch.float32
            ).unbind(
                0
            ),  # list of 4 dim tensors
        ]

        # all ways to define densities
        densities = [
            torch.randn(
                size=[num_volumes, 1, *size], device=device, dtype=torch.float32
            ),  # padded tensor
            torch.randn(
                size=[num_volumes, 1, *size], device=device, dtype=torch.float32
            ).unbind(
                0
            ),  # list of densities
        ]

        # bad ways to define densities
        bad_densities = [
            None,  # omitted
            torch.randn(
                size=[num_volumes, 1, 1, *size], device=device, dtype=torch.float32
            ),  # 6-dim tensor
            torch.randn(
                size=[num_volumes, 1, 1, *size], device=device, dtype=torch.float32
            ).unbind(
                0
            ),  # list of 5-dim densities
        ]

        # all possible ways to define the voxels sizes
        vox_sizes = [
            torch.Tensor([1.0, 1.0, 1.0]),
            [1.0, 1.0, 1.0],
            torch.Tensor([1.0, 1.0, 1.0])[None].repeat(num_volumes, 1),
            torch.Tensor([1.0])[None].repeat(num_volumes, 1),
            1.0,
            torch.Tensor([1.0]),
        ]

        # all possible ways to define the volume translations
        vol_translations = [
            torch.Tensor([1.0, 1.0, 1.0]),
            [1.0, 1.0, 1.0],
            torch.Tensor([1.0, 1.0, 1.0])[None].repeat(num_volumes, 1),
        ]

        # wrong ways to define voxel sizes
        bad_vox_sizes = [
            torch.Tensor([1.0, 1.0, 1.0, 1.0]),
            [1.0, 1.0, 1.0, 1.0],
            torch.Tensor([]),
            None,
        ]

        # wrong ways to define the volume translations
        bad_vol_translations = [
            torch.Tensor([1.0, 1.0]),
            [1.0, 1.0],
            1.0,
            torch.Tensor([1.0, 1.0, 1.0])[None].repeat(num_volumes + 1, 1),
        ]

        def zip_with_ok_indicator(good, bad):
            return zip([*good, *bad], [*([True] * len(good)), *([False] * len(bad))])

        for features_, features_ok in zip_with_ok_indicator(features, bad_features):
            for densities_, densities_ok in zip_with_ok_indicator(
                densities, bad_densities
            ):
                for vox_size, size_ok in zip_with_ok_indicator(
                    vox_sizes, bad_vox_sizes
                ):
                    for vol_translation, trans_ok in zip_with_ok_indicator(
                        vol_translations, bad_vol_translations
                    ):
                        if (
                            size_ok and trans_ok and features_ok and densities_ok
                        ):  # if all entries are good we check that this doesnt throw
                            Volumes(
                                features=features_,
                                densities=densities_,
                                voxel_size=vox_size,
                                volume_translation=vol_translation,
                            )

                        else:  # otherwise we check for ValueError
                            self.assertRaises(
                                ValueError,
                                Volumes,
                                features=features_,
                                densities=densities_,
                                voxel_size=vox_size,
                                volume_translation=vol_translation,
                            )
