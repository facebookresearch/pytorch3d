# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Callable

import torch
from pytorch3d.common.compat import meshgrid_ij
from pytorch3d.ops import eyes
from pytorch3d.renderer import (
    MonteCarloRaysampler,
    MultinomialRaysampler,
    NDCGridRaysampler,
    NDCMultinomialRaysampler,
)
from pytorch3d.renderer.cameras import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    OrthographicCameras,
    PerspectiveCameras,
)
from pytorch3d.renderer.implicit.raysampling import (
    _jiggle_within_stratas,
    _safe_multinomial,
)
from pytorch3d.renderer.implicit.utils import (
    ray_bundle_to_ray_points,
    ray_bundle_variables_to_ray_points,
)
from pytorch3d.transforms import Rotate

from .common_testing import TestCaseMixin
from .test_cameras import init_random_cameras


class TestNDCRaysamplerConvention(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_ndc_convention(
        self,
        h=428,
        w=760,
    ):
        device = torch.device("cuda")

        camera = init_random_cameras(PerspectiveCameras, 1, random_z=True).to(device)

        depth_map = torch.ones((1, 1, h, w)).to(device)

        xyz = ray_bundle_to_ray_points(
            NDCGridRaysampler(
                image_width=w,
                image_height=h,
                n_pts_per_ray=1,
                min_depth=1.0,
                max_depth=1.0,
            )(camera)._replace(lengths=depth_map[:, 0, ..., None])
        ).view(1, -1, 3)

        # project pointcloud
        xy = camera.transform_points(xyz)[:, :, :2].squeeze()

        xy_grid = self._get_ndc_grid(h, w, device)

        self.assertClose(
            xy,
            xy_grid,
            atol=1e-4,
        )

    def _get_ndc_grid(self, h, w, device):
        if w >= h:
            range_x = w / h
            range_y = 1.0
        else:
            range_x = 1.0
            range_y = h / w

        half_pix_width = range_x / w
        half_pix_height = range_y / h

        min_x = range_x - half_pix_width
        max_x = -range_x + half_pix_width
        min_y = range_y - half_pix_height
        max_y = -range_y + half_pix_height

        y_grid, x_grid = meshgrid_ij(
            torch.linspace(min_y, max_y, h, dtype=torch.float32),
            torch.linspace(min_x, max_x, w, dtype=torch.float32),
        )

        x_points = x_grid.contiguous().view(-1).to(device)
        y_points = y_grid.contiguous().view(-1).to(device)
        xy = torch.stack((x_points, y_points), dim=1)
        return xy


class TestRaysampling(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    @staticmethod
    def raysampler(
        raysampler_type,
        camera_type,
        n_pts_per_ray: int,
        batch_size: int,
        image_width: int,
        image_height: int,
    ) -> Callable[[], None]:
        """
        Used for benchmarks.
        """
        device = torch.device("cuda")

        # init raysamplers
        raysampler = TestRaysampling.init_raysampler(
            raysampler_type=raysampler_type,
            min_x=-1.0,
            max_x=1.0,
            min_y=-1.0,
            max_y=1.0,
            image_width=image_width,
            image_height=image_height,
            min_depth=1.0,
            max_depth=10.0,
            n_pts_per_ray=n_pts_per_ray,
        ).to(device)

        # init a batch of random cameras
        cameras = init_random_cameras(camera_type, batch_size, random_z=True).to(device)

        def run_raysampler() -> None:
            raysampler(cameras=cameras)
            torch.cuda.synchronize()

        return run_raysampler

    @staticmethod
    def init_raysampler(
        raysampler_type,
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        image_width=10,
        image_height=20,
        min_depth=1.0,
        max_depth=10.0,
        n_pts_per_ray=10,
        n_rays_total=None,
        n_rays_per_image=None,
    ):
        raysampler_params = {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "n_pts_per_ray": n_pts_per_ray,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "n_rays_total": n_rays_total,
            "n_rays_per_image": n_rays_per_image,
        }

        if issubclass(raysampler_type, MultinomialRaysampler):
            raysampler_params.update(
                {"image_width": image_width, "image_height": image_height}
            )
        elif issubclass(raysampler_type, MonteCarloRaysampler):
            raysampler_params["n_rays_per_image"] = (
                image_width * image_height
                if (n_rays_total is None) and (n_rays_per_image is None)
                else n_rays_per_image
            )
        else:
            raise ValueError(str(raysampler_type))

        if issubclass(raysampler_type, NDCMultinomialRaysampler):
            # NDCGridRaysampler does not use min/max_x/y
            for k in ("min_x", "max_x", "min_y", "max_y"):
                del raysampler_params[k]

        # instantiate the raysampler
        raysampler = raysampler_type(**raysampler_params)

        return raysampler

    def test_raysamplers(
        self,
        batch_size=25,
        min_x=-1.0,
        max_x=1.0,
        min_y=-1.0,
        max_y=1.0,
        image_width=10,
        image_height=20,
        min_depth=1.0,
        max_depth=10.0,
    ):
        """
        Tests the shapes and outputs of MC and GridRaysamplers for randomly
        generated cameras and different numbers of points per ray.
        """

        device = torch.device("cuda")

        for n_pts_per_ray in (100, 1):

            for raysampler_type in (
                MonteCarloRaysampler,
                MultinomialRaysampler,
                NDCMultinomialRaysampler,
            ):

                raysampler = TestRaysampling.init_raysampler(
                    raysampler_type=raysampler_type,
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    image_width=image_width,
                    image_height=image_height,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    n_pts_per_ray=n_pts_per_ray,
                )

                if issubclass(raysampler_type, NDCMultinomialRaysampler):
                    # adjust the gt bounds for NDCGridRaysampler
                    if image_width >= image_height:
                        range_x = image_width / image_height
                        range_y = 1.0
                    else:
                        range_x = 1.0
                        range_y = image_height / image_width

                    half_pix_width = range_x / image_width
                    half_pix_height = range_y / image_height
                    min_x_ = range_x - half_pix_width
                    max_x_ = -range_x + half_pix_width
                    min_y_ = range_y - half_pix_height
                    max_y_ = -range_y + half_pix_height
                else:
                    min_x_ = min_x
                    max_x_ = max_x
                    min_y_ = min_y
                    max_y_ = max_y

                # carry out the test over several camera types
                for cam_type in (
                    FoVPerspectiveCameras,
                    FoVOrthographicCameras,
                    OrthographicCameras,
                    PerspectiveCameras,
                ):

                    # init a batch of random cameras
                    cameras = init_random_cameras(
                        cam_type, batch_size, random_z=True
                    ).to(device)

                    # call the raysampler
                    ray_bundle = raysampler(cameras=cameras)

                    # check the shapes of the raysampler outputs
                    self._check_raysampler_output_shapes(
                        raysampler,
                        ray_bundle,
                        batch_size,
                        image_width,
                        image_height,
                        n_pts_per_ray,
                    )

                    # check the points sampled along each ray
                    self._check_raysampler_ray_points(
                        raysampler,
                        cameras,
                        ray_bundle,
                        min_x_,
                        max_x_,
                        min_y_,
                        max_y_,
                        image_width,
                        image_height,
                        min_depth,
                        max_depth,
                    )

                    # check the output direction vectors
                    self._check_raysampler_ray_directions(
                        cameras, raysampler, ray_bundle
                    )

    def _check_grid_shape(self, grid, batch_size, spatial_size, n_pts_per_ray, dim):
        """
        A helper for checking the desired size of a variable output by a raysampler.
        """
        tgt_shape = [
            x for x in [batch_size, *spatial_size, n_pts_per_ray, dim] if x > 0
        ]
        self.assertTrue(all(sz1 == sz2 for sz1, sz2 in zip(grid.shape, tgt_shape)))

    def _check_raysampler_output_shapes(
        self,
        raysampler,
        ray_bundle,
        batch_size,
        image_width,
        image_height,
        n_pts_per_ray,
    ):
        """
        Checks the shapes of raysampler outputs.
        """

        if isinstance(raysampler, MultinomialRaysampler):
            spatial_size = [image_height, image_width]
        elif isinstance(raysampler, MonteCarloRaysampler):
            spatial_size = [image_height * image_width]
        else:
            raise ValueError(str(type(raysampler)))

        self._check_grid_shape(ray_bundle.xys, batch_size, spatial_size, 0, 2)
        self._check_grid_shape(ray_bundle.origins, batch_size, spatial_size, 0, 3)
        self._check_grid_shape(ray_bundle.directions, batch_size, spatial_size, 0, 3)
        self._check_grid_shape(
            ray_bundle.lengths, batch_size, spatial_size, n_pts_per_ray, 0
        )

    def _check_raysampler_ray_points(
        self,
        raysampler,
        cameras,
        ray_bundle,
        min_x,
        max_x,
        min_y,
        max_y,
        image_width,
        image_height,
        min_depth,
        max_depth,
    ):
        """
        Check rays_points_world and rays_zs outputs of raysamplers.
        """

        batch_size = cameras.R.shape[0]

        # convert to ray points
        rays_points_world = ray_bundle_variables_to_ray_points(
            ray_bundle.origins, ray_bundle.directions, ray_bundle.lengths
        )
        n_pts_per_ray = rays_points_world.shape[-2]

        # check that the outputs if ray_bundle_variables_to_ray_points and
        # ray_bundle_to_ray_points match
        rays_points_world_ = ray_bundle_to_ray_points(ray_bundle)
        self.assertClose(rays_points_world, rays_points_world_)

        # check that the depth of each ray point in camera coords
        # matches the expected linearly-spaced depth
        depth_expected = torch.linspace(
            min_depth,
            max_depth,
            n_pts_per_ray,
            dtype=torch.float32,
            device=rays_points_world.device,
        )
        ray_points_camera = (
            cameras.get_world_to_view_transform()
            .transform_points(rays_points_world.view(batch_size, -1, 3))
            .view(batch_size, -1, n_pts_per_ray, 3)
        )
        self.assertClose(
            ray_points_camera[..., 2],
            depth_expected[None, None, :].expand_as(ray_points_camera[..., 2]),
            atol=1e-4,
        )

        # check also that rays_zs is consistent with depth_expected
        self.assertClose(
            ray_bundle.lengths.view(batch_size, -1, n_pts_per_ray),
            depth_expected[None, None, :].expand_as(ray_points_camera[..., 2]),
            atol=1e-6,
        )

        # project the world ray points back to screen space
        ray_points_projected = cameras.transform_points(
            rays_points_world.view(batch_size, -1, 3)
        ).view(rays_points_world.shape)

        # check that ray_xy matches rays_points_projected xy
        rays_xy_projected = ray_points_projected[..., :2].view(
            batch_size, -1, n_pts_per_ray, 2
        )
        self.assertClose(
            ray_bundle.xys.view(batch_size, -1, 1, 2).expand_as(rays_xy_projected),
            rays_xy_projected,
            atol=1e-4,
        )

        # check that projected world points' xy coordinates
        # range correctly between [minx/y, max/y]
        if isinstance(raysampler, MultinomialRaysampler):
            # get the expected coordinates along each grid axis
            ys, xs = [
                torch.linspace(
                    low, high, sz, dtype=torch.float32, device=rays_points_world.device
                )
                for low, high, sz in (
                    (min_y, max_y, image_height),
                    (min_x, max_x, image_width),
                )
            ]
            # compare expected xy with the output xy
            for dim, gt_axis in zip(
                (0, 1), (xs[None, None, :, None], ys[None, :, None, None])
            ):
                self.assertClose(
                    ray_points_projected[..., dim],
                    gt_axis.expand_as(ray_points_projected[..., dim]),
                    atol=1e-4,
                )

        elif isinstance(raysampler, MonteCarloRaysampler):
            # check that the randomly sampled locations
            # are within the allowed bounds for both x and y axes
            for dim, axis_bounds in zip((0, 1), ((min_x, max_x), (min_y, max_y))):
                self.assertTrue(
                    (
                        (ray_points_projected[..., dim] <= axis_bounds[1])
                        & (ray_points_projected[..., dim] >= axis_bounds[0])
                    ).all()
                )

                # also check that x,y along each ray is constant
                if n_pts_per_ray > 1:
                    self.assertClose(
                        ray_points_projected[..., :2].std(dim=-2),
                        torch.zeros_like(ray_points_projected[..., 0, :2]),
                        atol=1e-5,
                    )

        else:
            raise ValueError(str(type(raysampler)))

    def _check_raysampler_ray_directions(self, cameras, raysampler, ray_bundle):
        """
        Check the rays_directions_world output of raysamplers.
        """

        batch_size = cameras.R.shape[0]
        n_pts_per_ray = ray_bundle.lengths.shape[-1]
        spatial_size = ray_bundle.xys.shape[1:-1]
        n_rays_per_image = spatial_size.numel()

        # obtain the ray points in world coords
        rays_points_world = cameras.unproject_points(
            torch.cat(
                (
                    ray_bundle.xys.view(batch_size, n_rays_per_image, 1, 2).expand(
                        batch_size, n_rays_per_image, n_pts_per_ray, 2
                    ),
                    ray_bundle.lengths.view(
                        batch_size, n_rays_per_image, n_pts_per_ray, 1
                    ),
                ),
                dim=-1,
            ).view(batch_size, -1, 3)
        ).view(batch_size, -1, n_pts_per_ray, 3)

        # reshape to common testing size
        rays_directions_world_normed = torch.nn.functional.normalize(
            ray_bundle.directions.view(batch_size, -1, 3), dim=-1
        )

        # check that the l2-normed difference of all consecutive planes
        # of points in world coords matches ray_directions_world
        rays_directions_world_ = torch.nn.functional.normalize(
            rays_points_world[:, :, -1:] - rays_points_world[:, :, :-1], dim=-1
        )
        self.assertClose(
            rays_directions_world_normed[:, :, None].expand_as(rays_directions_world_),
            rays_directions_world_,
            atol=1e-4,
        )

        # check the ray directions rotated using camera rotation matrix
        # match the ray directions of a camera with trivial extrinsics
        cameras_trivial_extrinsic = cameras.clone()
        cameras_trivial_extrinsic.R = eyes(
            N=batch_size, dim=3, dtype=cameras.R.dtype, device=cameras.device
        )
        cameras_trivial_extrinsic.T = torch.zeros_like(cameras.T)

        # make sure we get the same random rays in case we call the
        # MonteCarloRaysampler twice below
        with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
            torch.random.manual_seed(42)
            ray_bundle_world_fix_seed = raysampler(cameras=cameras)
            torch.random.manual_seed(42)
            ray_bundle_camera_fix_seed = raysampler(cameras=cameras_trivial_extrinsic)

        rays_directions_camera_fix_seed_ = Rotate(
            cameras.R, device=cameras.R.device
        ).transform_points(ray_bundle_world_fix_seed.directions.view(batch_size, -1, 3))

        self.assertClose(
            rays_directions_camera_fix_seed_,
            ray_bundle_camera_fix_seed.directions.view(batch_size, -1, 3),
            atol=1e-5,
        )

    @unittest.skipIf(
        torch.__version__[:4] == "1.5.", "non persistent buffer needs PyTorch 1.6"
    )
    def test_load_state_different_resolution(self):
        # check that we can load the state of one ray sampler into
        # another with different image size.
        module1 = NDCGridRaysampler(
            image_width=20,
            image_height=30,
            n_pts_per_ray=40,
            min_depth=1.2,
            max_depth=2.3,
        )
        module2 = NDCGridRaysampler(
            image_width=22,
            image_height=32,
            n_pts_per_ray=42,
            min_depth=1.2,
            max_depth=2.3,
        )
        state = module1.state_dict()
        module2.load_state_dict(state)

    def test_jiggle(self):
        # random data which is in ascending order along the last dimension
        scale = 180
        data = scale * torch.cumsum(torch.rand(8, 3, 4, 20), dim=-1)

        out = _jiggle_within_stratas(data)
        self.assertTupleEqual(out.shape, data.shape)

        # Check `out` is in ascending order
        self.assertGreater((out[..., 1:] - out[..., :-1]).min(), 0)

        self.assertConstant(out[..., :-1] < data[..., 1:], True)
        self.assertConstant(data[..., :-1] < out[..., 1:], True)

        jiggles = out - data
        # jiggles is random between -scale/2 and scale/2
        self.assertLess(jiggles.min(), -0.4 * scale)
        self.assertGreater(jiggles.min(), -0.5 * scale)
        self.assertGreater(jiggles.max(), 0.4 * scale)
        self.assertLess(jiggles.max(), 0.5 * scale)

    def test_safe_multinomial(self):
        mask = [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ]
        tmask = torch.tensor(mask, dtype=torch.float32)

        for _ in range(5):
            random_scalar = torch.rand(1)
            samples = _safe_multinomial(tmask * random_scalar, 3)
            self.assertTupleEqual(samples.shape, (4, 3))

            # samples[0] is exactly determined
            self.assertConstant(samples[0], 0)

            self.assertGreaterEqual(samples[1].min(), 0)
            self.assertLessEqual(samples[1].max(), 1)

            # samples[2] is exactly determined
            self.assertSetEqual(set(samples[2].tolist()), {0, 1, 2})

            # samples[3] has enough sources, so must contain 3 distinct values.
            self.assertLessEqual(samples[3].max(), 3)
            self.assertEqual(len(set(samples[3].tolist())), 3)

    def test_heterogeneous_sampling(self, batch_size=8):
        """
        Test that the output of heterogeneous sampling has the first dimension equal
        to n_rays_total and second to 1 and that ray_bundle elements from different
        sampled cameras are different and equal for same sampled cameras.
        """
        cameras = init_random_cameras(PerspectiveCameras, batch_size, random_z=True)
        for n_rays_total in [2, 3, 17, 21, 32]:
            for cls in (MultinomialRaysampler, MonteCarloRaysampler):
                with self.subTest(cls.__name__ + ", n_rays_total=" + str(n_rays_total)):
                    raysampler = self.init_raysampler(
                        cls, n_rays_total=n_rays_total, n_rays_per_image=None
                    )
                    ray_bundle = raysampler(cameras)

                    # test weather they are of the correct shape
                    for attr in ("origins", "directions", "lengths", "xys"):
                        tensor = getattr(ray_bundle, attr)
                        assert tensor.shape[:2] == torch.Size(
                            (n_rays_total, 1)
                        ), tensor.shape

                    # if two camera ids are same than origins should also be the same
                    # directions and xys are always different and lengths equal
                    for i1, (origin1, dir1, len1, id1) in enumerate(
                        zip(
                            ray_bundle.origins,
                            ray_bundle.directions,
                            ray_bundle.lengths,
                            torch.repeat_interleave(
                                ray_bundle.camera_ids, ray_bundle.camera_counts
                            ),
                        )
                    ):
                        for i2, (origin2, dir2, len2, id2) in enumerate(
                            zip(
                                ray_bundle.origins,
                                ray_bundle.directions,
                                ray_bundle.lengths,
                                torch.repeat_interleave(
                                    ray_bundle.camera_ids, ray_bundle.camera_counts
                                ),
                            )
                        ):
                            if i1 == i2:
                                continue
                            assert torch.allclose(
                                origin1, origin2, rtol=1e-4, atol=1e-4
                            ) == (id1 == id2), (origin1, origin2, id1, id2)
                            assert not torch.allclose(dir1, dir2), (dir1, dir2)
                            self.assertClose(len1, len2), (len1, len2)
