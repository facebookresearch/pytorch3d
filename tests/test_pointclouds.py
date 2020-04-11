# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


import unittest

import numpy as np
import torch
from common_testing import TestCaseMixin
from pytorch3d.structures.pointclouds import Pointclouds


class TestPointclouds(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def init_cloud(
        num_clouds: int = 3,
        max_points: int = 100,
        channels: int = 4,
        lists_to_tensors: bool = False,
        with_normals: bool = True,
        with_features: bool = True,
    ):
        """
        Function to generate a Pointclouds object of N meshes with
        random number of points.

        Args:
            num_clouds: Number of clouds to generate.
            channels: Number of features.
            max_points: Max number of points per cloud.
            lists_to_tensors: Determines whether the generated clouds should be
                              constructed from lists (=False) or
                              tensors (=True) of points/normals/features.
            with_normals: bool whether to include normals
            with_features: bool whether to include features

        Returns:
            Pointclouds object.
        """
        device = torch.device("cuda:0")
        p = torch.randint(max_points, size=(num_clouds,))
        if lists_to_tensors:
            p.fill_(p[0])

        points_list = [
            torch.rand((i, 3), device=device, dtype=torch.float32) for i in p
        ]
        normals_list, features_list = None, None
        if with_normals:
            normals_list = [
                torch.rand((i, 3), device=device, dtype=torch.float32) for i in p
            ]
        if with_features:
            features_list = [
                torch.rand((i, channels), device=device, dtype=torch.float32) for i in p
            ]

        if lists_to_tensors:
            points_list = torch.stack(points_list)
            if with_normals:
                normals_list = torch.stack(normals_list)
            if with_features:
                features_list = torch.stack(features_list)

        return Pointclouds(points_list, normals=normals_list, features=features_list)

    def test_simple(self):
        device = torch.device("cuda:0")
        points = [
            torch.tensor(
                [[0.1, 0.3, 0.5], [0.5, 0.2, 0.1], [0.6, 0.8, 0.7]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [[0.1, 0.3, 0.3], [0.6, 0.7, 0.8], [0.2, 0.3, 0.4], [0.1, 0.5, 0.3]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [
                    [0.7, 0.3, 0.6],
                    [0.2, 0.4, 0.8],
                    [0.9, 0.5, 0.2],
                    [0.2, 0.3, 0.4],
                    [0.9, 0.3, 0.8],
                ],
                dtype=torch.float32,
                device=device,
            ),
        ]
        clouds = Pointclouds(points)

        self.assertClose(
            (clouds.packed_to_cloud_idx()).cpu(),
            torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]),
        )
        self.assertClose(
            clouds.cloud_to_packed_first_idx().cpu(), torch.tensor([0, 3, 7])
        )
        self.assertClose(clouds.num_points_per_cloud().cpu(), torch.tensor([3, 4, 5]))
        self.assertClose(
            clouds.padded_to_packed_idx().cpu(),
            torch.tensor([0, 1, 2, 5, 6, 7, 8, 10, 11, 12, 13, 14]),
        )

    def test_all_constructions(self):
        public_getters = [
            "points_list",
            "points_packed",
            "packed_to_cloud_idx",
            "cloud_to_packed_first_idx",
            "num_points_per_cloud",
            "points_padded",
            "padded_to_packed_idx",
        ]
        public_normals_getters = ["normals_list", "normals_packed", "normals_padded"]
        public_features_getters = [
            "features_list",
            "features_packed",
            "features_padded",
        ]

        lengths = [3, 4, 2]
        max_len = max(lengths)
        C = 4

        points_data = [torch.zeros((max_len, 3)).uniform_() for i in lengths]
        normals_data = [torch.zeros((max_len, 3)).uniform_() for i in lengths]
        features_data = [torch.zeros((max_len, C)).uniform_() for i in lengths]
        for length, p, n, f in zip(lengths, points_data, normals_data, features_data):
            p[length:] = 0.0
            n[length:] = 0.0
            f[length:] = 0.0
        points_list = [d[:length] for length, d in zip(lengths, points_data)]
        normals_list = [d[:length] for length, d in zip(lengths, normals_data)]
        features_list = [d[:length] for length, d in zip(lengths, features_data)]
        points_packed = torch.cat(points_data)
        normals_packed = torch.cat(normals_data)
        features_packed = torch.cat(features_data)
        test_cases_inputs = [
            ("list_0_0", points_list, None, None),
            ("list_1_0", points_list, normals_list, None),
            ("list_0_1", points_list, None, features_list),
            ("list_1_1", points_list, normals_list, features_list),
            ("padded_0_0", points_data, None, None),
            ("padded_1_0", points_data, normals_data, None),
            ("padded_0_1", points_data, None, features_data),
            ("padded_1_1", points_data, normals_data, features_data),
            ("emptylist_emptylist_emptylist", [], [], []),
        ]
        false_cases_inputs = [
            ("list_packed", points_list, normals_packed, features_packed, ValueError),
            ("packed_0", points_packed, None, None, ValueError),
        ]

        for name, points, normals, features in test_cases_inputs:
            with self.subTest(name=name):
                p = Pointclouds(points, normals, features)
                for method in public_getters:
                    self.assertIsNotNone(getattr(p, method)())
                for method in public_normals_getters:
                    if normals is None or p.isempty():
                        self.assertIsNone(getattr(p, method)())
                for method in public_features_getters:
                    if features is None or p.isempty():
                        self.assertIsNone(getattr(p, method)())

        for name, points, normals, features, error in false_cases_inputs:
            with self.subTest(name=name):
                with self.assertRaises(error):
                    Pointclouds(points, normals, features)

    def test_simple_random_clouds(self):
        # Define the test object either from lists or tensors.
        for with_normals in (False, True):
            for with_features in (False, True):
                for lists_to_tensors in (False, True):
                    N = 10
                    cloud = self.init_cloud(
                        N,
                        lists_to_tensors=lists_to_tensors,
                        with_normals=with_normals,
                        with_features=with_features,
                    )
                    points_list = cloud.points_list()
                    normals_list = cloud.normals_list()
                    features_list = cloud.features_list()

                    # Check batch calculations.
                    points_padded = cloud.points_padded()
                    normals_padded = cloud.normals_padded()
                    features_padded = cloud.features_padded()
                    points_per_cloud = cloud.num_points_per_cloud()

                    if not with_normals:
                        self.assertIsNone(normals_list)
                        self.assertIsNone(normals_padded)
                    if not with_features:
                        self.assertIsNone(features_list)
                        self.assertIsNone(features_padded)
                    for n in range(N):
                        p = points_list[n].shape[0]
                        self.assertClose(points_padded[n, :p, :], points_list[n])
                        if with_normals:
                            norms = normals_list[n].shape[0]
                            self.assertEqual(p, norms)
                            self.assertClose(normals_padded[n, :p, :], normals_list[n])
                        if with_features:
                            f = features_list[n].shape[0]
                            self.assertEqual(p, f)
                            self.assertClose(
                                features_padded[n, :p, :], features_list[n]
                            )
                        if points_padded.shape[1] > p:
                            self.assertTrue(points_padded[n, p:, :].eq(0).all())
                            if with_features:
                                self.assertTrue(features_padded[n, p:, :].eq(0).all())
                        self.assertEqual(points_per_cloud[n], p)

                    # Check compute packed.
                    points_packed = cloud.points_packed()
                    packed_to_cloud = cloud.packed_to_cloud_idx()
                    cloud_to_packed = cloud.cloud_to_packed_first_idx()
                    normals_packed = cloud.normals_packed()
                    features_packed = cloud.features_packed()
                    if not with_normals:
                        self.assertIsNone(normals_packed)
                    if not with_features:
                        self.assertIsNone(features_packed)

                    cur = 0
                    for n in range(N):
                        p = points_list[n].shape[0]
                        self.assertClose(
                            points_packed[cur : cur + p, :], points_list[n]
                        )
                        if with_normals:
                            self.assertClose(
                                normals_packed[cur : cur + p, :], normals_list[n]
                            )
                        if with_features:
                            self.assertClose(
                                features_packed[cur : cur + p, :], features_list[n]
                            )
                        self.assertTrue(packed_to_cloud[cur : cur + p].eq(n).all())
                        self.assertTrue(cloud_to_packed[n] == cur)
                        cur += p

    def test_allempty(self):
        clouds = Pointclouds([], [])
        self.assertEqual(len(clouds), 0)
        self.assertIsNone(clouds.normals_list())
        self.assertIsNone(clouds.features_list())
        self.assertEqual(clouds.points_padded().shape[0], 0)
        self.assertIsNone(clouds.normals_padded())
        self.assertIsNone(clouds.features_padded())
        self.assertEqual(clouds.points_packed().shape[0], 0)
        self.assertIsNone(clouds.normals_packed())
        self.assertIsNone(clouds.features_packed())

    def test_empty(self):
        N, P, C = 10, 100, 2
        device = torch.device("cuda:0")
        points_list = []
        normals_list = []
        features_list = []
        valid = torch.randint(2, size=(N,), dtype=torch.uint8, device=device)
        for n in range(N):
            if valid[n]:
                p = torch.randint(
                    3, high=P, size=(1,), dtype=torch.int32, device=device
                )[0]
                points = torch.rand((p, 3), dtype=torch.float32, device=device)
                normals = torch.rand((p, 3), dtype=torch.float32, device=device)
                features = torch.rand((p, C), dtype=torch.float32, device=device)
            else:
                points = torch.tensor([], dtype=torch.float32, device=device)
                normals = torch.tensor([], dtype=torch.float32, device=device)
                features = torch.tensor([], dtype=torch.int64, device=device)
            points_list.append(points)
            normals_list.append(normals)
            features_list.append(features)

        for with_normals in (False, True):
            for with_features in (False, True):
                this_features, this_normals = None, None
                if with_normals:
                    this_normals = normals_list
                if with_features:
                    this_features = features_list
                clouds = Pointclouds(
                    points=points_list, normals=this_normals, features=this_features
                )
                points_padded = clouds.points_padded()
                normals_padded = clouds.normals_padded()
                features_padded = clouds.features_padded()
                if not with_normals:
                    self.assertIsNone(normals_padded)
                if not with_features:
                    self.assertIsNone(features_padded)
                points_per_cloud = clouds.num_points_per_cloud()
                for n in range(N):
                    p = len(points_list[n])
                    if p > 0:
                        self.assertClose(points_padded[n, :p, :], points_list[n])
                        if with_normals:
                            self.assertClose(normals_padded[n, :p, :], normals_list[n])
                        if with_features:
                            self.assertClose(
                                features_padded[n, :p, :], features_list[n]
                            )
                        if points_padded.shape[1] > p:
                            self.assertTrue(points_padded[n, p:, :].eq(0).all())
                            if with_normals:
                                self.assertTrue(normals_padded[n, p:, :].eq(0).all())
                            if with_features:
                                self.assertTrue(features_padded[n, p:, :].eq(0).all())
                    self.assertTrue(points_per_cloud[n] == p)

    def test_clone_list(self):
        N = 5
        clouds = self.init_cloud(N, 100, 5)
        for force in (False, True):
            if force:
                clouds.points_packed()

            new_clouds = clouds.clone()

            # Check cloned and original objects do not share tensors.
            self.assertSeparate(new_clouds.points_list()[0], clouds.points_list()[0])
            self.assertSeparate(new_clouds.normals_list()[0], clouds.normals_list()[0])
            self.assertSeparate(
                new_clouds.features_list()[0], clouds.features_list()[0]
            )
            for attrib in [
                "points_packed",
                "normals_packed",
                "features_packed",
                "points_padded",
                "normals_padded",
                "features_padded",
            ]:
                self.assertSeparate(
                    getattr(new_clouds, attrib)(), getattr(clouds, attrib)()
                )

            self.assertCloudsEqual(clouds, new_clouds)

    def test_clone_tensor(self):
        N = 5
        clouds = self.init_cloud(N, 100, 5, lists_to_tensors=True)
        for force in (False, True):
            if force:
                clouds.points_packed()

            new_clouds = clouds.clone()

            # Check cloned and original objects do not share tensors.
            self.assertSeparate(new_clouds.points_list()[0], clouds.points_list()[0])
            self.assertSeparate(new_clouds.normals_list()[0], clouds.normals_list()[0])
            self.assertSeparate(
                new_clouds.features_list()[0], clouds.features_list()[0]
            )
            for attrib in [
                "points_packed",
                "normals_packed",
                "features_packed",
                "points_padded",
                "normals_padded",
                "features_padded",
            ]:
                self.assertSeparate(
                    getattr(new_clouds, attrib)(), getattr(clouds, attrib)()
                )

            self.assertCloudsEqual(clouds, new_clouds)

    def assertCloudsEqual(self, cloud1, cloud2):
        N = len(cloud1)
        self.assertEqual(N, len(cloud2))

        for i in range(N):
            self.assertClose(cloud1.points_list()[i], cloud2.points_list()[i])
            self.assertClose(cloud1.normals_list()[i], cloud2.normals_list()[i])
            self.assertClose(cloud1.features_list()[i], cloud2.features_list()[i])
        has_normals = cloud1.normals_list() is not None
        self.assertTrue(has_normals == (cloud2.normals_list() is not None))
        has_features = cloud1.features_list() is not None
        self.assertTrue(has_features == (cloud2.features_list() is not None))

        # check padded & packed
        self.assertClose(cloud1.points_padded(), cloud2.points_padded())
        self.assertClose(cloud1.points_packed(), cloud2.points_packed())
        if has_normals:
            self.assertClose(cloud1.normals_padded(), cloud2.normals_padded())
            self.assertClose(cloud1.normals_packed(), cloud2.normals_packed())
        if has_features:
            self.assertClose(cloud1.features_padded(), cloud2.features_padded())
            self.assertClose(cloud1.features_packed(), cloud2.features_packed())
        self.assertClose(cloud1.packed_to_cloud_idx(), cloud2.packed_to_cloud_idx())
        self.assertClose(
            cloud1.cloud_to_packed_first_idx(), cloud2.cloud_to_packed_first_idx()
        )
        self.assertClose(cloud1.num_points_per_cloud(), cloud2.num_points_per_cloud())
        self.assertClose(cloud1.packed_to_cloud_idx(), cloud2.packed_to_cloud_idx())
        self.assertClose(cloud1.padded_to_packed_idx(), cloud2.padded_to_packed_idx())
        self.assertTrue(all(cloud1.valid == cloud2.valid))
        self.assertTrue(cloud1.equisized == cloud2.equisized)

    def test_offset(self):
        def naive_offset(clouds, offsets_packed):
            new_points_packed = clouds.points_packed() + offsets_packed
            new_points_list = list(
                new_points_packed.split(clouds.num_points_per_cloud().tolist(), 0)
            )
            return Pointclouds(
                points=new_points_list,
                normals=clouds.normals_list(),
                features=clouds.features_list(),
            )

        N = 5
        clouds = self.init_cloud(N, 100, 10)
        all_p = clouds.points_packed().size(0)
        points_per_cloud = clouds.num_points_per_cloud()
        for force in (False, True):
            if force:
                clouds._compute_packed(refresh=True)
                clouds._compute_padded()
                clouds.padded_to_packed_idx()

            deform = torch.rand((all_p, 3), dtype=torch.float32, device=clouds.device)
            new_clouds_naive = naive_offset(clouds, deform)

            new_clouds = clouds.offset(deform)

            points_cumsum = torch.cumsum(points_per_cloud, 0).tolist()
            points_cumsum.insert(0, 0)
            for i in range(N):
                self.assertClose(
                    new_clouds.points_list()[i],
                    clouds.points_list()[i]
                    + deform[points_cumsum[i] : points_cumsum[i + 1]],
                )
                self.assertClose(
                    clouds.normals_list()[i], new_clouds_naive.normals_list()[i]
                )
                self.assertClose(
                    clouds.features_list()[i], new_clouds_naive.features_list()[i]
                )
            self.assertCloudsEqual(new_clouds, new_clouds_naive)

    def test_scale(self):
        def naive_scale(cloud, scale):
            if not torch.is_tensor(scale):
                scale = torch.full(len(cloud), scale)
            new_points_list = [
                scale[i] * points.clone()
                for (i, points) in enumerate(cloud.points_list())
            ]
            return Pointclouds(
                new_points_list, cloud.normals_list(), cloud.features_list()
            )

        N = 5
        clouds = self.init_cloud(N, 100, 10)
        for force in (False, True):
            if force:
                clouds._compute_packed(refresh=True)
                clouds._compute_padded()
                clouds.padded_to_packed_idx()
            scales = torch.rand(N)
            new_clouds_naive = naive_scale(clouds, scales)
            new_clouds = clouds.scale(scales)
            for i in range(N):
                self.assertClose(
                    scales[i] * clouds.points_list()[i], new_clouds.points_list()[i]
                )
                self.assertClose(
                    clouds.normals_list()[i], new_clouds_naive.normals_list()[i]
                )
                self.assertClose(
                    clouds.features_list()[i], new_clouds_naive.features_list()[i]
                )
            self.assertCloudsEqual(new_clouds, new_clouds_naive)

    def test_extend_list(self):
        N = 10
        clouds = self.init_cloud(N, 100, 10)
        for force in (False, True):
            if force:
                # force some computes to happen
                clouds._compute_packed(refresh=True)
                clouds._compute_padded()
                clouds.padded_to_packed_idx()
            new_clouds = clouds.extend(N)
            self.assertEqual(len(clouds) * 10, len(new_clouds))
            for i in range(len(clouds)):
                for n in range(N):
                    self.assertClose(
                        clouds.points_list()[i], new_clouds.points_list()[i * N + n]
                    )
                    self.assertClose(
                        clouds.normals_list()[i], new_clouds.normals_list()[i * N + n]
                    )
                    self.assertClose(
                        clouds.features_list()[i], new_clouds.features_list()[i * N + n]
                    )
                    self.assertTrue(clouds.valid[i] == new_clouds.valid[i * N + n])
            self.assertAllSeparate(
                clouds.points_list()
                + new_clouds.points_list()
                + clouds.normals_list()
                + new_clouds.normals_list()
                + clouds.features_list()
                + new_clouds.features_list()
            )
            self.assertIsNone(new_clouds._points_packed)
            self.assertIsNone(new_clouds._normals_packed)
            self.assertIsNone(new_clouds._features_packed)
            self.assertIsNone(new_clouds._points_padded)
            self.assertIsNone(new_clouds._normals_padded)
            self.assertIsNone(new_clouds._features_padded)

        with self.assertRaises(ValueError):
            clouds.extend(N=-1)

    def test_to_list(self):
        cloud = self.init_cloud(5, 100, 10)
        device = torch.device("cuda:1")

        new_cloud = cloud.to(device)
        self.assertTrue(new_cloud.device == device)
        self.assertTrue(cloud.device == torch.device("cuda:0"))
        for attrib in [
            "points_padded",
            "points_packed",
            "normals_padded",
            "normals_packed",
            "features_padded",
            "features_packed",
            "num_points_per_cloud",
            "cloud_to_packed_first_idx",
            "padded_to_packed_idx",
        ]:
            self.assertClose(
                getattr(new_cloud, attrib)().cpu(), getattr(cloud, attrib)().cpu()
            )
        for i in range(len(cloud)):
            self.assertClose(
                cloud.points_list()[i].cpu(), new_cloud.points_list()[i].cpu()
            )
            self.assertClose(
                cloud.normals_list()[i].cpu(), new_cloud.normals_list()[i].cpu()
            )
            self.assertClose(
                cloud.features_list()[i].cpu(), new_cloud.features_list()[i].cpu()
            )
        self.assertTrue(all(cloud.valid.cpu() == new_cloud.valid.cpu()))
        self.assertTrue(cloud.equisized == new_cloud.equisized)
        self.assertTrue(cloud._N == new_cloud._N)
        self.assertTrue(cloud._P == new_cloud._P)
        self.assertTrue(cloud._C == new_cloud._C)

    def test_to_tensor(self):
        cloud = self.init_cloud(5, 100, 10, lists_to_tensors=True)
        device = torch.device("cuda:1")

        new_cloud = cloud.to(device)
        self.assertTrue(new_cloud.device == device)
        self.assertTrue(cloud.device == torch.device("cuda:0"))
        for attrib in [
            "points_padded",
            "points_packed",
            "normals_padded",
            "normals_packed",
            "features_padded",
            "features_packed",
            "num_points_per_cloud",
            "cloud_to_packed_first_idx",
            "padded_to_packed_idx",
        ]:
            self.assertClose(
                getattr(new_cloud, attrib)().cpu(), getattr(cloud, attrib)().cpu()
            )
        for i in range(len(cloud)):
            self.assertClose(
                cloud.points_list()[i].cpu(), new_cloud.points_list()[i].cpu()
            )
            self.assertClose(
                cloud.normals_list()[i].cpu(), new_cloud.normals_list()[i].cpu()
            )
            self.assertClose(
                cloud.features_list()[i].cpu(), new_cloud.features_list()[i].cpu()
            )
        self.assertTrue(all(cloud.valid.cpu() == new_cloud.valid.cpu()))
        self.assertTrue(cloud.equisized == new_cloud.equisized)
        self.assertTrue(cloud._N == new_cloud._N)
        self.assertTrue(cloud._P == new_cloud._P)
        self.assertTrue(cloud._C == new_cloud._C)

    def test_split(self):
        clouds = self.init_cloud(5, 100, 10)
        split_sizes = [2, 3]
        split_clouds = clouds.split(split_sizes)
        self.assertEqual(len(split_clouds[0]), 2)
        self.assertTrue(
            split_clouds[0].points_list()
            == [clouds.get_cloud(0)[0], clouds.get_cloud(1)[0]]
        )
        self.assertEqual(len(split_clouds[1]), 3)
        self.assertTrue(
            split_clouds[1].points_list()
            == [clouds.get_cloud(2)[0], clouds.get_cloud(3)[0], clouds.get_cloud(4)[0]]
        )

        split_sizes = [2, 0.3]
        with self.assertRaises(ValueError):
            clouds.split(split_sizes)

    def test_get_cloud(self):
        clouds = self.init_cloud(2, 100, 10)
        for i in range(len(clouds)):
            points, normals, features = clouds.get_cloud(i)
            self.assertClose(points, clouds.points_list()[i])
            self.assertClose(normals, clouds.normals_list()[i])
            self.assertClose(features, clouds.features_list()[i])

        with self.assertRaises(ValueError):
            clouds.get_cloud(5)
        with self.assertRaises(ValueError):
            clouds.get_cloud(0.2)

    def test_get_bounding_boxes(self):
        device = torch.device("cuda:0")
        points_list = []
        for size in [10]:
            points = torch.rand((size, 3), dtype=torch.float32, device=device)
            points_list.append(points)

        mins = torch.min(points, dim=0)[0]
        maxs = torch.max(points, dim=0)[0]
        bboxes_gt = torch.stack([mins, maxs], dim=1).unsqueeze(0)
        clouds = Pointclouds(points_list)
        bboxes = clouds.get_bounding_boxes()
        self.assertClose(bboxes_gt, bboxes)

    def test_padded_to_packed_idx(self):
        device = torch.device("cuda:0")
        points_list = []
        npoints = [10, 20, 30]
        for p in npoints:
            points = torch.rand((p, 3), dtype=torch.float32, device=device)
            points_list.append(points)

        clouds = Pointclouds(points_list)

        padded_to_packed_idx = clouds.padded_to_packed_idx()
        points_packed = clouds.points_packed()
        points_padded = clouds.points_padded()
        points_padded_flat = points_padded.view(-1, 3)

        self.assertClose(points_padded_flat[padded_to_packed_idx], points_packed)

        idx = padded_to_packed_idx.view(-1, 1).expand(-1, 3)
        self.assertClose(points_padded_flat.gather(0, idx), points_packed)

    def test_getitem(self):
        device = torch.device("cuda:0")
        clouds = self.init_cloud(3, 10, 100)

        def check_equal(selected, indices):
            for selectedIdx, index in indices:
                self.assertClose(
                    selected.points_list()[selectedIdx], clouds.points_list()[index]
                )
                self.assertClose(
                    selected.normals_list()[selectedIdx], clouds.normals_list()[index]
                )
                self.assertClose(
                    selected.features_list()[selectedIdx], clouds.features_list()[index]
                )

        # int index
        index = 1
        clouds_selected = clouds[index]
        self.assertEqual(len(clouds_selected), 1)
        check_equal(clouds_selected, [(0, 1)])

        # list index
        index = [1, 2]
        clouds_selected = clouds[index]
        self.assertEqual(len(clouds_selected), len(index))
        check_equal(clouds_selected, enumerate(index))

        # slice index
        index = slice(0, 2, 1)
        clouds_selected = clouds[index]
        self.assertEqual(len(clouds_selected), 2)
        check_equal(clouds_selected, [(0, 0), (1, 1)])

        # bool tensor
        index = torch.tensor([1, 0, 1], dtype=torch.bool, device=device)
        clouds_selected = clouds[index]
        self.assertEqual(len(clouds_selected), index.sum())
        check_equal(clouds_selected, [(0, 0), (1, 2)])

        # int tensor
        index = torch.tensor([1, 2], dtype=torch.int64, device=device)
        clouds_selected = clouds[index]
        self.assertEqual(len(clouds_selected), index.numel())
        check_equal(clouds_selected, enumerate(index.tolist()))

        # invalid index
        index = torch.tensor([1, 0, 1], dtype=torch.float32, device=device)
        with self.assertRaises(IndexError):
            clouds_selected = clouds[index]
        index = 1.2
        with self.assertRaises(IndexError):
            clouds_selected = clouds[index]

    def test_update_padded(self):
        N, P, C = 5, 100, 4
        for with_normfeat in (True, False):
            for with_new_normfeat in (True, False):
                clouds = self.init_cloud(
                    N, P, C, with_normals=with_normfeat, with_features=with_normfeat
                )

                num_points_per_cloud = clouds.num_points_per_cloud()

                # initialize new points, normals, features
                new_points = torch.rand(
                    clouds.points_padded().shape, device=clouds.device
                )
                new_points_list = [
                    new_points[i, : num_points_per_cloud[i]] for i in range(N)
                ]
                new_normals, new_normals_list = None, None
                new_features, new_features_list = None, None
                if with_new_normfeat:
                    new_normals = torch.rand(
                        clouds.points_padded().shape, device=clouds.device
                    )
                    new_normals_list = [
                        new_normals[i, : num_points_per_cloud[i]] for i in range(N)
                    ]
                    feat_shape = [
                        clouds.points_padded().shape[0],
                        clouds.points_padded().shape[1],
                        C,
                    ]
                    new_features = torch.rand(feat_shape, device=clouds.device)
                    new_features_list = [
                        new_features[i, : num_points_per_cloud[i]] for i in range(N)
                    ]

                # update
                new_clouds = clouds.update_padded(new_points, new_normals, new_features)
                self.assertIsNone(new_clouds._points_list)
                self.assertIsNone(new_clouds._points_packed)

                self.assertEqual(new_clouds.equisized, clouds.equisized)
                self.assertTrue(all(new_clouds.valid == clouds.valid))

                self.assertClose(new_clouds.points_padded(), new_points)
                self.assertClose(new_clouds.points_packed(), torch.cat(new_points_list))
                for i in range(N):
                    self.assertClose(new_clouds.points_list()[i], new_points_list[i])

                if with_new_normfeat:
                    for i in range(N):
                        self.assertClose(
                            new_clouds.normals_list()[i], new_normals_list[i]
                        )
                        self.assertClose(
                            new_clouds.features_list()[i], new_features_list[i]
                        )
                    self.assertClose(new_clouds.normals_padded(), new_normals)
                    self.assertClose(
                        new_clouds.normals_packed(), torch.cat(new_normals_list)
                    )
                    self.assertClose(new_clouds.features_padded(), new_features)
                    self.assertClose(
                        new_clouds.features_packed(), torch.cat(new_features_list)
                    )
                else:
                    if with_normfeat:
                        for i in range(N):
                            self.assertClose(
                                new_clouds.normals_list()[i], clouds.normals_list()[i]
                            )
                            self.assertClose(
                                new_clouds.features_list()[i], clouds.features_list()[i]
                            )
                            self.assertNotSeparate(
                                new_clouds.normals_list()[i], clouds.normals_list()[i]
                            )
                            self.assertNotSeparate(
                                new_clouds.features_list()[i], clouds.features_list()[i]
                            )

                        self.assertClose(
                            new_clouds.normals_padded(), clouds.normals_padded()
                        )
                        self.assertClose(
                            new_clouds.normals_packed(), clouds.normals_packed()
                        )
                        self.assertClose(
                            new_clouds.features_padded(), clouds.features_padded()
                        )
                        self.assertClose(
                            new_clouds.features_packed(), clouds.features_packed()
                        )
                        self.assertNotSeparate(
                            new_clouds.normals_padded(), clouds.normals_padded()
                        )
                        self.assertNotSeparate(
                            new_clouds.features_padded(), clouds.features_padded()
                        )
                    else:
                        self.assertIsNone(new_clouds.normals_list())
                        self.assertIsNone(new_clouds.features_list())
                        self.assertIsNone(new_clouds.normals_padded())
                        self.assertIsNone(new_clouds.features_padded())
                        self.assertIsNone(new_clouds.normals_packed())
                        self.assertIsNone(new_clouds.features_packed())

                for attrib in [
                    "num_points_per_cloud",
                    "cloud_to_packed_first_idx",
                    "padded_to_packed_idx",
                ]:
                    self.assertClose(
                        getattr(new_clouds, attrib)(), getattr(clouds, attrib)()
                    )

    def test_inside_box(self):
        def inside_box_naive(cloud, box_min, box_max):
            return (cloud >= box_min.view(1, 3)) * (cloud <= box_max.view(1, 3))

        N, P, C = 5, 100, 4

        clouds = self.init_cloud(N, P, C, with_normals=False, with_features=False)
        device = clouds.device

        # box of shape Nx2x3
        box_min = torch.rand((N, 1, 3), device=device)
        box_max = box_min + torch.rand((N, 1, 3), device=device)
        box = torch.cat([box_min, box_max], dim=1)

        within_box = clouds.inside_box(box)

        within_box_naive = []
        for i, cloud in enumerate(clouds.points_list()):
            within_box_naive.append(inside_box_naive(cloud, box[i, 0], box[i, 1]))
        within_box_naive = torch.cat(within_box_naive, 0)
        self.assertTrue(within_box.eq(within_box_naive).all())

        # box of shape 2x3
        box2 = box[0, :]

        within_box2 = clouds.inside_box(box2)

        within_box_naive2 = []
        for cloud in clouds.points_list():
            within_box_naive2.append(inside_box_naive(cloud, box2[0], box2[1]))
        within_box_naive2 = torch.cat(within_box_naive2, 0)
        self.assertTrue(within_box2.eq(within_box_naive2).all())

        # box of shape 1x2x3
        box3 = box2.expand(1, 2, 3)

        within_box3 = clouds.inside_box(box3)
        self.assertTrue(within_box2.eq(within_box3).all())

        # invalid box
        invalid_box = torch.cat(
            [box_min, box_min - torch.rand((N, 1, 3), device=device)], dim=1
        )
        with self.assertRaisesRegex(ValueError, "Input box is invalid"):
            clouds.inside_box(invalid_box)

        # invalid box shapes
        invalid_box = box[0].expand(2, 2, 3)
        with self.assertRaisesRegex(ValueError, "Input box dimension is"):
            clouds.inside_box(invalid_box)
        invalid_box = torch.rand((5, 8, 9, 3), device=device)
        with self.assertRaisesRegex(ValueError, "Input box must be of shape"):
            clouds.inside_box(invalid_box)

    @staticmethod
    def compute_packed_with_init(
        num_clouds: int = 10, max_p: int = 100, features: int = 300
    ):
        clouds = TestPointclouds.init_cloud(num_clouds, max_p, features)
        torch.cuda.synchronize()

        def compute_packed():
            clouds._compute_packed(refresh=True)
            torch.cuda.synchronize()

        return compute_packed

    @staticmethod
    def compute_padded_with_init(
        num_clouds: int = 10, max_p: int = 100, features: int = 300
    ):
        clouds = TestPointclouds.init_cloud(num_clouds, max_p, features)
        torch.cuda.synchronize()

        def compute_padded():
            clouds._compute_padded(refresh=True)
            torch.cuda.synchronize()

        return compute_padded
