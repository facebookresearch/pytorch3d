# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import random
import unittest
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from pytorch3d.io import save_obj
from pytorch3d.ops.iou_box3d import _box_planes, _box_triangles, box3d_overlap
from pytorch3d.transforms.rotation_conversions import random_rotation

from .common_testing import get_random_cuda_device, get_tests_dir, TestCaseMixin


OBJECTRON_TO_PYTORCH3D_FACE_IDX = [0, 4, 6, 2, 1, 5, 7, 3]
DATA_DIR = get_tests_dir() / "data"
DEBUG = False
DOT_EPS = 1e-3
AREA_EPS = 1e-4

UNIT_BOX = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]


class TestIoU3D(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    @staticmethod
    def create_box(xyz, whl):
        x, y, z = xyz
        w, h, le = whl

        verts = torch.tensor(
            [
                [x - w / 2.0, y - h / 2.0, z - le / 2.0],
                [x + w / 2.0, y - h / 2.0, z - le / 2.0],
                [x + w / 2.0, y + h / 2.0, z - le / 2.0],
                [x - w / 2.0, y + h / 2.0, z - le / 2.0],
                [x - w / 2.0, y - h / 2.0, z + le / 2.0],
                [x + w / 2.0, y - h / 2.0, z + le / 2.0],
                [x + w / 2.0, y + h / 2.0, z + le / 2.0],
                [x - w / 2.0, y + h / 2.0, z + le / 2.0],
            ],
            device=xyz.device,
            dtype=torch.float32,
        )
        return verts

    @staticmethod
    def _box3d_overlap_naive_batched(boxes1, boxes2):
        """
        Wrapper around box3d_overlap_naive to support
        batched input
        """
        N = boxes1.shape[0]
        M = boxes2.shape[0]
        vols = torch.zeros((N, M), dtype=torch.float32, device=boxes1.device)
        ious = torch.zeros((N, M), dtype=torch.float32, device=boxes1.device)
        for n in range(N):
            for m in range(M):
                vol, iou = box3d_overlap_naive(boxes1[n], boxes2[m])
                vols[n, m] = vol
                ious[n, m] = iou
        return vols, ious

    @staticmethod
    def _box3d_overlap_sampling_batched(boxes1, boxes2, num_samples: int):
        """
        Wrapper around box3d_overlap_sampling to support
        batched input
        """
        N = boxes1.shape[0]
        M = boxes2.shape[0]
        ious = torch.zeros((N, M), dtype=torch.float32, device=boxes1.device)
        for n in range(N):
            for m in range(M):
                iou = box3d_overlap_sampling(boxes1[n], boxes2[m])
                ious[n, m] = iou
        return ious

    def _test_iou(self, overlap_fn, device):

        box1 = torch.tensor(
            UNIT_BOX,
            dtype=torch.float32,
            device=device,
        )

        # 1st test: same box, iou = 1.0
        vol, iou = overlap_fn(box1[None], box1[None])
        self.assertClose(vol, torch.tensor([[1.0]], device=vol.device, dtype=vol.dtype))
        self.assertClose(iou, torch.tensor([[1.0]], device=vol.device, dtype=vol.dtype))

        # 2nd test
        dd = random.random()
        box2 = box1 + torch.tensor([[0.0, dd, 0.0]], device=device)
        vol, iou = overlap_fn(box1[None], box2[None])
        self.assertClose(
            vol, torch.tensor([[1 - dd]], device=vol.device, dtype=vol.dtype)
        )
        # symmetry
        vol, iou = overlap_fn(box2[None], box1[None])
        self.assertClose(
            vol, torch.tensor([[1 - dd]], device=vol.device, dtype=vol.dtype)
        )

        # 3rd test
        dd = random.random()
        box2 = box1 + torch.tensor([[dd, 0.0, 0.0]], device=device)
        vol, _ = overlap_fn(box1[None], box2[None])
        self.assertClose(
            vol, torch.tensor([[1 - dd]], device=vol.device, dtype=vol.dtype)
        )
        # symmetry
        vol, _ = overlap_fn(box2[None], box1[None])
        self.assertClose(
            vol, torch.tensor([[1 - dd]], device=vol.device, dtype=vol.dtype)
        )

        # 4th test
        ddx, ddy, ddz = random.random(), random.random(), random.random()
        box2 = box1 + torch.tensor([[ddx, ddy, ddz]], device=device)
        vol, _ = overlap_fn(box1[None], box2[None])
        self.assertClose(
            vol,
            torch.tensor(
                [[(1 - ddx) * (1 - ddy) * (1 - ddz)]],
                device=vol.device,
                dtype=vol.dtype,
            ),
        )
        # symmetry
        vol, _ = overlap_fn(box2[None], box1[None])
        self.assertClose(
            vol,
            torch.tensor(
                [[(1 - ddx) * (1 - ddy) * (1 - ddz)]],
                device=vol.device,
                dtype=vol.dtype,
            ),
        )

        # Also check IoU is 1 when computing overlap with the same shifted box
        vol, iou = overlap_fn(box2[None], box2[None])
        self.assertClose(iou, torch.tensor([[1.0]], device=vol.device, dtype=vol.dtype))

        # 5th test
        ddx, ddy, ddz = random.random(), random.random(), random.random()
        box2 = box1 + torch.tensor([[ddx, ddy, ddz]], device=device)
        RR = random_rotation(dtype=torch.float32, device=device)
        box1r = box1 @ RR.transpose(0, 1)
        box2r = box2 @ RR.transpose(0, 1)
        vol, _ = overlap_fn(box1r[None], box2r[None])
        self.assertClose(
            vol,
            torch.tensor(
                [[(1 - ddx) * (1 - ddy) * (1 - ddz)]],
                device=vol.device,
                dtype=vol.dtype,
            ),
        )
        # symmetry
        vol, _ = overlap_fn(box2r[None], box1r[None])
        self.assertClose(
            vol,
            torch.tensor(
                [[(1 - ddx) * (1 - ddy) * (1 - ddz)]],
                device=vol.device,
                dtype=vol.dtype,
            ),
        )

        # 6th test
        ddx, ddy, ddz = random.random(), random.random(), random.random()
        box2 = box1 + torch.tensor([[ddx, ddy, ddz]], device=device)
        RR = random_rotation(dtype=torch.float32, device=device)
        TT = torch.rand((1, 3), dtype=torch.float32, device=device)
        box1r = box1 @ RR.transpose(0, 1) + TT
        box2r = box2 @ RR.transpose(0, 1) + TT
        vol, _ = overlap_fn(box1r[None], box2r[None])
        self.assertClose(
            vol,
            torch.tensor(
                [[(1 - ddx) * (1 - ddy) * (1 - ddz)]],
                device=vol.device,
                dtype=vol.dtype,
            ),
            atol=1e-7,
        )
        # symmetry
        vol, _ = overlap_fn(box2r[None], box1r[None])
        self.assertClose(
            vol,
            torch.tensor(
                [[(1 - ddx) * (1 - ddy) * (1 - ddz)]],
                device=vol.device,
                dtype=vol.dtype,
            ),
            atol=1e-7,
        )

        # 7th test: hand coded example and test with meshlab output

        # Meshlab procedure to compute volumes of shapes
        # 1. Load a shape, then Filters
        #       -> Remeshing, Simplification, Reconstruction -> Convex Hull
        # 2. Select the convex hull shape (This is important!)
        # 3. Then Filters -> Quality Measure and Computation -> Compute Geometric Measures
        # 3. Check for "Mesh Volume" in the stdout
        box1r = torch.tensor(
            [
                [3.1673, -2.2574, 0.4817],
                [4.6470, 0.2223, 2.4197],
                [5.2200, 1.1844, 0.7510],
                [3.7403, -1.2953, -1.1869],
                [-4.9316, 2.5724, 0.4856],
                [-3.4519, 5.0521, 2.4235],
                [-2.8789, 6.0142, 0.7549],
                [-4.3586, 3.5345, -1.1831],
            ],
            device=device,
        )
        box2r = torch.tensor(
            [
                [0.5623, 4.0647, 3.4334],
                [3.3584, 4.3191, 1.1791],
                [3.0724, -5.9235, -0.3315],
                [0.2763, -6.1779, 1.9229],
                [-2.0773, 4.6121, 0.2213],
                [0.7188, 4.8665, -2.0331],
                [0.4328, -5.3761, -3.5436],
                [-2.3633, -5.6305, -1.2893],
            ],
            device=device,
        )
        # from Meshlab:
        vol_inters = 33.558529
        vol_box1 = 65.899010
        vol_box2 = 156.386719
        iou_mesh = vol_inters / (vol_box1 + vol_box2 - vol_inters)

        vol, iou = overlap_fn(box1r[None], box2r[None])
        self.assertClose(vol, torch.tensor([[vol_inters]], device=device), atol=1e-1)
        self.assertClose(iou, torch.tensor([[iou_mesh]], device=device), atol=1e-1)
        # symmetry
        vol, iou = overlap_fn(box2r[None], box1r[None])
        self.assertClose(vol, torch.tensor([[vol_inters]], device=device), atol=1e-1)
        self.assertClose(iou, torch.tensor([[iou_mesh]], device=device), atol=1e-1)

        # 8th test: compare with sampling
        # create box1
        ctrs = torch.rand((2, 3), device=device)
        whl = torch.rand((2, 3), device=device) * 10.0 + 1.0
        # box8a & box8b
        box8a = self.create_box(ctrs[0], whl[0])
        box8b = self.create_box(ctrs[1], whl[1])
        RR1 = random_rotation(dtype=torch.float32, device=device)
        TT1 = torch.rand((1, 3), dtype=torch.float32, device=device)
        RR2 = random_rotation(dtype=torch.float32, device=device)
        TT2 = torch.rand((1, 3), dtype=torch.float32, device=device)
        box1r = box8a @ RR1.transpose(0, 1) + TT1
        box2r = box8b @ RR2.transpose(0, 1) + TT2
        vol, iou = overlap_fn(box1r[None], box2r[None])
        iou_sampling = self._box3d_overlap_sampling_batched(
            box1r[None], box2r[None], num_samples=10000
        )
        self.assertClose(iou, iou_sampling, atol=1e-2)
        # symmetry
        vol, iou = overlap_fn(box2r[None], box1r[None])
        self.assertClose(iou, iou_sampling, atol=1e-2)

        # 9th test: non overlapping boxes, iou = 0.0
        box2 = box1 + torch.tensor([[0.0, 100.0, 0.0]], device=device)
        vol, iou = overlap_fn(box1[None], box2[None])
        self.assertClose(vol, torch.tensor([[0.0]], device=vol.device, dtype=vol.dtype))
        self.assertClose(iou, torch.tensor([[0.0]], device=vol.device, dtype=vol.dtype))
        # symmetry
        vol, iou = overlap_fn(box2[None], box1[None])
        self.assertClose(vol, torch.tensor([[0.0]], device=vol.device, dtype=vol.dtype))
        self.assertClose(iou, torch.tensor([[0.0]], device=vol.device, dtype=vol.dtype))

        # 10th test: Non coplanar verts in a plane
        box10 = box1 + torch.rand((8, 3), dtype=torch.float32, device=device)
        msg = "Plane vertices are not coplanar"
        with self.assertRaisesRegex(ValueError, msg):
            overlap_fn(box10[None], box10[None])

        # 11th test: Skewed bounding boxes but all verts are coplanar
        box_skew_1 = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [-2, -2, 2],
                [2, -2, 2],
                [2, 2, 2],
                [-2, 2, 2],
            ],
            dtype=torch.float32,
            device=device,
        )
        box_skew_2 = torch.tensor(
            [
                [2.015995, 0.695233, 2.152806],
                [2.832533, 0.663448, 1.576389],
                [2.675445, -0.309592, 1.407520],
                [1.858907, -0.277806, 1.983936],
                [-0.413922, 3.161758, 2.044343],
                [2.852230, 3.034615, -0.261321],
                [2.223878, -0.857545, -0.936800],
                [-1.042273, -0.730402, 1.368864],
            ],
            dtype=torch.float32,
            device=device,
        )
        vol1 = 14.000
        vol2 = 14.000005
        vol_inters = 5.431122
        iou = vol_inters / (vol1 + vol2 - vol_inters)

        vols, ious = overlap_fn(box_skew_1[None], box_skew_2[None])
        self.assertClose(vols, torch.tensor([[vol_inters]], device=device), atol=1e-1)
        self.assertClose(ious, torch.tensor([[iou]], device=device), atol=1e-1)
        # symmetry
        vols, ious = overlap_fn(box_skew_2[None], box_skew_1[None])
        self.assertClose(vols, torch.tensor([[vol_inters]], device=device), atol=1e-1)
        self.assertClose(ious, torch.tensor([[iou]], device=device), atol=1e-1)

        # 12th test: Zero area bounding box (from GH issue #992)
        box12a = torch.tensor(
            [
                [-1.0000, -1.0000, -0.5000],
                [1.0000, -1.0000, -0.5000],
                [1.0000, 1.0000, -0.5000],
                [-1.0000, 1.0000, -0.5000],
                [-1.0000, -1.0000, 0.5000],
                [1.0000, -1.0000, 0.5000],
                [1.0000, 1.0000, 0.5000],
                [-1.0000, 1.0000, 0.5000],
            ],
            device=device,
            dtype=torch.float32,
        )

        box12b = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float32,
        )
        msg = "Planes have zero areas"
        with self.assertRaisesRegex(ValueError, msg):
            overlap_fn(box12a[None], box12b[None])
        # symmetry
        with self.assertRaisesRegex(ValueError, msg):
            overlap_fn(box12b[None], box12a[None])

        # 13th test: From GH issue #992
        # Zero area coplanar face after intersection
        ctrs = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]])
        whl = torch.tensor([[2.0, 2.0, 2.0], [2.0, 2, 2]])
        box13a = TestIoU3D.create_box(ctrs[0], whl[0])
        box13b = TestIoU3D.create_box(ctrs[1], whl[1])
        vol, iou = overlap_fn(box13a[None], box13b[None])
        self.assertClose(vol, torch.tensor([[2.0]], device=vol.device, dtype=vol.dtype))

        # 14th test: From GH issue #992
        # Random rotation, same boxes, iou should be 1.0
        corners = (
            torch.tensor(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )
            * 0.5
        )
        yaw = torch.tensor(0.185)
        Rot = torch.tensor(
            [
                [torch.cos(yaw), 0.0, torch.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-torch.sin(yaw), 0.0, torch.cos(yaw)],
            ],
            dtype=torch.float32,
            device=device,
        )
        corners = (Rot.mm(corners.t())).t()
        vol, iou = overlap_fn(corners[None], corners[None])
        self.assertClose(
            iou, torch.tensor([[1.0]], device=vol.device, dtype=vol.dtype), atol=1e-2
        )

        # 15th test: From GH issue #1082
        box15a = torch.tensor(
            [
                [-2.5629019, 4.13995749, -1.76344576],
                [1.92329434, 4.28127117, -1.86155124],
                [1.86994571, 5.97489644, -1.86155124],
                [-2.61625053, 5.83358276, -1.76344576],
                [-2.53123587, 4.14095496, -0.31397536],
                [1.95496037, 4.28226864, -0.41208084],
                [1.90161174, 5.97589391, -0.41208084],
                [-2.5845845, 5.83458023, -0.31397536],
            ],
            device=device,
            dtype=torch.float32,
        )

        box15b = torch.tensor(
            [
                [-2.6256125, 4.13036357, -1.82893437],
                [1.87201008, 4.25296695, -1.82893437],
                [1.82562476, 5.95458116, -1.82893437],
                [-2.67199782, 5.83197777, -1.82893437],
                [-2.6256125, 4.13036357, -0.40095884],
                [1.87201008, 4.25296695, -0.40095884],
                [1.82562476, 5.95458116, -0.40095884],
                [-2.67199782, 5.83197777, -0.40095884],
            ],
            device=device,
            dtype=torch.float32,
        )
        vol, iou = overlap_fn(box15a[None], box15b[None])
        self.assertClose(
            iou, torch.tensor([[0.91]], device=vol.device, dtype=vol.dtype), atol=1e-2
        )
        # symmetry
        vol, iou = overlap_fn(box15b[None], box15a[None])
        self.assertClose(
            iou, torch.tensor([[0.91]], device=vol.device, dtype=vol.dtype), atol=1e-2
        )

        # 16th test: From GH issue 1287
        box16a = torch.tensor(
            [
                [-167.5847, -70.6167, -2.7927],
                [-166.7333, -72.4264, -2.7927],
                [-166.7333, -72.4264, -4.5927],
                [-167.5847, -70.6167, -4.5927],
                [-163.0605, -68.4880, -2.7927],
                [-162.2090, -70.2977, -2.7927],
                [-162.2090, -70.2977, -4.5927],
                [-163.0605, -68.4880, -4.5927],
            ],
            device=device,
            dtype=torch.float32,
        )

        box16b = torch.tensor(
            [
                [-167.5847, -70.6167, -2.7927],
                [-166.7333, -72.4264, -2.7927],
                [-166.7333, -72.4264, -4.5927],
                [-167.5847, -70.6167, -4.5927],
                [-163.0605, -68.4880, -2.7927],
                [-162.2090, -70.2977, -2.7927],
                [-162.2090, -70.2977, -4.5927],
                [-163.0605, -68.4880, -4.5927],
            ],
            device=device,
            dtype=torch.float32,
        )
        vol, iou = overlap_fn(box16a[None], box16b[None])
        self.assertClose(
            iou, torch.tensor([[1.0]], device=vol.device, dtype=vol.dtype), atol=1e-2
        )
        # symmetry
        vol, iou = overlap_fn(box16b[None], box16a[None])
        self.assertClose(
            iou, torch.tensor([[1.0]], device=vol.device, dtype=vol.dtype), atol=1e-2
        )

        # 17th test: From GH issue 1287
        box17a = torch.tensor(
            [
                [-33.94158, -4.51639, 0.96941],
                [-34.67156, -2.65437, 0.96941],
                [-34.67156, -2.65437, -0.95367],
                [-33.94158, -4.51639, -0.95367],
                [-38.75954, -6.40521, 0.96941],
                [-39.48952, -4.54319, 0.96941],
                [-39.48952, -4.54319, -0.95367],
                [-38.75954, -6.40521, -0.95367],
            ],
            device=device,
            dtype=torch.float32,
        )

        box17b = torch.tensor(
            [
                [-33.94159, -4.51638, 0.96939],
                [-34.67158, -2.65437, 0.96939],
                [-34.67158, -2.65437, -0.95368],
                [-33.94159, -4.51638, -0.95368],
                [-38.75954, -6.40523, 0.96939],
                [-39.48953, -4.54321, 0.96939],
                [-39.48953, -4.54321, -0.95368],
                [-38.75954, -6.40523, -0.95368],
            ],
            device=device,
            dtype=torch.float32,
        )
        vol, iou = overlap_fn(box17a[None], box17b[None])
        self.assertClose(
            iou, torch.tensor([[1.0]], device=vol.device, dtype=vol.dtype), atol=1e-2
        )
        # symmetry
        vol, iou = overlap_fn(box17b[None], box17a[None])
        self.assertClose(
            iou, torch.tensor([[1.0]], device=vol.device, dtype=vol.dtype), atol=1e-2
        )

        # 18th test: From GH issue 1287
        box18a = torch.tensor(
            [
                [-105.6248, -32.7026, -1.2279],
                [-106.4690, -30.8895, -1.2279],
                [-106.4690, -30.8895, -3.0279],
                [-105.6248, -32.7026, -3.0279],
                [-110.1575, -34.8132, -1.2279],
                [-111.0017, -33.0001, -1.2279],
                [-111.0017, -33.0001, -3.0279],
                [-110.1575, -34.8132, -3.0279],
            ],
            device=device,
            dtype=torch.float32,
        )
        box18b = torch.tensor(
            [
                [-105.5094, -32.9504, -1.0641],
                [-106.4272, -30.9793, -1.0641],
                [-106.4272, -30.9793, -3.1916],
                [-105.5094, -32.9504, -3.1916],
                [-110.0421, -35.0609, -1.0641],
                [-110.9599, -33.0899, -1.0641],
                [-110.9599, -33.0899, -3.1916],
                [-110.0421, -35.0609, -3.1916],
            ],
            device=device,
            dtype=torch.float32,
        )
        # from Meshlab
        vol_inters = 17.108501
        vol_box1 = 18.000067
        vol_box2 = 23.128527
        iou_mesh = vol_inters / (vol_box1 + vol_box2 - vol_inters)
        vol, iou = overlap_fn(box18a[None], box18b[None])
        self.assertClose(
            iou,
            torch.tensor([[iou_mesh]], device=vol.device, dtype=vol.dtype),
            atol=1e-2,
        )
        self.assertClose(
            vol,
            torch.tensor([[vol_inters]], device=vol.device, dtype=vol.dtype),
            atol=1e-2,
        )
        # symmetry
        vol, iou = overlap_fn(box18b[None], box18a[None])
        self.assertClose(
            iou,
            torch.tensor([[iou_mesh]], device=vol.device, dtype=vol.dtype),
            atol=1e-2,
        )
        self.assertClose(
            vol,
            torch.tensor([[vol_inters]], device=vol.device, dtype=vol.dtype),
            atol=1e-2,
        )

        # 19th example: From GH issue 1287
        box19a = torch.tensor(
            [
                [-59.4785, -15.6003, 0.4398],
                [-60.2263, -13.6928, 0.4398],
                [-60.2263, -13.6928, -1.3909],
                [-59.4785, -15.6003, -1.3909],
                [-64.1743, -17.4412, 0.4398],
                [-64.9221, -15.5337, 0.4398],
                [-64.9221, -15.5337, -1.3909],
                [-64.1743, -17.4412, -1.3909],
            ],
            device=device,
            dtype=torch.float32,
        )
        box19b = torch.tensor(
            [
                [-59.4874, -15.5775, -0.1512],
                [-60.2174, -13.7155, -0.1512],
                [-60.2174, -13.7155, -1.9820],
                [-59.4874, -15.5775, -1.9820],
                [-64.1832, -17.4185, -0.1512],
                [-64.9132, -15.5564, -0.1512],
                [-64.9132, -15.5564, -1.9820],
                [-64.1832, -17.4185, -1.9820],
            ],
            device=device,
            dtype=torch.float32,
        )
        # from Meshlab
        vol_inters = 12.505723
        vol_box1 = 18.918238
        vol_box2 = 18.468531
        iou_mesh = vol_inters / (vol_box1 + vol_box2 - vol_inters)
        vol, iou = overlap_fn(box19a[None], box19b[None])
        self.assertClose(
            iou,
            torch.tensor([[iou_mesh]], device=vol.device, dtype=vol.dtype),
            atol=1e-2,
        )
        self.assertClose(
            vol,
            torch.tensor([[vol_inters]], device=vol.device, dtype=vol.dtype),
            atol=1e-2,
        )
        # symmetry
        vol, iou = overlap_fn(box19b[None], box19a[None])
        self.assertClose(
            iou,
            torch.tensor([[iou_mesh]], device=vol.device, dtype=vol.dtype),
            atol=1e-2,
        )
        self.assertClose(
            vol,
            torch.tensor([[vol_inters]], device=vol.device, dtype=vol.dtype),
            atol=1e-2,
        )

    def _test_real_boxes(self, overlap_fn, device):
        data_filename = "./real_boxes.pkl"
        with open(DATA_DIR / data_filename, "rb") as f:
            example = pickle.load(f)

        verts1 = torch.FloatTensor(example["verts1"])
        verts2 = torch.FloatTensor(example["verts2"])
        boxes = torch.stack((verts1, verts2)).to(device)

        iou_expected = torch.eye(2).to(device)
        vol, iou = overlap_fn(boxes, boxes)
        self.assertClose(iou, iou_expected)

    def test_iou_naive(self):
        device = get_random_cuda_device()
        self._test_iou(self._box3d_overlap_naive_batched, device)
        self._test_compare_objectron(self._box3d_overlap_naive_batched, device)
        self._test_real_boxes(self._box3d_overlap_naive_batched, device)

    def test_iou_cpu(self):
        device = torch.device("cpu")
        self._test_iou(box3d_overlap, device)
        self._test_compare_objectron(box3d_overlap, device)
        self._test_real_boxes(box3d_overlap, device)

    def test_iou_cuda(self):
        device = torch.device("cuda:0")
        self._test_iou(box3d_overlap, device)
        self._test_compare_objectron(box3d_overlap, device)
        self._test_real_boxes(box3d_overlap, device)

    def _test_compare_objectron(self, overlap_fn, device):
        # Load saved objectron data
        data_filename = "./objectron_vols_ious.pt"
        objectron_vals = torch.load(DATA_DIR / data_filename)
        boxes1 = objectron_vals["boxes1"]
        boxes2 = objectron_vals["boxes2"]
        vols_objectron = objectron_vals["vols"]
        ious_objectron = objectron_vals["ious"]

        boxes1 = boxes1.to(device=device, dtype=torch.float32)
        boxes2 = boxes2.to(device=device, dtype=torch.float32)

        # Convert vertex orderings from Objectron to PyTorch3D convention
        idx = torch.tensor(
            OBJECTRON_TO_PYTORCH3D_FACE_IDX, dtype=torch.int64, device=device
        )
        boxes1 = boxes1.index_select(index=idx, dim=1)
        boxes2 = boxes2.index_select(index=idx, dim=1)

        # Run PyTorch3D version
        vols, ious = overlap_fn(boxes1, boxes2)

        # Check values match
        self.assertClose(vols_objectron, vols.cpu())
        self.assertClose(ious_objectron, ious.cpu())

    def test_batched_errors(self):
        N, M = 5, 10
        boxes1 = torch.randn((N, 8, 3))
        boxes2 = torch.randn((M, 10, 3))
        with self.assertRaisesRegex(ValueError, "(8, 3)"):
            box3d_overlap(boxes1, boxes2)

    def test_box_volume(self):
        device = torch.device("cuda:0")
        box1 = torch.tensor(
            [
                [3.1673, -2.2574, 0.4817],
                [4.6470, 0.2223, 2.4197],
                [5.2200, 1.1844, 0.7510],
                [3.7403, -1.2953, -1.1869],
                [-4.9316, 2.5724, 0.4856],
                [-3.4519, 5.0521, 2.4235],
                [-2.8789, 6.0142, 0.7549],
                [-4.3586, 3.5345, -1.1831],
            ],
            dtype=torch.float32,
            device=device,
        )
        box2 = torch.tensor(
            [
                [0.5623, 4.0647, 3.4334],
                [3.3584, 4.3191, 1.1791],
                [3.0724, -5.9235, -0.3315],
                [0.2763, -6.1779, 1.9229],
                [-2.0773, 4.6121, 0.2213],
                [0.7188, 4.8665, -2.0331],
                [0.4328, -5.3761, -3.5436],
                [-2.3633, -5.6305, -1.2893],
            ],
            dtype=torch.float32,
            device=device,
        )

        box3 = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.float32,
            device=device,
        )

        RR = random_rotation(dtype=torch.float32, device=device)
        TT = torch.rand((1, 3), dtype=torch.float32, device=device)
        box4 = box3 @ RR.transpose(0, 1) + TT

        self.assertClose(box_volume(box1).cpu(), torch.tensor(65.899010), atol=1e-3)
        self.assertClose(box_volume(box2).cpu(), torch.tensor(156.386719), atol=1e-3)
        self.assertClose(box_volume(box3).cpu(), torch.tensor(1.0), atol=1e-3)
        self.assertClose(box_volume(box4).cpu(), torch.tensor(1.0), atol=1e-3)

    def test_box_planar_dir(self):
        device = torch.device("cuda:0")
        box1 = torch.tensor(
            UNIT_BOX,
            dtype=torch.float32,
            device=device,
        )

        n1 = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            device=device,
            dtype=torch.float32,
        )

        RR = random_rotation(dtype=torch.float32, device=device)
        TT = torch.rand((1, 3), dtype=torch.float32, device=device)
        box2 = box1 @ RR.transpose(0, 1) + TT
        n2 = n1 @ RR.transpose(0, 1)

        self.assertClose(box_planar_dir(box1), n1)
        self.assertClose(box_planar_dir(box2), n2)

    @staticmethod
    def iou_naive(N: int, M: int, device="cpu"):
        box = torch.tensor(
            [UNIT_BOX],
            dtype=torch.float32,
            device=device,
        )
        boxes1 = box + torch.randn((N, 1, 3), device=device)
        boxes2 = box + torch.randn((M, 1, 3), device=device)

        def output():
            vol, iou = TestIoU3D._box3d_overlap_naive_batched(boxes1, boxes2)

        return output

    @staticmethod
    def iou(N: int, M: int, device="cpu"):
        box = torch.tensor(
            [UNIT_BOX],
            dtype=torch.float32,
            device=device,
        )
        boxes1 = box + torch.randn((N, 1, 3), device=device)
        boxes2 = box + torch.randn((M, 1, 3), device=device)

        def output():
            vol, iou = box3d_overlap(boxes1, boxes2)

        return output

    @staticmethod
    def iou_sampling(N: int, M: int, num_samples: int, device="cpu"):
        box = torch.tensor(
            [UNIT_BOX],
            dtype=torch.float32,
            device=device,
        )
        boxes1 = box + torch.randn((N, 1, 3), device=device)
        boxes2 = box + torch.randn((M, 1, 3), device=device)

        def output():
            _ = TestIoU3D._box3d_overlap_sampling_batched(boxes1, boxes2, num_samples)

        return output


# -------------------------------------------------- #
#                NAIVE IMPLEMENTATION                #
# -------------------------------------------------- #

"""
The main functions below are:
* box3d_overlap_naive: which computes the exact IoU of box1 and box2
* box3d_overlap_sampling: which computes an approximate IoU of box1 and box2
    by sampling points within the boxes

Note that both implementations currently do not support batching.
"""
# -------------------------------------------------- #
# Throughout this implementation, we assume that boxes
# are defined by their 8 corners in the following order
#
#        (4) +---------+. (5)
#            | ` .     |  ` .
#            | (0) +---+-----+ (1)
#            |     |   |     |
#        (7) +-----+---+. (6)|
#            ` .   |     ` . |
#            (3) ` +---------+ (2)
#
# -------------------------------------------------- #

# -------------------------------------------------- #
#       HELPER FUNCTIONS FOR EXACT SOLUTION          #
# -------------------------------------------------- #


def get_tri_verts(box: torch.Tensor) -> torch.Tensor:
    """
    Return the vertex coordinates forming the triangles of the box.
    The computation here resembles the Meshes data structure.
    But since we only want this tiny functionality, we abstract it out.
    Args:
        box: tensor of shape (8, 3)
    Returns:
        tri_verts: tensor of shape (12, 3, 3)
    """
    device = box.device
    faces = torch.tensor(_box_triangles, device=device, dtype=torch.int64)  # (12, 3)
    tri_verts = box[faces]  # (12, 3, 3)
    return tri_verts


def get_plane_verts(box: torch.Tensor) -> torch.Tensor:
    """
    Return the vertex coordinates forming the planes of the box.
    The computation here resembles the Meshes data structure.
    But since we only want this tiny functionality, we abstract it out.
    Args:
        box: tensor of shape (8, 3)
    Returns:
        plane_verts: tensor of shape (6, 4, 3)
    """
    device = box.device
    faces = torch.tensor(_box_planes, device=device, dtype=torch.int64)  # (6, 4)
    plane_verts = box[faces]  # (6, 4, 3)
    return plane_verts


def get_tri_center_normal(tris: torch.Tensor) -> torch.Tensor:
    """
    Returns the center and normal of triangles
    Args:
        tris: tensor of shape (T, 3, 3)
    Returns:
        center: tensor of shape (T, 3)
        normal: tensor of shape (T, 3)
    """
    add_dim0 = False
    if tris.ndim == 2:
        tris = tris.unsqueeze(0)
        add_dim0 = True

    ctr = tris.mean(1)  # (T, 3)
    normals = torch.zeros_like(ctr)

    v0, v1, v2 = tris.unbind(1)  # 3 x (T, 3)

    # unvectorized solution
    T = tris.shape[0]
    for t in range(T):
        ns = torch.zeros((3, 3), device=tris.device)
        ns[0] = torch.cross(v0[t] - ctr[t], v1[t] - ctr[t], dim=-1)
        ns[1] = torch.cross(v0[t] - ctr[t], v2[t] - ctr[t], dim=-1)
        ns[2] = torch.cross(v1[t] - ctr[t], v2[t] - ctr[t], dim=-1)

        i = torch.norm(ns, dim=-1).argmax()
        normals[t] = ns[i]

    if add_dim0:
        ctr = ctr[0]
        normals = normals[0]
    normals = F.normalize(normals, dim=-1)
    return ctr, normals


def get_plane_center_normal(planes: torch.Tensor) -> torch.Tensor:
    """
    Returns the center and normal of planes
    Args:
        planes: tensor of shape (P, 4, 3)
    Returns:
        center: tensor of shape (P, 3)
        normal: tensor of shape (P, 3)
    """
    add_dim0 = False
    if planes.ndim == 2:
        planes = planes.unsqueeze(0)
        add_dim0 = True

    ctr = planes.mean(1)  # (P, 3)
    normals = torch.zeros_like(ctr)

    v0, v1, v2, v3 = planes.unbind(1)  # 4 x (P, 3)

    # unvectorized solution
    P = planes.shape[0]
    for t in range(P):
        ns = torch.zeros((6, 3), device=planes.device)
        ns[0] = torch.cross(v0[t] - ctr[t], v1[t] - ctr[t], dim=-1)
        ns[1] = torch.cross(v0[t] - ctr[t], v2[t] - ctr[t], dim=-1)
        ns[2] = torch.cross(v0[t] - ctr[t], v3[t] - ctr[t], dim=-1)
        ns[3] = torch.cross(v1[t] - ctr[t], v2[t] - ctr[t], dim=-1)
        ns[4] = torch.cross(v1[t] - ctr[t], v3[t] - ctr[t], dim=-1)
        ns[5] = torch.cross(v2[t] - ctr[t], v3[t] - ctr[t], dim=-1)

        i = torch.norm(ns, dim=-1).argmax()
        normals[t] = ns[i]

    if add_dim0:
        ctr = ctr[0]
        normals = normals[0]
    normals = F.normalize(normals, dim=-1)
    return ctr, normals


def box_planar_dir(
    box: torch.Tensor, dot_eps: float = DOT_EPS, area_eps: float = AREA_EPS
) -> torch.Tensor:
    """
    Finds the unit vector n which is perpendicular to each plane in the box
    and points towards the inside of the box.
    The planes are defined by `_box_planes`.
    Since the shape is convex, we define the interior to be the direction
    pointing to the center of the shape.
    Args:
       box: tensor of shape (8, 3) of the vertices of the 3D box
    Returns:
       n: tensor of shape (6,) of the unit vector orthogonal to the face pointing
          towards the interior of the shape
    """
    assert box.shape[0] == 8 and box.shape[1] == 3

    # center point of each box
    box_ctr = box.mean(0).view(1, 3)

    # box planes
    plane_verts = get_plane_verts(box)  # (6, 4, 3)
    v0, v1, v2, v3 = plane_verts.unbind(1)
    plane_ctr, n = get_plane_center_normal(plane_verts)

    # Check all verts are coplanar
    if (
        not (
            F.normalize(v3 - v0, dim=-1).unsqueeze(1).bmm(n.unsqueeze(2)).abs()
            < dot_eps
        )
        .all()
        .item()
    ):
        msg = "Plane vertices are not coplanar"
        raise ValueError(msg)

    # Check all faces have non zero area
    area1 = torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    area2 = torch.cross(v3 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2
    if (area1 < area_eps).any().item() or (area2 < area_eps).any().item():
        msg = "Planes have zero areas"
        raise ValueError(msg)

    # We can write:  `box_ctr = plane_ctr + a * e0 + b * e1 + c * n`, (1).
    # With <e0, n> = 0 and <e1, n> = 0, where <.,.> refers to the dot product,
    # since that e0 is orthogonal to n. Same for e1.
    """
    # Below is how one would solve for (a, b, c)
    # Solving for (a, b)
    numF = verts.shape[0]
    A = torch.ones((numF, 2, 2), dtype=torch.float32, device=device)
    B = torch.ones((numF, 2), dtype=torch.float32, device=device)
    A[:, 0, 1] = (e0 * e1).sum(-1)
    A[:, 1, 0] = (e0 * e1).sum(-1)
    B[:, 0] = ((box_ctr - plane_ctr) * e0).sum(-1)
    B[:, 1] = ((box_ctr - plane_ctr) * e1).sum(-1)
    ab = torch.linalg.solve(A, B)  # (numF, 2)
    a, b = ab.unbind(1)
    # solving for c
    c = ((box_ctr - plane_ctr - a.view(numF, 1) * e0 - b.view(numF, 1) * e1) * n).sum(-1)
    """
    # Since we know that <e0, n> = 0 and <e1, n> = 0 (e0 and e1 are orthogonal to n),
    # the above solution is equivalent to
    direc = F.normalize(box_ctr - plane_ctr, dim=-1)  # (6, 3)
    c = (direc * n).sum(-1)
    # If c is negative, then we revert the direction of n such that n points "inside"
    negc = c < 0.0
    n[negc] *= -1.0
    # c[negc] *= -1.0
    # Now (a, b, c) is the solution to (1)

    return n


def tri_verts_area(tri_verts: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of the triangle faces in tri_verts
    Args:
        tri_verts: tensor of shape (T, 3, 3)
    Returns:
        areas: the area of the triangles (T, 1)
    """
    add_dim = False
    if tri_verts.ndim == 2:
        tri_verts = tri_verts.unsqueeze(0)
        add_dim = True

    v0, v1, v2 = tri_verts.unbind(1)
    areas = torch.cross(v1 - v0, v2 - v0, dim=-1).norm(dim=-1) / 2.0

    if add_dim:
        areas = areas[0]
    return areas


def box_volume(box: torch.Tensor) -> torch.Tensor:
    """
    Computes the volume of each box in boxes.
    The volume of each box is the sum of all the tetrahedrons
    formed by the faces of the box. The face of the box is the base of
    that tetrahedron and the center point of the box is the apex.
    In other words, vol(box) = sum_i A_i * d_i / 3,
    where A_i is the area of the i-th face and d_i is the
    distance of the apex from the face.
    We use the equivalent dot/cross product formulation.
    Read https://en.wikipedia.org/wiki/Tetrahedron#Volume

    Args:
        box: tensor of shape (8, 3) containing the vertices
            of the 3D box
    Returns:
        vols: the volume of the box
    """
    assert box.shape[0] == 8 and box.shape[1] == 3

    # Compute the center point of each box
    ctr = box.mean(0).view(1, 1, 3)

    # Extract the coordinates of the faces for each box
    tri_verts = get_tri_verts(box)
    # Set the origin of the coordinate system to coincide
    # with the apex of the tetrahedron to simplify the volume calculation
    # See https://en.wikipedia.org/wiki/Tetrahedron#Volume
    tri_verts = tri_verts - ctr

    # Compute the volume of each box using the dot/cross product formula
    vols = torch.sum(
        tri_verts[:, 0] * torch.cross(tri_verts[:, 1], tri_verts[:, 2], dim=-1),
        dim=-1,
    )
    vols = (vols.abs() / 6.0).sum()

    return vols


def coplanar_tri_faces(tri1: torch.Tensor, tri2: torch.Tensor, eps: float = DOT_EPS):
    """
    Determines whether two triangle faces in 3D are coplanar
    Args:
        tri1: tensor of shape (3, 3) of the vertices of the 1st triangle
        tri2: tensor of shape (3, 3) of the vertices of the 2nd triangle
    Returns:
        is_coplanar: bool
    """
    tri1_ctr, tri1_n = get_tri_center_normal(tri1)
    tri2_ctr, tri2_n = get_tri_center_normal(tri2)

    check1 = tri1_n.dot(tri2_n).abs() > 1 - eps  # checks if parallel

    dist12 = torch.norm(tri1.unsqueeze(1) - tri2.unsqueeze(0), dim=-1)
    dist12_argmax = dist12.argmax()
    i1 = dist12_argmax // 3
    i2 = dist12_argmax % 3
    assert dist12[i1, i2] == dist12.max()

    check2 = (
        F.normalize(tri1[i1] - tri2[i2], dim=0).dot(tri1_n).abs() < eps
    ) or F.normalize(tri1[i1] - tri2[i2], dim=0).dot(tri2_n).abs() < eps

    return check1 and check2


def coplanar_tri_plane(
    tri: torch.Tensor, plane: torch.Tensor, n: torch.Tensor, eps: float = DOT_EPS
):
    """
    Determines whether two triangle faces in 3D are coplanar
    Args:
            tri: tensor of shape (3, 3) of the vertices of the triangle
            plane: tensor of shape (4, 3) of the vertices of the plane
            n: tensor of shape (3,) of the unit "inside" direction on the plane
    Returns:
            is_coplanar: bool
    """
    tri_ctr, tri_n = get_tri_center_normal(tri)

    check1 = tri_n.dot(n).abs() > 1 - eps  # checks if parallel

    dist12 = torch.norm(tri.unsqueeze(1) - plane.unsqueeze(0), dim=-1)
    dist12_argmax = dist12.argmax()
    i1 = dist12_argmax // 4
    i2 = dist12_argmax % 4
    assert dist12[i1, i2] == dist12.max()

    check2 = F.normalize(tri[i1] - plane[i2], dim=0).dot(n).abs() < eps

    return check1 and check2


def is_inside(
    plane: torch.Tensor,
    n: torch.Tensor,
    points: torch.Tensor,
    return_proj: bool = True,
):
    """
    Computes whether point is "inside" the plane.
    The definition of "inside" means that the point
    has a positive component in the direction of the plane normal defined by n.
    For example,
                  plane
                    |
                    |         . (A)
                    |--> n
                    |
         .(B)       |

    Point (A) is "inside" the plane, while point (B) is "outside" the plane.
    Args:
      plane: tensor of shape (4,3) of vertices of a box plane
      n: tensor of shape (3,) of the unit "inside" direction on the plane
      points: tensor of shape (P, 3) of coordinates of a point
      return_proj: bool whether to return the projected point on the plane
    Returns:
      is_inside: bool of shape (P,) of whether point is inside
      p_proj: tensor of shape (P, 3) of the projected point on plane
    """
    device = plane.device
    v0, v1, v2, v3 = plane.unbind(0)
    plane_ctr = plane.mean(0)
    e0 = F.normalize(v0 - plane_ctr, dim=0)
    e1 = F.normalize(v1 - plane_ctr, dim=0)
    if not torch.allclose(e0.dot(n), torch.zeros((1,), device=device), atol=1e-2):
        raise ValueError("Input n is not perpendicular to the plane")
    if not torch.allclose(e1.dot(n), torch.zeros((1,), device=device), atol=1e-2):
        raise ValueError("Input n is not perpendicular to the plane")

    add_dim = False
    if points.ndim == 1:
        points = points.unsqueeze(0)
        add_dim = True

    assert points.shape[1] == 3
    # Every point p can be written as p = ctr + a e0 + b e1 + c n

    # If return_proj is True, we need to solve for (a, b)
    p_proj = None
    if return_proj:
        # solving for (a, b)
        A = torch.tensor(
            [[1.0, e0.dot(e1)], [e0.dot(e1), 1.0]], dtype=torch.float32, device=device
        )
        B = torch.zeros((2, points.shape[0]), dtype=torch.float32, device=device)
        B[0, :] = torch.sum((points - plane_ctr.view(1, 3)) * e0.view(1, 3), dim=-1)
        B[1, :] = torch.sum((points - plane_ctr.view(1, 3)) * e1.view(1, 3), dim=-1)
        ab = A.inverse() @ B  # (2, P)
        p_proj = plane_ctr.view(1, 3) + ab.transpose(0, 1) @ torch.stack(
            (e0, e1), dim=0
        )

    # solving for c
    # c = (point - ctr - a * e0 - b * e1).dot(n)
    direc = torch.sum((points - plane_ctr.view(1, 3)) * n.view(1, 3), dim=-1)
    ins = direc >= 0.0

    if add_dim:
        assert p_proj.shape[0] == 1
        p_proj = p_proj[0]

    return ins, p_proj


def plane_edge_point_of_intersection(plane, n, p0, p1, eps: float = DOT_EPS):
    """
    Finds the point of intersection between a box plane and
    a line segment connecting (p0, p1).
    The plane is assumed to be infinite long.
    Args:
      plane: tensor of shape (4, 3) of the coordinates of the vertices defining the plane
      n: tensor of shape (3,) of the unit direction perpendicular on the plane
          (Note that we could compute n but since it's computed in the main
          body of the function, we save time by feeding it in. For the purpose
          of this function, it's not important that n points "inside" the shape.)
      p0, p1: tensors of shape (3,), (3,)
    Returns:
      p: tensor of shape (3,) of the coordinates of the point of intersection
      a: scalar such that p = p0 + a*(p1-p0)
    """
    # The point of intersection can be parametrized
    # p = p0 + a (p1 - p0) where a in [0, 1]
    # We want to find a such that p is on plane
    # <p - ctr, n> = 0

    # if segment (p0, p1) is parallel to plane (it can only be on it)
    direc = F.normalize(p1 - p0, dim=0)
    if direc.dot(n).abs() < eps:
        return (p1 + p0) / 2.0, 0.5
    else:
        ctr = plane.mean(0)
        a = -(p0 - ctr).dot(n) / ((p1 - p0).dot(n))
        p = p0 + a * (p1 - p0)
        return p, a


"""
The three following functions support clipping a triangle face by a plane.
They contain the following cases: (a) the triangle has one point "outside" the plane and
(b) the triangle has two points "outside" the plane.
This logic follows the logic of clipping triangles when they intersect the image plane while
rendering.
"""


def clip_tri_by_plane_oneout(
    plane: torch.Tensor,
    n: torch.Tensor,
    vout: torch.Tensor,
    vin1: torch.Tensor,
    vin2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Case (a).
    Clips triangle by plane when vout is outside plane, and vin1, vin2, is inside
    In this case, only one vertex of the triangle is outside the plane.
    Clip the triangle into a quadrilateral, and then split into two triangles
    Args:
        plane: tensor of shape (4, 3) of the coordinates of the vertices forming the plane
        n: tensor of shape (3,) of the unit "inside" direction of the plane
        vout, vin1, vin2: tensors of shape (3,) of the points forming the triangle, where
            vout is "outside" the plane and vin1, vin2 are "inside"
    Returns:
        verts: tensor of shape (4, 3) containing the new vertices formed after clipping the
            original intersecting triangle (vout, vin1, vin2)
        faces: tensor of shape (2, 3) defining the vertex indices forming the two new triangles
            which are "inside" the plane formed after clipping
    """
    device = plane.device
    # point of intersection between plane and (vin1, vout)
    pint1, a1 = plane_edge_point_of_intersection(plane, n, vin1, vout)
    assert a1 >= -0.0001 and a1 <= 1.0001, a1
    # point of intersection between plane and (vin2, vout)
    pint2, a2 = plane_edge_point_of_intersection(plane, n, vin2, vout)
    assert a2 >= -0.0001 and a2 <= 1.0001, a2

    verts = torch.stack((vin1, pint1, pint2, vin2), dim=0)  # 4x3
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3]], dtype=torch.int64, device=device
    )  # 2x3
    return verts, faces


def clip_tri_by_plane_twoout(
    plane: torch.Tensor,
    n: torch.Tensor,
    vout1: torch.Tensor,
    vout2: torch.Tensor,
    vin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Case (b).
    Clips face by plane when vout1, vout2 are outside plane, and vin1 is inside
    In this case, only one vertex of the triangle is inside the plane.
    Args:
        plane: tensor of shape (4, 3) of the coordinates of the vertices forming the plane
        n: tensor of shape (3,) of the unit "inside" direction of the plane
        vout1, vout2, vin: tensors of shape (3,) of the points forming the triangle, where
            vin is "inside" the plane and vout1, vout2 are "outside"
    Returns:
        verts: tensor of shape (3, 3) containing the new vertices formed after clipping the
            original intersectiong triangle (vout, vin1, vin2)
        faces: tensor of shape (1, 3) defining the vertex indices forming
            the single new triangle which is "inside" the plane formed after clipping
    """
    device = plane.device
    # point of intersection between plane and (vin, vout1)
    pint1, a1 = plane_edge_point_of_intersection(plane, n, vin, vout1)
    assert a1 >= -0.0001 and a1 <= 1.0001, a1
    # point of intersection between plane and (vin, vout2)
    pint2, a2 = plane_edge_point_of_intersection(plane, n, vin, vout2)
    assert a2 >= -0.0001 and a2 <= 1.0001, a2

    verts = torch.stack((vin, pint1, pint2), dim=0)  # 3x3
    faces = torch.tensor(
        [
            [0, 1, 2],
        ],
        dtype=torch.int64,
        device=device,
    )  # 1x3
    return verts, faces


def clip_tri_by_plane(plane, n, tri_verts) -> Union[List, torch.Tensor]:
    """
    Clip a trianglular face defined by tri_verts with a plane of inside "direction" n.
    This function computes whether the triangle has one or two
    or none points "outside" the plane.
    Args:
       plane: tensor of shape (4, 3) of the vertex coordinates of the plane
       n: tensor of shape (3,) of the unit "inside" direction of the plane
       tri_verts: tensor of shape (3, 3) of the vertex coordiantes of the the triangle faces
    Returns:
        tri_verts: tensor of shape (K, 3, 3) of the vertex coordinates of the triangles formed
            after clipping. All K triangles are now "inside" the plane.
    """
    if coplanar_tri_plane(tri_verts, plane, n):
        return tri_verts.view(1, 3, 3)

    v0, v1, v2 = tri_verts.unbind(0)
    isin0, _ = is_inside(plane, n, v0)
    isin1, _ = is_inside(plane, n, v1)
    isin2, _ = is_inside(plane, n, v2)

    if isin0 and isin1 and isin2:
        # all in, no clipping, keep the old triangle face
        return tri_verts.view(1, 3, 3)
    elif (not isin0) and (not isin1) and (not isin2):
        # all out, delete triangle
        return []
    else:
        if isin0:
            if isin1:  # (isin0, isin1, not isin2)
                verts, faces = clip_tri_by_plane_oneout(plane, n, v2, v0, v1)
                return verts[faces]
            elif isin2:  # (isin0, not isin1, isin2)
                verts, faces = clip_tri_by_plane_oneout(plane, n, v1, v0, v2)
                return verts[faces]
            else:  # (isin0, not isin1, not isin2)
                verts, faces = clip_tri_by_plane_twoout(plane, n, v1, v2, v0)
                return verts[faces]
        else:
            if isin1 and isin2:  # (not isin0, isin1, isin2)
                verts, faces = clip_tri_by_plane_oneout(plane, n, v0, v1, v2)
                return verts[faces]
            elif isin1:  # (not isin0, isin1, not isin2)
                verts, faces = clip_tri_by_plane_twoout(plane, n, v0, v2, v1)
                return verts[faces]
            elif isin2:  # (not isin0, not isin1, isin2)
                verts, faces = clip_tri_by_plane_twoout(plane, n, v0, v1, v2)
                return verts[faces]

    # Should not be reached
    return []


# -------------------------------------------------- #
#               MAIN: BOX3D_OVERLAP                  #
# -------------------------------------------------- #


def box3d_overlap_naive(box1: torch.Tensor, box2: torch.Tensor):
    """
    Computes the intersection of 3D boxes1 and boxes2.
    Inputs boxes1, boxes2 are tensors of shape (8, 3) containing
    the 8 corners of the boxes, as follows

        (4) +---------+. (5)
            | ` .     |  ` .
            | (0) +---+-----+ (1)
            |     |   |     |
        (7) +-----+---+. (6)|
            ` .   |     ` . |
            (3) ` +---------+ (2)

    Args:
        box1: tensor of shape (8, 3) of the coordinates of the 1st box
        box2: tensor of shape (8, 3) of the coordinates of the 2nd box
    Returns:
        vol: the volume of the intersecting convex shape
        iou: the intersection over union which is simply
            `iou = vol / (vol1 + vol2 - vol)`
    """
    device = box1.device

    # For boxes1 we compute the unit directions n1 corresponding to quad_faces
    n1 = box_planar_dir(box1)  # (6, 3)
    # For boxes2 we compute the unit directions n2 corresponding to quad_faces
    n2 = box_planar_dir(box2)

    # We define triangle faces
    vol1 = box_volume(box1)
    vol2 = box_volume(box2)

    tri_verts1 = get_tri_verts(box1)  # (12, 3, 3)
    plane_verts1 = get_plane_verts(box1)  # (6, 4, 3)
    tri_verts2 = get_tri_verts(box2)  # (12, 3, 3)
    plane_verts2 = get_plane_verts(box2)  # (6, 4, 3)

    num_planes = plane_verts1.shape[0]  # (=6) based on our definition of planes

    # Every triangle in box1 will be compared to each plane in box2.
    # If the triangle is fully outside or fully inside, then it will remain as is
    # If the triangle intersects with the (infinite) plane, it will be broken into
    # subtriangles such that each subtriangle is either fully inside or outside the plane.

    # Tris in Box1 -> Planes in Box2
    for pidx in range(num_planes):
        plane = plane_verts2[pidx]
        nplane = n2[pidx]
        tri_verts_updated = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)
        for i in range(tri_verts1.shape[0]):
            tri = clip_tri_by_plane(plane, nplane, tri_verts1[i])
            if len(tri) > 0:
                tri_verts_updated = torch.cat((tri_verts_updated, tri), dim=0)
        tri_verts1 = tri_verts_updated

    # Tris in Box2 -> Planes in Box1
    for pidx in range(num_planes):
        plane = plane_verts1[pidx]
        nplane = n1[pidx]
        tri_verts_updated = torch.zeros((0, 3, 3), dtype=torch.float32, device=device)
        for i in range(tri_verts2.shape[0]):
            tri = clip_tri_by_plane(plane, nplane, tri_verts2[i])
            if len(tri) > 0:
                tri_verts_updated = torch.cat((tri_verts_updated, tri), dim=0)
        tri_verts2 = tri_verts_updated

    # remove triangles that are coplanar from the intersection as
    # otherwise they would be doublecounting towards the volume
    # this happens only if the original 3D boxes have common planes
    # Since the resulting shape is convex and specifically composed of planar segments,
    # each planar segment can belong either on box1 or box2 but not both.
    # Without loss of generality, we assign shared planar segments to box1
    keep2 = torch.ones((tri_verts2.shape[0],), device=device, dtype=torch.bool)
    for i1 in range(tri_verts1.shape[0]):
        for i2 in range(tri_verts2.shape[0]):
            if (
                coplanar_tri_faces(tri_verts1[i1], tri_verts2[i2])
                and tri_verts_area(tri_verts1[i1]) > AREA_EPS
            ):
                keep2[i2] = 0
    keep2 = keep2.nonzero()[:, 0]
    tri_verts2 = tri_verts2[keep2]

    # intersecting shape
    num_faces = tri_verts1.shape[0] + tri_verts2.shape[0]
    num_verts = num_faces * 3  # V=F*3
    overlap_faces = torch.arange(num_verts).view(num_faces, 3)  # Fx3
    overlap_tri_verts = torch.cat((tri_verts1, tri_verts2), dim=0)  # Fx3x3
    overlap_verts = overlap_tri_verts.view(num_verts, 3)  # Vx3

    # the volume of the convex hull defined by (overlap_verts, overlap_faces)
    # can be defined as the sum of all the tetrahedrons formed where for each tetrahedron
    # the base is the triangle and the apex is the center point of the convex hull
    # See the math here: https://en.wikipedia.org/wiki/Tetrahedron#Volume

    # we compute the center by computing the center point of each face
    # and then averaging the face centers
    ctr = overlap_tri_verts.mean(1).mean(0)
    tetras = overlap_tri_verts - ctr.view(1, 1, 3)
    vol = torch.sum(
        tetras[:, 0] * torch.cross(tetras[:, 1], tetras[:, 2], dim=-1), dim=-1
    )
    vol = (vol.abs() / 6.0).sum()

    iou = vol / (vol1 + vol2 - vol)

    if DEBUG:
        # save shapes
        tri_faces = torch.tensor(_box_triangles, device=device, dtype=torch.int64)
        save_obj("/tmp/output/shape1.obj", box1, tri_faces)
        save_obj("/tmp/output/shape2.obj", box2, tri_faces)
        if len(overlap_verts) > 0:
            save_obj("/tmp/output/inters_shape.obj", overlap_verts, overlap_faces)
    return vol, iou


# -------------------------------------------------- #
#       HELPER FUNCTIONS FOR SAMPLING SOLUTION       #
# -------------------------------------------------- #


def is_point_inside_box(box: torch.Tensor, points: torch.Tensor):
    """
    Determines whether points are inside the boxes
    Args:
        box: tensor of shape (8, 3) of the corners of the boxes
        points: tensor of shape (P, 3) of the points
    Returns:
        inside: bool tensor of shape (P,)
    """
    device = box.device
    P = points.shape[0]

    n = box_planar_dir(box)  # (6, 3)
    box_planes = get_plane_verts(box)  # (6, 4)
    num_planes = box_planes.shape[0]  # = 6

    # a point p is inside the box if it "inside" all planes of the box
    # so we run the checks
    ins = torch.zeros((P, num_planes), device=device, dtype=torch.bool)
    for i in range(num_planes):
        is_in, _ = is_inside(box_planes[i], n[i], points, return_proj=False)
        ins[:, i] = is_in
    ins = ins.all(dim=1)
    return ins


def sample_points_within_box(box: torch.Tensor, num_samples: int = 10):
    """
    Sample points within a box defined by its 8 coordinates
    Args:
        box: tensor of shape (8, 3) of the box coordinates
        num_samples: int defining the number of samples
    Returns:
        points: (num_samples, 3) of points inside the box
    """
    assert box.shape[0] == 8 and box.shape[1] == 3
    xyzmin = box.min(0).values.view(1, 3)
    xyzmax = box.max(0).values.view(1, 3)

    uvw = torch.rand((num_samples, 3), device=box.device)
    points = uvw * (xyzmax - xyzmin) + xyzmin

    # because the box is not axis aligned we need to check wether
    # the points are within the box
    num_points = 0
    samples = []
    while num_points < num_samples:
        inside = is_point_inside_box(box, points)
        samples.append(points[inside].view(-1, 3))
        num_points += inside.sum()

    samples = torch.cat(samples, dim=0)
    return samples[1:num_samples]


# -------------------------------------------------- #
#          MAIN: BOX3D_OVERLAP_SAMPLING              #
# -------------------------------------------------- #


def box3d_overlap_sampling(
    box1: torch.Tensor, box2: torch.Tensor, num_samples: int = 10000
):
    """
    Computes the intersection of two boxes by sampling points
    """
    vol1 = box_volume(box1)
    vol2 = box_volume(box2)

    points1 = sample_points_within_box(box1, num_samples=num_samples)
    points2 = sample_points_within_box(box2, num_samples=num_samples)

    isin21 = is_point_inside_box(box1, points2)
    num21 = isin21.sum()
    isin12 = is_point_inside_box(box2, points1)
    num12 = isin12.sum()

    assert num12 <= num_samples
    assert num21 <= num_samples

    inters = (vol1 * num12 + vol2 * num21) / 2.0
    union = vol1 * num_samples + vol2 * num_samples - inters
    return inters / union
