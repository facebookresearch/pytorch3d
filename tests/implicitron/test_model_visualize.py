# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import os
import unittest
from typing import Tuple

import torch
from pytorch3d.implicitron.dataset.json_index_dataset import JsonIndexDataset
from pytorch3d.implicitron.dataset.visualize import get_implicitron_sequence_pointcloud

from pytorch3d.implicitron.models.visualization.render_flyaround import render_flyaround
from pytorch3d.implicitron.tools.config import expand_args_fields
from pytorch3d.implicitron.tools.point_cloud_utils import render_point_cloud_pytorch3d
from pytorch3d.renderer.cameras import CamerasBase
from tests.common_testing import interactive_testing_requested
from visdom import Visdom

from .common_resources import get_skateboard_data


class TestModelVisualize(unittest.TestCase):
    def test_flyaround_one_sequence(
        self,
        image_size: int = 256,
    ):
        if not interactive_testing_requested():
            return
        category = "skateboard"
        stack = contextlib.ExitStack()
        dataset_root, path_manager = stack.enter_context(get_skateboard_data())
        self.addCleanup(stack.close)
        frame_file = os.path.join(dataset_root, category, "frame_annotations.jgz")
        sequence_file = os.path.join(dataset_root, category, "sequence_annotations.jgz")
        subset_lists_file = os.path.join(dataset_root, category, "set_lists.json")
        expand_args_fields(JsonIndexDataset)
        train_dataset = JsonIndexDataset(
            frame_annotations_file=frame_file,
            sequence_annotations_file=sequence_file,
            subset_lists_file=subset_lists_file,
            dataset_root=dataset_root,
            image_height=image_size,
            image_width=image_size,
            box_crop=True,
            load_point_clouds=True,
            path_manager=path_manager,
            subsets=[
                "train_known",
            ],
        )

        # select few sequences to visualize
        sequence_names = list(train_dataset.seq_annots.keys())

        # select the first sequence name
        show_sequence_name = sequence_names[0]

        output_dir = os.path.split(os.path.abspath(__file__))[0]

        visdom_show_preds = Visdom().check_connection()

        for load_dataset_pointcloud in [True, False]:

            model = _PointcloudRenderingModel(
                train_dataset,
                show_sequence_name,
                device="cuda:0",
                load_dataset_pointcloud=load_dataset_pointcloud,
            )

            video_path = os.path.join(
                output_dir,
                f"load_pcl_{load_dataset_pointcloud}",
            )

            os.makedirs(output_dir, exist_ok=True)

            for output_video_frames_dir in [None, video_path]:
                render_flyaround(
                    train_dataset,
                    show_sequence_name,
                    model,
                    video_path,
                    n_flyaround_poses=10,
                    fps=5,
                    max_angle=2 * math.pi,
                    trajectory_type="circular_lsq_fit",
                    trajectory_scale=1.1,
                    scene_center=(0.0, 0.0, 0.0),
                    up=(0.0, 1.0, 0.0),
                    traj_offset=1.0,
                    n_source_views=1,
                    visdom_show_preds=visdom_show_preds,
                    visdom_environment="test_model_visalize",
                    visdom_server="http://127.0.0.1",
                    visdom_port=8097,
                    num_workers=10,
                    seed=None,
                    video_resize=None,
                    visualize_preds_keys=[
                        "images_render",
                        "depths_render",
                        "masks_render",
                        "_all_source_images",
                    ],
                    output_video_frames_dir=output_video_frames_dir,
                )


class _PointcloudRenderingModel(torch.nn.Module):
    def __init__(
        self,
        train_dataset: JsonIndexDataset,
        sequence_name: str,
        render_size: Tuple[int, int] = (400, 400),
        device=None,
        load_dataset_pointcloud: bool = False,
        max_frames: int = 30,
        num_workers: int = 10,
    ):
        super().__init__()
        self._render_size = render_size
        point_cloud, _ = get_implicitron_sequence_pointcloud(
            train_dataset,
            sequence_name=sequence_name,
            mask_points=True,
            max_frames=max_frames,
            num_workers=num_workers,
            load_dataset_point_cloud=load_dataset_pointcloud,
        )
        self._point_cloud = point_cloud.to(device)

    def forward(
        self,
        camera: CamerasBase,
        **kwargs,
    ):
        image_render, mask_render, depth_render = render_point_cloud_pytorch3d(
            camera[0],
            self._point_cloud,
            render_size=self._render_size,
            point_radius=1e-2,
            topk=10,
            bg_color=0.0,
        )
        return {
            "images_render": image_render.clamp(0.0, 1.0),
            "masks_render": mask_render,
            "depths_render": depth_render,
        }
