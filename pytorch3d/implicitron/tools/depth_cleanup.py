# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as Fu
from pytorch3d.ops import wmean
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds


def cleanup_eval_depth(
    point_cloud: Pointclouds,
    camera: CamerasBase,
    depth: torch.Tensor,
    mask: torch.Tensor,
    sigma: float = 0.01,
    image=None,
):

    ba, _, H, W = depth.shape

    pcl = point_cloud.points_padded()
    n_pts = point_cloud.num_points_per_cloud()
    pcl_mask = (
        torch.arange(pcl.shape[1], dtype=torch.int64, device=pcl.device)[None]
        < n_pts[:, None]
    ).type_as(pcl)

    pcl_proj = camera.transform_points(pcl, eps=1e-2)[..., :-1]
    pcl_depth = camera.get_world_to_view_transform().transform_points(pcl)[..., -1]

    depth_and_idx = torch.cat(
        (
            depth,
            torch.arange(H * W).view(1, 1, H, W).expand(ba, 1, H, W).type_as(depth),
        ),
        dim=1,
    )

    depth_and_idx_sampled = Fu.grid_sample(
        depth_and_idx, -pcl_proj[:, None], mode="nearest"
    )[:, :, 0].view(ba, 2, -1)

    depth_sampled, idx_sampled = depth_and_idx_sampled.split([1, 1], dim=1)
    df = (depth_sampled[:, 0] - pcl_depth).abs()

    # the threshold is a sigma-multiple of the standard deviation of the depth
    mu = wmean(depth.view(ba, -1, 1), mask.view(ba, -1)).view(ba, 1)
    std = (
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        wmean((depth.view(ba, -1) - mu).view(ba, -1, 1) ** 2, mask.view(ba, -1))
        .clamp(1e-4)
        .sqrt()
        .view(ba, -1)
    )
    good_df_thr = std * sigma
    good_depth = (df <= good_df_thr).float() * pcl_mask

    # perc_kept = good_depth.sum(dim=1) / pcl_mask.sum(dim=1).clamp(1)
    # print(f'Kept {100.0 * perc_kept.mean():1.3f} % points')

    good_depth_raster = torch.zeros_like(depth).view(ba, -1)
    good_depth_raster.scatter_add_(1, torch.round(idx_sampled[:, 0]).long(), good_depth)

    good_depth_mask = (good_depth_raster.view(ba, 1, H, W) > 0).float()

    # if float(torch.rand(1)) > 0.95:
    #     depth_ok = depth * good_depth_mask

    #     # visualize
    #     visdom_env = 'depth_cleanup_dbg'
    #     from visdom import Visdom
    #     # from tools.vis_utils import make_depth_image
    #     from pytorch3d.vis.plotly_vis import plot_scene
    #     viz = Visdom()

    #     show_pcls = {
    #         'pointclouds': point_cloud,
    #     }
    #     for d, nm in zip(
    #         (depth, depth_ok),
    #         ('pointclouds_unproj', 'pointclouds_unproj_ok'),
    #     ):
    #         pointclouds_unproj = get_rgbd_point_cloud(
    #             camera, image, d,
    #         )
    #         if int(pointclouds_unproj.num_points_per_cloud()) > 0:
    #             show_pcls[nm] = pointclouds_unproj

    #     scene_dict = {'1': {
    #         **show_pcls,
    #         'cameras': camera,
    #     }}
    #     scene = plot_scene(
    #         scene_dict,
    #         pointcloud_max_points=5000,
    #         pointcloud_marker_size=1.5,
    #         camera_scale=1.0,
    #     )
    #     viz.plotlyplot(scene, env=visdom_env, win='scene')

    #     # depth_image_ok = make_depth_image(depths_ok, masks)
    #     # viz.images(depth_image_ok, env=visdom_env, win='depth_ok')
    #     # depth_image = make_depth_image(depths, masks)
    #     # viz.images(depth_image, env=visdom_env, win='depth')
    #     # # viz.images(rgb_rendered, env=visdom_env, win='images_render')
    #     # viz.images(images, env=visdom_env, win='images')
    #     import pdb; pdb.set_trace()

    return good_depth_mask
