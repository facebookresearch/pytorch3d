# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import math
from typing import Optional, Tuple

import torch
from pytorch3d.implicitron.tools import utils
from pytorch3d.implicitron.tools.circle_fitting import fit_circle_in_3d
from pytorch3d.renderer import look_at_view_transform, PerspectiveCameras
from pytorch3d.transforms import Scale


logger = logging.getLogger(__name__)


def generate_eval_video_cameras(
    train_cameras,
    n_eval_cams: int = 100,
    trajectory_type: str = "figure_eight",
    trajectory_scale: float = 0.2,
    scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    focal_length: Optional[torch.Tensor] = None,
    principal_point: Optional[torch.Tensor] = None,
    time: Optional[torch.Tensor] = None,
    infer_up_as_plane_normal: bool = True,
    traj_offset: Optional[Tuple[float, float, float]] = None,
    traj_offset_canonical: Optional[Tuple[float, float, float]] = None,
    remove_outliers_rate: float = 0.0,
) -> PerspectiveCameras:
    """
    Generate a camera trajectory rendering a scene from multiple viewpoints.

    Args:
        train_cameras: The set of cameras from the training dataset object.
        n_eval_cams: Number of cameras in the trajectory.
        trajectory_type: The type of the camera trajectory. Can be one of:
            circular_lsq_fit: Camera centers follow a trajectory obtained
                by fitting a 3D circle to train_cameras centers.
                All cameras are looking towards scene_center.
            figure_eight: Figure-of-8 trajectory around the center of the
                central camera of the training dataset.
            trefoil_knot: Same as 'figure_eight', but the trajectory has a shape
                of a trefoil knot (https://en.wikipedia.org/wiki/Trefoil_knot).
            figure_eight_knot: Same as 'figure_eight', but the trajectory has a shape
                of a figure-eight knot
                (https://en.wikipedia.org/wiki/Figure-eight_knot_(mathematics)).
        trajectory_scale: The extent of the trajectory.
        scene_center: The center of the scene in world coordinates which all
            the cameras from the generated trajectory look at.
        up: The "circular_lsq_fit" vector of the scene (=the normal of the scene floor).
            Active for the `trajectory_type="circular"`.
        focal_length: The focal length of the output cameras. If `None`, an average
            focal length of the train_cameras is used.
        principal_point: The principal point of the output cameras. If `None`, an average
            principal point of all train_cameras is used.
        time: Defines the total length of the generated camera trajectory. All possible
            trajectories (set with the `trajectory_type` argument) are periodic with
            the period of `time=2pi`.
            E.g. setting `trajectory_type=circular_lsq_fit` and `time=4pi`, will generate
            a trajectory of camera poses rotating the total of 720 deg around the object.
        infer_up_as_plane_normal: Infer the camera `up` vector automatically as the normal
            of the plane fit to the optical centers of `train_cameras`.
        traj_offset: 3D offset vector added to each point of the trajectory.
        traj_offset_canonical: 3D offset vector expressed in the local coordinates of
            the estimated trajectory which is added to each point of the trajectory.
        remove_outliers_rate: the number between 0 and 1; if > 0,
            some outlier train_cameras will be removed from trajectory estimation;
            the filtering is based on camera center coordinates; top and
            bottom `remove_outliers_rate` cameras on each dimension are removed.
    Returns:
        Batch of camera instances which can be used as the test dataset
    """
    if remove_outliers_rate > 0.0:
        train_cameras = _remove_outlier_cameras(train_cameras, remove_outliers_rate)

    if trajectory_type in ("figure_eight", "trefoil_knot", "figure_eight_knot"):
        cam_centers = train_cameras.get_camera_center()
        # get the nearest camera center to the mean of centers
        mean_camera_idx = (
            ((cam_centers - cam_centers.mean(dim=0)[None]) ** 2)
            .sum(dim=1)
            .min(dim=0)
            .indices
        )
        # generate the knot trajectory in canonical coords
        if time is None:
            time = torch.linspace(0, 2 * math.pi, n_eval_cams + 1)[:n_eval_cams]
        else:
            assert time.numel() == n_eval_cams
        if trajectory_type == "trefoil_knot":
            traj = _trefoil_knot(time)
        elif trajectory_type == "figure_eight_knot":
            traj = _figure_eight_knot(time)
        elif trajectory_type == "figure_eight":
            traj = _figure_eight(time)
        else:
            raise ValueError(f"bad trajectory type: {trajectory_type}")
        traj[:, 2] -= traj[:, 2].max()

        # transform the canonical knot to the coord frame of the mean camera
        mean_camera = PerspectiveCameras(
            **{
                k: getattr(train_cameras, k)[[int(mean_camera_idx)]]
                for k in ("focal_length", "principal_point", "R", "T")
            }
        )
        traj_trans = Scale(cam_centers.std(dim=0).mean() * trajectory_scale).compose(
            mean_camera.get_world_to_view_transform().inverse()
        )

        if traj_offset_canonical is not None:
            traj_trans = traj_trans.translate(
                torch.FloatTensor(traj_offset_canonical)[None].to(traj)
            )

        traj = traj_trans.transform_points(traj)

        plane_normal = _fit_plane(cam_centers)[:, 0]
        if infer_up_as_plane_normal:
            up = _disambiguate_normal(plane_normal, up)

    elif trajectory_type == "circular_lsq_fit":
        ### fit plane to the camera centers

        # get the center of the plane as the median of the camera centers
        cam_centers = train_cameras.get_camera_center()

        if time is not None:
            angle = time
        else:
            angle = torch.linspace(0, 2.0 * math.pi, n_eval_cams).to(cam_centers)

        fit = fit_circle_in_3d(
            cam_centers,
            angles=angle,
            offset=(
                angle.new_tensor(traj_offset_canonical)
                if traj_offset_canonical is not None
                else None
            ),
            up=angle.new_tensor(up),
        )
        traj = fit.generated_points

        # scalethe trajectory
        _t_mu = traj.mean(dim=0, keepdim=True)
        traj = (traj - _t_mu) * trajectory_scale + _t_mu

        plane_normal = fit.normal

        if infer_up_as_plane_normal:
            up = _disambiguate_normal(plane_normal, up)

    else:
        raise ValueError(f"Uknown trajectory_type {trajectory_type}.")

    if traj_offset is not None:
        traj = traj + torch.FloatTensor(traj_offset)[None].to(traj)

    # point all cameras towards the center of the scene
    R, T = look_at_view_transform(
        eye=traj,
        at=(scene_center,),  # (1, 3)
        up=(up,),  # (1, 3)
        device=traj.device,
    )

    # get the average focal length and principal point
    if focal_length is None:
        focal_length = train_cameras.focal_length.mean(dim=0).repeat(n_eval_cams, 1)
    if principal_point is None:
        principal_point = train_cameras.principal_point.mean(dim=0).repeat(
            n_eval_cams, 1
        )

    test_cameras = PerspectiveCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R,
        T=T,
        device=focal_length.device,
    )

    # _visdom_plot_scene(
    #     train_cameras,
    #     test_cameras,
    # )

    return test_cameras


def _remove_outlier_cameras(
    cameras: PerspectiveCameras, outlier_rate: float
) -> PerspectiveCameras:
    keep_indices = utils.get_inlier_indicators(
        cameras.get_camera_center(), dim=0, outlier_rate=outlier_rate
    )
    # pyre-fixme[6]: For 1st param expected `Union[List[int], int, BoolTensor,
    #  LongTensor]` but got `Tensor`.
    clean_cameras = cameras[keep_indices]
    logger.info(
        "Filtered outlier cameras when estimating the trajectory: "
        f"{len(cameras)} → {len(clean_cameras)}"
    )
    # pyre-fixme[7]: Expected `PerspectiveCameras` but got `CamerasBase`.
    return clean_cameras


def _disambiguate_normal(normal, up):
    up_t = torch.tensor(up).to(normal)
    flip = (up_t * normal).sum().sign()
    up = normal * flip
    up = up.tolist()
    return up


def _fit_plane(x):
    x = x - x.mean(dim=0)[None]
    cov = (x.t() @ x) / x.shape[0]
    _, e_vec = torch.linalg.eigh(cov)
    return e_vec


def _visdom_plot_scene(
    train_cameras,
    test_cameras,
) -> None:
    from pytorch3d.vis.plotly_vis import plot_scene

    p = plot_scene(
        {
            "scene": {
                "train_cams": train_cameras,
                "test_cams": test_cameras,
            }
        }
    )
    from visdom import Visdom

    viz = Visdom()
    viz.plotlyplot(p, env="cam_traj_dbg", win="cam_trajs")


def _figure_eight_knot(t: torch.Tensor, z_scale: float = 0.5):
    x = (2 + (2 * t).cos()) * (3 * t).cos()
    y = (2 + (2 * t).cos()) * (3 * t).sin()
    z = (4 * t).sin() * z_scale
    return torch.stack((x, y, z), dim=-1)


def _trefoil_knot(t: torch.Tensor, z_scale: float = 0.5):
    x = t.sin() + 2 * (2 * t).sin()
    y = t.cos() - 2 * (2 * t).cos()
    z = -(3 * t).sin() * z_scale
    return torch.stack((x, y, z), dim=-1)


def _figure_eight(t: torch.Tensor, z_scale: float = 0.5):
    x = t.cos()
    y = (2 * t).sin() / 2
    z = t.sin() * z_scale
    return torch.stack((x, y, z), dim=-1)
