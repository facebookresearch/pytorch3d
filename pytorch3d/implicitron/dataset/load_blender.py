# @lint-ignore-every LICENSELINT
# Adapted from https://github.com/bmild/nerf/blob/master/load_blender.py
# Copyright (c) 2020 bmild

# pyre-unsafe
import json
import os

import numpy as np
import torch
from PIL import Image


def translate_by_t_along_z(t):
    tform = np.eye(4).astype(np.float32)
    tform[2][3] = t
    return tform


def rotate_by_phi_along_x(phi):
    tform = np.eye(4).astype(np.float32)
    tform[1, 1] = tform[2, 2] = np.cos(phi)
    tform[1, 2] = -np.sin(phi)
    tform[2, 1] = -tform[1, 2]
    return tform


def rotate_by_theta_along_y(theta):
    tform = np.eye(4).astype(np.float32)
    tform[0, 0] = tform[2, 2] = np.cos(theta)
    tform[0, 2] = -np.sin(theta)
    tform[2, 0] = -tform[0, 2]
    return tform


def pose_spherical(theta, phi, radius):
    c2w = translate_by_t_along_z(radius)
    c2w = rotate_by_phi_along_x(phi / 180.0 * np.pi) @ c2w
    c2w = rotate_by_theta_along_y(theta / 180 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def _local_path(path_manager, path):
    if path_manager is None:
        return path
    return path_manager.get_local_path(path)


def load_blender_data(
    basedir,
    half_res=False,
    testskip=1,
    debug=False,
    path_manager=None,
    focal_length_in_screen_space=False,
):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        path = os.path.join(basedir, f"transforms_{s}.json")
        with open(_local_path(path_manager, path)) as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s == "train" or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            imgs.append(np.array(Image.open(_local_path(path_manager, fname))))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs) / 255.0).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    if focal_length_in_screen_space:
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    else:
        focal = 1 / np.tan(0.5 * camera_angle_x)

    render_poses = torch.stack(
        [
            torch.from_numpy(pose_spherical(angle, -30.0, 4.0))
            for angle in np.linspace(-180, 180, 40 + 1)[:-1]
        ],
        0,
    )

    # In debug mode, return extremely tiny images
    if debug:
        import cv2

        H = H // 32
        W = W // 32
        if focal_length_in_screen_space:
            focal = focal / 32.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(25, 25), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)
        poses = torch.from_numpy(poses)
        return imgs, poses, render_poses, [H, W, focal], i_split

    if half_res:
        import cv2

        # TODO: resize images using INTER_AREA (cv2)
        H = H // 2
        W = W // 2
        if focal_length_in_screen_space:
            focal = focal / 2.0
        imgs = [
            torch.from_numpy(
                cv2.resize(imgs[i], dsize=(400, 400), interpolation=cv2.INTER_AREA)
            )
            for i in range(imgs.shape[0])
        ]
        imgs = torch.stack(imgs, 0)

    poses = torch.from_numpy(poses)

    return imgs, poses, render_poses, [H, W, focal], i_split
