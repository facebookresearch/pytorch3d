# @lint-ignore-every LICENSELINT
# Adapted from https://github.com/bmild/nerf/blob/master/load_llff.py
# Copyright (c) 2020 bmild

# pyre-unsafe
import logging
import os
import warnings

import numpy as np

from PIL import Image


# Slightly modified version of LLFF data loading code
#  see https://github.com/Fyusion/LLFF for original

logger = logging.getLogger(__name__)


def _minify(basedir, path_manager, factors=(), resolutions=()):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, "images_{}".format(r))
        if not _exists(path_manager, imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, "images_{}x{}".format(r[1], r[0]))
        if not _exists(path_manager, imgdir):
            needtoload = True
    if not needtoload:
        return
    assert path_manager is None

    from subprocess import check_output

    imgdir = os.path.join(basedir, "images")
    imgs = [os.path.join(imgdir, f) for f in sorted(_ls(path_manager, imgdir))]
    imgs = [f for f in imgs if f.endswith("JPG", "jpg", "png", "jpeg", "PNG")]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = "images_{}".format(r)
            resizearg = "{}%".format(100.0 / r)
        else:
            name = "images_{}x{}".format(r[1], r[0])
            resizearg = "{}x{}".format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        logger.info(f"Minifying {r}, {basedir}")

        os.makedirs(imgdir)
        check_output("cp {}/* {}".format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split(".")[-1]
        args = " ".join(
            ["mogrify", "-resize", resizearg, "-format", "png", "*.{}".format(ext)]
        )
        logger.info(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != "png":
            check_output("rm {}/*.{}".format(imgdir, ext), shell=True)
            logger.info("Removed duplicates")
        logger.info("Done")


def _load_data(
    basedir, factor=None, width=None, height=None, load_imgs=True, path_manager=None
):
    poses_arr = np.load(
        _local_path(path_manager, os.path.join(basedir, "poses_bounds.npy"))
    )
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [
        os.path.join(basedir, "images", f)
        for f in sorted(_ls(path_manager, os.path.join(basedir, "images")))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ][0]

    def imread(f):
        return np.array(Image.open(f))

    sh = imread(_local_path(path_manager, img0)).shape

    sfx = ""

    if factor is not None:
        sfx = "_{}".format(factor)
        _minify(basedir, path_manager, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, path_manager, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, path_manager, resolutions=[[height, width]])
        sfx = "_{}x{}".format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, "images" + sfx)
    if not _exists(path_manager, imgdir):
        raise ValueError(f"{imgdir} does not exist, returning")

    imgfiles = [
        _local_path(path_manager, os.path.join(imgdir, f))
        for f in sorted(_ls(path_manager, imgdir))
        if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
    ]
    if poses.shape[-1] != len(imgfiles):
        raise ValueError(
            "Mismatch between imgs {} and poses {} !!!!".format(
                len(imgfiles), poses.shape[-1]
            )
        )

    sh = imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    if not load_imgs:
        return poses, bds

    imgs = imgs = [imread(f)[..., :3] / 255.0 for f in imgfiles]
    imgs = np.stack(imgs, -1)

    logger.info(f"Loaded image data, shape {imgs.shape}")
    return poses, bds, imgs


def normalize(x):
    denom = np.linalg.norm(x)
    if denom < 0.001:
        warnings.warn("unsafe normalize()")
    return x / denom


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.0])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


def spherify_poses(poses, bds):
    def add_row_to_homogenize_transform(p):
        r"""Add the last row to homogenize 3 x 4 transformation matrices."""
        return np.concatenate(
            [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
        )

    # p34_to_44 = lambda p: np.concatenate(
    #     [p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1
    # )

    p34_to_44 = add_row_to_homogenize_transform

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(
            -np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0)
        )
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([0.1, 0.2, 0.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1.0 / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2 - zh**2)
    new_poses = []

    for th in np.linspace(0.0, 2.0 * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.0])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate(
        [new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1
    )
    poses_reset = np.concatenate(
        [
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape),
        ],
        -1,
    )

    return poses_reset, new_poses, bds


def _local_path(path_manager, path):
    if path_manager is None:
        return path
    return path_manager.get_local_path(path)


def _ls(path_manager, path):
    if path_manager is None:
        return os.listdir(path)
    return path_manager.ls(path)


def _exists(path_manager, path):
    if path_manager is None:
        return os.path.exists(path)
    return path_manager.exists(path)


def load_llff_data(
    basedir,
    factor=8,
    recenter=True,
    bd_factor=0.75,
    spherify=False,
    path_zflat=False,
    path_manager=None,
):
    poses, bds, imgs = _load_data(
        basedir, factor=factor, path_manager=path_manager
    )  # factor=8 downsamples original imgs by 8x
    logger.info(f"Loaded {basedir}, {bds.min()}, {bds.max()}")

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds
