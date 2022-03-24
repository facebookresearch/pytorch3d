#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Script to visualize a previously trained model. Example call:

    projects/implicitron_trainer/visualize_reconstruction.py
    exp_dir='./exps/checkpoint_dir' visdom_show_preds=True visdom_port=8097
    n_eval_cameras=40 render_size="[64,64]" video_size="[256,256]"
"""

import math
import os
import random
import sys
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as Fu
from experiment import init_model
from omegaconf import OmegaConf
from pytorch3d.implicitron.dataset.dataset_zoo import dataset_zoo
from pytorch3d.implicitron.dataset.implicitron_dataset import (
    FrameData,
    ImplicitronDataset,
)
from pytorch3d.implicitron.dataset.utils import is_train_frame
from pytorch3d.implicitron.models.base import EvaluationMode
from pytorch3d.implicitron.tools.configurable import get_default_args
from pytorch3d.implicitron.tools.eval_video_trajectory import (
    generate_eval_video_cameras,
)
from pytorch3d.implicitron.tools.video_writer import VideoWriter
from pytorch3d.implicitron.tools.vis_utils import (
    get_visdom_connection,
    make_depth_image,
)
from tqdm import tqdm


def render_sequence(
    dataset: ImplicitronDataset,
    sequence_name: str,
    model: torch.nn.Module,
    video_path,
    n_eval_cameras=40,
    fps=20,
    max_angle=2 * math.pi,
    trajectory_type="circular_lsq_fit",
    trajectory_scale=1.1,
    scene_center=(0.0, 0.0, 0.0),
    up=(0.0, -1.0, 0.0),
    traj_offset=0.0,
    n_source_views=9,
    viz_env="debug",
    visdom_show_preds=False,
    visdom_server="http://127.0.0.1",
    visdom_port=8097,
    num_workers=10,
    seed=None,
    video_resize=None,
):
    if seed is None:
        seed = hash(sequence_name)
    print(f"Loading all data of sequence '{sequence_name}'.")
    seq_idx = list(dataset.sequence_indices_in_order(sequence_name))
    train_data = _load_whole_dataset(dataset, seq_idx, num_workers=num_workers)
    assert all(train_data.sequence_name[0] == sn for sn in train_data.sequence_name)
    sequence_set_name = "train" if is_train_frame(train_data.frame_type)[0] else "test"
    print(f"Sequence set = {sequence_set_name}.")
    train_cameras = train_data.camera
    time = torch.linspace(0, max_angle, n_eval_cameras + 1)[:n_eval_cameras]
    test_cameras = generate_eval_video_cameras(
        train_cameras,
        time=time,
        n_eval_cams=n_eval_cameras,
        trajectory_type=trajectory_type,
        trajectory_scale=trajectory_scale,
        scene_center=scene_center,
        up=up,
        focal_length=None,
        principal_point=torch.zeros(n_eval_cameras, 2),
        traj_offset_canonical=[0.0, 0.0, traj_offset],
    )

    # sample the source views reproducibly
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        source_views_i = torch.randperm(len(seq_idx))[:n_source_views]
    # add the first dummy view that will get replaced with the target camera
    source_views_i = Fu.pad(source_views_i, [1, 0])
    source_views = [seq_idx[i] for i in source_views_i.tolist()]
    batch = _load_whole_dataset(dataset, source_views, num_workers=num_workers)
    assert all(batch.sequence_name[0] == sn for sn in batch.sequence_name)

    preds_total = []
    for n in tqdm(range(n_eval_cameras), total=n_eval_cameras):
        # set the first batch camera to the target camera
        for k in ("R", "T", "focal_length", "principal_point"):
            getattr(batch.camera, k)[0] = getattr(test_cameras[n], k)

        # Move to cuda
        net_input = batch.cuda()
        with torch.no_grad():
            preds = model(**{**net_input, "evaluation_mode": EvaluationMode.EVALUATION})

            # make sure we dont overwrite something
            assert all(k not in preds for k in net_input.keys())
            preds.update(net_input)  # merge everything into one big dict

            # Render the predictions to images
            rendered_pred = images_from_preds(preds)
            preds_total.append(rendered_pred)

            # show the preds every 5% of the export iterations
            if visdom_show_preds and (
                n % max(n_eval_cameras // 20, 1) == 0 or n == n_eval_cameras - 1
            ):
                viz = get_visdom_connection(server=visdom_server, port=visdom_port)
                show_predictions(
                    preds_total,
                    sequence_name=batch.sequence_name[0],
                    viz=viz,
                    viz_env=viz_env,
                )

    print(f"Exporting videos for sequence {sequence_name} ...")
    generate_prediction_videos(
        preds_total,
        sequence_name=batch.sequence_name[0],
        viz=viz,
        viz_env=viz_env,
        fps=fps,
        video_path=video_path,
        resize=video_resize,
    )


def _load_whole_dataset(dataset, idx, num_workers=10):
    load_all_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, idx),
        batch_size=len(idx),
        num_workers=num_workers,
        shuffle=False,
        collate_fn=FrameData.collate,
    )
    return next(iter(load_all_dataloader))


def images_from_preds(preds):
    imout = {}
    for k in (
        "image_rgb",
        "images_render",
        "fg_probability",
        "masks_render",
        "depths_render",
        "depth_map",
        "_all_source_images",
    ):
        if k == "_all_source_images" and "image_rgb" in preds:
            src_ims = preds["image_rgb"][1:].cpu().detach().clone()
            v = _stack_images(src_ims, None)[None]
        else:
            if k not in preds or preds[k] is None:
                print(f"cant show {k}")
                continue
            v = preds[k].cpu().detach().clone()
        if k.startswith("depth"):
            mask_resize = Fu.interpolate(
                preds["masks_render"],
                size=preds[k].shape[2:],
                mode="nearest",
            )
            v = make_depth_image(preds[k], mask_resize)
        if v.shape[1] == 1:
            v = v.repeat(1, 3, 1, 1)
        imout[k] = v.detach().cpu()

    return imout


def _stack_images(ims, size):
    ba = ims.shape[0]
    H = int(np.ceil(np.sqrt(ba)))
    W = H
    n_add = H * W - ba
    if n_add > 0:
        ims = torch.cat((ims, torch.zeros_like(ims[:1]).repeat(n_add, 1, 1, 1)))

    ims = ims.view(H, W, *ims.shape[1:])
    cated = torch.cat([torch.cat(list(row), dim=2) for row in ims], dim=1)
    if size is not None:
        cated = Fu.interpolate(cated[None], size=size, mode="bilinear")[0]
    return cated.clamp(0.0, 1.0)


def show_predictions(
    preds,
    sequence_name,
    viz,
    viz_env="visualizer",
    predicted_keys=(
        "images_render",
        "masks_render",
        "depths_render",
        "_all_source_images",
    ),
    n_samples=10,
    one_image_width=200,
):
    """Given a list of predictions visualize them into a single image using visdom."""
    assert isinstance(preds, list)

    pred_all = []
    # Randomly choose a subset of the rendered images, sort by ordr in the sequence
    n_samples = min(n_samples, len(preds))
    pred_idx = sorted(random.sample(list(range(len(preds))), n_samples))
    for predi in pred_idx:
        # Make the concatentation for the same camera vertically
        pred_all.append(
            torch.cat(
                [
                    torch.nn.functional.interpolate(
                        preds[predi][k].cpu(),
                        scale_factor=one_image_width / preds[predi][k].shape[3],
                        mode="bilinear",
                    ).clamp(0.0, 1.0)
                    for k in predicted_keys
                ],
                dim=2,
            )
        )
    # Concatenate the images horizontally
    pred_all_cat = torch.cat(pred_all, dim=3)[0]
    viz.image(
        pred_all_cat,
        win="show_predictions",
        env=viz_env,
        opts={"title": f"pred_{sequence_name}"},
    )


def generate_prediction_videos(
    preds,
    sequence_name,
    viz,
    viz_env="visualizer",
    predicted_keys=(
        "images_render",
        "masks_render",
        "depths_render",
        "_all_source_images",
    ),
    fps=20,
    video_path="/tmp/video",
    resize=None,
):
    """Given a list of predictions create and visualize rotating videos of the
    objects using visdom.
    """
    assert isinstance(preds, list)

    # make sure the target video directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # init a video writer for each predicted key
    vws = {}
    for k in predicted_keys:
        vws[k] = VideoWriter(out_path=f"{video_path}_{sequence_name}_{k}.mp4", fps=fps)

    for rendered_pred in tqdm(preds):
        for k in predicted_keys:
            vws[k].write_frame(
                rendered_pred[k][0].detach().cpu().numpy(),
                resize=resize,
            )

    for k in predicted_keys:
        vws[k].get_video(quiet=True)
        print(f"Generated {vws[k].out_path}.")
        viz.video(
            videofile=vws[k].out_path,
            env=viz_env,
            win=k,  # we reuse the same window otherwise visdom dies
            opts={"title": sequence_name + " " + k},
        )


def export_scenes(
    exp_dir: str = "",
    restrict_sequence_name: Optional[str] = None,
    output_directory: Optional[str] = None,
    render_size: Tuple[int, int] = (512, 512),
    video_size: Optional[Tuple[int, int]] = None,
    split: str = "train",  # train | test
    n_source_views: int = 9,
    n_eval_cameras: int = 40,
    visdom_server="http://127.0.0.1",
    visdom_port=8097,
    visdom_show_preds: bool = False,
    visdom_env: Optional[str] = None,
    gpu_idx: int = 0,
):
    # In case an output directory is specified use it. If no output_directory
    # is specified create a vis folder inside the experiment directory
    if output_directory is None:
        output_directory = os.path.join(exp_dir, "vis")
    else:
        output_directory = output_directory
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Set the random seeds
    torch.manual_seed(0)
    np.random.seed(0)

    # Get the config from the experiment_directory,
    # and overwrite relevant fields
    config = _get_config_from_experiment_directory(exp_dir)
    config.gpu_idx = gpu_idx
    config.exp_dir = exp_dir
    # important so that the CO3D dataset gets loaded in full
    config.dataset_args.test_on_train = False
    # Set the rendering image size
    config.generic_model_args.render_image_width = render_size[0]
    config.generic_model_args.render_image_height = render_size[1]
    if restrict_sequence_name is not None:
        config.dataset_args.restrict_sequence_name = restrict_sequence_name

    # Set up the CUDA env for the visualization
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_idx)

    # Load the previously trained model
    model, _, _ = init_model(config, force_load=True, load_model_only=True)
    model.cuda()
    model.eval()

    # Setup the dataset
    dataset = dataset_zoo(**config.dataset_args)[split]

    # iterate over the sequences in the dataset
    for sequence_name in dataset.sequence_names():
        with torch.no_grad():
            render_sequence(
                dataset,
                sequence_name,
                model,
                video_path="{}/video".format(output_directory),
                n_source_views=n_source_views,
                visdom_show_preds=visdom_show_preds,
                n_eval_cameras=n_eval_cameras,
                visdom_server=visdom_server,
                visdom_port=visdom_port,
                viz_env=f"visualizer_{config.visdom_env}"
                if visdom_env is None
                else visdom_env,
                video_resize=video_size,
            )


def _get_config_from_experiment_directory(experiment_directory):
    cfg_file = os.path.join(experiment_directory, "expconfig.yaml")
    config = OmegaConf.load(cfg_file)
    return config


def main(argv):
    # automatically parses arguments of export_scenes
    cfg = OmegaConf.create(get_default_args(export_scenes))
    cfg.update(OmegaConf.from_cli())
    with torch.no_grad():
        export_scenes(**cfg)


if __name__ == "__main__":
    main(sys.argv)
