#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings

import hydra
import numpy as np
import torch
from nerf.dataset import get_nerf_datasets, trivial_collate
from nerf.eval_video_utils import generate_eval_video_cameras
from nerf.nerf_renderer import RadianceFieldRenderer
from nerf.stats import Stats
from omegaconf import DictConfig
from PIL import Image


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="lego")
def main(cfg: DictConfig):
    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        warnings.warn(
            "Please note that although executing on CPU is supported,"
            + "the testing is unlikely to finish in reasonable time."
        )
        device = "cpu"

    # Initialize the Radiance Field model.
    model = RadianceFieldRenderer(
        image_size=cfg.data.image_size,
        n_pts_per_ray=cfg.raysampler.n_pts_per_ray,
        n_pts_per_ray_fine=cfg.raysampler.n_pts_per_ray,
        n_rays_per_image=cfg.raysampler.n_rays_per_image,
        min_depth=cfg.raysampler.min_depth,
        max_depth=cfg.raysampler.max_depth,
        stratified=cfg.raysampler.stratified,
        stratified_test=cfg.raysampler.stratified_test,
        chunk_size_test=cfg.raysampler.chunk_size_test,
        n_harmonic_functions_xyz=cfg.implicit_function.n_harmonic_functions_xyz,
        n_harmonic_functions_dir=cfg.implicit_function.n_harmonic_functions_dir,
        n_hidden_neurons_xyz=cfg.implicit_function.n_hidden_neurons_xyz,
        n_hidden_neurons_dir=cfg.implicit_function.n_hidden_neurons_dir,
        n_layers_xyz=cfg.implicit_function.n_layers_xyz,
        density_noise_std=cfg.implicit_function.density_noise_std,
    )

    # Move the model to the relevant device.
    model.to(device)

    # Resume from the checkpoint.
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise ValueError(f"Model checkpoint {checkpoint_path} does not exist!")

    print(f"Loading checkpoint {checkpoint_path}.")
    loaded_data = torch.load(checkpoint_path, weights_only=True)
    # Do not load the cached xy grid.
    # - this allows setting an arbitrary evaluation image size.
    state_dict = {
        k: v
        for k, v in loaded_data["model"].items()
        if "_grid_raysampler._xy_grid" not in k
    }
    model.load_state_dict(state_dict, strict=False)

    # Load the test data.
    if cfg.test.mode == "evaluation":
        _, _, test_dataset = get_nerf_datasets(
            dataset_name=cfg.data.dataset_name,
            image_size=cfg.data.image_size,
        )
    elif cfg.test.mode == "export_video":
        train_dataset, _, _ = get_nerf_datasets(
            dataset_name=cfg.data.dataset_name,
            image_size=cfg.data.image_size,
        )
        test_dataset = generate_eval_video_cameras(
            train_dataset,
            trajectory_type=cfg.test.trajectory_type,
            up=cfg.test.up,
            scene_center=cfg.test.scene_center,
            n_eval_cams=cfg.test.n_frames,
            trajectory_scale=cfg.test.trajectory_scale,
        )
        # store the video in directory (checkpoint_file - extension + '_video')
        export_dir = os.path.splitext(checkpoint_path)[0] + "_video"
        os.makedirs(export_dir, exist_ok=True)
    else:
        raise ValueError(f"Unknown test mode {cfg.test_mode}.")

    # Init the test dataloader.
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    if cfg.test.mode == "evaluation":
        # Init the test stats object.
        eval_stats = ["mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"]
        stats = Stats(eval_stats)
        stats.new_epoch()
    elif cfg.test.mode == "export_video":
        # Init the frame buffer.
        frame_paths = []

    # Set the model to the eval mode.
    model.eval()

    # Run the main testing loop.
    for batch_idx, test_batch in enumerate(test_dataloader):
        test_image, test_camera, camera_idx = test_batch[0].values()
        if test_image is not None:
            test_image = test_image.to(device)
        test_camera = test_camera.to(device)

        # Activate eval mode of the model (lets us do a full rendering pass).
        model.eval()
        with torch.no_grad():
            test_nerf_out, test_metrics = model(
                None,  # we do not use pre-cached cameras
                test_camera,
                test_image,
            )

        if cfg.test.mode == "evaluation":
            # Update stats with the validation metrics.
            stats.update(test_metrics, stat_set="test")
            stats.print(stat_set="test")

        elif cfg.test.mode == "export_video":
            # Store the video frame.
            frame = test_nerf_out["rgb_fine"][0].detach().cpu()
            frame_path = os.path.join(export_dir, f"frame_{batch_idx:05d}.png")
            print(f"Writing {frame_path}.")
            Image.fromarray((frame.numpy() * 255.0).astype(np.uint8)).save(frame_path)
            frame_paths.append(frame_path)

    if cfg.test.mode == "evaluation":
        print(f"Final evaluation metrics on '{cfg.data.dataset_name}':")
        for stat in eval_stats:
            stat_value = stats.stats["test"][stat].get_epoch_averages()[0]
            print(f"{stat:15s}: {stat_value:1.4f}")

    elif cfg.test.mode == "export_video":
        # Convert the exported frames to a video.
        video_path = os.path.join(export_dir, "video.mp4")
        ffmpeg_bin = "ffmpeg"
        frame_regexp = os.path.join(export_dir, "frame_%05d.png")
        ffmcmd = (
            "%s -r %d -i %s -vcodec h264 -f mp4 -y -b 2000k -pix_fmt yuv420p %s"
            % (ffmpeg_bin, cfg.test.fps, frame_regexp, video_path)
        )
        ret = os.system(ffmcmd)
        if ret != 0:
            raise RuntimeError("ffmpeg failed!")


if __name__ == "__main__":
    main()
