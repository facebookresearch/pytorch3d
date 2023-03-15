# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.dataset_base import FrameData
from pytorch3d.implicitron.dataset.utils import _get_bbox_from_mask
from pytorch3d.io import IO
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures.pointclouds import Pointclouds


@dataclass
class BlobLoader:
    """
    A loader for correctly (according to setup) loading blobs for FrameData.
    Beware that modification done in place

    Args:
        dataset_root: The root folder of the dataset; all the paths in jsons are
                specified relative to this root (but not json paths themselves).
        load_images: Enable loading the frame RGB data.
        load_depths: Enable loading the frame depth maps.
        load_depth_masks: Enable loading the frame depth map masks denoting the
            depth values used for evaluation (the points consistent across views).
        load_masks: Enable loading frame foreground masks.
        load_point_clouds: Enable loading sequence-level point clouds.
        max_points: Cap on the number of loaded points in the point cloud;
            if reached, they are randomly sampled without replacement.
        mask_images: Whether to mask the images with the loaded foreground masks;
            0 value is used for background.
        mask_depths: Whether to mask the depth maps with the loaded foreground
            masks; 0 value is used for background.
        image_height: The height of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        image_width: The width of the returned images, masks, and depth maps;
            aspect ratio is preserved during cropping/resizing.
        box_crop: Enable cropping of the image around the bounding box inferred
            from the foreground region of the loaded segmentation mask; masks
            and depth maps are cropped accordingly; cameras are corrected.
        box_crop_mask_thr: The threshold used to separate pixels into foreground
            and background based on the foreground_probability mask; if no value
            is greater than this threshold, the loader lowers it and repeats.
        box_crop_context: The amount of additional padding added to each
            dimension of the cropping bounding box, relative to box size.
    """

    dataset_root: str = ""
    load_images: bool = True
    load_depths: bool = True
    load_depth_masks: bool = True
    load_masks: bool = True
    load_point_clouds: bool = False
    max_points: int = 0
    mask_images: bool = False
    mask_depths: bool = False
    image_height: Optional[int] = 800
    image_width: Optional[int] = 800
    box_crop: bool = True
    box_crop_mask_thr: float = 0.4
    box_crop_context: float = 0.3
    path_manager: Any = None

    def load_(
        self,
        frame_data: FrameData,
        entry: types.FrameAnnotation,
        seq_annotation: types.SequenceAnnotation,
        bbox_xywh: Optional[torch.Tensor] = None,
    ) -> FrameData:
        """Main method for loader.
        FrameData modification done inplace
        if bbox_xywh not provided bbox will be calculated from mask
        """
        (
            frame_data.fg_probability,
            frame_data.mask_path,
            frame_data.bbox_xywh,
        ) = self._load_fg_probability(entry, bbox_xywh)

        if self.load_images and entry.image is not None:
            # original image size
            frame_data.image_size_hw = _safe_as_tensor(entry.image.size, torch.long)

            (
                frame_data.image_rgb,
                frame_data.image_path,
            ) = self._load_images(entry, frame_data.fg_probability)

        if self.load_depths and entry.depth is not None:
            (
                frame_data.depth_map,
                frame_data.depth_path,
                frame_data.depth_mask,
            ) = self._load_mask_depth(entry, frame_data.fg_probability)

        if self.load_point_clouds and seq_annotation.point_cloud is not None:
            pcl_path = self._fix_point_cloud_path(seq_annotation.point_cloud.path)
            frame_data.sequence_point_cloud = _load_pointcloud(
                self._local_path(pcl_path), max_points=self.max_points
            )
            frame_data.sequence_point_cloud_path = pcl_path

        clamp_bbox_xyxy = None
        if self.box_crop:
            clamp_bbox_xyxy = frame_data.crop_by_bbox_(self.box_crop_context)

        scale = (
            min(
                self.image_height / entry.image.size[0],
                # pyre-ignore
                self.image_width / entry.image.size[1],
            )
            if self.image_height is not None and self.image_width is not None
            else 1.0
        )

        if self.image_height is not None and self.image_width is not None:
            optional_scale = frame_data.resize_frame_(
                self.image_height, self.image_width
            )
            scale = optional_scale or scale

        # creating camera taking to account bbox and resize scale
        if entry.viewpoint is not None:
            frame_data.camera = self._get_pytorch3d_camera(
                entry, scale, clamp_bbox_xyxy
            )
        return frame_data

    def _load_fg_probability(
        self,
        entry: types.FrameAnnotation,
        bbox_xywh: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[str], Optional[torch.Tensor]]:
        fg_probability = None
        full_path = None

        if (self.load_masks) and entry.mask is not None:
            full_path = os.path.join(self.dataset_root, entry.mask.path)
            fg_probability = _load_mask(self._local_path(full_path))
            # we can use provided bbox_xywh or calculate it based on mask
            if bbox_xywh is None:
                bbox_xywh = _get_bbox_from_mask(fg_probability, self.box_crop_mask_thr)
            if fg_probability.shape[-2:] != entry.image.size:
                raise ValueError(
                    f"bad mask size: {fg_probability.shape[-2:]} vs {entry.image.size}!"
                )

        return (
            _safe_as_tensor(fg_probability, torch.float),
            full_path,
            _safe_as_tensor(bbox_xywh, torch.long),
        )

    def _load_images(
        self,
        entry: types.FrameAnnotation,
        fg_probability: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str]:
        assert self.dataset_root is not None and entry.image is not None
        path = os.path.join(self.dataset_root, entry.image.path)
        image_rgb = _load_image(self._local_path(path))

        if image_rgb.shape[-2:] != entry.image.size:
            raise ValueError(
                f"bad image size: {image_rgb.shape[-2:]} vs {entry.image.size}!"
            )

        if self.mask_images:
            assert fg_probability is not None
            image_rgb *= fg_probability

        return image_rgb, path

    def _load_mask_depth(
        self,
        entry: types.FrameAnnotation,
        fg_probability: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        entry_depth = entry.depth
        assert entry_depth is not None
        path = os.path.join(self.dataset_root, entry_depth.path)
        depth_map = _load_depth(self._local_path(path), entry_depth.scale_adjustment)

        if self.mask_depths:
            assert fg_probability is not None
            depth_map *= fg_probability

        if self.load_depth_masks:
            assert entry_depth.mask_path is not None
            mask_path = os.path.join(self.dataset_root, entry_depth.mask_path)
            depth_mask = _load_depth_mask(self._local_path(mask_path))
        else:
            depth_mask = torch.ones_like(depth_map)

        return torch.tensor(depth_map), path, torch.tensor(depth_mask)

    def _get_pytorch3d_camera(
        self,
        entry: types.FrameAnnotation,
        scale: float,
        clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> PerspectiveCameras:
        entry_viewpoint = entry.viewpoint
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

        half_image_size_wh_orig = (
            torch.tensor(list(reversed(entry.image.size)), dtype=torch.float) / 2.0
        )

        # first, we convert from the dataset's NDC convention to pixels
        format = entry_viewpoint.intrinsics_format
        if format.lower() == "ndc_norm_image_bounds":
            # this is e.g. currently used in CO3D for storing intrinsics
            rescale = half_image_size_wh_orig
        elif format.lower() == "ndc_isotropic":
            rescale = half_image_size_wh_orig.min()
        else:
            raise ValueError(f"Unknown intrinsics format: {format}")

        # principal point and focal length in pixels
        principal_point_px = half_image_size_wh_orig - principal_point * rescale
        focal_length_px = focal_length * rescale

        # changing principal_point according to bbox_crop
        if clamp_bbox_xyxy is not None:
            principal_point_px -= clamp_bbox_xyxy[:2]

        # now, convert from pixels to PyTorch3D v0.5+ NDC convention
        if self.image_height is None or self.image_width is None:
            out_size = list(reversed(entry.image.size))
        else:
            out_size = [self.image_width, self.image_height]

        half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
        half_min_image_size_output = half_image_size_output.min()

        # rescaled principal point and focal length in ndc
        principal_point = (
            half_image_size_output - principal_point_px * scale
        ) / half_min_image_size_output
        focal_length = focal_length_px * scale / half_min_image_size_output

        return PerspectiveCameras(
            focal_length=focal_length[None],
            principal_point=principal_point[None],
            R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
            T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
        )

    def _fix_point_cloud_path(self, path: str) -> str:
        """
        Fix up a point cloud path from the dataset.
        Some files in Co3Dv2 have an accidental absolute path stored.
        """
        unwanted_prefix = (
            "/large_experiments/p3/replay/datasets/co3d/co3d45k_220512/export_v23/"
        )
        if path.startswith(unwanted_prefix):
            path = path[len(unwanted_prefix) :]
        return os.path.join(self.dataset_root, path)

    def _local_path(self, path: str) -> str:
        if self.path_manager is None:
            return path
        return self.path_manager.get_local_path(path)


def _load_image(path) -> np.ndarray:
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    im = im.transpose((2, 0, 1))
    im = im.astype(np.float32) / 255.0
    return im


def _load_mask(path) -> np.ndarray:
    with Image.open(path) as pil_im:
        mask = np.array(pil_im)
    mask = mask.astype(np.float32) / 255.0
    return mask[None]  # fake feature channel


def _load_depth(path, scale_adjustment) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth file name "%s"' % path)

    d = _load_16big_png_depth(path) * scale_adjustment
    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel


def _load_16big_png_depth(depth_png) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def _load_1bit_png_mask(file: str) -> np.ndarray:
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def _load_depth_mask(path: str) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth mask file name "%s"' % path)
    m = _load_1bit_png_mask(path)
    return m[None]  # fake feature channel


def _safe_as_tensor(data, dtype):
    return torch.tensor(data, dtype=dtype) if data is not None else None


# NOTE this cache is per-worker; they are implemented as processes.
# each batch is loaded and collated by a single worker;
# since sequences tend to co-occur within batches, this is useful.
@functools.lru_cache(maxsize=256)
def _load_pointcloud(pcl_path: Union[str, Path], max_points: int = 0) -> Pointclouds:
    pcl = IO().load_pointcloud(pcl_path)
    if max_points > 0:
        pcl = pcl.subsample(max_points)

    return pcl
