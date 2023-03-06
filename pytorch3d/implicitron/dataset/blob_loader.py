# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.dataset_base import FrameData
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

    dataset_root: str
    load_images: bool
    load_depths: bool
    load_depth_masks: bool
    load_masks: bool
    load_point_clouds: bool
    max_points: int
    mask_images: bool
    mask_depths: bool
    image_height: Optional[int]
    image_width: Optional[int]
    box_crop: bool
    box_crop_mask_thr: float
    box_crop_context: float
    path_manager: Any = None

    def load(
        self,
        frame_data: FrameData,
        entry: types.FrameAnnotation,
        seq_annotation: types.SequenceAnnotation,
    ) -> FrameData:
        """Main method for loader.
        FrameData modification done inplace
        """
        (
            frame_data.fg_probability,
            frame_data.mask_path,
            frame_data.bbox_xywh,
            clamp_bbox_xyxy,
            frame_data.crop_bbox_xywh,
        ) = self._load_crop_fg_probability(entry)

        scale = 1.0
        if self.load_images and entry.image is not None:
            # original image size
            frame_data.image_size_hw = _safe_as_tensor(entry.image.size, torch.long)

            (
                frame_data.image_rgb,
                frame_data.image_path,
                frame_data.mask_crop,
                scale,
            ) = self._load_crop_images(
                entry, frame_data.fg_probability, clamp_bbox_xyxy
            )

        if self.load_depths and entry.depth is not None:
            (
                frame_data.depth_map,
                frame_data.depth_path,
                frame_data.depth_mask,
            ) = self._load_mask_depth(entry, clamp_bbox_xyxy, frame_data.fg_probability)

        if entry.viewpoint is not None:
            frame_data.camera = self._get_pytorch3d_camera(
                entry,
                scale,
                clamp_bbox_xyxy,
            )

        if self.load_point_clouds and seq_annotation.point_cloud is not None:
            pcl_path = self._fix_point_cloud_path(seq_annotation.point_cloud.path)
            frame_data.sequence_point_cloud = _load_pointcloud(
                self._local_path(pcl_path), max_points=self.max_points
            )
            frame_data.sequence_point_cloud_path = pcl_path
        return frame_data

    def _load_crop_fg_probability(
        self, entry: types.FrameAnnotation
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[str],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        fg_probability = None
        full_path = None
        bbox_xywh = None
        clamp_bbox_xyxy = None
        crop_box_xywh = None

        if (self.load_masks or self.box_crop) and entry.mask is not None:
            full_path = os.path.join(self.dataset_root, entry.mask.path)
            mask = _load_mask(self._local_path(full_path))

            if mask.shape[-2:] != entry.image.size:
                raise ValueError(
                    f"bad mask size: {mask.shape[-2:]} vs {entry.image.size}!"
                )

            bbox_xywh = torch.tensor(_get_bbox_from_mask(mask, self.box_crop_mask_thr))

            if self.box_crop:
                clamp_bbox_xyxy = _clamp_box_to_image_bounds_and_round(
                    _get_clamp_bbox(
                        bbox_xywh,
                        image_path=entry.image.path,
                        box_crop_context=self.box_crop_context,
                    ),
                    image_size_hw=tuple(mask.shape[-2:]),
                )
                crop_box_xywh = _bbox_xyxy_to_xywh(clamp_bbox_xyxy)

                mask = _crop_around_box(mask, clamp_bbox_xyxy, full_path)

            fg_probability, _, _ = self._resize_image(mask, mode="nearest")

        return fg_probability, full_path, bbox_xywh, clamp_bbox_xyxy, crop_box_xywh

    def _load_crop_images(
        self,
        entry: types.FrameAnnotation,
        fg_probability: Optional[torch.Tensor],
        clamp_bbox_xyxy: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor, float]:
        assert self.dataset_root is not None and entry.image is not None
        path = os.path.join(self.dataset_root, entry.image.path)
        image_rgb = _load_image(self._local_path(path))

        if image_rgb.shape[-2:] != entry.image.size:
            raise ValueError(
                f"bad image size: {image_rgb.shape[-2:]} vs {entry.image.size}!"
            )

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            image_rgb = _crop_around_box(image_rgb, clamp_bbox_xyxy, path)

        image_rgb, scale, mask_crop = self._resize_image(image_rgb)

        if self.mask_images:
            assert fg_probability is not None
            image_rgb *= fg_probability

        return image_rgb, path, mask_crop, scale

    def _load_mask_depth(
        self,
        entry: types.FrameAnnotation,
        clamp_bbox_xyxy: Optional[torch.Tensor],
        fg_probability: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        entry_depth = entry.depth
        assert entry_depth is not None
        path = os.path.join(self.dataset_root, entry_depth.path)
        depth_map = _load_depth(self._local_path(path), entry_depth.scale_adjustment)

        if self.box_crop:
            assert clamp_bbox_xyxy is not None
            depth_bbox_xyxy = _rescale_bbox(
                clamp_bbox_xyxy, entry.image.size, depth_map.shape[-2:]
            )
            depth_map = _crop_around_box(depth_map, depth_bbox_xyxy, path)

        depth_map, _, _ = self._resize_image(depth_map, mode="nearest")

        if self.mask_depths:
            assert fg_probability is not None
            depth_map *= fg_probability

        if self.load_depth_masks:
            assert entry_depth.mask_path is not None
            mask_path = os.path.join(self.dataset_root, entry_depth.mask_path)
            depth_mask = _load_depth_mask(self._local_path(mask_path))

            if self.box_crop:
                assert clamp_bbox_xyxy is not None
                depth_mask_bbox_xyxy = _rescale_bbox(
                    clamp_bbox_xyxy, entry.image.size, depth_mask.shape[-2:]
                )
                depth_mask = _crop_around_box(
                    depth_mask, depth_mask_bbox_xyxy, mask_path
                )

            depth_mask, _, _ = self._resize_image(depth_mask, mode="nearest")
        else:
            depth_mask = torch.ones_like(depth_map)

        return depth_map, path, depth_mask

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
        if self.box_crop:
            assert clamp_bbox_xyxy is not None
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

    def _resize_image(
        self, image, mode="bilinear"
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        image_height, image_width = self.image_height, self.image_width
        if image_height is None or image_width is None:
            # skip the resizing
            imre_ = torch.from_numpy(image)
            return imre_, 1.0, torch.ones_like(imre_[:1])
        # takes numpy array, returns pytorch tensor
        minscale = min(
            image_height / image.shape[-2],
            image_width / image.shape[-1],
        )
        imre = torch.nn.functional.interpolate(
            torch.from_numpy(image)[None],
            scale_factor=minscale,
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
            recompute_scale_factor=True,
        )[0]
        imre_ = torch.zeros(image.shape[0], image_height, image_width)
        imre_[:, 0 : imre.shape[1], 0 : imre.shape[2]] = imre
        mask = torch.zeros(1, image_height, image_width)
        mask[:, 0 : imre.shape[1], 0 : imre.shape[2]] = 1.0
        return imre_, minscale, mask


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


def _get_bbox_from_mask(
    mask, thr, decrease_quant: float = 0.05
) -> Tuple[int, int, int, int]:
    # bbox in xywh
    masks_for_box = np.zeros_like(mask)
    while masks_for_box.sum() <= 1.0:
        masks_for_box = (mask > thr).astype(np.float32)
        thr -= decrease_quant
    if thr <= 0.0:
        warnings.warn(
            f"Empty masks_for_bbox (thr={thr}) => using full image.", stacklevel=1
        )

    x0, x1 = _get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = _get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0


def _crop_around_box(tensor, bbox, impath: str = ""):
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox = _clamp_box_to_image_bounds_and_round(
        bbox,
        image_size_hw=tensor.shape[-2:],
    )
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"
    return tensor


def _clamp_box_to_image_bounds_and_round(
    bbox_xyxy: torch.Tensor,
    image_size_hw: Tuple[int, int],
) -> torch.LongTensor:
    bbox_xyxy = bbox_xyxy.clone()
    bbox_xyxy[[0, 2]] = torch.clamp(bbox_xyxy[[0, 2]], 0, image_size_hw[-1])
    bbox_xyxy[[1, 3]] = torch.clamp(bbox_xyxy[[1, 3]], 0, image_size_hw[-2])
    if not isinstance(bbox_xyxy, torch.LongTensor):
        bbox_xyxy = bbox_xyxy.round().long()
    return bbox_xyxy  # pyre-ignore [7]


def _get_clamp_bbox(
    bbox: torch.Tensor,
    box_crop_context: float = 0.0,
    image_path: str = "",
) -> torch.Tensor:
    # box_crop_context: rate of expansion for bbox
    # returns possibly expanded bbox xyxy as float

    bbox = bbox.clone()  # do not edit bbox in place

    # increase box size
    if box_crop_context > 0.0:
        c = box_crop_context
        bbox = bbox.float()
        bbox[0] -= bbox[2] * c / 2
        bbox[1] -= bbox[3] * c / 2
        bbox[2] += bbox[2] * c
        bbox[3] += bbox[3] * c

    if (bbox[2:] <= 1.0).any():
        raise ValueError(
            f"squashed image {image_path}!! The bounding box contains no pixels."
        )

    bbox[2:] = torch.clamp(bbox[2:], 2)  # set min height, width to 2 along both axes
    bbox_xyxy = _bbox_xywh_to_xyxy(bbox, clamp_size=2)

    return bbox_xyxy


def _bbox_xyxy_to_xywh(xyxy: torch.Tensor) -> torch.Tensor:
    wh = xyxy[2:] - xyxy[:2]
    xywh = torch.cat([xyxy[:2], wh])
    return xywh


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


def _rescale_bbox(bbox: torch.Tensor, orig_res, new_res) -> torch.Tensor:
    assert bbox is not None
    assert np.prod(orig_res) > 1e-8
    # average ratio of dimensions
    rel_size = (new_res[0] / orig_res[0] + new_res[1] / orig_res[1]) / 2.0
    return bbox * rel_size


def _load_1bit_png_mask(file: str) -> np.ndarray:
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def _load_depth_mask(path: str) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth mask file name "%s"' % path)
    m = _load_1bit_png_mask(path)
    return m[None]  # fake feature channel


def _get_1d_bounds(arr) -> Tuple[int, int]:
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1] + 1


def _bbox_xywh_to_xyxy(
    xywh: torch.Tensor, clamp_size: Optional[int] = None
) -> torch.Tensor:
    xyxy = xywh.clone()
    if clamp_size is not None:
        xyxy[2:] = torch.clamp(xyxy[2:], clamp_size)
    xyxy[2:] += xyxy[:2]
    return xyxy


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
