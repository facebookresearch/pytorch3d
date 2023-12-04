# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import functools
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import torch
from PIL import Image

from pytorch3d.io import IO
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures.pointclouds import Pointclouds

DATASET_TYPE_TRAIN = "train"
DATASET_TYPE_TEST = "test"
DATASET_TYPE_KNOWN = "known"
DATASET_TYPE_UNKNOWN = "unseen"


class GenericWorkaround:
    """
    OmegaConf.structured has a weirdness when you try to apply
    it to a dataclass whose first base class is a Generic which is not
    Dict. The issue is with a function called get_dict_key_value_types
    in omegaconf/_utils.py.
    For example this fails:

        @dataclass(eq=False)
        class D(torch.utils.data.Dataset[int]):
            a: int = 3

        OmegaConf.structured(D)

    We avoid the problem by adding this class as an extra base class.
    """

    pass


def is_known_frame_scalar(frame_type: str) -> bool:
    """
    Given a single frame type corresponding to a single frame, return whether
    the frame is a known frame.
    """
    return frame_type.endswith(DATASET_TYPE_KNOWN)


def is_known_frame(
    frame_type: List[str], device: Optional[str] = None
) -> torch.BoolTensor:
    """
    Given a list `frame_type` of frame types in a batch, return a tensor
    of boolean flags expressing whether the corresponding frame is a known frame.
    """
    # pyre-fixme[7]: Expected `BoolTensor` but got `Tensor`.
    return torch.tensor(
        [is_known_frame_scalar(ft) for ft in frame_type],
        dtype=torch.bool,
        device=device,
    )


def is_train_frame(
    frame_type: List[str], device: Optional[str] = None
) -> torch.BoolTensor:
    """
    Given a list `frame_type` of frame types in a batch, return a tensor
    of boolean flags expressing whether the corresponding frame is a training frame.
    """
    # pyre-fixme[7]: Expected `BoolTensor` but got `Tensor`.
    return torch.tensor(
        [ft.startswith(DATASET_TYPE_TRAIN) for ft in frame_type],
        dtype=torch.bool,
        device=device,
    )


def get_bbox_from_mask(
    mask: np.ndarray, thr: float, decrease_quant: float = 0.05
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

    x0, x1 = get_1d_bounds(masks_for_box.sum(axis=-2))
    y0, y1 = get_1d_bounds(masks_for_box.sum(axis=-1))

    return x0, y0, x1 - x0, y1 - y0


def crop_around_box(
    tensor: torch.Tensor, bbox: torch.Tensor, impath: str = ""
) -> torch.Tensor:
    # bbox is xyxy, where the upper bound is corrected with +1
    bbox = clamp_box_to_image_bounds_and_round(
        bbox,
        image_size_hw=tuple(tensor.shape[-2:]),
    )
    tensor = tensor[..., bbox[1] : bbox[3], bbox[0] : bbox[2]]
    assert all(c > 0 for c in tensor.shape), f"squashed image {impath}"
    return tensor


def clamp_box_to_image_bounds_and_round(
    bbox_xyxy: torch.Tensor,
    image_size_hw: Tuple[int, int],
) -> torch.LongTensor:
    bbox_xyxy = bbox_xyxy.clone()
    bbox_xyxy[[0, 2]] = torch.clamp(bbox_xyxy[[0, 2]], 0, image_size_hw[-1])
    bbox_xyxy[[1, 3]] = torch.clamp(bbox_xyxy[[1, 3]], 0, image_size_hw[-2])
    if not isinstance(bbox_xyxy, torch.LongTensor):
        bbox_xyxy = bbox_xyxy.round().long()
    return bbox_xyxy  # pyre-ignore [7]


T = TypeVar("T", bound=torch.Tensor)


def bbox_xyxy_to_xywh(xyxy: T) -> T:
    wh = xyxy[2:] - xyxy[:2]
    xywh = torch.cat([xyxy[:2], wh])
    return xywh  # pyre-ignore


def get_clamp_bbox(
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
    bbox_xyxy = bbox_xywh_to_xyxy(bbox, clamp_size=2)

    return bbox_xyxy


def rescale_bbox(
    bbox: torch.Tensor,
    orig_res: Union[Tuple[int, int], torch.LongTensor],
    new_res: Union[Tuple[int, int], torch.LongTensor],
) -> torch.Tensor:
    assert bbox is not None
    assert np.prod(orig_res) > 1e-8
    # average ratio of dimensions
    # pyre-ignore
    rel_size = (new_res[0] / orig_res[0] + new_res[1] / orig_res[1]) / 2.0
    return bbox * rel_size


def bbox_xywh_to_xyxy(
    xywh: torch.Tensor, clamp_size: Optional[int] = None
) -> torch.Tensor:
    xyxy = xywh.clone()
    if clamp_size is not None:
        xyxy[2:] = torch.clamp(xyxy[2:], clamp_size)
    xyxy[2:] += xyxy[:2]
    return xyxy


def get_1d_bounds(arr: np.ndarray) -> Tuple[int, int]:
    nz = np.flatnonzero(arr)
    return nz[0], nz[-1] + 1


def resize_image(
    image: Union[np.ndarray, torch.Tensor],
    image_height: Optional[int],
    image_width: Optional[int],
    mode: str = "bilinear",
) -> Tuple[torch.Tensor, float, torch.Tensor]:

    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)

    if image_height is None or image_width is None:
        # skip the resizing
        return image, 1.0, torch.ones_like(image[:1])
    # takes numpy array or tensor, returns pytorch tensor
    minscale = min(
        image_height / image.shape[-2],
        image_width / image.shape[-1],
    )
    imre = torch.nn.functional.interpolate(
        image[None],
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


def transpose_normalize_image(image: np.ndarray) -> np.ndarray:
    im = np.atleast_3d(image).transpose((2, 0, 1))
    return im.astype(np.float32) / 255.0


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))

    return transpose_normalize_image(im)


def load_mask(path: str) -> np.ndarray:
    with Image.open(path) as pil_im:
        mask = np.array(pil_im)

    return transpose_normalize_image(mask)


def load_depth(path: str, scale_adjustment: float) -> np.ndarray:
    if path.lower().endswith(".exr"):
        # NOTE: environment variable OPENCV_IO_ENABLE_OPENEXR must be set to 1
        # You will have to accept these vulnerabilities by using OpenEXR:
        # https://github.com/opencv/opencv/issues/21326
        import cv2

        d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[..., 0]
        d[d > 1e9] = 0.0
    elif path.lower().endswith(".png"):
        d = load_16big_png_depth(path)
    else:
        raise ValueError('unsupported depth file name "%s"' % path)

    d = d * scale_adjustment

    d[~np.isfinite(d)] = 0.0
    return d[None]  # fake feature channel


def load_16big_png_depth(depth_png: str) -> np.ndarray:
    with Image.open(depth_png) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def load_1bit_png_mask(file: str) -> np.ndarray:
    with Image.open(file) as pil_im:
        mask = (np.array(pil_im.convert("L")) > 0.0).astype(np.float32)
    return mask


def load_depth_mask(path: str) -> np.ndarray:
    if not path.lower().endswith(".png"):
        raise ValueError('unsupported depth mask file name "%s"' % path)
    m = load_1bit_png_mask(path)
    return m[None]  # fake feature channel


def safe_as_tensor(data, dtype):
    return torch.tensor(data, dtype=dtype) if data is not None else None


def _convert_ndc_to_pixels(
    focal_length: torch.Tensor,
    principal_point: torch.Tensor,
    image_size_wh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point_px = half_image_size - principal_point * rescale
    focal_length_px = focal_length * rescale
    return focal_length_px, principal_point_px


def _convert_pixels_to_ndc(
    focal_length_px: torch.Tensor,
    principal_point_px: torch.Tensor,
    image_size_wh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half_image_size = image_size_wh / 2
    rescale = half_image_size.min()
    principal_point = (half_image_size - principal_point_px) / rescale
    focal_length = focal_length_px / rescale
    return focal_length, principal_point


def adjust_camera_to_bbox_crop_(
    camera: PerspectiveCameras,
    image_size_wh: torch.Tensor,
    clamp_bbox_xywh: torch.Tensor,
) -> None:
    if len(camera) != 1:
        raise ValueError("Adjusting currently works with singleton cameras camera only")

    focal_length_px, principal_point_px = _convert_ndc_to_pixels(
        camera.focal_length[0],
        camera.principal_point[0],
        image_size_wh,
    )
    principal_point_px_cropped = principal_point_px - clamp_bbox_xywh[:2]

    focal_length, principal_point_cropped = _convert_pixels_to_ndc(
        focal_length_px,
        principal_point_px_cropped,
        clamp_bbox_xywh[2:],
    )

    camera.focal_length = focal_length[None]
    camera.principal_point = principal_point_cropped[None]


def adjust_camera_to_image_scale_(
    camera: PerspectiveCameras,
    original_size_wh: torch.Tensor,
    new_size_wh: torch.LongTensor,
) -> PerspectiveCameras:
    focal_length_px, principal_point_px = _convert_ndc_to_pixels(
        camera.focal_length[0],
        camera.principal_point[0],
        original_size_wh,
    )

    # now scale and convert from pixels to NDC
    image_size_wh_output = new_size_wh.float()
    scale = (image_size_wh_output / original_size_wh).min(dim=-1, keepdim=True).values
    focal_length_px_scaled = focal_length_px * scale
    principal_point_px_scaled = principal_point_px * scale

    focal_length_scaled, principal_point_scaled = _convert_pixels_to_ndc(
        focal_length_px_scaled,
        principal_point_px_scaled,
        image_size_wh_output,
    )
    camera.focal_length = focal_length_scaled[None]
    camera.principal_point = principal_point_scaled[None]  # pyre-ignore


# NOTE this cache is per-worker; they are implemented as processes.
# each batch is loaded and collated by a single worker;
# since sequences tend to co-occur within batches, this is useful.
@functools.lru_cache(maxsize=256)
def load_pointcloud(pcl_path: Union[str, Path], max_points: int = 0) -> Pointclouds:
    pcl = IO().load_pointcloud(pcl_path)
    if max_points > 0:
        pcl = pcl.subsample(max_points)

    return pcl
