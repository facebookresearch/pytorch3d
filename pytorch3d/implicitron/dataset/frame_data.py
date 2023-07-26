# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    ClassVar,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import numpy as np
import torch

from pytorch3d.implicitron.dataset import types
from pytorch3d.implicitron.dataset.utils import (
    adjust_camera_to_bbox_crop_,
    adjust_camera_to_image_scale_,
    bbox_xyxy_to_xywh,
    clamp_box_to_image_bounds_and_round,
    crop_around_box,
    GenericWorkaround,
    get_bbox_from_mask,
    get_clamp_bbox,
    load_depth,
    load_depth_mask,
    load_image,
    load_mask,
    load_pointcloud,
    rescale_bbox,
    resize_image,
    safe_as_tensor,
)
from pytorch3d.implicitron.tools.config import registry, ReplaceableBase
from pytorch3d.renderer.camera_utils import join_cameras_as_batch
from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
from pytorch3d.structures.pointclouds import join_pointclouds_as_batch, Pointclouds


@dataclass
class FrameData(Mapping[str, Any]):
    """
    A type of the elements returned by indexing the dataset object.
    It can represent both individual frames and batches of thereof;
    in this documentation, the sizes of tensors refer to single frames;
    add the first batch dimension for the collation result.

    Args:
        frame_number: The number of the frame within its sequence.
            0-based continuous integers.
        sequence_name: The unique name of the frame's sequence.
        sequence_category: The object category of the sequence.
        frame_timestamp: The time elapsed since the start of a sequence in sec.
        image_size_hw: The size of the original image in pixels; (height, width)
            tensor of shape (2,). Note that it is optional, e.g. it can be `None`
            if the frame annotation has no size ans image_rgb has not [yet] been
            loaded. Image-less FrameData is valid but mutators like crop/resize
            may fail if the original image size cannot be deduced.
        effective_image_size_hw: The size of the image after mutations such as
            crop/resize in pixels; (height, width). if the image has not been mutated,
            it is equal to `image_size_hw`. Note that it is also optional, for the
            same reason as `image_size_hw`.
        image_path: The qualified path to the loaded image (with dataset_root).
        image_rgb: A Tensor of shape `(3, H, W)` holding the RGB image
            of the frame; elements are floats in [0, 1].
        mask_crop: A binary mask of shape `(1, H, W)` denoting the valid image
            regions. Regions can be invalid (mask_crop[i,j]=0) in case they
            are a result of zero-padding of the image after cropping around
            the object bounding box; elements are floats in {0.0, 1.0}.
        depth_path: The qualified path to the frame's depth map.
        depth_map: A float Tensor of shape `(1, H, W)` holding the depth map
            of the frame; values correspond to distances from the camera;
            use `depth_mask` and `mask_crop` to filter for valid pixels.
        depth_mask: A binary mask of shape `(1, H, W)` denoting pixels of the
            depth map that are valid for evaluation, they have been checked for
            consistency across views; elements are floats in {0.0, 1.0}.
        mask_path: A qualified path to the foreground probability mask.
        fg_probability: A Tensor of `(1, H, W)` denoting the probability of the
            pixels belonging to the captured object; elements are floats
            in [0, 1].
        bbox_xywh: The bounding box tightly enclosing the foreground object in the
            format (x0, y0, width, height). The convention assumes that
            `x0+width` and `y0+height` includes the boundary of the box.
            I.e., to slice out the corresponding crop from an image tensor `I`
            we execute `crop = I[..., y0:y0+height, x0:x0+width]`
        crop_bbox_xywh: The bounding box denoting the boundaries of `image_rgb`
            in the original image coordinates in the format (x0, y0, width, height).
            The convention is the same as for `bbox_xywh`. `crop_bbox_xywh` differs
            from `bbox_xywh` due to padding (which can happen e.g. due to
            setting `JsonIndexDataset.box_crop_context > 0`)
        camera: A PyTorch3D camera object corresponding the frame's viewpoint,
            corrected for cropping if it happened.
        camera_quality_score: The score proportional to the confidence of the
            frame's camera estimation (the higher the more accurate).
        point_cloud_quality_score: The score proportional to the accuracy of the
            frame's sequence point cloud (the higher the more accurate).
        sequence_point_cloud_path: The path to the sequence's point cloud.
        sequence_point_cloud: A PyTorch3D Pointclouds object holding the
            point cloud corresponding to the frame's sequence. When the object
            represents a batch of frames, point clouds may be deduplicated;
            see `sequence_point_cloud_idx`.
        sequence_point_cloud_idx: Integer indices mapping frame indices to the
            corresponding point clouds in `sequence_point_cloud`; to get the
            corresponding point cloud to `image_rgb[i]`, use
            `sequence_point_cloud[sequence_point_cloud_idx[i]]`.
        frame_type: The type of the loaded frame specified in
            `subset_lists_file`, if provided.
        meta: A dict for storing additional frame information.
    """

    frame_number: Optional[torch.LongTensor]
    sequence_name: Union[str, List[str]]
    sequence_category: Union[str, List[str]]
    frame_timestamp: Optional[torch.Tensor] = None
    image_size_hw: Optional[torch.LongTensor] = None
    effective_image_size_hw: Optional[torch.LongTensor] = None
    image_path: Union[str, List[str], None] = None
    image_rgb: Optional[torch.Tensor] = None
    # masks out padding added due to cropping the square bit
    mask_crop: Optional[torch.Tensor] = None
    depth_path: Union[str, List[str], None] = None
    depth_map: Optional[torch.Tensor] = None
    depth_mask: Optional[torch.Tensor] = None
    mask_path: Union[str, List[str], None] = None
    fg_probability: Optional[torch.Tensor] = None
    bbox_xywh: Optional[torch.Tensor] = None
    crop_bbox_xywh: Optional[torch.Tensor] = None
    camera: Optional[PerspectiveCameras] = None
    camera_quality_score: Optional[torch.Tensor] = None
    point_cloud_quality_score: Optional[torch.Tensor] = None
    sequence_point_cloud_path: Union[str, List[str], None] = None
    sequence_point_cloud: Optional[Pointclouds] = None
    sequence_point_cloud_idx: Optional[torch.Tensor] = None
    frame_type: Union[str, List[str], None] = None  # known | unseen
    meta: dict = field(default_factory=lambda: {})

    # NOTE that batching resets this attribute
    _uncropped: bool = field(init=False, default=True)

    def to(self, *args, **kwargs):
        new_params = {}
        for field_name in iter(self):
            value = getattr(self, field_name)
            if isinstance(value, (torch.Tensor, Pointclouds, CamerasBase)):
                new_params[field_name] = value.to(*args, **kwargs)
            else:
                new_params[field_name] = value
        frame_data = type(self)(**new_params)
        frame_data._uncropped = self._uncropped
        return frame_data

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    # the following functions make sure **frame_data can be passed to functions
    def __iter__(self):
        for f in fields(self):
            if f.name.startswith("_"):
                continue

            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return sum(1 for f in iter(self))

    def crop_by_metadata_bbox_(
        self,
        box_crop_context: float,
    ) -> None:
        """Crops the frame data in-place by (possibly expanded) bounding box.
        The bounding box is taken from the object state (usually taken from
        the frame annotation or estimated from the foregroubnd mask).
        If the expanded bounding box does not fit the image, it is clamped,
        i.e. the image is *not* padded.

        Args:
            box_crop_context: rate of expansion for bbox; 0 means no expansion,

        Raises:
            ValueError: If the object does not contain a bounding box (usually when no
                mask annotation is provided)
            ValueError: If the frame data have been cropped or resized, thus the intrinsic
                bounding box is not valid for the current image size.
            ValueError: If the frame does not have an image size (usually a corner case
                when no image has been loaded)
        """
        if self.bbox_xywh is None:
            raise ValueError(
                "Attempted cropping by metadata with empty bounding box. Consider either"
                " to remove_empty_masks or turn off box_crop in the dataset config."
            )

        if not self._uncropped:
            raise ValueError(
                "Trying to apply the metadata bounding box to already cropped "
                "or resized image; coordinates have changed."
            )

        self._crop_by_bbox_(
            box_crop_context,
            self.bbox_xywh,
        )

    def crop_by_given_bbox_(
        self,
        box_crop_context: float,
        bbox_xywh: torch.Tensor,
    ) -> None:
        """Crops the frame data in-place by (possibly expanded) bounding box.
        If the expanded bounding box does not fit the image, it is clamped,
        i.e. the image is *not* padded.

        Args:
            box_crop_context: rate of expansion for bbox; 0 means no expansion,
            bbox_xywh: bounding box in [x0, y0, width, height] format. If float
                tensor, values are floored (after converting to [x0, y0, x1, y1]).

        Raises:
            ValueError: If the frame does not have an image size (usually a corner case
                when no image has been loaded)
        """
        self._crop_by_bbox_(
            box_crop_context,
            bbox_xywh,
        )

    def _crop_by_bbox_(
        self,
        box_crop_context: float,
        bbox_xywh: torch.Tensor,
    ) -> None:
        """Crops the frame data in-place by (possibly expanded) bounding box.
        If the expanded bounding box does not fit the image, it is clamped,
        i.e. the image is *not* padded.

        Args:
            box_crop_context: rate of expansion for bbox; 0 means no expansion,
            bbox_xywh: bounding box in [x0, y0, width, height] format. If float
                tensor, values are floored (after converting to [x0, y0, x1, y1]).

        Raises:
            ValueError: If the frame does not have an image size (usually a corner case
                when no image has been loaded)
        """
        effective_image_size_hw = self.effective_image_size_hw
        if effective_image_size_hw is None:
            raise ValueError("Calling crop on image-less FrameData")

        bbox_xyxy = get_clamp_bbox(
            bbox_xywh,
            image_path=self.image_path,  # pyre-ignore
            box_crop_context=box_crop_context,
        )
        clamp_bbox_xyxy = clamp_box_to_image_bounds_and_round(
            bbox_xyxy,
            image_size_hw=tuple(self.effective_image_size_hw),  # pyre-ignore
        )
        crop_bbox_xywh = bbox_xyxy_to_xywh(clamp_bbox_xyxy)

        if self.fg_probability is not None:
            self.fg_probability = crop_around_box(
                self.fg_probability,
                clamp_bbox_xyxy,
                self.mask_path,  # pyre-ignore
            )
        if self.image_rgb is not None:
            self.image_rgb = crop_around_box(
                self.image_rgb,
                clamp_bbox_xyxy,
                self.image_path,  # pyre-ignore
            )

        depth_map = self.depth_map
        if depth_map is not None:
            clamp_bbox_xyxy_depth = rescale_bbox(
                clamp_bbox_xyxy, tuple(depth_map.shape[-2:]), effective_image_size_hw
            ).long()
            self.depth_map = crop_around_box(
                depth_map,
                clamp_bbox_xyxy_depth,
                self.depth_path,  # pyre-ignore
            )

        depth_mask = self.depth_mask
        if depth_mask is not None:
            clamp_bbox_xyxy_depth = rescale_bbox(
                clamp_bbox_xyxy, tuple(depth_mask.shape[-2:]), effective_image_size_hw
            ).long()
            self.depth_mask = crop_around_box(
                depth_mask,
                clamp_bbox_xyxy_depth,
                self.mask_path,  # pyre-ignore
            )

        # changing principal_point according to bbox_crop
        if self.camera is not None:
            adjust_camera_to_bbox_crop_(
                camera=self.camera,
                image_size_wh=effective_image_size_hw.flip(dims=[-1]),
                clamp_bbox_xywh=crop_bbox_xywh,
            )

        # pyre-ignore
        self.effective_image_size_hw = crop_bbox_xywh[..., 2:].flip(dims=[-1])
        self._uncropped = False

    def resize_frame_(self, new_size_hw: torch.LongTensor) -> None:
        """Resizes frame data in-place according to given dimensions.

        Args:
            new_size_hw: target image size [height, width], a LongTensor of shape (2,)

        Raises:
            ValueError: If the frame does not have an image size (usually a corner case
                when no image has been loaded)
        """

        effective_image_size_hw = self.effective_image_size_hw
        if effective_image_size_hw is None:
            raise ValueError("Calling resize on image-less FrameData")

        image_height, image_width = new_size_hw.tolist()

        if self.fg_probability is not None:
            self.fg_probability, _, _ = resize_image(
                self.fg_probability,
                image_height=image_height,
                image_width=image_width,
                mode="nearest",
            )

        if self.image_rgb is not None:
            self.image_rgb, _, self.mask_crop = resize_image(
                self.image_rgb, image_height=image_height, image_width=image_width
            )

        if self.depth_map is not None:
            self.depth_map, _, _ = resize_image(
                self.depth_map,
                image_height=image_height,
                image_width=image_width,
                mode="nearest",
            )

        if self.depth_mask is not None:
            self.depth_mask, _, _ = resize_image(
                self.depth_mask,
                image_height=image_height,
                image_width=image_width,
                mode="nearest",
            )

        if self.camera is not None:
            if self.image_size_hw is None:
                raise ValueError(
                    "image_size_hw has to be defined for resizing FrameData with cameras."
                )
            adjust_camera_to_image_scale_(
                camera=self.camera,
                original_size_wh=effective_image_size_hw.flip(dims=[-1]),
                new_size_wh=new_size_hw.flip(dims=[-1]),  # pyre-ignore
            )

        self.effective_image_size_hw = new_size_hw
        self._uncropped = False

    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """

        elem = batch[0]

        if isinstance(elem, cls):
            pointcloud_ids = [id(el.sequence_point_cloud) for el in batch]
            id_to_idx = defaultdict(list)
            for i, pc_id in enumerate(pointcloud_ids):
                id_to_idx[pc_id].append(i)

            sequence_point_cloud = []
            sequence_point_cloud_idx = -np.ones((len(batch),))
            for i, ind in enumerate(id_to_idx.values()):
                sequence_point_cloud_idx[ind] = i
                sequence_point_cloud.append(batch[ind[0]].sequence_point_cloud)
            assert (sequence_point_cloud_idx >= 0).all()

            override_fields = {
                "sequence_point_cloud": sequence_point_cloud,
                "sequence_point_cloud_idx": sequence_point_cloud_idx.tolist(),
            }
            # note that the pre-collate value of sequence_point_cloud_idx is unused

            collated = {}
            for f in fields(elem):
                if not f.init:
                    continue

                list_values = override_fields.get(
                    f.name, [getattr(d, f.name) for d in batch]
                )
                collated[f.name] = (
                    cls.collate(list_values)
                    if all(list_value is not None for list_value in list_values)
                    else None
                )
            return cls(**collated)

        elif isinstance(elem, Pointclouds):
            return join_pointclouds_as_batch(batch)

        elif isinstance(elem, CamerasBase):
            # TODO: don't store K; enforce working in NDC space
            return join_cameras_as_batch(batch)
        else:
            return torch.utils.data._utils.collate.default_collate(batch)


FrameDataSubtype = TypeVar("FrameDataSubtype", bound=FrameData)


class FrameDataBuilderBase(ReplaceableBase, Generic[FrameDataSubtype], ABC):
    """A base class for FrameDataBuilders that build a FrameData object, load and
    process the binary data (crop and resize). Implementations should parametrize
    the class with a subtype of FrameData and set frame_data_type class variable to
    that type. They have to also implement `build` method.
    """

    # To be initialised to FrameDataSubtype
    frame_data_type: ClassVar[Type[FrameDataSubtype]]

    @abstractmethod
    def build(
        self,
        frame_annotation: types.FrameAnnotation,
        sequence_annotation: types.SequenceAnnotation,
        *,
        load_blobs: bool = True,
        **kwargs,
    ) -> FrameDataSubtype:
        """An abstract method to build the frame data based on raw frame/sequence
        annotations, load the binary data and adjust them according to the metadata.
        """
        raise NotImplementedError()


class GenericFrameDataBuilder(FrameDataBuilderBase[FrameDataSubtype], ABC):
    """
    A class to build a FrameData object, load and process the binary data (crop and
    resize). This is an abstract class for extending to build FrameData subtypes. Most
    users need to use concrete `FrameDataBuilder` class instead.
    Beware that modifications of frame data are done in-place.

    Args:
        dataset_root: The root folder of the dataset; all paths in frame / sequence
            annotations are defined w.r.t. this root. Has to be set if any of the
            load_* flabs below is true.
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
        path_manager: Optionally a PathManager for interpreting paths in a special way.
    """

    dataset_root: Optional[str] = None
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

    def __post_init__(self) -> None:
        load_any_blob = (
            self.load_images
            or self.load_depths
            or self.load_depth_masks
            or self.load_masks
            or self.load_point_clouds
        )
        if load_any_blob and self.dataset_root is None:
            raise ValueError(
                "dataset_root must be set to load any blob data. "
                "Make sure it is set in either FrameDataBuilder or Dataset params."
            )

        if load_any_blob and not self._exists_in_dataset_root(""):
            raise ValueError(
                f"dataset_root is passed but {self.dataset_root} does not exist."
            )

    def build(
        self,
        frame_annotation: types.FrameAnnotation,
        sequence_annotation: types.SequenceAnnotation,
        *,
        load_blobs: bool = True,
        **kwargs,
    ) -> FrameDataSubtype:
        """Builds the frame data based on raw frame/sequence annotations, loads the
        binary data and adjust them according to the metadata. The processing includes:
            * if box_crop is set, the image/mask/depth are cropped with the bounding
                box provided or estimated from MaskAnnotation,
            * if image_height/image_width are set, the image/mask/depth are resized to
                fit that resolution. Note that the aspect ratio is preserved, and the
                (possibly cropped) image is pasted into the top-left corner. In the
                resulting frame_data, mask_crop field corresponds to the mask of the
                pasted image.

        Args:
            frame_annotation: frame annotation
            sequence_annotation: sequence annotation
            load_blobs: if the function should attempt loading the image, depth map
                and mask, and foreground mask

        Returns:
            The constructed FrameData object.
        """

        point_cloud = sequence_annotation.point_cloud

        frame_data = self.frame_data_type(
            frame_number=safe_as_tensor(frame_annotation.frame_number, torch.long),
            frame_timestamp=safe_as_tensor(
                frame_annotation.frame_timestamp, torch.float
            ),
            sequence_name=frame_annotation.sequence_name,
            sequence_category=sequence_annotation.category,
            camera_quality_score=safe_as_tensor(
                sequence_annotation.viewpoint_quality_score, torch.float
            ),
            point_cloud_quality_score=safe_as_tensor(
                point_cloud.quality_score, torch.float
            )
            if point_cloud is not None
            else None,
        )

        fg_mask_np: Optional[np.ndarray] = None
        mask_annotation = frame_annotation.mask
        if mask_annotation is not None:
            if load_blobs and self.load_masks:
                fg_mask_np, mask_path = self._load_fg_probability(frame_annotation)
                frame_data.mask_path = mask_path
                frame_data.fg_probability = safe_as_tensor(fg_mask_np, torch.float)

            bbox_xywh = mask_annotation.bounding_box_xywh
            if bbox_xywh is None and fg_mask_np is not None:
                bbox_xywh = get_bbox_from_mask(fg_mask_np, self.box_crop_mask_thr)

            frame_data.bbox_xywh = safe_as_tensor(bbox_xywh, torch.float)

        if frame_annotation.image is not None:
            image_size_hw = safe_as_tensor(frame_annotation.image.size, torch.long)
            frame_data.image_size_hw = image_size_hw  # original image size
            # image size after crop/resize
            frame_data.effective_image_size_hw = image_size_hw
            image_path = None
            dataset_root = self.dataset_root
            if frame_annotation.image.path is not None and dataset_root is not None:
                image_path = os.path.join(dataset_root, frame_annotation.image.path)
                frame_data.image_path = image_path

            if load_blobs and self.load_images:
                if image_path is None:
                    raise ValueError("Image path is required to load images.")

                image_np = load_image(self._local_path(image_path))
                frame_data.image_rgb = self._postprocess_image(
                    image_np, frame_annotation.image.size, frame_data.fg_probability
                )

        if (
            load_blobs
            and self.load_depths
            and frame_annotation.depth is not None
            and frame_annotation.depth.path is not None
        ):
            (
                frame_data.depth_map,
                frame_data.depth_path,
                frame_data.depth_mask,
            ) = self._load_mask_depth(frame_annotation, fg_mask_np)

        if load_blobs and self.load_point_clouds and point_cloud is not None:
            pcl_path = self._fix_point_cloud_path(point_cloud.path)
            frame_data.sequence_point_cloud = load_pointcloud(
                self._local_path(pcl_path), max_points=self.max_points
            )
            frame_data.sequence_point_cloud_path = pcl_path

        if frame_annotation.viewpoint is not None:
            frame_data.camera = self._get_pytorch3d_camera(frame_annotation)

        if self.box_crop:
            frame_data.crop_by_metadata_bbox_(self.box_crop_context)

        if self.image_height is not None and self.image_width is not None:
            new_size = (self.image_height, self.image_width)
            frame_data.resize_frame_(
                new_size_hw=torch.tensor(new_size, dtype=torch.long),  # pyre-ignore
            )

        return frame_data

    def _load_fg_probability(
        self, entry: types.FrameAnnotation
    ) -> Tuple[np.ndarray, str]:
        assert self.dataset_root is not None and entry.mask is not None
        full_path = os.path.join(self.dataset_root, entry.mask.path)
        fg_probability = load_mask(self._local_path(full_path))
        if fg_probability.shape[-2:] != entry.image.size:
            raise ValueError(
                f"bad mask size: {fg_probability.shape[-2:]} vs {entry.image.size}!"
            )

        return fg_probability, full_path

    def _postprocess_image(
        self,
        image_np: np.ndarray,
        image_size: Tuple[int, int],
        fg_probability: Optional[torch.Tensor],
    ) -> torch.Tensor:
        image_rgb = safe_as_tensor(image_np, torch.float)

        if image_rgb.shape[-2:] != image_size:
            raise ValueError(f"bad image size: {image_rgb.shape[-2:]} vs {image_size}!")

        if self.mask_images:
            assert fg_probability is not None
            image_rgb *= fg_probability

        return image_rgb

    def _load_mask_depth(
        self,
        entry: types.FrameAnnotation,
        fg_mask: Optional[np.ndarray],
    ) -> Tuple[torch.Tensor, str, torch.Tensor]:
        entry_depth = entry.depth
        dataset_root = self.dataset_root
        assert dataset_root is not None
        assert entry_depth is not None and entry_depth.path is not None
        path = os.path.join(dataset_root, entry_depth.path)
        depth_map = load_depth(self._local_path(path), entry_depth.scale_adjustment)

        if self.mask_depths:
            assert fg_mask is not None
            depth_map *= fg_mask

        mask_path = entry_depth.mask_path
        if self.load_depth_masks and mask_path is not None:
            mask_path = os.path.join(dataset_root, mask_path)
            depth_mask = load_depth_mask(self._local_path(mask_path))
        else:
            depth_mask = (depth_map > 0.0).astype(np.float32)

        return torch.tensor(depth_map), path, torch.tensor(depth_mask)

    def _get_pytorch3d_camera(
        self,
        entry: types.FrameAnnotation,
    ) -> PerspectiveCameras:
        entry_viewpoint = entry.viewpoint
        assert entry_viewpoint is not None
        # principal point and focal length
        principal_point = torch.tensor(
            entry_viewpoint.principal_point, dtype=torch.float
        )
        focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

        format = entry_viewpoint.intrinsics_format
        if entry_viewpoint.intrinsics_format == "ndc_norm_image_bounds":
            # legacy PyTorch3D NDC format
            # convert to pixels unequally and convert to ndc equally
            image_size_as_list = list(reversed(entry.image.size))
            image_size_wh = torch.tensor(image_size_as_list, dtype=torch.float)
            per_axis_scale = image_size_wh / image_size_wh.min()
            focal_length = focal_length * per_axis_scale
            principal_point = principal_point * per_axis_scale
        elif entry_viewpoint.intrinsics_format != "ndc_isotropic":
            raise ValueError(f"Unknown intrinsics format: {format}")

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
        assert self.dataset_root is not None
        return os.path.join(self.dataset_root, path)

    def _local_path(self, path: str) -> str:
        if self.path_manager is None:
            return path
        return self.path_manager.get_local_path(path)

    def _exists_in_dataset_root(self, relpath) -> bool:
        if not self.dataset_root:
            return False

        full_path = os.path.join(self.dataset_root, relpath)
        if self.path_manager is None:
            return os.path.exists(full_path)
        else:
            return self.path_manager.exists(full_path)


@registry.register
class FrameDataBuilder(GenericWorkaround, GenericFrameDataBuilder[FrameData]):
    """
    A concrete class to build a FrameData object, load and process the binary data (crop
    and resize). Beware that modifications of frame data are done in-place. Please see
    the documentation for `GenericFrameDataBuilder` for the description of parameters
    and methods.
    """

    frame_data_type: ClassVar[Type[FrameData]] = FrameData
