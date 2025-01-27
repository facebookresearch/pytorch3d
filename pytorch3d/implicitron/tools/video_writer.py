# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import os
import shutil
import subprocess
import tempfile
import warnings
from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from PIL import Image

_NO_TORCHVISION = False
try:
    import torchvision
except ImportError:
    _NO_TORCHVISION = True


_DEFAULT_FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

matplotlib.use("Agg")


class VideoWriter:
    """
    A class for exporting videos.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ffmpeg_bin: str = _DEFAULT_FFMPEG,
        out_path: str = "/tmp/video.mp4",
        fps: int = 20,
        output_format: str = "visdom",
        rmdir_allowed: bool = False,
        use_torchvision_video_writer: bool = False,
        **kwargs,
    ) -> None:
        """
        Args:
            cache_dir: A directory for storing the video frames. If `None`,
                a temporary directory will be used.
            ffmpeg_bin: The path to an `ffmpeg` executable.
            out_path: The path to the output video.
            fps: The speed of the generated video in frames-per-second.
            output_format: Format of the output video. Currently only `"visdom"`
                is supported.
            rmdir_allowed: If `True` delete and create `cache_dir` in case
                it is not empty.
            use_torchvision_video_writer: If `True` use `torchvision.io.write_video`
            to write the video
        """
        self.rmdir_allowed = rmdir_allowed
        self.output_format = output_format
        self.fps = fps
        self.out_path = out_path
        self.cache_dir = cache_dir
        self.ffmpeg_bin = ffmpeg_bin
        self.use_torchvision_video_writer = use_torchvision_video_writer
        self.frames = []
        self.regexp = "frame_%08d.png"
        self.frame_num = 0

        if self.use_torchvision_video_writer:
            assert not _NO_TORCHVISION, "torchvision not available"

        if self.cache_dir is not None:
            self.tmp_dir = None
            if os.path.isdir(self.cache_dir):
                if rmdir_allowed:
                    shutil.rmtree(self.cache_dir)
                else:
                    warnings.warn(
                        f"Warning: cache directory not empty ({self.cache_dir})."
                    )
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            self.tmp_dir = tempfile.TemporaryDirectory()
            self.cache_dir = self.tmp_dir.name

    def write_frame(
        self,
        frame: Union[matplotlib.figure.Figure, np.ndarray, Image.Image, str],
        resize: Optional[Union[float, Tuple[int, int]]] = None,
    ) -> None:
        """
        Write a frame to the video.

        Args:
            frame: An object containing the frame image.
            resize: Either a floating defining the image rescaling factor
                or a 2-tuple defining the size of the output image.
        """

        # pyre-fixme[6]: For 1st argument expected `Union[PathLike[str], str]` but
        #  got `Optional[str]`.
        outfile = os.path.join(self.cache_dir, self.regexp % self.frame_num)

        if isinstance(frame, matplotlib.figure.Figure):
            plt.savefig(outfile)
            im = Image.open(outfile)
        elif isinstance(frame, np.ndarray):
            if frame.dtype in (np.float64, np.float32, float):
                frame = (np.transpose(frame, (1, 2, 0)) * 255.0).astype(np.uint8)
            im = Image.fromarray(frame)
        elif isinstance(frame, Image.Image):
            im = frame
        elif isinstance(frame, str):
            im = Image.open(frame).convert("RGB")
        else:
            raise ValueError("Cant convert type %s" % str(type(frame)))

        if im is not None:
            if resize is not None:
                if isinstance(resize, float):
                    resize = [int(resize * s) for s in im.size]
            else:
                resize = im.size
            # make sure size is divisible by 2
            resize = tuple([resize[i] + resize[i] % 2 for i in (0, 1)])

            im = im.resize(resize, Image.ANTIALIAS)
            im.save(outfile)

        self.frames.append(outfile)
        self.frame_num += 1

    def get_video(self, quiet: bool = True) -> str:
        """
        Generate the video from the written frames.

        Args:
            quiet: If `True`, suppresses logging messages.

        Returns:
            video_path: The path to the generated video if any frames were added.
                Otherwise returns an empty string.
        """
        if self.frame_num == 0:
            return ""

        # pyre-fixme[6]: For 1st argument expected `Union[PathLike[str], str]` but
        #  got `Optional[str]`.
        regexp = os.path.join(self.cache_dir, self.regexp)

        if self.output_format == "visdom":  # works for ppt too
            # Video codec parameters
            video_codec = "h264"
            crf = "18"
            b = "2000k"
            pix_fmt = "yuv420p"

            if self.use_torchvision_video_writer:
                torchvision.io.write_video(
                    self.out_path,
                    torch.stack(
                        [torch.from_numpy(np.array(Image.open(f))) for f in self.frames]
                    ),
                    fps=self.fps,
                    video_codec=video_codec,
                    options={"crf": crf, "b": b, "pix_fmt": pix_fmt},
                )

            else:
                if shutil.which(self.ffmpeg_bin) is None:
                    raise ValueError(
                        f"Cannot find ffmpeg as `{self.ffmpeg_bin}`. "
                        + "Please set FFMPEG in the environment or ffmpeg_bin on this class."
                    )

                args = [
                    self.ffmpeg_bin,
                    "-r",
                    str(self.fps),
                    "-i",
                    regexp,
                    "-vcodec",
                    video_codec,
                    "-f",
                    "mp4",
                    "-y",
                    "-crf",
                    crf,
                    "-b",
                    b,
                    "-pix_fmt",
                    pix_fmt,
                    self.out_path,
                ]
                if quiet:
                    subprocess.check_call(
                        args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                else:
                    subprocess.check_call(args)
        else:
            raise ValueError("no such output type %s" % str(self.output_format))

        return self.out_path

    def __del__(self) -> None:
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
