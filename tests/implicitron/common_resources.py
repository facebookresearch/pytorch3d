# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import tempfile
import unittest
from pathlib import Path
from typing import Generator, Tuple
from zipfile import ZipFile

from iopath.common.file_io import PathManager


CO3D_MANIFOLD_PATH: str = "manifold://co3d/tree/extracted"
CO3DV2_MANIFOLD_PATH: str = "manifold://co3d/tree/v2/extracted"

INSIDE_RE_WORKER: bool = os.environ.get("INSIDE_RE_WORKER", False)


def get_path_manager(silence_logs: bool = False) -> PathManager:
    """
    Returns a path manager which can access manifold internally.

    Args:
        silence_logs: Whether to reduce log output from iopath library.
    """
    if silence_logs:
        logging.getLogger("iopath.fb.manifold").setLevel(logging.CRITICAL)
        logging.getLogger("iopath.common.file_io").setLevel(logging.CRITICAL)

    if INSIDE_RE_WORKER:
        raise ValueError("Cannot get to manifold from RE")

    path_manager = PathManager()

    if os.environ.get("FB_TEST", False):
        from iopath.fb.manifold import ManifoldPathHandler

        path_manager.register_handler(ManifoldPathHandler())

    return path_manager


@contextlib.contextmanager
def get_skateboard_data(
    avoid_manifold: bool = False, silence_logs: bool = False
) -> Generator[Tuple[str, PathManager], None, None]:
    """
    Context manager for accessing Co3D dataset by tests, at least for
    the first 5 skateboards. Internally, we want this to exercise the
    normal way to access the data directly manifold, but on an RE
    worker this is impossible so we use a workaround.

    Args:
        avoid_manifold: Use the method used by RE workers even locally.
        silence_logs: Whether to reduce log output from iopath library.

    Yields:
        dataset_root: (str) path to dataset root.
        path_manager: path_manager to access it with.
    """
    if silence_logs:
        logging.getLogger("iopath.fb.manifold").setLevel(logging.CRITICAL)
        logging.getLogger("iopath.common.file_io").setLevel(logging.CRITICAL)

    if not os.environ.get("FB_TEST", False):
        if os.getenv("FAIR_ENV_CLUSTER", "") == "":
            raise unittest.SkipTest("Unknown environment. Data not available.")
        yield "/datasets01/co3d/081922", PathManager()

    elif avoid_manifold or INSIDE_RE_WORKER:
        from libfb.py.parutil import get_file_path

        par_path = "skateboard_first_5"
        source = get_file_path(par_path)
        assert Path(source).is_file()
        with tempfile.TemporaryDirectory() as dest:
            with ZipFile(source) as f:
                f.extractall(dest)
            yield os.path.join(dest, "extracted"), PathManager()
    else:
        yield CO3D_MANIFOLD_PATH, get_path_manager()


def _provide_torchvision_weights(par_path: str, filename: str) -> None:
    """
    Ensure the weights files are available for a torchvision model.
    """
    # In OSS, torchvision looks for vgg16 weights in
    #   https://download.pytorch.org/models/vgg16-397923af.pth
    # Inside fbcode, this is replaced by asking iopath for
    #   manifold://torchvision/tree/models/vgg16-397923af.pth
    # (the code for this replacement is in
    #    fbcode/pytorch/vision/fb/_internally_replaced_utils.py )
    #
    # iopath does this by looking for the file at the cache location
    # and if it is not there getting it from manifold.
    # (the code for this is in
    #    fbcode/fair_infra/data/iopath/iopath/fb/manifold.py )
    #
    # On the remote execution worker, manifold is inaccessible.
    # We solve this by making the cached file available before iopath
    # looks.
    #
    # By default the cache location is
    #   ~/.torch/iopath_cache/manifold_cache/tree/models/vgg16-397923af.pth
    # But we can't write to the home directory on the RE worker.
    # We define FVCORE_CACHE to change the cache location to
    #  iopath_cache/manifold_cache/tree/models/vgg16-397923af.pth
    # (Without it, manifold caches in unstable temporary locations on RE.)
    #
    # The file we want has been copied from
    #    tree/models/vgg16-397923af.pth in the torchvision bucket
    # to
    #    tree/testing/vgg16-397923af.pth in the co3d bucket
    # and the TARGETS file copies it somewhere in the PAR which we
    # recover with get_file_path.
    # (It can't copy straight to a nested location, see
    #    https://fb.workplace.com/groups/askbuck/posts/2644615728920359/)
    # Here we symlink it to the new cache location.
    if INSIDE_RE_WORKER:
        from libfb.py.parutil import get_file_path

        os.environ["FVCORE_CACHE"] = "iopath_cache"

        source = Path(get_file_path(par_path))
        assert source.is_file()

        dest = Path("iopath_cache/manifold_cache/tree/models")
        if not dest.exists():
            dest.mkdir(parents=True)

        if not (dest / filename).is_symlink():
            try:
                (dest / filename).symlink_to(source)
            except FileExistsError:
                print("FileExistsError: no symlink created.")


def provide_lpips_vgg() -> None:
    """
    Ensure the weights files are available for lpips.LPIPS(net="vgg")
    to be called. Specifically, torchvision's vgg16.
    """
    _provide_torchvision_weights("vgg_weights_for_lpips", "vgg16-397923af.pth")


def provide_resnet34() -> None:
    """
    Ensure the weights files are available for

        torchvision.models.resnet34(pretrained=True)

    to be called.
    """
    _provide_torchvision_weights("resnet34_weights", "resnet34-b627a593.pth")
