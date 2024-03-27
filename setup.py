#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import runpy
import sys
import warnings
from typing import List, Optional

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CppExtension, CUDA_HOME, CUDAExtension


def get_existing_ccbin(nvcc_args: List[str]) -> Optional[str]:
    """
    Given a list of nvcc arguments, return the compiler if specified.

    Note from CUDA doc: Single value options and list options must have
    arguments, which must follow the name of the option itself by either
    one of more spaces or an equals character.
    """
    last_arg = None
    for arg in reversed(nvcc_args):
        if arg == "-ccbin":
            return last_arg
        if arg.startswith("-ccbin="):
            return arg[7:]
        last_arg = arg
    return None


def get_extensions():
    no_extension = os.getenv("PYTORCH3D_NO_EXTENSION", "0") == "1"
    if no_extension:
        msg = "SKIPPING EXTENSION BUILD. PYTORCH3D WILL NOT WORK!"
        print(msg, file=sys.stderr)
        warnings.warn(msg)
        return []

    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "pytorch3d", "csrc")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
    extension = CppExtension

    extra_compile_args = {"cxx": ["-std=c++17"]}
    define_macros = []
    include_dirs = [extensions_dir]

    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    force_no_cuda = os.getenv("PYTORCH3D_FORCE_NO_CUDA", "0") == "1"
    if (
        not force_no_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    ) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        # Thrust is only used for its tuple objects.
        # With CUDA 11.0 we can't use the cudatoolkit's version of cub.
        # We take the risk that CUB and Thrust are incompatible, because
        # we aren't using parts of Thrust which actually use CUB.
        define_macros += [("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        cub_home = os.environ.get("CUB_HOME", None)
        nvcc_args = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        if os.name != "nt":
            nvcc_args.append("-std=c++17")
        if cub_home is None:
            prefix = os.environ.get("CONDA_PREFIX", None)
            if prefix is not None and os.path.isdir(prefix + "/include/cub"):
                cub_home = prefix + "/include"

        if cub_home is None:
            warnings.warn(
                "The environment variable `CUB_HOME` was not found. "
                "NVIDIA CUB is required for compilation and can be downloaded "
                "from `https://github.com/NVIDIA/cub/releases`. You can unpack "
                "it to a location of your choice and set the environment variable "
                "`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
            )
        else:
            include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))
        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            nvcc_args.extend(nvcc_flags_env.split(" "))

        # This is needed for pytorch 1.6 and earlier. See e.g.
        # https://github.com/facebookresearch/pytorch3d/issues/436
        # It is harmless after https://github.com/pytorch/pytorch/pull/47404 .
        # But it can be problematic in torch 1.7.0 and 1.7.1
        if torch.__version__[:4] != "1.7.":
            CC = os.environ.get("CC", None)
            if CC is not None:
                existing_CC = get_existing_ccbin(nvcc_args)
                if existing_CC is None:
                    CC_arg = "-ccbin={}".format(CC)
                    nvcc_args.append(CC_arg)
                elif existing_CC != CC:
                    msg = f"Inconsistent ccbins: {CC} and {existing_CC}"
                    raise ValueError(msg)

        extra_compile_args["nvcc"] = nvcc_args

    sources = [os.path.join(extensions_dir, s) for s in sources]

    ext_modules = [
        extension(
            "pytorch3d._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Retrieve __version__ from the package.
__version__ = runpy.run_path("pytorch3d/__init__.py")["__version__"]


if os.getenv("PYTORCH3D_NO_NINJA", "0") == "1":

    class BuildExtension(torch.utils.cpp_extension.BuildExtension):
        def __init__(self, *args, **kwargs):
            super().__init__(use_ninja=False, *args, **kwargs)

else:
    BuildExtension = torch.utils.cpp_extension.BuildExtension

trainer = "pytorch3d.implicitron_trainer"

setup(
    name="pytorch3d",
    version=__version__,
    author="FAIR",
    url="https://github.com/facebookresearch/pytorch3d",
    description="PyTorch3D is FAIR's library of reusable components "
    "for deep Learning with 3D data.",
    packages=find_packages(
        exclude=("configs", "tests", "tests.*", "docs.*", "projects.*")
    )
    + [trainer],
    package_dir={trainer: "projects/implicitron_trainer"},
    install_requires=["fvcore", "iopath"],
    extras_require={
        "all": ["matplotlib", "tqdm>4.29.0", "imageio", "ipywidgets"],
        "dev": ["flake8", "usort"],
        "implicitron": [
            "hydra-core>=1.1",
            "visdom",
            "lpips",
            "tqdm>4.29.0",
            "matplotlib",
            "accelerate",
            "sqlalchemy>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            f"pytorch3d_implicitron_runner={trainer}.experiment:experiment",
            f"pytorch3d_implicitron_visualizer={trainer}.visualize_reconstruction:main",
        ]
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    package_data={
        "": ["*.json"],
    },
)
