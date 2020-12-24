#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import glob
import os
import runpy
import warnings

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "pytorch3d", "csrc")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True)
    extension = CppExtension

    extra_compile_args = {"cxx": ["-std=c++14"]}
    define_macros = []
    include_dirs = [extensions_dir]

    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
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
            "-std=c++14",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
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
        CC = os.environ.get("CC", None)
        if CC is not None:
            CC_arg = "-ccbin={}".format(CC)
            if CC_arg not in nvcc_args:
                if any(arg.startswith("-ccbin") for arg in nvcc_args):
                    raise ValueError("Inconsistent ccbins")
                nvcc_args.append(CC_arg)

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


setup(
    name="pytorch3d",
    version=__version__,
    author="FAIR",
    url="https://github.com/facebookresearch/pytorch3d",
    description="PyTorch3D is FAIR's library of reusable components "
    "for deep Learning with 3D data.",
    packages=find_packages(exclude=("configs", "tests", "tests.*")),
    install_requires=["torchvision>=0.4", "fvcore", "iopath"],
    extras_require={
        "all": ["matplotlib", "tqdm>4.29.0", "imageio", "ipywidgets"],
        "dev": ["flake8", "isort", "black==19.3b0"],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
