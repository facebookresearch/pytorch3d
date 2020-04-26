#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import glob
import os
import runpy

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "pytorch3d", "csrc")

    main_source = os.path.join(extensions_dir, "ext.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu"))

    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": ["-std=c++14"]}
    define_macros = []

    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    if (torch.cuda.is_available() and CUDA_HOME is not None) or force_cuda:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        nvcc_args = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            nvcc_args.extend(nvcc_flags_env.split(" "))

        # It's better if pytorch can do this by default ..
        CC = os.environ.get("CC", None)
        if CC is not None:
            CC_arg = "-ccbin={}".format(CC)
            if CC_arg not in nvcc_args:
                if any(arg.startswith("-ccbin") for arg in nvcc_args):
                    raise ValueError("Inconsistent ccbins")
                nvcc_args.append(CC_arg)

        extra_compile_args["nvcc"] = nvcc_args

    sources = [os.path.join(extensions_dir, s) for s in sources]

    include_dirs = [extensions_dir]

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
    packages=find_packages(exclude=("configs", "tests")),
    install_requires=["torchvision>=0.4", "fvcore"],
    extras_require={
        "all": ["matplotlib", "tqdm>4.29.0", "imageio", "ipywidgets"],
        "dev": ["flake8", "isort", "black==19.3b0"],
    },
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
)
