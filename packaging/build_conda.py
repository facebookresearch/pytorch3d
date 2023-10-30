# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import runpy
import subprocess
from typing import List

# required env vars:
# CU_VERSION: E.g. cu112
# JUST_TESTRUN: 1 to not set nvcc flags
# PYTORCH_VERSION: e.g. 1.12.0
# PYTHON_VERSION: e.g. 3.9

# should be run from pytorch3d root

CU_VERSION = os.environ["CU_VERSION"]
PYTORCH_VERSION = os.environ["PYTORCH_VERSION"]
pytorch_major_minor = tuple(int(i) for i in PYTORCH_VERSION.split(".")[:2])
source_root_dir = os.environ["PWD"]


def version_constraint(version):
    """
    Given version "11.3" returns " >=11.3,<11.4"
    """
    last_part = version.rindex(".") + 1
    upper = version[:last_part] + str(1 + int(version[last_part:]))
    return f" >={version},<{upper}"


def get_cuda_major_minor():
    if CU_VERSION == "cpu":
        raise ValueError("fn only for cuda builds")
    if len(CU_VERSION) != 5 or CU_VERSION[:2] != "cu":
        raise ValueError(f"Bad CU_VERSION {CU_VERSION}")
    major = CU_VERSION[2:4]
    minor = CU_VERSION[4]
    return major, minor


def setup_cuda():
    if CU_VERSION == "cpu":
        return
    major, minor = get_cuda_major_minor()
    os.environ["CUDA_HOME"] = f"/usr/local/cuda-{major}.{minor}/"
    os.environ["FORCE_CUDA"] = "1"

    basic_nvcc_flags = (
        "-gencode=arch=compute_50,code=sm_50 "
        "-gencode=arch=compute_60,code=sm_60 "
        "-gencode=arch=compute_70,code=sm_70 "
        "-gencode=arch=compute_75,code=sm_75 "
        "-gencode=arch=compute_50,code=compute_50"
    )
    if CU_VERSION == "cu102":
        nvcc_flags = "-gencode=arch=compute_35,code=sm_35 " + basic_nvcc_flags
    elif CU_VERSION < ("cu118"):
        nvcc_flags = (
            "-gencode=arch=compute_35,code=sm_35 "
            + "-gencode=arch=compute_80,code=sm_80 "
            + "-gencode=arch=compute_86,code=sm_86 "
            + basic_nvcc_flags
        )
    else:
        nvcc_flags = (
            "-gencode=arch=compute_80,code=sm_80 "
            + "-gencode=arch=compute_86,code=sm_86 "
            + "-gencode=arch=compute_90,code=sm_90 "
            + basic_nvcc_flags
        )

    if os.environ.get("JUST_TESTRUN", "0") != "1":
        os.environ["NVCC_FLAGS"] = nvcc_flags


def setup_conda_pytorch_constraint() -> List[str]:
    pytorch_constraint = f"- pytorch=={PYTORCH_VERSION}"
    os.environ["CONDA_PYTORCH_CONSTRAINT"] = pytorch_constraint
    os.environ["CONDA_PYTORCH_BUILD_CONSTRAINT"] = pytorch_constraint
    os.environ["PYTORCH_VERSION_NODOT"] = PYTORCH_VERSION.replace(".", "")

    if pytorch_major_minor < (1, 13):
        return ["-c", "pytorch"]
    else:
        return ["-c", "pytorch", "-c", "nvidia"]


def setup_conda_cudatoolkit_constraint():
    if CU_VERSION == "cpu":
        os.environ["CONDA_CPUONLY_FEATURE"] = "- cpuonly"
        os.environ["CONDA_CUDATOOLKIT_CONSTRAINT"] = ""
        return
    os.environ["CONDA_CPUONLY_FEATURE"] = ""

    if CU_VERSION in ("cu102", "cu110"):
        os.environ["CONDA_CUB_CONSTRAINT"] = "- nvidiacub"
    else:
        os.environ["CONDA_CUB_CONSTRAINT"] = ""

    major, minor = get_cuda_major_minor()
    version_clause = version_constraint(f"{major}.{minor}")
    if pytorch_major_minor < (1, 13):
        toolkit = f"- cudatoolkit {version_clause}"
    else:
        toolkit = f"- pytorch-cuda {version_clause}"
    os.environ["CONDA_CUDATOOLKIT_CONSTRAINT"] = toolkit


def do_build(start_args: List[str]):
    args = start_args.copy()

    test_flag = os.environ.get("TEST_FLAG")
    if test_flag is not None:
        args.append(test_flag)

    args.extend(["-c", "bottler", "-c", "fvcore", "-c", "iopath", "-c", "conda-forge"])
    args.append("--no-anaconda-upload")
    args.extend(["--python", os.environ["PYTHON_VERSION"]])
    args.append("packaging/pytorch3d")
    print(args)
    subprocess.check_call(args)


if __name__ == "__main__":
    args = ["conda", "build"]
    setup_cuda()

    init_path = source_root_dir + "/pytorch3d/__init__.py"
    build_version = runpy.run_path(init_path)["__version__"]
    os.environ["BUILD_VERSION"] = build_version

    os.environ["SOURCE_ROOT_DIR"] = source_root_dir
    args += setup_conda_pytorch_constraint()
    setup_conda_cudatoolkit_constraint()
    do_build(args)
