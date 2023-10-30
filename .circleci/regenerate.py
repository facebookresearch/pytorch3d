#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is adapted from the torchvision one.
"""

import os.path

import jinja2
import yaml
from packaging import version


# The CUDA versions which have pytorch conda packages available for linux for each
# version of pytorch.
CONDA_CUDA_VERSIONS = {
    "1.12.0": ["cu113", "cu116"],
    "1.12.1": ["cu113", "cu116"],
    "1.13.0": ["cu116", "cu117"],
    "1.13.1": ["cu116", "cu117"],
    "2.0.0": ["cu117", "cu118"],
    "2.0.1": ["cu117", "cu118"],
    "2.1.0": ["cu118", "cu121"],
}


def conda_docker_image_for_cuda(cuda_version):
    if cuda_version in ("cu101", "cu102", "cu111"):
        return None
    if len(cuda_version) != 5:
        raise ValueError("Unknown cuda version")
    return "pytorch/conda-builder:cuda" + cuda_version[2:]


def pytorch_versions_for_python(python_version):
    if python_version in ["3.8", "3.9"]:
        return list(CONDA_CUDA_VERSIONS)
    if python_version == "3.10":
        return [
            i
            for i in CONDA_CUDA_VERSIONS
            if version.Version(i) >= version.Version("1.11.0")
        ]
    if python_version == "3.11":
        return [
            i
            for i in CONDA_CUDA_VERSIONS
            if version.Version(i) >= version.Version("2.1.0")
        ]


def workflows(prefix="", filter_branch=None, upload=False, indentation=6):
    w = []
    for btype in ["conda"]:
        for python_version in ["3.8", "3.9", "3.10", "3.11"]:
            for pytorch_version in pytorch_versions_for_python(python_version):
                for cu_version in CONDA_CUDA_VERSIONS[pytorch_version]:
                    w += workflow_pair(
                        btype=btype,
                        python_version=python_version,
                        pytorch_version=pytorch_version,
                        cu_version=cu_version,
                        prefix=prefix,
                        upload=upload,
                        filter_branch=filter_branch,
                    )

    return indent(indentation, w)


def workflow_pair(
    *,
    btype,
    python_version,
    pytorch_version,
    cu_version,
    prefix="",
    upload=False,
    filter_branch,
):

    w = []
    py = python_version.replace(".", "")
    pyt = pytorch_version.replace(".", "")
    base_workflow_name = f"{prefix}linux_{btype}_py{py}_{cu_version}_pyt{pyt}"

    w.append(
        generate_base_workflow(
            base_workflow_name=base_workflow_name,
            python_version=python_version,
            pytorch_version=pytorch_version,
            cu_version=cu_version,
            btype=btype,
            filter_branch=filter_branch,
        )
    )

    if upload:
        w.append(
            generate_upload_workflow(
                base_workflow_name=base_workflow_name,
                btype=btype,
                cu_version=cu_version,
                filter_branch=filter_branch,
            )
        )

    return w


def generate_base_workflow(
    *,
    base_workflow_name,
    python_version,
    cu_version,
    pytorch_version,
    btype,
    filter_branch=None,
):

    d = {
        "name": base_workflow_name,
        "python_version": python_version,
        "cu_version": cu_version,
        "pytorch_version": pytorch_version,
        "context": "DOCKERHUB_TOKEN",
    }

    conda_docker_image = conda_docker_image_for_cuda(cu_version)
    if conda_docker_image is not None:
        d["conda_docker_image"] = conda_docker_image

    if filter_branch is not None:
        d["filters"] = {"branches": {"only": filter_branch}}

    return {f"binary_linux_{btype}": d}


def generate_upload_workflow(*, base_workflow_name, btype, cu_version, filter_branch):
    d = {
        "name": f"{base_workflow_name}_upload",
        "context": "org-member",
        "requires": [base_workflow_name],
    }

    if btype == "wheel":
        d["subfolder"] = cu_version + "/"

    if filter_branch is not None:
        d["filters"] = {"branches": {"only": filter_branch}}

    return {f"binary_{btype}_upload": d}


def indent(indentation, data_list):
    if len(data_list) == 0:
        return ""
    return ("\n" + " " * indentation).join(
        yaml.dump(data_list, default_flow_style=False).splitlines()
    )


if __name__ == "__main__":
    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d),
        lstrip_blocks=True,
        autoescape=False,
        keep_trailing_newline=True,
    )

    with open(os.path.join(d, "config.yml"), "w") as f:
        f.write(env.get_template("config.in.yml").render(workflows=workflows))
