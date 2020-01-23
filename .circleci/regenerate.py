#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
This script is adapted from the torchvision one.
There is no python2.7 nor macos.
TODO: python 3.8 when pytorch 1.4.
"""

import os.path
import jinja2
import yaml


def workflows(prefix="", filter_branch=None, upload=False, indentation=6):
    w = []
    # add "wheel" here for pypi
    for btype in ["conda"]:
        for python_version in ["3.6", "3.7", "3.8"]:
            for cu_version in ["cu92", "cu100", "cu101"]:
                w += workflow_pair(
                    btype=btype,
                    python_version=python_version,
                    cu_version=cu_version,
                    prefix=prefix,
                    upload=upload,
                    filter_branch=filter_branch,
                )

    return indent(indentation, w)


def workflow_pair(
    *, btype, python_version, cu_version, prefix="", upload=False, filter_branch
):

    w = []
    base_workflow_name = (
        f"{prefix}binary_linux_{btype}_py{python_version}_{cu_version}"
    )

    w.append(
        generate_base_workflow(
            base_workflow_name=base_workflow_name,
            python_version=python_version,
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
    *, base_workflow_name, python_version, cu_version, btype, filter_branch=None
):

    d = {
        "name": base_workflow_name,
        "python_version": python_version,
        "cu_version": cu_version,
        "build_version": "0.1.0",
        "pytorch_version": "1.4",
    }

    if cu_version == "cu92":
        d["wheel_docker_image"] = "pytorch/manylinux-cuda92"
    elif cu_version == "cu100":
        d["wheel_docker_image"] = "pytorch/manylinux-cuda100"

    if filter_branch is not None:
        d["filters"] = {"branches": {"only": filter_branch}}

    return {f"binary_linux_{btype}": d}


def generate_upload_workflow(
    *, base_workflow_name, btype, cu_version, filter_branch
):
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
    return ("\n" + " " * indentation).join(
        yaml.dump(data_list, default_flow_style=False).splitlines()
    )


if __name__ == "__main__":
    d = os.path.dirname(__file__)
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(d), lstrip_blocks=True, autoescape=False
    )

    with open(os.path.join(d, "config.yml"), "w") as f:
        f.write(env.get_template("config.in.yml").render(workflows=workflows))
