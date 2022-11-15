#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script makes the stubs for implicitron in docs/modules.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def paths_to_modules(paths):
    """
    Given an iterable of paths, return equivalent list of modules.
    """
    return [
        str(i.relative_to(ROOT_DIR))[:-3].replace("/", ".")
        for i in paths
        if "__pycache__" not in str(i)
    ]


def create_one_file(title, description, sources, dest_file):
    with open(dest_file, "w") as f:
        print(title, file=f)
        print("=" * len(title), file=f)
        print(file=f)
        print(description, file=f)
        for source in sources:
            if source.find("._") != -1:
                # ignore internal modules including __init__.py
                continue
            print(f"\n.. automodule:: {source}", file=f)
            print("    :members:", file=f)
            print("    :undoc-members:", file=f)
            print("    :show-inheritance:", file=f)


def iterate_directory(directory_path, dest):
    """
    Create a file for each module in the given path
    """
    toc = []
    if not dest.exists():
        dest.mkdir()
    for file in sorted(directory_path.glob("*.py")):
        if file.stem.startswith("_"):
            continue
        module = paths_to_modules([file])
        create_one_file(module[0], file.stem, module, dest / f"{file.stem}.rst")
        toc.append(file.stem)

    for subdir in directory_path.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == "fb":
            continue
        if subdir.name.startswith("_"):
            continue
        iterate_directory(subdir, dest / (subdir.name))
        toc.append(f"{subdir.name}/index")

    paths_to_modules_ = paths_to_modules([directory_path.with_suffix(".XX")])
    if len(paths_to_modules_) == 0:
        return
    title = paths_to_modules_[0]

    with open(dest / "index.rst", "w") as f:
        print(title, file=f)
        print("=" * len(title), file=f)
        print("\n.. toctree::\n", file=f)
        for item in toc:
            print(f"    {item}", file=f)


def make_directory_index(title: str, directory_path: Path):
    index_file = directory_path / "index.rst"
    directory_rsts = sorted(directory_path.glob("*.rst"))
    subdirs = sorted([f for f in directory_path.iterdir() if f.is_dir()])
    with open(index_file, "w") as f:
        print(title, file=f)
        print("=" * len(title), file=f)
        print("\n.. toctree::\n", file=f)
        for subdir in subdirs:
            print(f"    {subdir.stem}/index.rst", file=f)
        for rst in directory_rsts:
            if rst.stem == "index":
                continue
            print(f"    {rst.stem}", file=f)


def do_implicitron():
    DEST_DIR = Path(__file__).resolve().parent / "modules/implicitron"

    iterate_directory(ROOT_DIR / "pytorch3d/implicitron/models", DEST_DIR / "models")

    unwanted_tools = ["configurable", "depth_cleanup", "utils"]
    tools_sources = sorted(ROOT_DIR.glob("pytorch3d/implicitron/tools/*.py"))
    tools_modules = [
        str(i.relative_to(ROOT_DIR))[:-3].replace("/", ".")
        for i in tools_sources
        if i.stem not in unwanted_tools
    ]
    create_one_file(
        "pytorch3d.implicitron.tools",
        "Tools for implicitron",
        tools_modules,
        DEST_DIR / "tools.rst",
    )

    dataset_files = sorted(ROOT_DIR.glob("pytorch3d/implicitron/dataset/*.py"))
    basic_dataset = [
        "dataset_base",
        "dataset_map_provider",
        "data_loader_map_provider",
        "data_source",
        "scene_batch_sampler",
    ]
    basic_dataset_modules = [
        f"pytorch3d.implicitron.dataset.{i}" for i in basic_dataset
    ]
    create_one_file(
        "pytorch3d.implicitron.dataset in general",
        "Basics of data for implicitron",
        basic_dataset_modules,
        DEST_DIR / "data_basics.rst",
    )

    specific_dataset_files = [
        i for i in dataset_files if i.stem.find("_dataset_map_provider") != -1
    ]
    create_one_file(
        "pytorch3d.implicitron.dataset specific datasets",
        "specific datasets",
        paths_to_modules(specific_dataset_files),
        DEST_DIR / "datasets.rst",
    )

    evaluation_files = sorted(ROOT_DIR.glob("pytorch3d/implicitron/evaluation/*.py"))
    create_one_file(
        "pytorch3d.implicitron.evaluation",
        "evaluation",
        paths_to_modules(evaluation_files),
        DEST_DIR / "evaluation.rst",
    )

    make_directory_index("pytorch3d.implicitron", DEST_DIR)


def iterate_toplevel_module(name: str) -> None:
    dest_dir = Path(__file__).resolve().parent / "modules" / name
    iterate_directory(ROOT_DIR / "pytorch3d" / name, dest_dir)


do_implicitron()
iterate_toplevel_module("renderer")
iterate_toplevel_module("vis")
