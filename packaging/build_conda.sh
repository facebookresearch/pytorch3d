#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

VERSION=$(python -c "exec(open('${script_dir}/../pytorch3d/__init__.py').read()); print(__version__)")

# Prevent dev tag in the version string.
export BUILD_VERSION=$VERSION

export BUILD_TYPE=conda
setup_env "$VERSION"
export SOURCE_ROOT_DIR="$PWD"
setup_conda_pytorch_constraint
setup_conda_cudatoolkit_constraint
setup_visual_studio_constraint
# shellcheck disable=SC2086
conda build $CONDA_CHANNEL_FLAGS ${TEST_FLAG:-} -c defaults -c conda-forge --no-anaconda-upload -c fvcore --python "$PYTHON_VERSION" packaging/pytorch3d
