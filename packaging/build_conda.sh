#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

if [[ "$JUST_TESTRUN" == "1" ]]
then
    # We are not building for other users, we
    # are only trying to see if the tests pass.
    # So save time by only building for our own GPU.
    unset NVCC_FLAGS
fi

# shellcheck disable=SC2086
conda build $CONDA_CHANNEL_FLAGS ${TEST_FLAG:-} -c bottler -c fvcore -c iopath -c conda-forge --no-anaconda-upload --python "$PYTHON_VERSION" packaging/pytorch3d
