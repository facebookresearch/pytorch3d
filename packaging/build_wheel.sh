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

export BUILD_TYPE=wheel
setup_env "$VERSION"
setup_wheel_python
pip_install numpy
setup_pip_pytorch_version
download_nvidiacub_if_needed
python setup.py clean
IS_WHEEL=1 python setup.py bdist_wheel
