#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

VERSION=$(python -c "exec(open('${script_dir}/../pytorch3d/__init__.py').read()); print(__version__)")

export BUILD_TYPE=wheel
setup_env "$VERSION"
setup_wheel_python
pip_install numpy
setup_pip_pytorch_version
python setup.py clean
IS_WHEEL=1 python setup.py bdist_wheel
