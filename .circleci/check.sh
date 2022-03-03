#!/bin/bash -e
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script before committing config.yml to verify it is valid yaml.

python -c 'import yaml; yaml.safe_load(open("config.yml"))' && echo OK - valid yaml

msg="circleci not installed so can't check schema"
command -v circleci > /dev/null && (cd ..; circleci config validate) || echo "$msg"
