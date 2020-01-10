#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Run this script at project root by ". linter.sh" before you commit.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Running isort..."
isort -y -sp "${DIR}"

echo "Running black..."
black -l 80  "${DIR}"

echo "Running flake..."
flake8 "${DIR}"

echo "Running clang-format ..."
find "${DIR}" -regex ".*\.\(cpp\|c\|cc\|cu\|cuh\|cxx\|h\|hh\|hpp\|hxx\|tcc\|mm\|m\)" -print0 | xargs -0 clang-format -i

# Run pyre internally only.
if [[ -f tests/TARGETS ]]
then
  echo "Running pyre..."
  echo "To restart/kill pyre server, run 'pyre restart' or 'pyre kill' in fbcode/"
  ( cd ~/fbsource/fbcode; pyre -l vision/fair/pytorch3d/ )
fi
