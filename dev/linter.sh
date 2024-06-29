#!/bin/bash -e
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script at project root by "./dev/linter.sh" before you commit

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=$(dirname "${DIR}")

if [[ -f "${DIR}/TARGETS" ]]
then
  pyfmt "${DIR}"
else
# run usort externally only
  echo "Running usort..."
  usort  "${DIR}"
fi

echo "Running black..."
black "${DIR}"

echo "Running flake..."
flake8 "${DIR}" || true

echo "Running clang-format ..."
clangformat=$(command -v clang-format-8 || echo clang-format)
find "${DIR}" -regex ".*\.\(cpp\|c\|cc\|cu\|cuh\|cxx\|h\|hh\|hpp\|hxx\|tcc\|mm\|m\)" -print0 | xargs -0 "${clangformat}" -i

# Run arc and pyre internally only.
if [[ -f "${DIR}/TARGETS" ]]
then
  (cd "${DIR}"; command -v arc > /dev/null && arc lint) || true

  echo "Running pyre..."
  echo "To restart/kill pyre server, run 'pyre restart' or 'pyre kill' in fbcode/"
  ( cd ~/fbsource/fbcode; pyre -l vision/fair/pytorch3d/ )
fi
