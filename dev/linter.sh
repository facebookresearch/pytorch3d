#!/bin/bash -e
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script at project root by "./dev/linter.sh" before you commit

{
  V=$(black --version|cut '-d ' -f3)
  code='import distutils.version; assert "19.3" < distutils.version.LooseVersion("'$V'")'
  PYTHON=false
  command -v python > /dev/null && PYTHON=python
  command -v python3 > /dev/null && PYTHON=python3
  ${PYTHON} -c "${code}" 2> /dev/null
} || {
  echo "Linter requires black 19.3b0 or higher!"
  exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR=$(dirname "${DIR}")

if [[ -f "${DIR}/tests/TARGETS" ]]
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
if [[ -f "${DIR}/tests/TARGETS" ]]
then
  (cd "${DIR}"; command -v arc > /dev/null && arc lint) || true

  echo "Running pyre..."
  echo "To restart/kill pyre server, run 'pyre restart' or 'pyre kill' in fbcode/"
  ( cd ~/fbsource/fbcode; pyre -l vision/fair/pytorch3d/ )
fi
