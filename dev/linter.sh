#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Run this script at project root by "./dev/linter.sh" before you commit

{
  V=$(black --version|cut '-d ' -f3)
  code='import distutils.version; assert "19.3" < distutils.version.LooseVersion("'$V'")'
  python -c "${code}" 2> /dev/null
} || {
  echo "Linter requires black 19.3b0 or higher!"
  exit 1
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
DIR="${DIR}/.."

echo "Running isort..."
isort -y -sp "${DIR}"

echo "Running black..."
black -l 80 "${DIR}"

echo "Running flake..."
flake8 "${DIR}"

echo "Running clang-format ..."
find "${DIR}" -regex ".*\.\(cpp\|c\|cc\|cu\|cuh\|cxx\|h\|hh\|hpp\|hxx\|tcc\|mm\|m\)" -print0 | xargs -0 clang-format -i

(cd "${DIR}"; command -v arc > /dev/null && arc lint) || true
