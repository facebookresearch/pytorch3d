# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
from pathlib import Path
from typing import List


"""
This module outputs a list of tests for completion.
It has no dependencies.
"""


def get_test_files() -> List[Path]:
    root = Path(__file__).parent.parent
    dirs = ["tests", "projects/implicitron_trainer"]
    return [i for dir in dirs for i in (root / dir).glob("**/test*.py")]


def tests_from_file(path: Path, base: str) -> List[str]:
    """
    Returns all the tests in the given file, in format
    expected as arguments when running the tests.
    e.g.
        file_stem
        file_stem.TestFunctionality
        file_stem.TestFunctionality.test_f
        file_stem.TestFunctionality.test_g
    """
    with open(path) as f:
        node = ast.parse(f.read())
    out = [base]
    for cls in node.body:
        if not isinstance(cls, ast.ClassDef):
            continue
        if not cls.name.startswith("Test"):
            continue
        class_base = base + "." + cls.name
        out.append(class_base)
        for method in cls.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            if not method.name.startswith("test"):
                continue
            out.append(class_base + "." + method.name)
    return out


def main() -> None:
    files = get_test_files()
    test_root = Path(__file__).parent.parent
    all_tests = []
    for f in files:
        file_base = str(f.relative_to(test_root))[:-3].replace("/", ".")
        all_tests.extend(tests_from_file(f, file_base))
    for test in sorted(all_tests):
        print(test)


if __name__ == "__main__":
    main()
