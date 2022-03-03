#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import subprocess
import sys
from os.path import dirname, isfile, join


if __name__ == "__main__":
    # pyre-ignore[16]
    if len(sys.argv) > 1:
        # Parse from flags.
        # pyre-ignore[16]
        file_names = [
            join(dirname(__file__), n) for n in sys.argv if n.startswith("bm_")
        ]
    else:
        # Get all the benchmark files (starting with "bm_").
        bm_files = glob.glob(join(dirname(__file__), "bm_*.py"))
        file_names = sorted(
            f for f in bm_files if isfile(f) and not f.endswith("bm_main.py")
        )

    # Forward all important path information to the subprocesses through the
    # environment.
    os.environ["PATH"] = sys.path[0] + ":" + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        sys.path[0] + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    os.environ["PYTHONPATH"] = ":".join(sys.path)
    for file_name in file_names:
        subprocess.check_call([sys.executable, file_name])
