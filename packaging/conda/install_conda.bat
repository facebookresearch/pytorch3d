@REM Copyright (c) Meta Platforms, Inc. and affiliates.
@REM All rights reserved.
@REM
@REM This source code is licensed under the BSD-style license found in the
@REM LICENSE file in the root directory of this source tree.

start /wait "" "%miniconda_exe%" /S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /D=%tmp_conda%
