@REM Copyright (c) Facebook, Inc. and its affiliates.
@REM All rights reserved.
@REM
@REM This source code is licensed under the BSD-style license found in the
@REM LICENSE file in the root directory of this source tree.

:: Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
start /wait "" "%miniconda_exe%" /S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /D=%tmp_conda%
