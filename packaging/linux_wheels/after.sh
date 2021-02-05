#!/usr/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
set -ex
sudo chown -R "$USER" output
python publish.py
