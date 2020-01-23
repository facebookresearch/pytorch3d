#!/bin/bash -e
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Run this script before committing config.yml to verify it is valid yaml.

python -c 'import yaml; yaml.safe_load(open("config.yml"))' && echo OK
