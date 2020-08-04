#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

"""
Print the number of nightly builds
"""

from collections import Counter

import yaml


conf = yaml.safe_load(open("config.yml"))
jobs = conf["workflows"]["build_and_test"]["jobs"]


def jobtype(job):
    if isinstance(job, str):
        return job
    if len(job) == 1:
        [name] = job.keys()
        return name
    return "MULTIPLE PARTS"


for i, j in Counter(map(jobtype, jobs)).items():
    print(i, j)
print()
print(len(jobs))
