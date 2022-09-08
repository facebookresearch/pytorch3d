# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
import re


@contextlib.contextmanager
def intercept_logs(logger_name: str, regexp: str):
    # Intercept logs that match a regexp, from a given logger.
    intercepted_messages = []
    logger = logging.getLogger(logger_name)

    class LoggerInterceptor(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            if re.search(regexp, message):
                intercepted_messages.append(message)
            return True

    interceptor = LoggerInterceptor()
    logger.addFilter(interceptor)
    try:
        yield intercepted_messages
    finally:
        logger.removeFilter(interceptor)


def interactive_testing_requested() -> bool:
    """
    Certain tests are only useful when run interactively, and so are not regularly run.
    These are activated by this funciton returning True, which the user requests by
    setting the environment variable `PYTORCH3D_INTERACTIVE_TESTING` to 1.
    """
    return os.environ.get("PYTORCH3D_INTERACTIVE_TESTING", "") == "1"
