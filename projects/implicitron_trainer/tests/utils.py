# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import re
from typing import List


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
