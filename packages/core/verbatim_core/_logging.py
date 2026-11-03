"""Loggers of verbatim_core that don't follow the module path."""

import logging

# The one logger for records that hold customer text: questions, chunks, quotes and span texts.
# Records on it are DEBUG only, so an application can keep customer text out of its logs by
# leaving this logger at INFO, whatever level the module loggers have.
content_logger = logging.getLogger("verbatim_core.content")
