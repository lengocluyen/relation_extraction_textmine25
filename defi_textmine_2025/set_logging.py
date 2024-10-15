"""
See https://betterstack.com/community/questions/how-to-color-python-logging-output/
"""

import os
import logging

logging.basicConfig(level=logging.DEBUG)


CONSOLE_DEFAULT_FORMAT = (
    "%(asctime)s|%(levelname)s|(%(filename)s.py:%(lineno)d) - %(message)s"
)
FILE_DEFAULT_FORMAT = "%(asctime)s|%(levelname)s|%(filename)s.py - %(message)s - (%(pathname)s:%(lineno)d)"


class CustomFormatter(logging.Formatter):
    # grey = "\x1b[38;21m"
    # yellow = "\x1b[33;21m"
    # red = "\x1b[31;21m"
    # bold_red = "\x1b[31;1m"
    # reset = "\x1b[0m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = CONSOLE_DEFAULT_FORMAT
    date_format = "%H:%M:%S"
    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, self.date_format)
        return formatter.format(record)


def config_logging(log_file_path: str = None):
    # clear all handlers
    while logging.root.hasHandlers():
        logging.root.removeHandler(logging.root.handlers[0])
    dir_path = os.path.dirname(log_file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # logging.root
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(
            logging.Formatter(fmt=FILE_DEFAULT_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S")
        )
        file_handler.setLevel(logging.DEBUG)
        logging.root.addHandler(file_handler)

    console_handler = logging.StreamHandler(stream=None)
    console_handler.setFormatter(CustomFormatter())
    console_handler.setLevel(logging.INFO)
    logging.root.addHandler(console_handler)
