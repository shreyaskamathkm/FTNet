'''
Adapted from
https://detectron2.readthedocs.io/en/latest/_modules/detectron2/utils/logger.html
'''

import functools
import logging
import os
import sys
import time
from collections import Counter
from logging.handlers import RotatingFileHandler

from tabulate import tabulate
from termcolor import colored
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn

__all__ = ['setup_logger']


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        elif record.levelno == logging.CRITICAL:
            prefix = colored("CRITICAL", "blue", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@rank_zero_only
@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(save_dir: str,
                 name: str = 'iDINE',
                 distributed_rank: int = 0,
                 filename: str = "log.txt",
                 mode: str = 'w',
                 color: bool = True,
                 abbrev_name: str = None,
                 print_to_console: bool = True):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    # don't log results for the non-master process

    if distributed_rank > 0:
        return logger

    if abbrev_name is None:
        abbrev_name = "SG" if name == "segmentation" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")

    if print_to_console:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)  # 'a+' for add, 'w' for overwrite
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# @functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
# def setup_logger(save_dir: str,
#                  name: str = 'iDINE',
#                  distributed_rank: int = 0,
#                  level=logging.DEBUG,
#                  filename: str = "log.txt",
#                  mode: str = 'w',
#                  color: bool = True,
#                  log_size_mb=1000,
#                  abbrev_name: str = None,
#                  num_log_archives=10,
#                  stdout: bool = True):

#     # Get the Lightning logger and add handlers/formatter
#     if not stdout and filename is None:
#         raise ValueError('ConsoleLogger will have no handlers if stdout=False and file=None')

#     logger = logging.getLogger(name)
#     logger.setLevel(level)
#     logger.propagate = False

#     if distributed_rank > 0:
#         return logger

#     if abbrev_name is None:
#         abbrev_name = "SG" if name == "segmentation" else name

#     plain_formatter = logging.Formatter(
#         "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")

#     if stdout:
#         ch = logging.StreamHandler(stream=sys.stdout)
#         ch.setLevel(logging.DEBUG)
#         if color:
#             formatter = _ColorfulFormatter(
#                 colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
#                 datefmt="%m/%d %H:%M:%S",
#                 root_name=name,
#                 abbrev_name=str(abbrev_name),
#             )
#         else:
#             formatter = plain_formatter
#         ch.setFormatter(formatter)
#         logger.addHandler(ch)

#     if save_dir:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         # should_roll_over = os.path.isfile(os.path.join(save_dir, filename))
#         # if should_roll_over:
#         #     fh = RotatingFileHandler(os.path.join(save_dir, filename), mode='w', encoding='utf-8', backupCount=5)
#         #     fh.doRollover()
#         # else:
#             file_handler = RotatingFileHandler(os.path.join(save_dir, filename),
#                                                maxBytes=log_size_mb * 1024 * 1024,
#                                                backupCount=num_log_archives)
#             file_handler.setFormatter(logging.Formatter(
#                 "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S")
#             )
#             logger.addHandler(file_handler)
#     logger.debug('Initialized ConsoleLogger')

#     return logger
