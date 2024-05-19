import functools
import logging
import sys
from pathlib import Path

from rich.logging import RichHandler

__all__ = ["setup_logger"]

MINIMUM_GLOBAL_LEVEL = logging.DEBUG
GLOBAL_HANDLER = logging.StreamHandler(stream=sys.stdout)
LOG_FORMAT = (
    "[%(asctime)s] - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
)
LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


# @rank_zero_only
@functools.lru_cache  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    save_dir: Path,
    filename: str = "log.txt",
    mode: str = "w",
    distributed_rank: int = 0,
    print_to_console: bool = True,
):
    # logger = logging.getLogger(name)
    logger = logging.getLogger()
    logger.setLevel(MINIMUM_GLOBAL_LEVEL)
    logger.propagate = False

    if distributed_rank > 0:
        return logger

    if print_to_console and distributed_rank == 0:
        # ch = GLOBAL_HANDLER
        # ch.setLevel(MINIMUM_GLOBAL_LEVEL)
        # logger.addHandler(ch)

        ch = RichHandler()
        ch.setLevel(MINIMUM_GLOBAL_LEVEL)
        logger.addHandler(ch)

    if save_dir:
        if filename.endswith(".txt") or filename.endswith(".log"):
            filename = filename
        else:
            filename = save_dir / "log.txt"

        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)

        save_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(filename, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(LOG_FORMAT)
        logger.addHandler(fh)

    return logger
