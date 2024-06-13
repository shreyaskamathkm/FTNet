import functools
import logging
import sys
from pathlib import Path

from lightning.pytorch.utilities import rank_zero_only
from rich.logging import RichHandler

__all__ = ["setup_logger"]

MINIMUM_GLOBAL_LEVEL = logging.INFO
DEBUG_LEVEL = logging.DEBUG

GLOBAL_HANDLER = logging.StreamHandler(stream=sys.stdout)
# Define the log format
LOG_FORMAT = logging.Formatter(
    "[%(asctime)s] - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
)
LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
import logging.config

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": True,
    }
)


@rank_zero_only
@functools.lru_cache  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    save_dir: Path,
    filename: str = "log.txt",
    mode: str = "w",
    distributed_rank: int = 0,
    debug: bool = False,
):
    level = DEBUG_LEVEL if debug else MINIMUM_GLOBAL_LEVEL
    logger = logging.getLogger()

    logger.setLevel(level)
    logger.propagate = False

    if distributed_rank > 0:
        return logger

    if distributed_rank == 0:
        ch = RichHandler()
        ch.setLevel(level)
        logger.addHandler(ch)

    if save_dir:
        if not filename.endswith((".txt", ".log")):
            filename = "log.txt"

        if distributed_rank > 0:
            filename = f"{filename}.rank{distributed_rank}"

        save_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = save_dir / filename

        fh = logging.FileHandler(log_file_path, mode=mode)
        fh.setLevel(level)
        fh.setFormatter(LOG_FORMAT)
        logger.addHandler(fh)

    return logger
