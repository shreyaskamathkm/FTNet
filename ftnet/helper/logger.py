import functools
import logging
import sys
from pathlib import Path

from lightning.pytorch.utilities import rank_zero_only
from rich.logging import RichHandler

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
    """Sets up a structured logger with optional file and console output.

    Args:
        save_dir (Path, optional): Path to the directory for saving logs.
            If not provided, logs will only be written to the console.
        filename (str, optional): Name of the log file. Defaults to "log.txt".
            Will be appended with ".rank{rank}" if distributed_rank is greater than 0.
            If the filename doesn't end with ".txt" or ".log", it will be automatically
            changed to "log.txt" for consistency.
        mode (str, optional): Mode for opening the log file ("w" for overwrite,
            "a" for append). Defaults to "w".
        distributed_rank (int, optional): The rank of the process in a distributed
            training environment. Defaults to 0.
        debug (bool, optional): Whether to enable debug logging level. Defaults to False.

    Returns:
        logging.Logger: The configured logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(DEBUG_LEVEL if debug else MINIMUM_GLOBAL_LEVEL)
    logger.propagate = False

    if distributed_rank > 0:
        return logger

    if distributed_rank == 0:
        ch = RichHandler(show_time=False)
        ch.setLevel(logger.level)
        logger.addHandler(ch)

    if save_dir:
        if not filename.endswith((".txt", ".log")):
            filename = "log.txt"

        if distributed_rank > 0:
            filename = f"{filename}.rank{distributed_rank}"

        save_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = save_dir / filename

        fh = logging.FileHandler(log_file_path, mode=mode)
        fh.setLevel(logger.level)
        fh.setFormatter(LOG_FORMAT)
        logger.addHandler(fh)

    return logger
