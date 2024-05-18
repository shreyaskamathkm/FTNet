import functools
import logging
from pathlib import Path

from rich.logging import RichHandler
from pytorch_lightning.utilities import rank_zero_only

__all__ = ["setup_logger"]

MINIMUM_GLOBAL_LEVEL = logging.DEBUG
GLOBAL_HANDLER = logging.StreamHandler()
LOG_FORMAT = (
    "[%(asctime)s] - %(levelname)s - [%(name)s.%(funcName)s:%(lineno)d] - %(message)s"
)
LOG_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@rank_zero_only
@functools.lru_cache  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    save_dir: Path,
    filename: str = "log.txt",
    mode: str = "w",
    print_to_console: bool = True,
):
    # logger = logging.getLogger(name)
    logger = logging.getLogger()
    logger.setLevel(MINIMUM_GLOBAL_LEVEL)
    logger.propagate = False

    if print_to_console:
        ch = RichHandler()
        ch.setLevel(MINIMUM_GLOBAL_LEVEL)
        logger.addHandler(ch)

    if save_dir:
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = save_dir / filename
        fh = logging.FileHandler(log_file_path, mode=mode)
        fh.setLevel(MINIMUM_GLOBAL_LEVEL)
        # fh.setFormatter(LOG_FORMAT)
        logger.addHandler(fh)

    return logger
