import functools
import logging
from pathlib import Path

from rich.logging import RichHandler
from pytorch_lightning.utilities import rank_zero_only

__all__ = ["setup_logger"]


@rank_zero_only
@functools.lru_cache  # so that calling setup_logger multiple times won't add many handlers
def setup_logger(
    save_dir: Path,
    name: str = "FTNet",
    distributed_rank: int = 0,
    filename: str = "log.txt",
    mode: str = "w",
    abbrev_name: str = None,
    print_to_console: bool = True,
):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if distributed_rank > 0:
        return logger

    if not abbrev_name:
        abbrev_name = "SG" if name == "segmentation" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    if print_to_console:
        ch = RichHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(ch)

    if save_dir:
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = save_dir / filename
        fh = logging.FileHandler(log_file_path, mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger
