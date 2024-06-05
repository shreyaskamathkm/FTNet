#!/usr/bin/env python

#####################################################################################################################################################################
# FTNet                                                                                                                                                             #
# Copyright 2020 Tufts University.                                                                                                                                  #
# Please see LICENSE file for full terms.                                                                                                                           #
#####################################################################################################################################################################

import lightning as pl

from .cfg import FTNetArgs, parse_args
from .engine.setup_engine import test_model, train_model
from .helper import checkpoint, collect_env_info, get_rank, setup_logger


def main() -> None:
    args = parse_args()
    args = FTNetArgs.from_config(args.config)
    ckp = checkpoint(args.checkpoint.save_dir)

    logger = setup_logger(
        save_dir=ckp.get_path("logs"),
        distributed_rank=get_rank(),
        print_to_console=True,
    )
    if args.compute.seed:
        pl.seed_everything(args.compute.seed, workers=True)

    logger.info("Environment info:\n" + collect_env_info())

    if args.task.mode == "train":
        train_model(args, ckp)
    elif args.task.mode in ("test", "infer"):
        test_model(args, ckp)


if __name__ == "__main__":
    main()
