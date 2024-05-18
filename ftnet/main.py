#!/usr/bin/env python

#####################################################################################################################################################################
# FTNet                                                                                                                                                             #
# Copyright 2020 Tufts University.                                                                                                                                  #
# Please see LICENSE file for full terms.                                                                                                                           #
#####################################################################################################################################################################

import lightning as pl
from cfg import parse_args, FTNetArgs
from utils import collect_env_info, setup_logger, checkpoint
from engine.setup_engine import train_model, test_model


def main() -> None:
    args = parse_args()
    args = FTNetArgs.from_config(args.config)
    ckp = checkpoint(args.checkpoint.save_dir)

    logger = setup_logger(
        save_dir=ckp.get_path("logs"), print_to_console=args.task.debug
    )
    if args.compute.seed:
        pl.seed_everything(args.compute.seed, workers=True)

    logger.info("Environment info:\n" + collect_env_info())

    if args.task.mode == "train":
        train_model(args, ckp)
    elif args.task.mode == "test":
        test_model(args, ckp)


if __name__ == "__main__":
    main()
