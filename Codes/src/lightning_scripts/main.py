#!/usr/bin/env python

#####################################################################################################################################################################
# FTNet                                                                                                                                                             #
# Copyright 2020 Tufts University.                                                                                                                                  #                                                                                                                #
# Please see LICENSE file for full terms.                                                                                                                           #                                                                                                                                              #                                                                                                                                                 #
#####################################################################################################################################################################

import inspect
import os
import sys


os.system("wandb login ")
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.abspath(os.path.dirname(currentdir))
sys.path.append(parentdir)

from trainers import thermal_edge_trainer
from trainers import thermal_edge_trainer_train_only
from pytorch_lightning.plugins import DDPPlugin
from options import parse_args
from core.utils.filesystem import checkpoint
from core.utils import collect_env_info, get_rank, setup_logger
from core.callbacks import ProgressBar
import pytorch_lightning as pl



args = parse_args()
ckp = checkpoint(args)

# seeding
if args.seed is not None:
    pl.seed_everything(args.seed, workers=True)


log_dir = ckp.get_path("logs")

logger = setup_logger(name="pytorch_lightning",
                      save_dir=log_dir,
                      distributed_rank=get_rank(),
                      color=True,
                      abbrev_name=None,
                      print_to_console=True if args.debug else False)

logger.info("Environment info:\n" + collect_env_info())

if "train" in args.mode:
    ckpt_dir = ckp.get_path("ckpt")

    if len(os.listdir(ckpt_dir)) > 0:
        if args.resume is None:
            logger.info("Resuming from previous last checkpoint\n")
            args.resume = os.path.join(ckpt_dir, "last.ckpt")

    if args.wandb_id is None:
        wandb_text_file = os.path.join(log_dir, "wandb_id.txt")
        if os.path.isfile(wandb_text_file):
            with open(wandb_text_file, "r") as f:
                for line in f:  # Read line-by-line
                    args.wandb_id = (
                        line.strip()
                    )  # Strip the leading/trailing whitespaces and newline
            msg = ("-" * 50
                   + "\nUsing existing WANDB ID\n {} \n ".format(args.wandb_id)
                   + "-" * 50
                   )
        else:
            import wandb
            args.wandb_id = wandb.util.generate_id()
            with open(wandb_text_file, "a") as f:
                f.write(args.wandb_id)
            msg = (
                "-" * 50
                + "\nNew WANDB ID GENERATED\n {} \n Please use the above id for resuming\n".format(
                    args.wandb_id
                ) +
                "-" * 50
            )
        logger.info(msg)

    testTube_logger = pl.loggers.TestTubeLogger(save_dir=log_dir,
                                                name="TestTube_{}_{}".format(args.model, args.dataset)
                                                )

    wandb_logger = pl.loggers.wandb.WandbLogger(name="Model: {} Datset: {} Des : {} ".format(args.model,
                                                                                             args.dataset,
                                                                                             args.wandb_name_ext, ),
                                                id=args.wandb_id,
                                                project="Thermal Segmentation",
                                                entity="tufts",
                                                offline=args.debug,
                                                save_dir=log_dir,
                                                )

    if args.distributed_backend == "ddp":
        args.train_batch_size = max(1, int(args.train_batch_size / max(1, args.gpus)))
        args.val_batch_size = max(1, int(args.val_batch_size / max(1, args.gpus)))
        args.workers = max(1, int(args.workers / max(1, args.gpus)))

    checkpoint_callbacks = [pl.callbacks.LearningRateMonitor(), ProgressBar(logger)]

    if args.train_only:
        model = thermal_edge_trainer_train_only(args=args, ckp=ckp, logger=logger)
        checkpoint_callbacks.append(pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=ckpt_dir,
                                                                                  filename="{epoch}",
                                                                                  verbose=True,
                                                                                  every_n_val_epochs=1,
                                                                                  save_last=True,
                                                                                  )
                                    )
    else:
        model = thermal_edge_trainer(args=args, ckp=ckp, logger=logger)

        checkpoint_callbacks.append(pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=ckpt_dir,
                                                                                  filename="{epoch}-{val_avg_mIOU_Acc:.4f}",
                                                                                  save_top_k=3,
                                                                                  verbose=True,
                                                                                  every_n_val_epochs=1,
                                                                                  monitor="val_avg_mIOU_Acc",
                                                                                  mode="max",
                                                                                  save_last=True,
                                                                                  )
                                    )

        checkpoint_callbacks.append(pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=ckpt_dir,
                                                                                  filename="{epoch}-{val_mIOU:.4f}",
                                                                                  save_top_k=3,
                                                                                  verbose=True,
                                                                                  monitor="val_mIOU",
                                                                                  mode="max",
                                                                                  every_n_val_epochs=1,
                                                                                  )
                                    )

        checkpoint_callbacks.append(pl.callbacks.model_checkpoint.ModelCheckpoint(dirpath=ckpt_dir,
                                                                                  filename="{epoch}-{val_mAcc:.4f}",
                                                                                  save_top_k=3,
                                                                                  verbose=True,
                                                                                  monitor="val_mAcc",
                                                                                  mode="max",
                                                                                  every_n_val_epochs=1,
                                                                                  )
                                    )

    trainer = pl.Trainer(default_root_dir=args.save_dir,
                         resume_from_checkpoint=args.resume,
                         gpus=args.gpus,
                         num_nodes=args.num_nodes,
                         logger=[testTube_logger, wandb_logger],
                         max_epochs=args.epochs,
                         amp_level="O0",
                         sync_batchnorm=True,
                         distributed_backend=args.distributed_backend,
                         accumulate_grad_batches=args.accumulate_grad_batches,
                         callbacks=checkpoint_callbacks,
                         fast_dev_run=args.debug,
                         plugins=DDPPlugin(find_unused_parameters=True),
                         deterministic=True,
                         replace_sampler_ddp=False)

    trainer.fit(model)
    '''
    Note: train_test provides inconsistent results on ddp. Please avoid this on ddp
    '''
    if args.mode == "train_test":
        trainer_v2 = pl.Trainer(fast_dev_run=args.debug,
                                distributed_backend="dp",
                                gpus=1,
                                deterministic=True,
                                callbacks=ProgressBar(logger),
                                )

        ckpt_list = {}
        for _ckpt in range(len(trainer.checkpoint_callbacks)):
            logger.info("Testing: monitor metric: {}".format(trainer.checkpoint_callbacks[_ckpt].monitor
                                                             )
                        )
            ckpt_path = trainer.checkpoint_callbacks[_ckpt].best_model_path

            if os.path.isfile(ckpt_path):
                ckpt_list["{}".format(trainer.checkpoint_callbacks[_ckpt].monitor)] = ckpt_path
                if trainer.checkpoint_callbacks[_ckpt].last_model_path != "":
                    ckpt_list["last"] = trainer.checkpoint_callbacks[_ckpt].last_model_path

        for name, path in ckpt_list.items():
            logger.info("Best checkpoint path: {}".format(path))
            model = thermal_edge_trainer.load_from_checkpoint(checkpoint_path=path,
                                                              args=args,
                                                              ckp=ckp,
                                                              train=False,
                                                              logger=logger
                                                              )
            trainer_v2.test_name = name
            trainer_v2.test(model=model)

elif args.mode == "test":
    if args.test_checkpoint is not None:
        t_checkpoint = args.test_checkpoint

    elif args.test_monitor is not None and os.path.exists(args.test_monitor_path):
        best = 0.0
        logger.info("Searching in {}".format(args.test_monitor_path))
        for x in os.listdir(args.test_monitor_path):
            if args.test_monitor in x:
                val = float(x[-11:-5])
                if val >= best:
                    t_checkpoint = os.path.join(args.test_monitor_path, x)
                    logger.info("Found {}".format(t_checkpoint))
                    best = val
        logger.info("Final best checkpoint is {}".format(t_checkpoint))
    else:
        ValueError("Provide the checkpoint for testing")

    logger.info("Loading from {}".format(t_checkpoint))

    model = thermal_edge_trainer.load_from_checkpoint(checkpoint_path=t_checkpoint,
                                                      args=args,
                                                      ckp=ckp,
                                                      train=False,
                                                      logger=logger,
                                                      )

    trainer = pl.Trainer(gpus=args.gpus,
                         num_nodes=args.num_nodes,
                         max_epochs=1,
                         distributed_backend="dp",
                         amp_level="O0",
                         callbacks=[ProgressBar(logger)],
                         fast_dev_run=False,
                         progress_bar_refresh_rate=0,
                         deterministic=True,
                         replace_sampler_ddp=False,
                         )

    trainer.test(model=model)
