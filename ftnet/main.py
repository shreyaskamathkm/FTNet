#!/usr/bin/env python

#####################################################################################################################################################################
# FTNet                                                                                                                                                             #
# Copyright 2020 Tufts University.                                                                                                                                  #                                                                                                                #
# Please see LICENSE file for full terms.                                                                                                                           #                                                                                                                                              #                                                                                                                                                 #
#####################################################################################################################################################################

import os
import lightning as pl
from cfg import parse_args, FTNetArgs
from utils import collect_env_info, setup_logger, checkpoint
from utils.utils import get_rank
from engine import thermal_edge_trainer_train_only, thermal_edge_trainer

args = parse_args()
args = FTNetArgs.from_config(args.config)
ckp = checkpoint(args.checkpoint.save_dir)

if args.compute.seed:
    pl.seed_everything(args.compute.seed, workers=True)

logger = setup_logger(
    name="pytorch_lightning",
    save_dir=ckp.get_path("logs"),
    distributed_rank=get_rank(),
    print_to_console=True if args.task.debug else False,
)

logger.info("Environment info:\n" + collect_env_info())

if args.task.mode == "train":
    if len(list(ckp.get_path("ckpt").glob("**/*"))) > 0:
        if args.checkpoint.resume:
            logger.info("Resuming from previous last checkpoint\n")
            args.checkpoint.resume = ckp.get_path("ckpt") / "last.ckpt"

    if not args.wandb.wandb_id:
        wandb_text_file = ckp.get_path("logs") / "wandb_id.txt"
        if wandb_text_file.is_file():
            with open(wandb_text_file) as f:
                for line in f:  # Read line-by-line
                    args.wandb.wandb_id = line.strip()
        else:
            import wandb

            args.wandb.wandb_id = wandb.util.generate_id()
            with open(wandb_text_file, "a") as f:
                f.write(args.wandb.wandb_id)
        logger.info(
            "-" * 50
            + f"\nNew WANDB ID GENERATED\n {args.wandb.wandb_id}\n Please use the above id for resuming\n"
            + "-" * 50
        )

    tensorboard_logger = pl.loggers.TensorBoardLogger(
        save_dir=ckp.get_path("logs"),
        name=f"Tensorboard_{args.model.model}_{args.dataset.dataset}",
    )

    wandb_logger = pl.loggers.WandbLogger(
        name=f"Model: {args.model.model} Dataset: {args.dataset.dataset} Des : {args.wandb.wandb_name_ext}",
        id=args.wandb.wandb_id,
        project="Thermal Segmentation",
        offline=args.task.debug,
        save_dir=ckp.get_path("logs"),
    )

    checkpoint_callbacks = [
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.TQDMProgressBar(refresh_rate=10),
    ]

    if args.task.train_only:
        model = thermal_edge_trainer_train_only(args=args, ckp=ckp, logger=logger)
    else:
        model = thermal_edge_trainer(args=args, ckp=ckp, logger=logger)

    checkpoint_callbacks.append(model._add_callback(ckp=ckp))

    trainer = pl.Trainer(
        default_root_dir=args.save_dir,
        resume_from_checkpoint=args.resume,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        logger=[tensorboard_logger, wandb_logger],
        max_epochs=args.epochs,
        amp_level="O0",
        sync_batchnorm=True,
        distributed_backend=args.distributed_backend,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=checkpoint_callbacks,
        fast_dev_run=args.debug,
        plugins=pl.plugins.DDPPlugin(find_unused_parameters=True),
        deterministic=True,
        replace_sampler_ddp=False,
    )

    trainer.fit(model)

    if args.mode == "train_test":
        trainer_v2 = pl.Trainer(
            fast_dev_run=args.debug,
            distributed_backend="dp",
            gpus=1,
            deterministic=True,
        )

        ckpt_list = {}
        for cb in trainer.checkpoint_callbacks:
            ckpt_path = cb.best_model_path
            if os.path.isfile(ckpt_path):
                ckpt_list[cb.monitor] = ckpt_path
                if cb.last_model_path:
                    ckpt_list["last"] = cb.last_model_path

        for name, path in ckpt_list.items():
            model = thermal_edge_trainer.load_from_checkpoint(
                checkpoint_path=path, args=args, ckp=ckp, train=False, logger=logger
            )
            trainer_v2.test_name = name
            trainer_v2.test(model=model)

elif args.mode == "test":
    if args.test_checkpoint is not None:
        t_checkpoint = args.test_checkpoint
    elif args.test_monitor is not None and os.path.exists(args.test_monitor_path):
        best = 0.0
        logger.info(f"Searching in {args.test_monitor_path}")
        for x in os.listdir(args.test_monitor_path):
            if args.test_monitor in x:
                val = float(x[-11:-5])
                if val >= best:
                    t_checkpoint = os.path.join(args.test_monitor_path, x)
                    logger.info(f"Found {t_checkpoint}")
                    best = val
        logger.info(f"Final best checkpoint is {t_checkpoint}")
    else:
        ValueError("Provide the checkpoint for testing")

    logger.info(f"Loading from {t_checkpoint}")

    model = thermal_edge_trainer.load_from_checkpoint(
        checkpoint_path=t_checkpoint, args=args, ckp=ckp, train=False, logger=logger
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        max_epochs=1,
        distributed_backend="dp",
        amp_level="O0",
        fast_dev_run=False,
        progress_bar_refresh_rate=0,
        deterministic=True,
        replace_sampler_ddp=False,
    )

    trainer.test(model=model)
