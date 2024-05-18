import pathlib
from typing import List
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, TQDMProgressBar, Callback
from cfg import FTNetArgs
from utils import checkpoint
from .thermal_edge_trainer import SegmentationLightningModel as thermal_edge_trainer
from .thermal_edge_trainer_train_only import (
    SegmentationLightningModel as thermal_edge_trainer_train_only,
)
import logging

logger = logging.getLogger(__name__)


def setup_pl_loggers(
    args: FTNetArgs, ckp: checkpoint
) -> tuple[TensorBoardLogger, WandbLogger]:
    """Setup TensorBoard and Weights & Biases loggers."""
    tensorboard_logger = TensorBoardLogger(
        save_dir=ckp.get_path("logs"),
        name=f"Tensorboard_{args.model.name}_{args.dataset.name}",
    )

    wandb_logger = WandbLogger(
        name=f"Model: {args.model.name} Dataset: {args.dataset.name} Des : {args.wandb.wandb_name_ext}",
        id=args.wandb.wandb_id,
        project="Thermal Segmentation",
        offline=args.task.debug,
        save_dir=ckp.get_path("logs"),
    )
    return tensorboard_logger, wandb_logger


def setup_checkpoints_and_callbacks(args: FTNetArgs, ckp: checkpoint) -> List[Callback]:
    """Setup checkpoint directory and callbacks."""
    if not args.wandb.wandb_id:
        wandb_text_file = ckp.get_path("logs") / "wandb_id.txt"
        if wandb_text_file.is_file():
            with wandb_text_file.open() as f:
                args.wandb.wandb_id = f.read().strip()
        else:
            import wandb

            args.wandb.wandb_id = wandb.util.generate_id()
            with wandb_text_file.open("a") as f:
                f.write(args.wandb.wandb_id)

    checkpoint_callbacks = [
        LearningRateMonitor(),
        TQDMProgressBar(refresh_rate=10),
    ]
    return checkpoint_callbacks


def train_model(args: FTNetArgs, ckp: checkpoint) -> None:
    """Train the model."""
    tensorboard_logger, wandb_logger = setup_pl_loggers(args, ckp)
    checkpoint_callbacks = setup_checkpoints_and_callbacks(args, ckp)

    model = (
        thermal_edge_trainer_train_only(args=args, ckp=ckp)
        if args.task.train_only
        else thermal_edge_trainer(args=args, ckp=ckp)
    )

    checkpoint_callbacks.append(model._add_callback(ckp=ckp))

    trainer = Trainer(
        default_root_dir=ckp.get_path("save_dir"),
        devices=args.compute.devices,
        accelerator=args.compute.accelerator,
        num_nodes=args.compute.num_nodes,
        logger=[tensorboard_logger, wandb_logger],
        max_epochs=args.trainer.epochs,
        sync_batchnorm=True,
        accumulate_grad_batches=args.trainer.accumulate_grad_batches,
        callbacks=checkpoint_callbacks,
        fast_dev_run=args.task.debug,
        deterministic=True,
        reload_dataloaders_every_n_epochs=1,
    )

    trainer.fit(model)


def test_model(args: FTNetArgs, ckp: checkpoint) -> None:
    """Test the model with a specific checkpoint."""
    t_checkpoint = None
    if args.test_checkpoint:
        t_checkpoint = pathlib.Path(args.test_checkpoint)
    elif args.test_monitor and pathlib.Path(args.test_monitor_path).exists():
        best = 0.0
        for x in pathlib.Path(args.test_monitor_path).iterdir():
            if args.test_monitor in x.name:
                val = float(x.name[-11:-5])
                if val >= best:
                    t_checkpoint = x
                    best = val
        logger.info(f"Final best checkpoint is {t_checkpoint}")
    else:
        raise ValueError("Provide the checkpoint for testing")

    logger.info(f"Loading from {t_checkpoint}")

    model = thermal_edge_trainer.load_from_checkpoint(
        checkpoint_path=t_checkpoint, args=args, ckp=ckp, train=False
    )

    trainer = Trainer(
        devices=args.gpus,
        num_nodes=args.num_nodes,
        max_epochs=1,
        strategy="dp",
        fast_dev_run=False,
        progress_bar_refresh_rate=0,
        deterministic=True,
        replace_sampler_ddp=False,
    )

    trainer.test(model=model)
