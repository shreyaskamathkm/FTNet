import logging
from typing import List

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

from ..cfg import FTNetArgs
from ..helper import checkpoint
from .thermal_edge_trainer import SegmentationLightningModel

logger = logging.getLogger(__name__)


def setup_pl_loggers(args: FTNetArgs, ckp: checkpoint) -> tuple[TensorBoardLogger, WandbLogger]:
    logger.debug("Setting up  TensorBoard logger")
    """Setup TensorBoard and Weights & Biases loggers."""
    tensorboard_logger = TensorBoardLogger(
        save_dir=ckp.get_path("logs"),
        name=f"Tensorboard_{args.model.name}_{args.dataset.name}",
    )

    logger.debug("Setting up  Wandb logger")

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
    logger.debug(" Setting up Callbacks")

    return [
        LearningRateMonitor(),
        TQDMProgressBar(refresh_rate=10),
    ]


def train_model(args: FTNetArgs, ckp: checkpoint) -> None:
    """Train the model."""
    tensorboard_logger, wandb_logger = setup_pl_loggers(args, ckp)
    checkpoint_callbacks = setup_checkpoints_and_callbacks(args, ckp)

    model = SegmentationLightningModel(args=args, ckp=ckp)

    checkpoint_callbacks.extend(model.add_callback(ckp=ckp, train_only=args.task.train_only))

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
        deterministic="warn",
        reload_dataloaders_every_n_epochs=1,
        use_distributed_sampler=False,
        limit_train_batches=0.2 if args.task.debug else 1.0,
        limit_val_batches=0.0 if args.task.train_only else 1.0,
    )

    trainer.fit(model)


def test_model(args: FTNetArgs, ckp: checkpoint) -> None:
    """Test the model with a specific checkpoint."""
    if not args.checkpoint.test_checkpoint.exists():
        raise ValueError("Provide the checkpoint for testing")

    logger.info(f"Loading from {str(args.checkpoint.test_checkpoint)}")

    model = SegmentationLightningModel.load_from_checkpoint(
        checkpoint_path=args.checkpoint.test_checkpoint, args=args, ckp=ckp, train=False
    )

    trainer = Trainer(
        devices=args.compute.devices,
        accelerator=args.compute.accelerator,
        num_nodes=args.compute.num_nodes,
        inference_mode=True,
        max_epochs=1,
        fast_dev_run=False,
        deterministic=True,
        use_distributed_sampler=False,
    )

    trainer.test(model=model)
