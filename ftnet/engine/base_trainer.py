#!/usr/bin/env python
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.data.samplers import (
    make_batch_data_sampler,
    make_data_sampler,
    make_multiscale_batch_data_sampler,
)
from core.loss import get_segmentation_loss
from helper.model_helpers import save_model_summary
from helper.optimizer_scheduler_helper import make_optimizer, make_scheduler
from lightning.pytorch import LightningModule
from models import get_segmentation_model
from torchmetrics import ConfusionMatrix as pl_ConfusionMatrix
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import MulticlassJaccardIndex

from data import get_segmentation_dataset
from ftnet.helper.model_helpers import check_mismatch

logger = logging.getLogger(__name__)


class BaseTrainer(LightningModule):
    def __init__(
        self,
        args: Namespace,
        ckp: Callable,
        train: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the BaseTrainer with arguments for configuration.

        Args:
            args (Namespace): Argument namespace containing configuration parameters.
            ckp (Callable): Checkpoint handler.
            train (bool): Flag to indicate if the model is in training mode. Defaults to True.
            **kwargs (Any): Additional arguments.
        """

        super().__init__(**kwargs)
        self.ckp = ckp
        self.args = args

        logger.info("Setting Up Segmentation Model")
        self.model = get_segmentation_model(
            model_name=self.args.model.name,
            dataset=self.args.dataset.name,
            backbone=self.args.model.backbone,
            norm_layer=nn.BatchNorm2d,
            dilated=self.args.model.dilation,
            no_of_filters=self.args.model.no_of_filters,
            pretrained_base=self.args.model.pretrained_base,
            edge_extracts=self.args.model.edge_extracts,
            num_blocks=self.args.model.num_blocks,
        )

        if train:
            save_model_summary(self.model, Path(self.ckp.get_path("logs")))  # type: ignore

            self._preload_complete_data()

            if (
                self.args.checkpoint.pretrain_checkpoint
                and self.args.checkpoint.pretrain_checkpoint.exists()
            ):
                self.load_weights_from_checkpoint(
                    self.args.trainer.pretrain_checkpoint, pretrain=True
                )

            self.criterion = get_segmentation_loss(
                self.args.model.name,
                loss_weight=self.args.trainer.loss_weight,
                ignore_index=self.train_dataset.IGNORE_INDEX,
            )

    def _preload_complete_data(self):
        """Preloads the complete training and validation datasets."""

        data_kwargs = {
            "root": self.args.dataset.dataset_path,
            "base_size": self.args.dataset.base_size,
            "crop_size": self.args.dataset.crop_size,
        }

        self.train_dataset = get_segmentation_dataset(
            name=self.args.dataset.name,
            split="train",
            mode="train",
            sobel_edges=False,
            **data_kwargs,
        )

        self.val_dataset = get_segmentation_dataset(
            self.args.dataset.name,
            split="test" if self.args.dataset.name != "scutseg" else "val",
            mode="val",
            sobel_edges=False,
            **data_kwargs,
        )

    def configure_optimizers(self):
        """Configures the optimizer and scheduler for training.

        Returns:
            Tuple[List[Optimizer], List[Dict[str, Any]]]: The optimizer and the scheduler.
        """
        logger.info("Setting Up Optimizer")

        params: List[dict] = []
        if hasattr(self.model, "encoder"):
            params.append(
                {
                    "params": self.model.encoder.parameters(),
                    "lr": self.args.optimizer.lr,
                }
            )
        if hasattr(self.model, "exclusive"):
            params.extend(
                {
                    "params": getattr(self.model, module).parameters(),
                    "lr": self.args.optimizer.lr * 10,
                }
                for module in self.model.exclusive
            )
        if not params:
            params = list(filter(lambda x: x.requires_grad, self.model.parameters()))

        optimizer = make_optimizer(args=self.args, params=params)

        logger.info("Setting Up Scheduler")
        logger.info(
            f"Iterations per epoch computed for scheduler is {self.trainer.estimated_stepping_batches}"
        )

        scheduler = make_scheduler(
            args=self.args,
            optimizer=optimizer,
            iters_per_epoch=self.trainer.estimated_stepping_batches,
            last_epoch=-1,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": scheduler.__interval__}]

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Sets up the training DataLoader with appropriate batch size and data
        sampler.

        Returns:
            torch.utils.data.DataLoader: The training DataLoader.
        """

        self.distributed = max(1, self.trainer.num_devices, self.trainer.num_nodes)

        self.args.trainer.train_batch_size = max(
            1, int(self.args.trainer.train_batch_size / self.distributed)
        )
        self.args.trainer.val_batch_size = max(
            1, int(self.args.trainer.val_batch_size / self.distributed)
        )
        self.args.compute.workers = max(1, int(self.args.compute.workers / self.distributed))

        train_sampler = make_data_sampler(
            dataset=self.train_dataset,
            shuffle=True,
            distributed=self.distributed > 1,
        )

        train_batch_sampler = make_multiscale_batch_data_sampler(
            sampler=train_sampler,
            batch_size=self.args.trainer.train_batch_size,
            multiscale_step=1,
            scales=len(min(self.args.dataset.base_size, self.args.dataset.crop_size)),
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=self.args.compute.workers,
            pin_memory=True,
        )

        self.load_metrics(
            mode="train",
            num_class=self.train_dataset.NUM_CLASS,
            ignore_index=self.train_dataset.IGNORE_INDEX,
        )

        return train_loader

    def val_dataloader(self):
        """Sets up the validation DataLoader with appropriate batch size and
        data sampler.

        Returns:
            torch.utils.data.DataLoader: The validation DataLoader.
        """
        if not self.args.task.train_only:
            val_sampler = make_data_sampler(
                dataset=self.val_dataset,
                shuffle=False,
                distributed=self.distributed > 1,
            )

            val_batch_sampler = make_batch_data_sampler(
                val_sampler, batch_size=self.args.trainer.val_batch_size
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=self.val_dataset,
                batch_sampler=val_batch_sampler,
                num_workers=self.args.compute.workers,
                pin_memory=True,
            )

            self.load_metrics(
                mode="val",
                num_class=self.val_dataset.NUM_CLASS,
                ignore_index=self.val_dataset.IGNORE_INDEX,
            )
            return val_loader

        return None

    def test_dataloader(self):
        """Sets up the test DataLoader with appropriate batch size and data
        sampler.

        Returns:
            torch.utils.data.DataLoader: The test DataLoader.
        """
        if self.args.task.mode == "test":
            logger.debug(" Running Inference Only")

            self.test_dataset = get_segmentation_dataset(
                self.args.dataset.name,
                split="test",
                mode="test",
                root=self.args.dataset.dataset_path,
                base_size=None,
                crop_size=None,
            )
            self.load_metrics(
                mode="test",
                num_class=self.test_dataset.NUM_CLASS,
                ignore_index=self.test_dataset.IGNORE_INDEX,
            )

        elif self.args.task.mode == "infer":
            logger.debug(" Running Inference Only")
            self.test_dataset = get_segmentation_dataset(
                "load_image", root=self.args.dataset.dataset_path, dataset=self.args.dataset.name
            )

        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            num_workers=self.args.compute.workers,
            pin_memory=True,
        )

    def load_metrics(self, mode: str, num_class: int, ignore_index: int) -> None:
        """Loads the metrics for a specific mode (train, val, or test).

        Args:
            mode (str): The mode for which to load metrics (train, val, test).
            num_class (int): The number of classes in the dataset.
            ignore_index (int): The index to ignore during metric computation.
        """
        self._set_metric(
            mode,
            "confmat",
            pl_ConfusionMatrix(task="multiclass", num_classes=num_class).to(self.device),
        )
        self._set_metric(
            mode,
            "IOU",
            MulticlassJaccardIndex(
                num_classes=num_class, ignore_index=ignore_index, average="none"
            ).to(self.device),
        )
        self._set_metric(mode, "edge_accuracy", MeanMetric().to(self.device))
        self._set_metric(mode, "loss", MeanMetric().to(self.device))

    def _set_metric(self, mode: str, metric_name: str, metric) -> None:
        """Sets a specific metric for a mode (train, val, or test).

        Args:
            mode (str): The mode for which to set the metric (train, val, test).
            metric_name (str): The name of the metric.
            metric: The metric object.
        """
        setattr(self, f"{mode}_{metric_name}", metric)

    def _update_metrics(
        self,
        mode: str,
        edge_pred: torch.Tensor,
        class_map: torch.Tensor,
        target: torch.Tensor,
        loss_val: Union[torch.Tensor, None],
    ) -> None:
        """Updates metrics for a specific mode (train, val, or test).

        Args:
            mode (str): The mode for which to update metrics (train, val, test).
            edge_pred (torch.Tensor): The edge predictions.
            class_map (torch.Tensor): The class map predictions.
            target (torch.Tensor): The ground truth targets.
            loss_val (Union[torch.Tensor, None]): The loss value, if applicable.
        """
        getattr(self, f"{mode}_edge_accuracy").update(edge_pred)
        getattr(self, f"{mode}_confmat").update(preds=class_map, target=target)
        getattr(self, f"{mode}_IOU").update(preds=class_map, target=target)
        if loss_val is not None:
            getattr(self, f"{mode}_loss").update(loss_val)

    def _log_epoch_end(self, mode: str) -> None:
        """Logs the metrics at the end of an epoch for a specific mode (train,
        val, or test).

        Args:
            mode (str): The mode for which to log metrics (train, val, test).
        """
        confusion_matrix = getattr(self, f"{mode}_confmat").compute()
        iou = getattr(self, f"{mode}_IOU").compute()
        accuracy = self._compute_accuracy(confusion_matrix)

        log_dict = {
            f"{mode}_mIOU": torch.mean(iou),
            f"{mode}_mAcc": torch.mean(accuracy),
            f"{mode}_avg_mIOU_Acc": (torch.mean(iou) + torch.mean(accuracy)) / 2,
            f"{mode}_edge_accuracy": getattr(self, f"{mode}_edge_accuracy").compute(),
            f"{mode}_loss": getattr(self, f"{mode}_loss").compute(),
        }

        self.log_dict(log_dict, prog_bar=True)
        self._reset_metrics(mode)

    def _reset_metrics(self, mode: str) -> None:
        """Resets the metrics for a specific mode (train, val, or test).

        Args:
            mode (str): The mode for which to reset metrics (train, val, test).
        """
        getattr(self, f"{mode}_confmat").reset()
        getattr(self, f"{mode}_IOU").reset()
        getattr(self, f"{mode}_edge_accuracy").reset()
        getattr(self, f"{mode}_loss").reset()

    def _upsample_output(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Upsamples the output tensor to match the target tensor size if
        necessary.

        Args:
            output (torch.Tensor): The output tensor.
            target (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The upsampled output tensor.
        """
        if output.shape != target.shape:
            return F.interpolate(
                output,
                size=(target.shape[1], target.shape[2]),
                mode="bilinear",
                align_corners=True,
            )
        return output

    def _compute_accuracy(self, confusion_matrix: torch.Tensor) -> torch.Tensor:
        """Computes the accuracy from the confusion matrix.

        Args:
            confusion_matrix (torch.Tensor): The confusion matrix.

        Returns:
            torch.Tensor: The computed accuracy.
        """
        acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        acc[torch.isnan(acc)] = 0
        return acc

    def training_step(self, *args: Any, **kwargs: Any) -> None:
        """Abstract method for the training step.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def on_train_epoch_end(self) -> None:
        """Abstract method called at the end of a training epoch.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Abstract method for the validation step.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        """Abstract method called at the end of a validation epoch.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Abstract method for the test step.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        """Abstract method called at the end of a test epoch.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def load_weights_from_checkpoint(self, checkpoint: str, pretrain: bool = False) -> None:
        """Loads model weights from a checkpoint file.

        Args:
            checkpoint (str): Path to the checkpoint file.

        Raises:
            ValueError: If the model cannot be loaded from the specified location.
        """

        if hasattr(self.model, "custom_load_state_dict"):
            self.model.custom_load_state_dict(checkpoint)
            logger.info(f"Loading model weights from {checkpoint}.")

        elif hasattr(self.model, "url"):
            path = (
                Path(__file__).resolve().parent.parent / "model_downloads" / self.args.model_name
            )
            load_from = torch.hub.load_state_dict_from_url(self.model.url, model_dir=path)
            self.model.load_state_dict(load_from, strict=True)
            logger.info(path)

        elif Path(checkpoint).is_file():
            logger.info(f"Loading model weights from {checkpoint}.")
            checkpoint_dict = torch.load(
                checkpoint,
                map_location=lambda storage, loc: storage,
            )
            pretrained_dict = checkpoint_dict["state_dict"]
            model_dict = self.model.state_dict()
            model_dict = check_mismatch(model_dict, pretrained_dict)
            self.model.load_state_dict(
                model_dict,
                strict=pretrain,
            )
            logger.info("Pre-trained model loaded successfully")

        else:
            raise ValueError("Cannot load model from specified location")
