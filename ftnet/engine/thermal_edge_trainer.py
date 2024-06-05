#!/usr/bin/env python
import logging
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint

from ftnet.helper.img_saving_helper import save_all_images, save_pred
from ftnet.helper.utils import as_numpy

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

BatchNorm2d = nn.BatchNorm2d


class SegmentationLightningModel(BaseTrainer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def training_step(self, batch: Tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Performs a single training step.

        Args:
            batch (Tuple[torch.Tensor]): The input batch containing images, target masks, and edges.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss value for the batch.
        """
        images, target, edges, _ = batch
        loss_val, edge_map, class_map = self._step(images, target, edges)
        self._update_metrics(
            "train",
            torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3]),
            class_map,
            target,
            loss_val,
        )
        return loss_val

    def on_train_epoch_end(self) -> None:
        """Logs training metrics at the end of an epoch."""

        self._log_epoch_end("train")

    def _step(
        self, images: torch.Tensor, target: torch.Tensor, edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Performs a forward pass and computes the loss.

        Args:
            images (torch.Tensor): The input images.
            target (torch.Tensor): The ground truth segmentation masks.
            edges (torch.Tensor): The edge maps.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            The loss value, edge predictions, class map predictions, target, and edge map.
        """

        output = self.model(images)
        loss_val = self.criterion(output, (target, edges))
        class_map, edge_map = output
        class_map = class_map[0] if isinstance(class_map, (tuple, list)) else class_map
        class_map = torch.argmax(class_map.long(), 1)
        return loss_val, edge_map, class_map

    def validation_step(self, batch: Tuple[torch.Tensor], _) -> torch.Tensor:
        """Performs a single validation step.

        Args:
            batch (Tuple[torch.Tensor]): The input batch containing images, target masks, edges, and filenames.

        Returns:
            torch.Tensor: The validation loss value.
        """
        images, target, edges, filename = batch
        loss_val, edge_map, class_map = self._step(images, target, edges)
        self._update_metrics(
            "val",
            torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3]),
            class_map,
            target,
            loss_val,
        )

        if self.args.checkpoint.save_images:
            save_all_images(
                original=as_numpy(images),
                groundtruth=as_numpy(target),
                prediction=as_numpy(class_map),
                edge_map=as_numpy(edge_map),
                save_dir=self.ckp.get_path("segmented_images/val/"),
                filename=filename,
                current_epoch=self.current_epoch,
                dataset=self.val_dataset,
                save_images_as_subplots=self.args.checkpoint.save_images_as_subplots,
            )

        return loss_val

    def on_validation_epoch_end(self) -> None:
        """Logs validation metrics at the end of an epoch."""

        self._log_epoch_end("val")

    def test_step(self, batch):
        if self.args.task.mode == "test":
            images, target, edges, filename = batch
            loss_val, edge_map, class_map = self._step(images, target, edges)
            self._update_metrics(
                "test",
                torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3]),
                class_map,
                target,
                loss_val,
            )
            class_map = self._upsample_output(class_map, target)
            edge_map = self._upsample_output(edge_map, target)

            if self.args.checkpoint.save_images:
                save_all_images(
                    original=as_numpy(images),
                    groundtruth=as_numpy(target),
                    prediction=as_numpy(class_map),
                    edge_map=as_numpy(edge_map.squeeze()),
                    save_dir=self.ckp.get_path("segmented_images/test/"),
                    filename=filename,
                    current_epoch="",
                    dataset=self.val_dataset,
                    save_images_as_subplots=self.args.checkpoint.save_images_as_subplots,
                )

        elif self.args.task.mode == "infer":
            images, filename = batch
            output = self.model(images)
            class_map, edge_map = output
            class_map = class_map[0] if isinstance(class_map, (tuple, list)) else class_map
            class_map = torch.argmax(class_map.long(), 1)
            save_pred(
                save_dir=self.ckp.get_path("infer"),
                filename=filename,
                prediction=as_numpy(class_map.squeeze()),
                edges=as_numpy(edge_map.squeeze()),
                dataset=self.args.dataset.name,
            )
        return

    def on_test_epoch_end(self) -> None:
        if self.args.task.mode == "test":
            self._log_epoch_end("test")

    @staticmethod
    def add_callback(
        ckp: Callable,
        train_only: bool,
        save_weights_only: bool = True,
    ) -> Union[ModelCheckpoint, List[ModelCheckpoint]]:
        """Adds a checkpoint callback to save model checkpoints.

        Args:
            ckp: The checkpoint object.
            train_only: Boolean flag indicating if only training checkpoints are needed.
            save_weights_only: this is useful when the code is refactored.
        Returns:
            Union[ModelCheckpoint, List[ModelCheckpoint]]: The configured ModelCheckpoint callback(s).
        """
        checkpoints = []
        path = ckp.get_path("ckpt")

        filename_template = "{epoch}"
        checkpoints.append(
            ModelCheckpoint(
                dirpath=path,
                filename=filename_template,
                verbose=True,
                every_n_epochs=1,
                save_last=True,
                save_weights_only=False,
            )
        )

        if train_only:
            return checkpoints

        monitor_metrics = [
            ("val_avg_mIOU_Acc", "{epoch}-{val_avg_mIOU_Acc:.4f}"),
            ("val_mIOU", "{epoch}-{val_mIOU:.4f}"),
            ("val_mAcc", "{epoch}-{val_mAcc:.4f}"),
        ]

        for metric, filename_template in monitor_metrics:
            checkpoints.append(
                ModelCheckpoint(
                    dirpath=path,
                    filename=filename_template,
                    save_top_k=3,
                    verbose=True,
                    every_n_epochs=1,
                    monitor=metric,
                    mode="max",
                    save_last=False,
                    save_weights_only=save_weights_only,
                )
            )

        return checkpoints
