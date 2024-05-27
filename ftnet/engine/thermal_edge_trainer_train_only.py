#!/usr/bin/env python
import logging
from typing import Any, Tuple

import torch
import torch.nn as nn
from helper.utils import as_numpy
from lightning.pytorch.callbacks import ModelCheckpoint

from ftnet.helper.img_saving_helper import save_images

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
        loss_val, edge_pred, class_map, target, _ = self._step(images, target, edges)
        self._update_metrics("train", edge_pred, class_map, target, loss_val)
        return loss_val

    def on_train_epoch_end(self) -> None:
        """Logs training metrics at the end of an epoch."""

        self._log_epoch_end("train")

    def _step(
        self, images: torch.Tensor, target: torch.Tensor, edges: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        edge_pred = torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3])
        return loss_val, edge_pred, class_map, target, edge_map

    def validation_step(self, batch: Tuple[torch.Tensor], _) -> torch.Tensor:
        """Performs a single validation step.

        Args:
            batch (Tuple[torch.Tensor]): The input batch containing images, target masks, edges, and filenames.

        Returns:
            torch.Tensor: The validation loss value.
        """
        images, target, edges, filename = batch
        loss_val, edge_pred, class_map, target, edge_map = self._step(images, target, edges)
        self._update_metrics("val", edge_pred, class_map, target, loss_val)

        if self.args.checkpoint.save_images:
            save_images(
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

    # def test_step(self, batch, batch_idx):
    #     images, target, edges, filename = batch
    #     output = self.forward(images)

    #     class_map, edge_map = output
    #     class_map = class_map[0] if isinstance(class_map, (tuple, list)) else class_map
    #     class_map = self._upsample_output(class_map, target)
    #     edge_map = self._upsample_output(edge_map, target)

    #     class_map = torch.argmax(class_map.long(), 1)
    #     edge_pred = torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3])

    #     self.test_confmat.update(preds=class_map, target=target)
    #     self.test_IOU.update(preds=class_map, target=target)
    #     self.test_edge_accuracy.update(edge_pred)

    #     if self.hparams.args.save_images:
    #         image_dict = {
    #             "original": as_numpy(images),
    #             "groundtruth": as_numpy(target),
    #             "prediction": as_numpy(class_map),
    #             "edge_map": as_numpy(edge_map),
    #             "filename": filename,
    #         }

    #         self.save_edge_images(**image_dict)

    #     return

    # def on_test_epoch_end(self, outputs):
    #     def accuracy_(confusion_matrix):
    #         acc = confusion_matrix.diag() / confusion_matrix.sum(1)
    #         acc[torch.isnan(acc)] = 0
    #         return acc

    #     confusion_matrix = self.test_confmat.compute()
    #     iou = self.test_IOU.compute()
    #     accuracy = accuracy_(confusion_matrix)

    #     self.plot_confusion_matrix(confusion_matrix.cpu().numpy())

    #     log_dict = {
    #         "test_mIOU": torch.mean(iou),
    #         "test_mAcc": torch.mean(accuracy),
    #         "test_avg_mIOU_Acc": (torch.mean(iou) + torch.mean(accuracy)) / 2,
    #         "test_edge_accuracy": self.test_edge_accuracy.compute(),
    #     }

    #     per_class_IOU = iou.cpu().numpy() * 100
    #     save_to_json_pretty(log_dict, path=os.path.join(self.seg_dir, "Average.txt"))
    #     np.savetxt(os.path.join(self.seg_dir, "per_class_iou.txt"), per_class_IOU)

    #     self.log_dict(log_dict)
    #     self.test_confmat.reset()
    #     self.test_IOU.reset()
    #     self.test_edge_accuracy.reset()

    @staticmethod
    def add_callback(ckp) -> ModelCheckpoint:
        """Adds a checkpoint callback to save model checkpoints.

        Args:
            ckp: The checkpoint object.

        Returns:
            ModelCheckpoint: The configured ModelCheckpoint callback.
        """
        return ModelCheckpoint(
            dirpath=ckp.get_path("ckpt"),
            filename="{epoch}",
            verbose=True,
            every_n_epochs=1,
            save_last=True,
        )
