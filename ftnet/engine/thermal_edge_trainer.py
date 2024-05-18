#!/usr/bin/env python
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_trainer import BaseTrainer
from core.metrics import AverageMeter
from utils.json_extension import save_to_json_pretty
from utils.utils import as_numpy
from torchmetrics import ConfusionMatrix as pl_ConfusionMatrix
from torchmetrics.classification import MulticlassJaccardIndex
from lightning.pytorch.callbacks import ModelCheckpoint


BatchNorm2d = nn.BatchNorm2d


# https://github.com/hszhao/semseg/blob/master/tool/train.py


class SegmentationLightningModel(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(SegmentationLightningModel, self).__init__(*args, **kwargs)

    def load_metrics(self, mode, num_class, ignore_index):
        setattr(
            self,
            f"{mode}_confmat",
            pl_ConfusionMatrix(num_classes=num_class).to(self.device),
        )
        setattr(
            self,
            f"{mode}_IOU",
            MulticlassJaccardIndex(
                num_classes=num_class, ignore_index=ignore_index, reduction="none"
            ).to(self.device),
        )
        setattr(self, f"{mode}_edge_accuracy", AverageMeter().to(self.device))

    def training_step(self, batch, batch_idx):
        images, target, edges, _ = batch
        output = self.forward(images)
        loss_val = self.criterion(output, (target, edges))

        class_map, edge_map = output
        class_map = (
            class_map[0]
            if isinstance(class_map, tuple) or isinstance(class_map, list)
            else class_map
        )
        class_map = torch.argmax(class_map.long(), 1)
        edge_pred = torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3])

        self.train_edge_accuracy.update(edge_pred)
        self.train_confmat.update(preds=class_map, target=target)
        self.train_IOU.update(preds=class_map, target=target)

        log_dict = OrderedDict({"loss": loss_val})
        self.log("loss", loss_val)

        return log_dict

    def on_train_epoch_end(self, outputs):
        confusion_matrix = self.train_confmat.compute()
        iou = self.train_IOU.compute()
        accuracy = self.accuracy_(confusion_matrix)

        log_dict = {
            "train_mIOU": torch.mean(iou),
            "train_mAcc": torch.mean(accuracy),
            "train_avg_mIOU_Acc": (torch.mean(iou) + torch.mean(accuracy)) / 2,
            "train_edge_accuracy": self.train_edge_accuracy.compute(),
        }

        log_dict["loss"] = torch.stack([output["loss"] for output in outputs]).mean()

        self.log_dict(log_dict, prog_bar=True)
        self.train_confmat.reset()
        self.train_IOU.reset()
        self.train_edge_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        images, target, edges, filename = batch
        output = self.forward(images)
        loss_val = self.criterion(output, (target, edges))

        class_map, edge_map = output
        class_map = (
            class_map[0]
            if isinstance(class_map, tuple) or isinstance(class_map, list)
            else class_map
        )
        class_map = torch.argmax(class_map.long(), 1)
        edge_pred = torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3])

        self.val_edge_accuracy.update(edge_pred)
        self.val_confmat.update(preds=class_map, target=target)
        self.val_IOU.update(preds=class_map, target=target)

        if self.hparams.args.save_images:
            image_dict = {
                "original": as_numpy(images),
                "groundtruth": as_numpy(target),
                "prediction": as_numpy(class_map),
                "edge_map": as_numpy(edge_map),
                "filename": filename,
            }

            self.save_edge_images(**image_dict)

        log_dict = OrderedDict({"val_loss": loss_val})

        return log_dict

    def on_validation_epoch_end(self, outputs):
        confusion_matrix = self.val_confmat.compute()
        iou = self.val_IOU.compute()
        accuracy = self.accuracy_(confusion_matrix)

        log_dict = {
            "val_mIOU": torch.mean(iou),
            "val_mAcc": torch.mean(accuracy),
            "val_avg_mIOU_Acc": (torch.mean(iou) + torch.mean(accuracy)) / 2,
            "val_edge_accuracy": self.val_edge_accuracy.compute(),
        }

        log_dict["val_loss"] = torch.stack(
            [output["val_loss"] for output in outputs]
        ).mean()
        self.log_dict(log_dict)

        self.val_edge_accuracy.reset()
        self.val_confmat.reset()
        self.val_IOU.reset()

    def test_step(self, batch, batch_idx):
        def upsample_output(output, target):
            if output.shape != target.shape:
                return F.interpolate(
                    output,
                    size=(target.shape[1], target.shape[2]),
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                output

        self.seg_dir = self.ckp.get_path(
            f"Segmented_images/test/{self.trainer.test_name}"
            if hasattr(self.trainer, "test_name")
            else "Segmented_images/test/"
        )

        images, target, edges, filename = batch
        output = self.forward(images)

        class_map, edge_map = output
        class_map = (
            class_map[0]
            if isinstance(class_map, tuple) or isinstance(class_map, list)
            else class_map
        )
        class_map = upsample_output(class_map, target)
        edge_map = upsample_output(edge_map, target)

        class_map = torch.argmax(class_map.long(), 1)
        edge_pred = torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3])

        self.test_confmat.update(preds=class_map, target=target)
        self.test_IOU.update(preds=class_map, target=target)
        self.test_edge_accuracy.update(edge_pred)

        if self.hparams.args.save_images:
            image_dict = {
                "original": as_numpy(images),
                "groundtruth": as_numpy(target),
                "prediction": as_numpy(class_map),
                "edge_map": as_numpy(edge_map),
                "filename": filename,
            }

            self.save_edge_images(**image_dict)

        return

    def on_test_epoch_end(self, outputs):
        # https://stackoverflow.com/questions/20927368/how-to-normalize-a-confusion-matrix

        def accuracy_(confusion_matrix):
            acc = confusion_matrix.diag() / confusion_matrix.sum(1)
            acc[torch.isnan(acc)] = 0
            return acc

        confusion_matrix = self.test_confmat.compute()
        iou = self.test_IOU.compute()
        accuracy = accuracy_(confusion_matrix)

        self.plot_confusion_matrix(confusion_matrix.cpu().numpy())

        log_dict = {
            "test_mIOU": torch.mean(iou),
            "test_mAcc": torch.mean(accuracy),
            "test_avg_mIOU_Acc": (torch.mean(iou) + torch.mean(accuracy)) / 2,
            "test_edge_accuracy": self.test_edge_accuracy.compute(),
        }

        per_class_IOU = iou.cpu().numpy() * 100
        save_to_json_pretty(log_dict, path=os.path.join(self.seg_dir, "Average.txt"))
        np.savetxt(os.path.join(self.seg_dir, "per_class_iou.txt"), per_class_IOU)

        self.log_dict(log_dict)
        self.test_confmat.reset()
        self.test_IOU.reset()
        self.test_edge_accuracy.reset()

    @staticmethod
    def _add_callback(ckp):
        return [
            ModelCheckpoint(
                dirpath=ckp.get_path("ckpt"),
                filename="{epoch}-{val_avg_mIOU_Acc:.4f}",
                save_top_k=3,
                verbose=True,
                every_n_epochs=1,
                monitor="val_avg_mIOU_Acc",
                mode="max",
                save_last=True,
            ),
            ModelCheckpoint(
                dirpath=ckp.get_path("ckpt"),
                filename="{epoch}-{val_mIOU:.4f}",
                save_top_k=3,
                verbose=True,
                monitor="val_mIOU",
                mode="max",
                every_n_epochs=1,
            ),
            ModelCheckpoint(
                dirpath=ckp.get_path("ckpt"),
                filename="{epoch}-{val_mAcc:.4f}",
                save_top_k=3,
                verbose=True,
                monitor="val_mAcc",
                mode="max",
                every_n_epochs=1,
            ),
        ]
