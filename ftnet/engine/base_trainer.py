#!/usr/bin/env python
import inspect
import logging

# matplotlib.use('Agg')
import math
import os
from argparse import Namespace
from collections import OrderedDict
from typing import Callable
import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn

from core.data.samplers import (
    make_batch_data_sampler,
    make_data_sampler,
    make_multiscale_batch_data_sampler,
)
from core.loss import get_segmentation_loss
from models import get_segmentation_model
from utils.optimizer_scheduler_helper import make_optimizer, make_scheduler
from utils.utils import save_model_summary
from utils.visualize import get_color_palette
from data import get_segmentation_dataset

# https://github.com/hszhao/semseg/blob/master/tool/train.py


class BaseTrainer(pl.LightningModule):
    def __init__(
        self,
        args: Namespace,
        ckp: Callable,
        train: bool = True,
        logger: logging.Logger = None,
        **kwargs,
    ):
        super(BaseTrainer, self).__init__(**kwargs)
        self.custom_logger = logger
        self.save_hyperparameters()
        self.ckp = ckp

        self.seg_dir = self.ckp.get_path("Segmented_images")  # type: ignore

        self.model = get_segmentation_model(
            model_name=self.hparams.args.model.model,
            dataset=self.hparams.args.dataset.dataset,
            backbone=self.hparams.args.model.backbone,
            norm_layer=nn.BatchNorm2d,
            dilated=self.hparams.args.model.dilation,
            no_of_filters=self.hparams.args.model.no_of_filters,
            pretrained_base=self.hparams.args.model.pretrained_base,
            edge_extracts=self.hparams.args.model.edge_extracts,
            num_blocks=self.hparams.args.model.num_blocks,
        )

        if self.hparams.args.pretrain_checkpoint is not None:
            self.load_weights_from_checkpoint(self.hparams.args.pretrain_checkpoint)

        save_model_summary(self.model, self.ckp.get_path("logs"))  # type: ignore

        self._preload_complete_data()

        if train:
            self.criterion = get_segmentation_loss(
                self.hparams.args.model,
                loss_weight=self.hparams.args.loss_weight,
                ignore_index=self.train_dataset.IGNORE_INDEX,
                logger=self.custom_logger,
            )

    def _preload_complete_data(self):
        data_kwargs = {
            "logger": self.custom_logger,
            "root": self.hparams.args.dataset_path,
            "base_size": self.hparams.args.base_size,
            "crop_size": self.hparams.args.crop_size,
        }

        self.train_dataset = get_segmentation_dataset(
            name=self.hparams.args.dataset,
            split="train",
            mode="train",
            sobel_edges=False,
            **data_kwargs,
        )

        self.val_dataset = get_segmentation_dataset(
            self.hparams.args.dataset,
            split="test"
            if (self.hparams.args.dataset == "soda")
            or (self.hparams.args.dataset == "scutseg")
            else "val",
            mode="val",
            sobel_edges=False,
            **data_kwargs,
        )

    def configure_optimizers(self):
        self.custom_logger.info("Setting Up Optimizer")

        params = list()
        if hasattr(self.model, "encoder"):
            params.append(
                {"params": self.model.encoder.parameters(), "lr": self.hparams.args.lr}
            )
        if hasattr(self.model, "exclusive"):
            for module in self.model.exclusive:
                params.append(
                    {
                        "params": getattr(self.model, module).parameters(),
                        "lr": self.hparams.args.lr * 10,
                    }
                )

        if len(params) == 0:
            params = filter(lambda x: x.requires_grad, self.model.parameters())

        optimizer = make_optimizer(
            args=self.hparams.args, params=params, logger=self.custom_logger
        )

        self.custom_logger.info("Setting Up Scheduler")

        if self.trainer.use_ddp:
            processes = self.hparams.args.gpus * self.hparams.args.num_nodes
        elif self.trainer.use_ddp2:
            processes = self.hparams.args.num_nodes
        else:
            processes = 1
        iters_per_epoch = (
            math.ceil(
                len(self.train_dataset) / self.hparams.args.train_batch_size / processes
            )
            // self.trainer.accumulate_grad_batches
        )
        self.custom_logger.info(
            f"Iterations per epoch computed for scheduler is {iters_per_epoch}"
        )

        scheduler = make_scheduler(
            args=self.hparams.args,
            optimizer=optimizer,
            iters_per_epoch=iters_per_epoch,
            last_epoch=-1,
            logger=self.custom_logger,
        )

        return [optimizer], [
            {"scheduler": scheduler, "interval": scheduler.__interval__}
        ]

    def train_dataloader(self):
        train_sampler = make_data_sampler(
            dataset=self.train_dataset,
            shuffle=True,
            distributed=(self.trainer.use_ddp or self.trainer.use_ddp2),
        )

        train_batch_sampler = make_multiscale_batch_data_sampler(
            sampler=train_sampler,
            batch_size=self.hparams.args.train_batch_size,
            multiscale_step=1,
            scales=len(self.hparams.args.crop_size),
        )

        train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=self.hparams.args.workers,
            pin_memory=True,
        )

        self.load_metrics(
            mode="train",
            num_class=self.train_dataset.NUM_CLASS,
            ignore_index=self.train_dataset.IGNORE_INDEX,
        )

        return train_loader

    def val_dataloader(self):
        val_sampler = make_data_sampler(
            dataset=self.val_dataset,
            shuffle=False,
            distributed=(self.trainer.use_ddp or self.trainer.use_ddp2),
        )

        val_batch_sampler = make_batch_data_sampler(
            val_sampler, batch_size=self.hparams.args.val_batch_size
        )
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset,
            batch_sampler=val_batch_sampler,
            num_workers=self.hparams.args.workers,
            pin_memory=True,
        )

        self.load_metrics(
            mode="val",
            num_class=self.val_dataset.NUM_CLASS,
            ignore_index=self.val_dataset.IGNORE_INDEX,
        )
        return val_loader

    def test_dataloader(self):
        data_kwargs = {
            "logger": self.custom_logger,
            "root": self.hparams.args.dataset_path,
            "base_size": None,
        }

        self.test_dataset = get_segmentation_dataset(
            self.hparams.args.dataset, split="test", mode="testval", **data_kwargs
        )

        test_sampler = make_data_sampler(
            dataset=self.test_dataset,
            shuffle=False,
            distributed=(self.trainer.use_ddp or self.trainer.use_ddp2),
        )

        test_batch_sampler = make_batch_data_sampler(
            test_sampler, batch_size=self.hparams.args.test_batch_size
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_sampler=test_batch_sampler,
            num_workers=self.hparams.args.workers,
            pin_memory=True,
        )

        self.load_metrics(
            mode="test",
            num_class=self.test_dataset.NUM_CLASS,
            ignore_index=self.test_dataset.IGNORE_INDEX,
        )
        return test_loader

    def load_metrics(self, mode, num_class, ignore_index):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def training_epoch_end(self, outputs):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_epoch_end(self, outputs):
        raise NotImplementedError

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        raise NotImplementedError

    def accuracy_(self, confusion_matrix):
        acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        acc[torch.isnan(acc)] = 0
        return acc

    def plot_confusion_matrix(self, confusion_matrix):
        # https://github.com/reiinakano/scikit-plot/blob/2dd3e6a76df77edcbd724c4db25575f70abb57cb/scikitplot/metrics.py#L33
        cmn = confusion_matrix / confusion_matrix.sum(1)[:, None]
        cmn[np.isnan(cmn)] = 0

        name = self.trainer.test_name if hasattr(self.trainer, "test_name") else "Final"

        fig, ax = plt.subplots(figsize=(15, 15))
        sns.heatmap(
            cmn,
            annot=True,
            fmt=".1f",
            xticklabels=self.test_dataset.class_names,
            yticklabels=self.test_dataset.class_names,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        fig.savefig(
            self.hparams.args.save_dir + "/{}_{}.png".format(name, "confusion_matrix"),
            bbox_inches="tight",
        )
        plt.close()

        # pd.DataFrame(
        #     confusion_matrix,
        #     columns=self.test_dataset.class_names,
        #     index=self.test_dataset.class_names,
        # ).to_csv(
        #     self.hparams.args.save_dir + "/{}_{}.csv".format(name, "confusion_matrix")
        # )

    def save_images(self, original, groundtruth, prediction, filename):
        calframe = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
        if calframe == "validation_step":
            base_path = os.path.join(self.seg_dir, str(self.current_epoch))
            std = self.val_dataset.std
            mean = self.val_dataset.mean
        elif calframe == "test_step":
            base_path = self.seg_dir
            std = self.test_dataset.std
            mean = self.test_dataset.mean
        else:
            ValueError("Standard Deviation and Mean not found")

        original_img = np.clip(
            np.moveaxis(original, 1, 3) * std + mean, a_min=0, a_max=1
        )

        if not os.path.exists(base_path):
            os.makedirs(base_path, mode=0o770, exist_ok=True)

        if self.hparams.args.save_images_as_subplots:
            for i in range(original_img.shape[0]):
                fig = plt.figure(figsize=(8.5, 11))
                plt.subplot(1, 3, 1)
                plt.imshow(original_img[i])
                plt.subplot(1, 3, 2)
                plt.imshow(
                    np.array(
                        get_color_palette(groundtruth[i], self.hparams.args.dataset)
                    )
                )
                plt.subplot(1, 3, 3)
                plt.imshow(
                    np.array(
                        get_color_palette(prediction[i], self.hparams.args.dataset)
                    )
                )
                plt.axis("off")
                fig.savefig(
                    base_path + f"/{os.path.splitext(filename[i])[0]}.png",
                    bbox_inches="tight",
                )
                plt.close()
        else:
            for i in range(original_img.shape[0]):
                # plt.imsave(
                #     base_path + '/Img_{}.png'.format(os.path.splitext(filename[i])[0]), original_img[i])
                # plt.imsave(base_path + '/GD_{}.png'.format(os.path.splitext(filename[i])[0]),
                #            np.array(get_color_palette(groundtruth[i], self.hparams.args.dataset)))
                plt.imsave(
                    base_path + f"/Pred_{os.path.splitext(filename[i])[0]}.png",
                    np.array(
                        get_color_palette(prediction[i], self.hparams.args.dataset)
                    ),
                )

    def save_edge_images(self, original, groundtruth, prediction, edge_map, filename):
        calframe = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
        if calframe == "validation_step":
            base_path = os.path.join(self.seg_dir, str(self.current_epoch))
            std = self.val_dataset.std
            mean = self.val_dataset.mean

        elif calframe == "test_step":
            base_path = self.seg_dir
            std = self.test_dataset.std
            mean = self.test_dataset.mean
        else:
            ValueError("Standard Deviation and Mean not found")

        # original_img = np.clip(np.moveaxis(original, 1, 3), a_min=0, a_max=1)
        original_img = np.clip(
            np.moveaxis(original, 1, 3) * std + mean, a_min=0, a_max=1
        )

        if not os.path.exists(base_path):
            os.makedirs(base_path, mode=0o770, exist_ok=True)

        if self.hparams.args.save_images_as_subplots:
            for i in range(original_img.shape[0]):
                fig = plt.figure(figsize=(8.5, 11))
                plt.subplot(1, 4, 1)
                plt.imshow(original_img[i])
                plt.subplot(1, 4, 2)
                plt.imshow(
                    np.array(
                        get_color_palette(groundtruth[i], self.hparams.args.dataset)
                    )
                )
                plt.subplot(1, 4, 3)
                plt.imshow(
                    np.array(
                        get_color_palette(prediction[i], self.hparams.args.dataset)
                    )
                )
                plt.subplot(1, 4, 4)
                plt.imshow(np.array(edge_map[i][0]))
                plt.axis("off")
                fig.savefig(
                    base_path + f"/{os.path.splitext(filename[i])[0]}.png",
                    bbox_inches="tight",
                )
                plt.close()
        else:
            for i in range(original_img.shape[0]):
                # plt.imsave(
                #     base_path + '/Img_{}.png'.format(os.path.splitext(filename[i])[0]), original_img[i])
                # plt.imsave(base_path + '/GD_{}.png'.format(os.path.splitext(filename[i])[0]),
                #            np.array(get_color_palette(groundtruth[i], self.hparams.args.dataset)))
                plt.imsave(
                    base_path + f"/Pred_{os.path.splitext(filename[i])[0]}.png",
                    np.array(
                        get_color_palette(prediction[i], self.hparams.args.dataset)
                    ),
                )
                plt.imsave(
                    base_path + f"/Edges_{os.path.splitext(filename[i])[0]}.png",
                    np.array(edge_map[i][0]),
                )

    def load_weights_from_checkpoint(self, checkpoint: str) -> None:
        def check_mismatch(model_dict, pretrained_dict) -> None:
            pretrained_dict = {key[6:]: item for key, item in pretrained_dict.items()}
            temp_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if k in model_dict.keys():
                    if model_dict[k].shape != pretrained_dict[k].shape:
                        self.custom_logger.info(
                            f"Skip loading parameter: {k}, "
                            f"required shape: {model_dict[k].shape}, "
                            f"loaded shape: {pretrained_dict[k].shape}"
                        )
                        continue
                    else:
                        temp_dict[k] = v
            return temp_dict

        if hasattr(self.model, "custom_load_state_dict"):
            self.model.custom_load_state_dict(checkpoint)
            self.custom_logger.info(f"loading model weights from {checkpoint}.")

        elif hasattr(self.model, "url"):
            path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "model_downloads",
                    self.hparams.args.model_name,
                )
            )
            load_from = torch.utils.model_zoo.load_url(self.model.url, model_dir=path)
            self.model.load_state_dict(load_from, strict=True)
            self.custom_logger.info(path)

        elif os.path.isfile(checkpoint):
            self.custom_logger.info(f"loading model weights from {checkpoint}.")
            checkpoint = torch.load(
                checkpoint,
                map_location=lambda storage, loc: storage,
            )
            pretrained_dict = checkpoint["state_dict"]
            model_dict = self.model.state_dict()
            model_dict = check_mismatch(model_dict, pretrained_dict)
            self.model.load_state_dict(
                model_dict,
                strict=False
                if self.hparams.args.pretrain_checkpoint is not None
                else True,
            )
            self.custom_logger.info("Pre trained model loaded successfully")

        else:
            ValueError("Cannot load model from specified location")
