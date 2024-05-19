#!/usr/bin/env python
import logging
from argparse import Namespace
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, List

import torch
import torch.nn as nn
from core.data.samplers import (
    make_batch_data_sampler,
    make_data_sampler,
    make_multiscale_batch_data_sampler,
)
from core.loss import get_segmentation_loss
from helper.optimizer_scheduler_helper import make_optimizer, make_scheduler
from helper.utils import save_model_summary
from lightning.pytorch import LightningModule
from models import get_segmentation_model

from data import get_segmentation_dataset

logger = logging.getLogger(__name__)


class BaseTrainer(LightningModule):
    def __init__(
        self,
        args: Namespace,
        ckp: Callable,
        train: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ckp = ckp
        self.args = args
        self.seg_dir = Path(self.ckp.get_path("Segmented_images"))  # type: ignore

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

        if self.args.trainer.pretrain_checkpoint:
            self.load_weights_from_checkpoint(self.args.trainer.pretrain_checkpoint)

        save_model_summary(self.model, Path(self.ckp.get_path("logs")))  # type: ignore
        self._preload_complete_data()
        if train:
            self.criterion = get_segmentation_loss(
                self.args.model.name,
                loss_weight=self.args.trainer.loss_weight,
                ignore_index=self.train_dataset.IGNORE_INDEX,
            )

    def _preload_complete_data(self):
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
            split="test" if self.args.dataset.name == "scutseg" else "val",
            mode="val",
            sobel_edges=False,
            **data_kwargs,
        )

    def configure_optimizers(self):
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

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError

    def on_train_epoch_end(self, outputs: List[Any]) -> None:
        raise NotImplementedError

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError

    def on_validation_epoch_end(self, outputs: List[Any]) -> None:
        raise NotImplementedError

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        raise NotImplementedError

    def on_test_epoch_end(self, outputs: List[Any]) -> None:
        raise NotImplementedError

    def accuracy_(self, confusion_matrix: torch.Tensor) -> torch.Tensor:
        acc = confusion_matrix.diag() / confusion_matrix.sum(1)
        acc[torch.isnan(acc)] = 0
        return acc

    def load_weights_from_checkpoint(self, checkpoint: str) -> None:
        def check_mismatch(model_dict: dict, pretrained_dict: dict) -> dict:
            pretrained_dict = {key[6:]: item for key, item in pretrained_dict.items()}
            temp_dict = OrderedDict()
            for k, v in pretrained_dict.items():
                if k in model_dict:
                    if model_dict[k].shape != pretrained_dict[k].shape:
                        logger.info(
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
                strict=bool(self.args.pretrain_checkpoint),
            )
            logger.info("Pre-trained model loaded successfully")

        else:
            raise ValueError("Cannot load model from specified location")
