#!/usr/bin/env python
# https://github.com/PyTorchLightning/pytorch-lightning/discussions/6182
# https://github.com/zhutmost/neuralzip/blob/master/apputil/progressbar.py
import inspect
import logging
import math
# matplotlib.use('Agg')
import os
from argparse import Namespace
from collections import OrderedDict
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from core.data.samplers import (make_batch_data_sampler, make_data_sampler,
                                make_multiscale_batch_data_sampler)
from core.loss import get_segmentation_loss
from torchmetrics import ConfusionMatrix as pl_ConfusionMatrix
from core.models import get_segmentation_model
from core.utils import collect_env_info, plot_tensors, setup_logger
from core.utils.filesystem import checkpoint
from core.utils.optimizer_scheduler_helper import (make_optimizer,
                                                   make_scheduler)
from core.utils.utils import as_numpy, save_model_summary, to_python_float
from core.utils.visualize import get_color_pallete
from datasets import get_segmentation_dataset
from options import parse_args
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TestTubeLogger
from core.metrics import pl_IOU
from core.metrics import AverageMeter


BatchNorm2d = nn.BatchNorm2d

# https://github.com/hszhao/semseg/blob/master/tool/train.py


class SegmentationTrainLightningModel(LightningModule):
    def __init__(self, args: Namespace,
                 ckp: Callable,
                 train: bool = True,
                 name: str = None,
                 logger: logging.Logger = None,
                 ** kwargs):

        super(SegmentationTrainLightningModel, self).__init__(**kwargs)
        self.custom_logger = logger
        self.save_hyperparameters()
        self.ckp = ckp

        self.model = get_segmentation_model(model_name=self.hparams.args.model,
                                            dataset=self.hparams.args.dataset,
                                            backbone=self.hparams.args.backbone,
                                            norm_layer=BatchNorm2d,
                                            no_of_filters=self.hparams.args.no_of_filters,
                                            pretrained_base=self.hparams.args.pretrained_base,
                                            edge_extracts=self.hparams.args.edge_extracts,
                                            num_blocks=self.hparams.args.num_blocks)

        save_model_summary(self.model, self.ckp.get_path('logs/'))

        self._preload_data()
        self.criterion = get_segmentation_loss(self.hparams.args.model,
                                               loss_weight=self.hparams.args.loss_weight,
                                               ignore_index=self.train_dataset.IGNORE_INDEX,
                                               logger=self.custom_logger)

    def _preload_data(self):
        data_kwargs = {'logger': self.custom_logger,
                       'root': self.hparams.args.dataset_path,
                       'base_size': self.hparams.args.base_size,
                       'crop_size': self.hparams.args.crop_size}

        self.train_dataset = get_segmentation_dataset(name=self.hparams.args.dataset,
                                                      split='train',
                                                      mode='train',
                                                      sobel_edges=True if self.hparams.args.model == 'mcnet' else False,
                                                      ** data_kwargs)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.custom_logger.info('Setting Up Optimizer')

        params = list()
        if hasattr(self.model, 'encoder'):
            params.append({'params': self.model.encoder.parameters(), 'lr': self.hparams.args.lr})
        if hasattr(self.model, 'exclusive'):
            for module in self.model.exclusive:
                params.append({'params': getattr(self.model, module).parameters(), 'lr': self.hparams.args.lr * 10})

        if len(params) == 0:
            params = filter(lambda x: x.requires_grad, self.model.parameters())

        optimizer = make_optimizer(args=self.hparams.args,
                                   params=params,
                                   logger=self.custom_logger)

        self.custom_logger.info('Setting Up Scheduler')

        if self.trainer.use_ddp:
            processes = self.hparams.args.gpus * self.hparams.args.num_nodes
        elif self.trainer.use_ddp2:
            processes = self.hparams.args.num_nodes
        else:
            processes = 1
        iters_per_epoch = math.ceil(len(self.train_dataset) / self.hparams.args.train_batch_size /
                                    processes) // self.trainer.accumulate_grad_batches
        self.custom_logger.info('Iterations per epoch computed for scheduler is {}'.format(iters_per_epoch))

        scheduler = make_scheduler(args=self.hparams.args,
                                   optimizer=optimizer,
                                   iters_per_epoch=iters_per_epoch,
                                   last_epoch=-1,
                                   logger=self.custom_logger)

        return [optimizer], [{'scheduler': scheduler, 'interval': scheduler.__interval__}]

    def load_metrics(self, mode, num_class, ignore_index):
        setattr(self, f'{mode}_IOU', pl_IOU(num_classes=num_class, ignore_index=ignore_index).to(self.device))
        setattr(self, f'{mode}_edge_accuracy', AverageMeter().to(self.device))

    def train_dataloader(self):
        train_sampler = make_data_sampler(dataset=self.train_dataset,
                                          shuffle=True,
                                          distributed=(self.trainer.use_ddp or self.trainer.use_ddp2))

        train_batch_sampler = make_multiscale_batch_data_sampler(sampler=train_sampler,
                                                                 batch_size=self.hparams.args.train_batch_size,
                                                                 multiscale_step=1,
                                                                 scales=len(self.hparams.args.crop_size))

        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_sampler=train_batch_sampler,
                                                   num_workers=self.hparams.args.workers,
                                                   pin_memory=True)

        self.load_metrics(mode='train', num_class=self.train_dataset.NUM_CLASS, ignore_index=self.train_dataset.IGNORE_INDEX)
        return train_loader

    def training_step(self, batch, batch_idx):
        images, target, edges, _ = batch
        output = self.forward(images)
        loss_val = self.criterion(output, (target, edges))

        class_map, edge_map = output
        class_map = class_map[0] if isinstance(class_map, tuple) or isinstance(class_map, list) else class_map

        edge_pred = torch.mean(((edge_map > 0) == edges).float(), dim=[1, 2, 3])

        self.train_edge_accuracy.update(edge_pred)
        self.train_IOU.update(preds=class_map, target=target)

        log_dict = OrderedDict({'loss': loss_val})
        self.log('loss', loss_val)

        return log_dict

    def training_epoch_end(self, outputs):
        mean_iou, mean_accuracy, all_accuracy = self.train_IOU.compute_mean()

        log_dict = {"train_mIOU": mean_iou,
                    "train_mAcc": mean_accuracy,
                    "train_avg_mIOU_Acc": (mean_iou + mean_accuracy) / 2,
                    "train_edge_accuracy": self.train_edge_accuracy.compute()}

        log_dict["loss"] = torch.stack([output["loss"] for output in outputs]).mean()

        self.log_dict(log_dict)
        self.train_IOU.reset()
        self.train_edge_accuracy.reset()
