import logging
import warnings
from argparse import Namespace
from typing import Any, List, Union

import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, StepLR, MultiStepLR, OneCycleLR

# Assuming these are defined elsewhere in the codebase
from core.optimizers.adabound import AdaBound
from core.schedulers.lr_scheduler import WarmupMultiStepLR, WarmupPolyLR

logger = logging.getLogger(__name__)

__all__ = ["make_optimizer", "make_scheduler"]


def make_optimizer(args: Namespace, params: Union[List[dict], Any]) -> Optimizer:
    optimizer_name = args.optimizer.name.lower()
    kwargs = {}

    if optimizer_name == "sgd":
        optimizer_function = optim.SGD
        kwargs |= {
            "momentum": args.optimizer.momentum,
            "nesterov": args.optimizer.nesterov,
        }
        logger.info("Optimizer SGD is being used")
    elif optimizer_name == "adam":
        optimizer_function = optim.Adam
        kwargs |= {
            "betas": (args.optimizer.beta1, args.optimizer.beta2),
            "eps": args.optimizer.epsilon,
        }
        logger.info("Optimizer ADAM is being used")
    elif optimizer_name == "rmsprop":
        optimizer_function = optim.RMSprop
        kwargs |= {"eps": args.optimizer.epsilon}
        logger.info("Optimizer RMSprop is being used")
    elif optimizer_name == "adabound":
        optimizer_function = AdaBound
        kwargs |= {
            "eps": args.optimizer.epsilon,
            "betas": (args.optimizer.beta1, args.optimizer.beta2),
        }
        logger.info("Optimizer AdaBound is being used")
    elif optimizer_name == "adamw":
        optimizer_function = optim.AdamW
        kwargs |= {
            "eps": args.optimizer.epsilon,
            "betas": (args.optimizer.beta1, args.optimizer.beta2),
        }
        logger.info("Optimizer AdamW is being used")
    else:
        raise ValueError(f"Unsupported optimizer name: {args.optimizer.name}")

    if isinstance(params, list):
        if all("weight_decay" not in p for p in params):
            kwargs["weight_decay"] = args.weight_decay
    else:
        kwargs["lr"] = args.optimizer.lr

    return optimizer_function(params, **kwargs)


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return list(self.base_lrs)


def make_scheduler(
    args: Namespace,
    optimizer: Optimizer,
    iters_per_epoch: int = None,
    logger: logging.Logger = None,
    last_epoch: int = -1,
) -> _LRScheduler:
    scheduler_type = args.scheduler_type.lower()

    if scheduler_type == "step":
        scheduler = StepLR(
            optimizer, step_size=args.lr_decay, gamma=args.gamma, last_epoch=last_epoch
        )
        scheduler.__setattr__("__interval__", "epoch")
        logger.info("Loading Step scheduler")
    elif "step" in scheduler_type:
        milestones = list(map(int, scheduler_type.split("_")[1:]))
        scheduler = MultiStepLR(
            optimizer, milestones=milestones, gamma=args.gamma, last_epoch=last_epoch
        )
        scheduler.__setattr__("__interval__", "epoch")
        logger.info("Loading Multi step scheduler")
    elif scheduler_type == "poly_warmstartup":
        # https://github.com/Tony-Y/pytorch_warmup
        scheduler = WarmupPolyLR(
            optimizer,
            power=0.9,
            epochs=args.epochs,
            steps_per_epoch=iters_per_epoch,
            warmup_factor=args.warmup_factor,
            warmup_iters=args.warmup_iters,
            warmup_method=args.warmup_method,
        )
        scheduler.__setattr__("__interval__", "step")
        logger.info("Loading Warm Startup scheduler")
    elif scheduler_type == "multistep_warmstartup":
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=args.milestones,
            gamma=0.1,
            warmup_factor=args.warmup_factor,
            warmup_iters=args.warmup_iters,
            warmup_method=args.warmup_method,
        )
        logger.info("Loading multi step Warm Startup scheduler")
        warnings.warn("This is not in compliance with pytorch lightning")
        scheduler.__setattr__("__interval__", "step")
    elif scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=iters_per_epoch,
            last_epoch=last_epoch,
        )
        scheduler.__setattr__("__interval__", "step")
        logger.info("Loading OneCycle scheduler")
    else:
        scheduler = ConstantLR(optimizer)
        scheduler.__setattr__("__interval__", "epoch")
        logger.info("Loading Constant scheduler")

    return scheduler
