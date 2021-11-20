import warnings
import math
import torch.optim as optim
from core.optimizers.adabound import AdaBound
from core.optimizers.adamw import AdamW
from core.schedulers.cyclic_warm_restart import CyclicLRWithRestarts
from core.schedulers.iteration_polyLR import IterationPolyLR
from core.schedulers.lr_scheduler import WarmupMultiStepLR, WarmupPolyLR
from core.schedulers.OneCycle_LR import OneCycle
from core.schedulers.Poly_LR import Poly
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

__all__ = ['make_optimizer', 'make_scheduler']


def make_optimizer(args, params, logger=None):

    if args.optimizer.lower() == 'sgd':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum,
                  'nesterov': args.nesterov}
        if logger is not None:
            logger.info('Optimizer SGD is being used')

    elif args.optimizer.lower() == 'adam':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
        if logger is not None:
            logger.info('Optimizer ADAM is being used')

    elif args.optimizer.lower() == 'rmsprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}
        if logger is not None:
            logger.info('Optimizer RMSprop is being used')

    elif args.optimizer.lower() == 'adabound':
        optimizer_function = AdaBound
        kwargs = {'eps': args.epsilon,
                  'betas': (args.beta1, args.beta2)}
        if logger is not None:
            logger.info('Optimizer AdaBound is being used')

    elif args.optimizer.lower() == 'adamw':
        optimizer_function = optim.AdamW
        kwargs = {'eps': args.epsilon,
                  'betas': (args.beta1, args.beta2)}
        if logger is not None:
            logger.info('Optimizer AdamW is being used')

    if isinstance(params, list):
        if not any("weight_decay" in s for s in params):
            kwargs['weight_decay'] = args.weight_decay
    else:
        kwargs['lr'] = args.lr

    return optimizer_function(params, **kwargs)


class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]


def make_scheduler(args, optimizer,
                   iters_per_epoch=None,
                   logger=None,
                   last_epoch=-1):

    if args.scheduler_type.lower() == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay,
                                        gamma=args.gamma,
                                        last_epoch=last_epoch
                                        )
        scheduler.__setattr__('__interval__', 'epoch')
        if logger is not None:
            logger.info('Loading Step scheduler')

    elif args.scheduler_type.find('step') >= 0:
        milestones = args.scheduler_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.gamma,
            last_epoch=last_epoch
        )

        scheduler.__setattr__('__interval__', 'epoch')
        if logger is not None:
            logger.info('Loading Multi step scheduler ')

    # elif args.scheduler_type.lower() == 'reduce_on_plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(
    #         optimizer,
    #     )
    #     scheduler.__setattr__('__interval__', 'epoch')

    elif args.scheduler_type.lower() == 'poly_warmstartup':
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1038
        scheduler = WarmupPolyLR(optimizer,
                                 power=0.9,
                                 epochs=args.epochs,
                                 steps_per_epoch=iters_per_epoch,
                                 warmup_factor=args.warmup_factor,
                                 warmup_iters=args.warmup_iters,
                                 warmup_method=args.warmup_method)

        scheduler.__setattr__('__interval__', 'step')
        if logger is not None:
            logger.info('Loading Warm Startup scheduler')

    elif args.scheduler_type.lower() == 'multistep_warmstartup':
        # =============================================================================
        #         https://github.com/Tony-Y/pytorch_warmup
        # =============================================================================
        scheduler = WarmupMultiStepLR(optimizer,
                                      milestones=args.milestones,
                                      gamma=0.1,
                                      warmup_factor=args.warmup_factor,
                                      warmup_iters=args.warmup_iters,
                                      warmup_method=args.warmup_method)

        if logger is not None:
            logger.info('Loading multi step Warm Startup scheduler')
        warnings.warn('This is not in compliance with pytorch lightning')
        scheduler.__setattr__('__interval__', 'step')

    elif args.scheduler_type.lower() == 'onecycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer,
                                            max_lr=args.lr,
                                            epochs=args.epochs,
                                            steps_per_epoch=iters_per_epoch,
                                            # pct_start=0.3,
                                            # anneal_strategy='cos',
                                            # cycle_momentum=True,
                                            # base_momentum=0.85,
                                            # max_momentum=0.95,
                                            # div_factor=25.0,
                                            # final_div_factor=10000.0,
                                            last_epoch=last_epoch)
        scheduler.__setattr__('__interval__', 'step')
        if logger is not None:
            logger.info('Loading OneCycle scheduler')

    else:
        scheduler = ConstantLR(optimizer)
        scheduler.__setattr__('__interval__', 'epoch')
        logger.info('Loading Constant scheduler')

    return scheduler
