# !/usr/bin/env python
import argparse
import warnings
import platform


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description='FTNet Semantic Segmentation Training With Pytorch')

# =============================================================================
# Models
# =============================================================================
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'test', 'train_test'],
                        help='Status of the trainer')

    parser.add_argument('--train_only',
                        type=str2bool,
                        default=False,
                        help='Set this for training Cityscapes dataset only')

    parser.add_argument('--model',
                        type=str,
                        default='ftnet',
                        help='Model name (default: ftnet)')

    parser.add_argument('--backbone',
                        type=str,
                        default='resnext50_32x4d',
                        help='Backbone name (default: resnext50_32x4d)')

    parser.add_argument('--pretrained-base',
                        type=str2bool,
                        default=True,
                        help='Use pretrained ImageNet backbone')

    parser.add_argument('--dilation',
                        type=str2bool,
                        default=False,
                        help='Use dilated backbone')
# =============================================================================
# Data and Dataloader
# =============================================================================
    parser.add_argument('--dataset',
                        type=str,
                        default='soda',
                        choices=['cityscapes_thermal_combine', 'soda', 'mfn', 'scutseg'],
                        help='Dataset to be utilized (default: soda)')

    parser.add_argument('--dataset-path',
                        type=str,
                        default='./Dataset/',
                        help='Path to the dataset folder')

    parser.add_argument('--base-size',
                        type=str,
                        default='300',
                        help='Base image size.\
                        Note: First value is considered for validation and testing. \
                        Please provide multiple size by using a + operator for example 256+512+768')

    parser.add_argument('--crop-size',
                        type=str,
                        default='256',
                        help='Crop image size  \
                        Note: First value is considered for validation and testing. \
                        Please provide multiple size by using a + operator for example 256+512+768')

    parser.add_argument('--workers',
                        type=int,
                        default=16,
                        help='Total number of workers for dataloader')

    parser.add_argument('--no-of-filters',
                        type=int,
                        default=128,
                        help='Number of filter for the FTNet')

    parser.add_argument('--edge-extracts',
                        type=str,
                        default='3',
                        help='The position of the encoder from which the edges needs to be extracted')

    parser.add_argument('--num-blocks',
                        type=int,
                        default=2,
                        help='Total number of residual units per stream')

    parser.add_argument('--train-batch-size',
                        type=int,
                        default=16,
                        help='Input batch size for training (default: 16)')

    parser.add_argument('--val-batch-size',
                        type=int,
                        default=4,
                        help='Input batch size for validation (default: 8)')

    parser.add_argument('--test-batch-size',
                        type=int,
                        default=1,
                        help='Input batch size for testing (default: 1)')

    parser.add_argument('--accumulate-grad-batches',
                        type=int,
                        default=1,
                        help='Number of batches to be accumulated doing a backwards pass')

    parser.add_argument('--test-monitor',
                        type=str,
                        default='val_mIOU',
                        help='The metric with best value to be tested')

    parser.add_argument('--test-monitor-path',
                        type=str,
                        help='Path to the checkpoint folder')

# =============================================================================
# WandB
# =============================================================================

    parser.add_argument('--wandb-id',
                        type=str,
                        default=None,
                        help='Sets the version, mainly used to resume a previous run.')

    parser.add_argument('--wandb-name-ext',
                        type=str,
                        default='None',
                        help='Name_extension_wandb')

# =============================================================================
# Training hyper params
# =============================================================================

    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to train (default: 100)')

    parser.add_argument('--loss-weight',
                        type=int,
                        default=1,
                        help='Auxiliary loss weight')

# =============================================================================
# Optimizer and scheduler parameters
# =============================================================================
    parser.add_argument('--optimizer',
                        default='SGD',
                        choices=('SGD', 'ADAM', 'RMSprop', 'AdaBound'),
                        help='Optimizer to use (SGD | ADAM | RMSprop | AdaBound)')

    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate (default: 0.01)')

    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='Momentum (default: 0.9)')

    parser.add_argument('--nesterov',
                        type=str2bool,
                        default=False,
                        help='Set Nesterov')

    parser.add_argument('--weight-decay',
                        type=float,
                        default=0.0001,
                        help='Weight-decay (default: 5e-4)')

    parser.add_argument('--beta1',
                        type=float,
                        default=0.9,
                        help='Beta1 parameter in optimizer')

    parser.add_argument('--beta2',
                        type=float,
                        default=0.999,
                        help='Beta2 parameter in optimizer')

    parser.add_argument('--epsilon',
                        type=float,
                        default=1e-8,
                        help='Epsilon for numerical stability')

    parser.add_argument('--scheduler-type',
                        type=str,
                        default='poly_warmstartup',
                        choices=('step', 'multistep_90_160',
                                 'poly_warmstartup', 'multistep_warmstartup', 'onecycle'),
                        help='Learning rate decay type')

    parser.add_argument('--warmup-iters',
                        type=int,
                        default=0,
                        help='Warmup iteration')

    parser.add_argument('--warmup-factor',
                        type=float,
                        default=1.0 / 3,
                        help='Warmup factor for the scheduler')

    parser.add_argument('--warmup-method',
                        type=str,
                        default='linear',
                        help='Method of warmup')

    parser.add_argument('--gamma',
                        type=float,
                        default=0.5,
                        help='Learning rate decay factor for step decay')

# =============================================================================
# Checkpoint and log
# =============================================================================
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Provide the path of the checkpoint to be resumed. \
                        If not provided, the save directory will be utilized to continue training')

    parser.add_argument('--save-dir',
                        default='./../../Results/',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--test-checkpoint',
                        default=None,
                        help='Checkpoint for testing')

    parser.add_argument('--save-images',
                        type=str2bool,
                        default=False,
                        help='Save validation and testing Images')

    parser.add_argument('--save-images-as-subplots',
                        type=str2bool,
                        default=False,
                        help='Save Validation and Testing Images as subplots or complete images. \
                            Note: save-images needs to be set as True')

# =============================================================================
# MISC
# =============================================================================
    parser.add_argument('--debug',
                        type=str2bool,
                        default=True,
                        help='Enable debugging mode')

    parser.add_argument('--seed', type=int, default=123,
                        help='Seed for the process')

    parser.add_argument('--num-nodes',
                        default=1,
                        type=int,
                        help='Number of Nodes available for computing(default=1)')

    parser.add_argument('--gpus',
                        default=1,
                        type=int,
                        help='If set to None, all the gpus are used else specific gpu is used')

    parser.add_argument('--distributed-backend',
                        type=str,
                        default='dp',
                        choices=('dp', 'ddp', 'ddp2', 'horovod'),
                        help='supports three options dp, ddp, ddp2')

    args = parser.parse_args()

    def _split(args):
        return list(map(lambda x: int(x), args.split('+')))

    args.base_size = _split(args.base_size)
    args.crop_size = _split(args.crop_size)

    args.edge_extracts = _split(args.edge_extracts)

    assert all(earlier >= later for earlier, later in zip(
        args.base_size, args.base_size[1:])), "Base size should be in descending order"

    return args
