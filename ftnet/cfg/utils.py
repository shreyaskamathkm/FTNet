# import argparse

import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="FTNet Semantic Segmentation Training With Pytorch"
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to the TOML configuration file",
    )

    args = parser.parse_args()

    # if args.config:
    #     config_dict = toml.load(args.config)
    # else:
    #     config_dict = {}

    # # Models
    # parser.add_argument('--mode', type=str, default=config_dict.get('mode', 'train'), choices=['train', 'test', 'train_test'], help='Status of the trainer')
    # parser.add_argument('--train_only', action='store_true', default=config_dict.get('train_only', False), help='Set this for training Cityscapes dataset only')
    # parser.add_argument('--model', type=str, default=config_dict.get('model', 'ftnet'), help='Model name (default: ftnet)')
    # parser.add_argument('--backbone', type=str, default=config_dict.get('backbone', 'resnext50_32x4d'), help='Backbone name (default: resnext50_32x4d)')
    # parser.add_argument('--pretrained_base', action='store_true', default=config_dict.get('pretrained_base', False), help='Use pretrained ImageNet backbone')
    # parser.add_argument('--dilation', action='store_true', default=config_dict.get('dilation', False), help='Use dilated backbone')

    # # Data and Dataloader
    # parser.add_argument('--dataset', type=str, default=config_dict.get('dataset', 'soda'), choices=['cityscapes_thermal_combine', 'soda', 'mfn', 'scutseg'], help='Dataset to be utilized (default: soda)')
    # parser.add_argument('--dataset-path', type=str, default=config_dict.get('dataset_path', './Dataset/'), help='Path to the dataset folder')
    # parser.add_argument('--base-size', type=str, default=config_dict.get('base_size', '300'), help='Base image size.')
    # parser.add_argument('--crop-size', type=str, default=config_dict.get('crop_size', '256'), help='Crop image size')
    # parser.add_argument('--workers', type=int, default=config_dict.get('workers', 16), help='Total number of workers for dataloader')
    # parser.add_argument('--no_of_filters', type=int, default=config_dict.get('no_of_filters', 128), help='Number of filter for the FTNet')
    # parser.add_argument('--edge-extracts', type=str, default=config_dict.get('edge_extracts', '3'), help='The position of the encoder from which the edges needs to be extracted')
    # parser.add_argument('--num-blocks', type=int, default=config_dict.get('num_blocks', 2), help='Total number of residual units per stream')
    # parser.add_argument('--train-batch-size', type=int, default=config_dict.get('train_batch_size', 16), help='Input batch size for training')
    # parser.add_argument('--val-batch-size', type=int, default=config_dict.get('val_batch_size', 4), help='Input batch size for validation')
    # parser.add_argument('--test-batch-size', type=int, default=config_dict.get('test_batch_size', 1), help='Input batch size for testing')
    # parser.add_argument('--accumulate-grad-batches', type=int, default=config_dict.get('accumulate_grad_batches', 1), help='Number of batches to be accumulated doing a backwards pass')
    # parser.add_argument('--test-monitor', type=str, default=config_dict.get('test_monitor', 'val_mIOU'), help='The metric with best value to be tested')
    # parser.add_argument('--test-monitor-path', type=str, default=config_dict.get('test_monitor_path'), help='Path to the checkpoint folder')

    # # WandB
    # parser.add_argument('--wandb-id', type=str, default=config_dict.get('wandb_id'), help='Sets the version, mainly used to resume a previous run.')
    # parser.add_argument('--wandb-name-ext', type=str, default=config_dict.get('wandb_name_ext', 'None'), help='Name_extension_wandb')

    # # Training hyper params
    # parser.add_argument('--epochs', type=int, default=config_dict.get('epochs', 100), help='Number of epochs to train')
    # parser.add_argument('--loss-weight', type=int, default=config_dict.get('loss_weight', 1), help='Auxiliary loss weight')

    # # Optimizer and scheduler parameters
    # parser.add_argument('--optimizer', default=config_dict.get('optimizer', 'SGD'), choices=('SGD', 'ADAM', 'RMSprop', 'AdaBound'), help='Optimizer to use')
    # parser.add_argument('--lr', type=float, default=config_dict.get('lr', 0.01), help='Learning rate')
    # parser.add_argument('--momentum', type=float, default=config_dict.get('momentum', 0.9), help='Momentum')
    # parser.add_argument('--nesterov', action='store_true', default=config_dict.get('nesterov', False), help='Set Nesterov')
    # parser.add_argument('--weight-decay', type=float, default=config_dict.get('weight_decay', 0.0001), help='Weight-decay')
    # parser.add_argument('--beta1', type=float, default=config_dict.get('beta1', 0.9), help='Beta1 parameter in optimizer')
    # parser.add_argument('--beta2', type=float, default=config_dict.get('beta2', 0.999), help='Beta2 parameter in optimizer')
    # parser.add_argument('--epsilon', type=float, default=config_dict.get('epsilon', 1e-8), help='Epsilon for numerical stability')
    # parser.add_argument('--scheduler-type', type=str, default=config_dict.get('scheduler_type', 'poly_warmstartup'), choices=('step', 'multistep_90_160', 'poly_warmstartup', 'multistep_warmstartup', 'onecycle'), help='Learning rate decay type')
    # parser.add_argument('--warmup-iters', type=int, default=config_dict.get('warmup_iters', 0), help='Warmup iteration')
    # parser.add_argument('--warmup-factor', type=float, default=config_dict.get('warmup_factor', 1.0 / 3), help='Warmup factor for the scheduler')
    # parser.add_argument('--warmup-method', type=str, default=config_dict.get('warmup_method', 'linear'), help='Method of warmup')
    # parser.add_argument('--gamma', type=float, default=config_dict.get('gamma', 0.5), help='Learning rate decay factor for step decay')

    # # Checkpoint and log
    # parser.add_argument('--resume', type=str, default=config_dict.get('resume'), help='Provide the path of the checkpoint to be resumed.')
    # parser.add_argument('--save-dir', default=config_dict.get('save_dir', './../../Results/',), help='Directory for saving checkpoint models')
    # parser.add_argument('--test-checkpoint', default=config_dict.get('test_checkpoint'), help='Checkpoint for testing')
    # parser.add_argument('--save-images', action='store_true', default=config_dict.get('save_images'), help='Save validation and testing Images')
    # parser.add_argument('--save-images-as-subplots', action='store_true', default=config_dict.get('save_images_as_subplots'), help='Save Validation and Testing Images as subplots or complete images.')

    # # MISC
    # parser.add_argument('--debug', action='store_true', default=config_dict.get('debug'), help='Enable debugging mode')
    # parser.add_argument('--seed', type=int, default=config_dict.get('seed', 123), help='Seed for the process')
    # parser.add_argument('--num-nodes', default=config_dict.get('num_nodes', 1), type=int, help='Number of Nodes available for computing')
    # parser.add_argument('--gpus', default=config_dict.get('gpus', 1), type=int, help='Number of GPUs to use')
    # parser.add_argument('--distributed-backend', type=str, default=config_dict.get('distributed_backend', 'dp'), choices=('dp', 'ddp', 'ddp2', 'horovod'), help='Supports three options dp, ddp, ddp2')

    # args = parser.parse_args(remaining_args)

    # def _split(args):
    #     return list(map(lambda x: int(x), args.split('+')))

    # args.base_size = _split(args.base_size)
    # args.crop_size = _split(args.crop_size)

    # args.edge_extracts = _split(args.edge_extracts)

    # assert all(earlier >= later for earlier, later in zip(
    #     args.base_size, args.base_size[1:])), "Base size should be in descending order"

    return args
