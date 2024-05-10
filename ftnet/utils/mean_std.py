# =============================================================================
# https://news.ycombinator.com/item?id=16609033
# =============================================================================

import os

# cur_path = os.path.abspath(os.path.dirname("__file__"))
# root_path = os.path.split(cur_path)[0]

# sys.path.append(root_path)

import lib.data.dataloader.qtransforms as qtrans
from lib.data.dataloader import get_classification_dataset
from lib.data.dataloader.prefetch_generator.dataloader_v2 import (
    DataLoaderX as DataLoader,
)
from options import parse_args

args = parse_args()

# =============================================================================
# Setting up
# =============================================================================
# Data loading code
traindir = os.path.join(args.dataset_path, "train")
valdir = os.path.join(args.dataset_path, "valid")

# =============================================================================
# 0.2126 * R + 0.7152 * G + 0.0722 * B
# =============================================================================

train_dataset = get_classification_dataset(
    name=args.dataset,
    root=traindir,
    train=True,
    download=True,
    transform=qtrans.ToQTensor(),
)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True,
)

mean, std = [], []
for i, (input, target) in enumerate(train_loader):
    mean.append(input.mean(dim=[0, 2, 3]))
    mean.append(input.std(dim=[0, 2, 3]))
