from pydantic import BaseModel
from typing import List
from pathlib import Path
import toml
from types import SimpleNamespace


class TaskArgs(BaseModel):
    mode: str = "train"
    train_only: bool = False
    debug: bool = False


class TrainingHyperParamsArgs(BaseModel):
    loss_weight: int = 1
    epochs: int = 100
    train_batch_size: int = 16
    val_batch_size: int = 4
    accumulate_grad_batches: int = 1


class TestingHyperParamsArgs(BaseModel):
    test_batch_size: int = 1
    test_monitor: str = "val_mIOU"
    test_monitor_path: str = None


class ModelArgs(BaseModel):
    model: str = "ftnet"
    backbone: str = "resnext50_32x4d"
    pretrained_base: bool = False
    dilation: bool = False
    no_of_filters: int = 128
    edge_extracts: List[int] = [3]
    num_blocks: int = 2


class DataLoaderArgs(BaseModel):
    dataset: str = "soda"
    dataset_path: Path = Path("./Dataset/")
    base_size: List[int] = [300]
    crop_size: List[int] = [256]


class WandBArgs(BaseModel):
    wandb_id: str = None
    wandb_name_ext: str = "None"


class OptimizerArgs(BaseModel):
    optimizer: str = "SGD"
    lr: float = 0.01
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


class SchedulerArgs(BaseModel):
    scheduler_type: str = "poly_warmstartup"
    warmup_iters: int = 0
    warmup_factor: float = 1.0 / 3
    warmup_method: str = "linear"
    gamma: float = 0.5


class CheckpointLogArgs(BaseModel):
    resume: Path = None
    save_dir: Path = Path("./../../Results/")
    test_checkpoint: str = None
    save_images: bool = False
    save_images_as_subplots: bool = False


class ComputeArgs(BaseModel):
    debug: bool = False
    seed: int = 123
    num_nodes: int = 1
    gpus: int = 1
    distributed_backend: str = "dp"
    workers: int = 16


class FTNetArgs(BaseModel):
    task: TaskArgs = TaskArgs()
    trainer: TrainingHyperParamsArgs = TrainingHyperParamsArgs()
    model: ModelArgs = ModelArgs()
    dataset: DataLoaderArgs = DataLoaderArgs()
    wandb: WandBArgs = WandBArgs()
    optimizer: OptimizerArgs = OptimizerArgs()
    scheduler: SchedulerArgs = SchedulerArgs()
    checkpoint: CheckpointLogArgs = CheckpointLogArgs()
    compute: ComputeArgs = ComputeArgs()

    @classmethod
    def from_config(cls, config: SimpleNamespace):
        if isinstance(
            config, str
        ):  # If a string is provided, assume it's a TOML file path
            config_dict = toml.load(config)
        elif isinstance(config, dict):  # If a dictionary is provided, use it directly
            config_dict = config
        else:
            raise ValueError(
                "Invalid input type. Please provide either a TOML file path or a dictionary."
            )

        # Check for extra keys in config_dict
        allowed_keys = set(cls.model_fields.keys())
        extra_keys = set(config_dict) - allowed_keys
        if extra_keys:
            raise ValueError(
                f"Unexpected keys found in config: {', '.join(extra_keys)}"
            )

        return cls.model_validate(config_dict)
