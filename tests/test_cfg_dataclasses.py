"""Tests for FTNet configuration dataclasses."""

import pytest
from pydantic import ValidationError

from ftnet.cfg.cfg_dataclasses import (
    CheckpointLogArgs,
    ComputeArgs,
    DataLoaderArgs,
    FTNetArgs,
    ModelArgs,
    OptimizerArgs,
    SchedulerArgs,
    TaskArgs,
    TrainingHyperParamsArgs,
    WandBArgs,
)


class TestTaskArgs:
    def test_defaults(self):
        args = TaskArgs()
        assert args.mode == "train"
        assert args.train_only is False
        assert args.debug is False

    def test_valid_modes(self):
        for mode in ("train", "test", "infer"):
            assert TaskArgs(mode=mode).mode == mode

    def test_invalid_mode(self):
        with pytest.raises(ValidationError):
            TaskArgs(mode="invalid")


class TestModelArgs:
    def test_defaults(self):
        args = ModelArgs()
        assert args.name == "ftnet"
        assert args.backbone == "resnext50_32x4d"
        assert args.no_of_filters == 128
        assert args.edge_extracts == [3]
        assert args.num_blocks == 2

    def test_custom_values(self):
        args = ModelArgs(backbone="resnet50", no_of_filters=256, num_blocks=4)
        assert args.backbone == "resnet50"
        assert args.no_of_filters == 256
        assert args.num_blocks == 4


class TestDataLoaderArgs:
    def test_defaults(self):
        args = DataLoaderArgs()
        assert args.name == "soda"

    def test_valid_dataset_names(self):
        valid = ["cityscapes_thermal_combine", "cityscapes_thermal_split", "soda", "mfn", "scutseg"]
        for name in valid:
            assert DataLoaderArgs(name=name).name == name

    def test_invalid_dataset_name(self):
        with pytest.raises(ValidationError):
            DataLoaderArgs(name="nonexistent_dataset")


class TestFTNetArgs:
    def test_defaults(self):
        args = FTNetArgs()
        assert isinstance(args.task, TaskArgs)
        assert isinstance(args.trainer, TrainingHyperParamsArgs)
        assert isinstance(args.model, ModelArgs)
        assert isinstance(args.dataset, DataLoaderArgs)
        assert isinstance(args.wandb, WandBArgs)
        assert isinstance(args.optimizer, OptimizerArgs)
        assert isinstance(args.scheduler, SchedulerArgs)
        assert isinstance(args.checkpoint, CheckpointLogArgs)
        assert isinstance(args.compute, ComputeArgs)

    def test_cityscapes_combine_requires_train_only(self):
        """cityscapes_thermal_combine must have train_only=True."""
        with pytest.raises(ValidationError, match="requires train_only=True"):
            FTNetArgs(
                dataset=DataLoaderArgs(name="cityscapes_thermal_combine"),
                task=TaskArgs(train_only=False),
            )

    def test_cityscapes_combine_with_train_only_passes(self):
        """cityscapes_thermal_combine with train_only=True should be valid."""
        args = FTNetArgs(
            dataset=DataLoaderArgs(name="cityscapes_thermal_combine"),
            task=TaskArgs(train_only=True),
        )
        assert args.dataset.name == "cityscapes_thermal_combine"
        assert args.task.train_only is True

    def test_other_datasets_without_train_only_passes(self):
        """Non-cityscapes_thermal_combine datasets should not require train_only."""
        for name in ("soda", "mfn", "scutseg"):
            args = FTNetArgs(
                dataset=DataLoaderArgs(name=name),
                task=TaskArgs(train_only=False),
            )
            assert args.dataset.name == name

    def test_from_config_with_dict(self):
        config = {"task": {"mode": "train"}, "dataset": {"name": "soda"}}
        args = FTNetArgs.from_config(config)
        assert args.task.mode == "train"
        assert args.dataset.name == "soda"

    def test_from_config_rejects_extra_keys(self):
        with pytest.raises(ValueError, match="Unexpected keys"):
            FTNetArgs.from_config({"nonexistent_section": {}})

    def test_from_config_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid input type"):
            FTNetArgs.from_config(object())  # type: ignore[arg-type]
