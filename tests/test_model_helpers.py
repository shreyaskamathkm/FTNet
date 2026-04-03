"""Tests for model helper utilities."""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from ftnet.helper.model_helpers import check_mismatch, save_model_summary, total_gradient


class TestCheckMismatch:
    def _make_state_dict(self, keys_shapes: dict) -> dict:
        return {k: torch.zeros(shape) for k, shape in keys_shapes.items()}

    def test_matching_keys_and_shapes(self):
        model_dict = self._make_state_dict({"layer.weight": (4, 4), "layer.bias": (4,)})
        # pretrained keys have a 6-char prefix (e.g. "model.")
        pretrained_dict = self._make_state_dict(
            {"model.layer.weight": (4, 4), "model.layer.bias": (4,)}
        )
        result = check_mismatch(model_dict, pretrained_dict)
        assert set(result.keys()) == {"layer.weight", "layer.bias"}

    def test_shape_mismatch_skips_key(self):
        model_dict = self._make_state_dict({"layer.weight": (4, 4)})
        pretrained_dict = self._make_state_dict({"model.layer.weight": (8, 8)})
        result = check_mismatch(model_dict, pretrained_dict)
        assert "layer.weight" not in result

    def test_missing_key_in_model_skipped(self):
        model_dict = self._make_state_dict({"layer.weight": (4, 4)})
        pretrained_dict = self._make_state_dict({"model.other.weight": (4, 4)})
        result = check_mismatch(model_dict, pretrained_dict)
        assert len(result) == 0

    def test_returns_ordered_dict(self):
        model_dict = self._make_state_dict({"layer.weight": (4, 4)})
        pretrained_dict = self._make_state_dict({"model.layer.weight": (4, 4)})
        result = check_mismatch(model_dict, pretrained_dict)
        assert isinstance(result, OrderedDict)

    def test_empty_pretrained(self):
        model_dict = self._make_state_dict({"layer.weight": (4, 4)})
        result = check_mismatch(model_dict, {})
        assert len(result) == 0


class TestSaveModelSummary:
    def test_creates_file(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        save_model_summary(model, tmp_path)
        summary_file = tmp_path / "model.txt"
        assert summary_file.exists()

    def test_file_contains_param_count(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        save_model_summary(model, tmp_path)
        content = (tmp_path / "model.txt").read_text()
        assert "Total number of parameters" in content

    def test_file_contains_model_repr(self, tmp_path: Path):
        model = nn.Linear(4, 2)
        save_model_summary(model, tmp_path)
        content = (tmp_path / "model.txt").read_text()
        assert "Linear" in content


class TestTotalGradient:
    def test_with_gradients(self):
        model = nn.Linear(4, 2)
        output = model(torch.randn(3, 4))
        output.sum().backward()
        norm = total_gradient(model.parameters())
        assert isinstance(norm, float)
        assert norm > 0

    def test_no_gradients_returns_zero(self):
        model = nn.Linear(4, 2)
        # No backward pass — no gradients
        norm = total_gradient(model.parameters())
        assert norm == 0.0
