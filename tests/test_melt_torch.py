"""Tests for the torch-native rewrite of MeltManager.melt and the shared
gradient-preserving apply helper _apply_mean_broadcast."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pytest
import torch

from melt_manager import MeltManager, _apply_mean_broadcast


def test_apply_mean_broadcast_shape():
    x = torch.randn(2, 4, 10)
    out = _apply_mean_broadcast(x, [[3, 2, 5], [4, 4, 2]])
    assert out.shape == x.shape


def test_apply_mean_broadcast_correct_means():
    x = torch.tensor([[[1.0, 2.0, 5.0, 7.0, 11.0, 13.0]]])
    out = _apply_mean_broadcast(x, [[2, 1, 3]])
    expected = torch.tensor([[[1.5, 1.5, 5.0, 31.0 / 3, 31.0 / 3, 31.0 / 3]]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_apply_mean_broadcast_per_channel():
    x = torch.tensor([[
        [1.0, 2.0, 5.0],
        [10.0, 20.0, 100.0],
    ]])
    out = _apply_mean_broadcast(x, [[2, 1]])
    expected = torch.tensor([[
        [1.5, 1.5, 5.0],
        [15.0, 15.0, 100.0],
    ]])
    assert torch.allclose(out, expected)


def test_apply_mean_broadcast_per_item_segmentation():
    x = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0]],
        [[10.0, 20.0, 30.0, 40.0]],
    ])
    out = _apply_mean_broadcast(x, [[4], [2, 2]])
    expected = torch.tensor([
        [[2.5, 2.5, 2.5, 2.5]],
        [[15.0, 15.0, 35.0, 35.0]],
    ])
    assert torch.allclose(out, expected)


def test_apply_mean_broadcast_gradient_flows():
    x = torch.arange(6.0).reshape(1, 1, 6).requires_grad_(True)
    out = _apply_mean_broadcast(x, [[3, 3]])
    assert out.grad_fn is not None
    out.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_apply_mean_broadcast_gradient_pattern():
    x = torch.arange(6.0).reshape(1, 1, 6).requires_grad_(True)
    out = _apply_mean_broadcast(x, [[2, 1, 3]])
    out[0, 0, 3].backward()
    expected = torch.tensor([[[0.0, 0.0, 0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3]]])
    assert torch.allclose(x.grad, expected, atol=1e-6)


def test_melt_returns_tensor_with_grad():
    mm = MeltManager(skip_prob=0.0)
    x = torch.randn(2, 16, 32, requires_grad=True)
    out = mm.melt(x, step=50_000)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert out.grad_fn is not None


def test_melt_skip_prob_one_passes_through():
    mm = MeltManager(skip_prob=1.0)
    x = torch.randn(2, 16, 32)
    out = mm.melt(x, step=0)
    assert torch.equal(out, x)


def test_melt_step_sync():
    mm = MeltManager()
    mm.melt(torch.randn(1, 4, 16), step=12345)
    assert mm._current_training_step == 12345


def test_melt_reduction_with_monkey_patched_segments(monkeypatch):
    mm = MeltManager(skip_prob=0.0)
    monkeypatch.setattr(mm, "_sample_segments_to_T", lambda proportion, T: [3, 5])
    x = torch.randn(1, 8, 8)
    out = mm.melt(x, step=10_000)
    expected = _apply_mean_broadcast(x, [[3, 5]])
    assert torch.allclose(out, expected, atol=1e-6)


def test_p_tgt_sum_tolerance():
    p_tgt = [0.1 + 1e-13, 0.45, 0.25, 0.2]
    MeltManager(p_tgt=p_tgt)


def test_p_tgt_sum_strictly_wrong_raises():
    with pytest.raises(ValueError):
        MeltManager(p_tgt=[0.1, 0.5, 0.25, 0.2])
