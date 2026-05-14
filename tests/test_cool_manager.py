"""Tests for cool_manager.CoolManager.

The DP itself is exercised indirectly. We mainly check that batched
``[B, C, T]`` input is handled, the apply re-engages autograd, and the
output values match the paper's per-segment-mean recipe.
"""
import sys

sys.path.insert(0, "/home/morg/students/dortirosh/audio_ml_tau_final")

import math

import numpy as np
import pytest
import torch

from cool_manager import CoolManager
from sched_dfr import SchedDFR


def _expected_means_per_segment(x: torch.Tensor, lengths_per_item):
    """Reference implementation of the per-segment mean-broadcast apply."""
    B, C, T = x.shape
    out = torch.zeros_like(x)
    for b in range(B):
        start = 0
        for s in lengths_per_item[b]:
            end = start + s
            out[b, :, start:end] = x[b, :, start:end].mean(dim=-1, keepdim=True)
            start = end
    return out


def test_compress_basic_shape():
    cm = CoolManager(down_sample_ratio=2.0, max_compression=4)
    x = torch.randn(2, 4, 16)
    out = cm.compress(x)
    assert out.shape == x.shape


def test_compress_segment_count():
    cm = CoolManager(down_sample_ratio=2.0, max_compression=4)
    B, C, T = 3, 8, 20
    x = torch.randn(B, C, T)
    cm.compress(x)
    dfr = SchedDFR(2.0, 4)
    expected_segments = int(math.ceil(T / 2.0))
    for b in range(B):
        arr = np.ascontiguousarray(x[b].numpy().T.astype(np.float32))
        encoded = dfr.optimal_down_sample(arr)
        assert sum(encoded.encoding_lengths) == T
        assert len(encoded.encoding_lengths) == expected_segments


def test_compress_deterministic():
    cm = CoolManager(down_sample_ratio=2.0, max_compression=4)
    x = torch.randn(2, 4, 16)
    out_a = cm.compress(x)
    out_b = cm.compress(x)
    assert torch.allclose(out_a, out_b)


def test_compress_gradient_flow():
    cm = CoolManager(down_sample_ratio=2.0, max_compression=4)
    x = torch.randn(2, 4, 16, requires_grad=True)
    out = cm.compress(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert (x.grad.abs() > 0).any()


def test_compress_uses_mean_broadcast():
    cm = CoolManager(down_sample_ratio=2.0, max_compression=4)
    B, C, T = 2, 3, 12
    x = torch.randn(B, C, T)
    out = cm.compress(x)

    dfr = SchedDFR(2.0, 4)
    lengths_per_item = []
    for b in range(B):
        arr = np.ascontiguousarray(x[b].numpy().T.astype(np.float32))
        encoded = dfr.optimal_down_sample(arr)
        lengths_per_item.append(list(encoded.encoding_lengths))

    expected = _expected_means_per_segment(x, lengths_per_item)
    assert torch.allclose(out, expected, atol=1e-6)


def test_compress_parallel_parity_with_sequential():
    """joblib n_jobs>1 must produce bit-identical output to n_jobs=1."""
    torch.manual_seed(0)
    x = torch.randn(8, 4, 16)
    cm_seq = CoolManager(down_sample_ratio=2.0, max_compression=4, n_jobs=1)
    cm_par = CoolManager(down_sample_ratio=2.0, max_compression=4, n_jobs=4)
    out_seq = cm_seq.compress(x)
    out_par = cm_par.compress(x)
    assert torch.allclose(out_seq, out_par)


def test_compress_deepcopy_before_use():
    """Lightning save_hyperparameters deepcopies __init__ args; with n_jobs>1
    we must remain deepcopy-friendly at construction time (lazy pool)."""
    import copy
    cm = CoolManager(down_sample_ratio=2.0, max_compression=4, n_jobs=4)
    cm_copy = copy.deepcopy(cm)
    assert cm_copy.n_jobs == 4
    x = torch.randn(2, 3, 8)
    out_orig = cm.compress(x)
    out_copy = cm_copy.compress(x)
    assert torch.allclose(out_orig, out_copy)
