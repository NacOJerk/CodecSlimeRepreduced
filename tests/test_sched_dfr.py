"""Regression tests for the optimized SchedDFR DP.

The optimized implementation (precomputed L[j,s] table + vectorized cost
build) must produce identical `encoding_lengths` and `encoded_data` as the
original triple-loop reference implementation that recomputes
`_difference_loss` inside the innermost DP loop. Paper fidelity is the
standing rule for this project.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from sched_dfr import SchedDFR, EncodedData


def _reference_optimal_down_sample(raw_output: np.ndarray,
                                   down_sample_ratio: float,
                                   max_compression: int) -> EncodedData:
    """Verbatim port of the pre-optimization DP. Used only as a comparison
    target; recomputes _difference_loss inside the inner s-loop."""
    t = raw_output.shape[0]
    t_tag = int(np.ceil(t / down_sample_ratio))
    optimal_distance = [[-np.inf for _ in range(t_tag + 1)] for _ in range(t + 1)]
    chosen_s_array = [[0 for _ in range(t_tag + 1)] for _ in range(t + 1)]
    optimal_distance[0][0] = 0
    for i in range(1, t_tag + 1):
        for j in range(1, t + 1):
            max_so_far = -np.inf
            chosen_s = 0
            max_s = min(j - i + 1, max_compression)
            for s in range(1, max_s + 1):
                diff = optimal_distance[j - s][i - 1] - SchedDFR._difference_loss(raw_output, j - 1, s)
                if diff >= max_so_far:
                    max_so_far = diff
                    chosen_s = s
            optimal_distance[j][i] = max_so_far
            chosen_s_array[j][i] = chosen_s

    final_s_choices = []
    current_j, current_i = t, t_tag
    while current_j > 0:
        final_s_choices.append(chosen_s_array[current_j][current_i])
        current_j -= chosen_s_array[current_j][current_i]
        current_i -= 1
    final_s_choices.reverse()
    return EncodedData(SchedDFR.down_sample(raw_output, final_s_choices), final_s_choices)


@pytest.mark.parametrize("t,d,ratio,u,seed", [
    (16, 4, 2.0, 4, 0),
    (32, 8, 2.0, 4, 1),
    (50, 16, 2.0, 4, 2),
    (100, 32, 2.0, 4, 3),
    (80, 1024, 2.0, 4, 4),       # realistic D matching the codec encoder
    (200, 1024, 2.0, 4, 5),
    (37, 1024, 2.0, 4, 6),       # odd t to stress the i+s boundary
    (16, 4, 1.6, 4, 7),          # non-2 ratio
    (16, 4, 1.0, 1, 8),          # degenerate ratio=1, U=1 (no compression possible)
])
def test_optimized_dp_matches_reference(t, d, ratio, u, seed):
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((t, d), dtype=np.float64).astype(np.float32)

    expected = _reference_optimal_down_sample(raw, ratio, u)
    actual = SchedDFR(ratio, u).optimal_down_sample(raw)

    assert actual.encoding_lengths == expected.encoding_lengths, \
        f"encoding_lengths drift on seed={seed}: {actual.encoding_lengths} vs {expected.encoding_lengths}"
    np.testing.assert_allclose(actual.encoded_data, expected.encoded_data, rtol=1e-5, atol=1e-6)


def test_precompute_cost_matches_per_pair_loss():
    """The precomputed L[j, s] table must equal _difference_loss(raw, j-1, s) cell-by-cell."""
    rng = np.random.default_rng(42)
    t, d, U = 50, 16, 4
    raw = rng.standard_normal((t, d), dtype=np.float64).astype(np.float32)
    L = SchedDFR._precompute_cost(raw, U)
    for j in range(1, t + 1):
        for s in range(1, min(j, U) + 1):
            expected = SchedDFR._difference_loss(raw, j - 1, s)
            np.testing.assert_allclose(L[j, s], expected, rtol=1e-5, atol=1e-6,
                                       err_msg=f"L[{j},{s}] drift")


def test_encoding_lengths_sum_to_t():
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((73, 64), dtype=np.float32)
    enc = SchedDFR(2.0, 4).optimal_down_sample(raw)
    assert sum(enc.encoding_lengths) == 73
