"""CoolManager: ScheDFR-driven downsampler used by the Cool fine-tuning stage.

The DP scheduler in ``sched_dfr.SchedDFR`` is numpy-only and operates on a
single ``[T, C]`` array. ``CoolManager.compress`` bridges it to the batched
``[B, C, T]`` tensor expected by ``CoolMeltWrapper`` and reuses
``melt_manager._apply_mean_broadcast`` for the differentiable per-segment
mean apply, matching the paper recipe: the DP runs inside the forward to
pick the schedule, gradients flow through the mean-broadcast apply only.

The DP itself is the same algorithm whether ``n_jobs == 1`` (a sequential
Python loop) or ``n_jobs > 1`` (joblib over batch items): each item runs the
exact same numpy ``optimal_down_sample`` and returns the same lengths.
``n_jobs`` is a pure runtime knob, not a paper deviation.
"""
from typing import List, Optional

import numpy as np
import torch
from joblib import Parallel, delayed

from sched_dfr import SchedDFR
from melt_manager import _apply_mean_broadcast


def _dp_one(arr_T_C: np.ndarray, down_sample_ratio: float, max_compression: int) -> List[int]:
    """Worker entry point: build a fresh SchedDFR per call so loky workers
    do not need to pickle the parent's instance."""
    local = SchedDFR(down_sample_ratio, max_compression)
    return list(local.optimal_down_sample(arr_T_C).encoding_lengths)


class CoolManager:
    def __init__(self,
                 down_sample_ratio: float = 2.0,
                 max_compression: int = 4,
                 n_jobs: int = 1):
        self.down_sample_ratio = float(down_sample_ratio)
        self.max_compression = int(max_compression)
        self.n_jobs = int(n_jobs)
        self._dfr = SchedDFR(self.down_sample_ratio, self.max_compression)
        # Pool is created lazily so the manager stays deepcopy-friendly until
        # first use. Lightning's save_hyperparameters deepcopies __init__ args
        # of CoolMeltWrapper, and joblib.Parallel contains thread locks that
        # break the copy. Lazy creation keeps the object picklable at
        # construction time.
        self._pool: Optional[Parallel] = None

    def _get_pool(self) -> Optional[Parallel]:
        if self.n_jobs == 1:
            return None
        if self._pool is None:
            self._pool = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)
        return self._pool

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the per-item DP-optimal downsample to a batched tensor.

        Args:
            x: float tensor of shape ``[B, C, T]``.

        Returns:
            Tensor of the same shape, with each (DP-chosen) segment along T
            replaced by its per-channel mean broadcast back to its length.
        """
        with torch.no_grad():
            x_np = x.detach().to(torch.float32).cpu().numpy()
        B = x_np.shape[0]
        items = [np.ascontiguousarray(x_np[b].T) for b in range(B)]
        pool = self._get_pool()
        if pool is None:
            lengths_per_item: List[List[int]] = [
                list(self._dfr.optimal_down_sample(arr).encoding_lengths)
                for arr in items
            ]
        else:
            lengths_per_item = pool(
                delayed(_dp_one)(arr, self.down_sample_ratio, self.max_compression)
                for arr in items
            )
        return _apply_mean_broadcast(x, lengths_per_item)
