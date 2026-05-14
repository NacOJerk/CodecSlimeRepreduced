"""CoolManager: ScheDFR-driven downsampler used by the Cool fine-tuning stage.

The DP scheduler in ``sched_dfr.SchedDFR`` is numpy-only and operates on a
single ``[T, C]`` array. ``CoolManager.compress`` bridges it to the batched
``[B, C, T]`` tensor expected by ``CoolMeltWrapper`` and reuses
``melt_manager._apply_mean_broadcast`` for the differentiable per-segment
mean apply, matching the paper recipe: the DP runs inside the forward to
pick the schedule, gradients flow through the mean-broadcast apply only.
"""
from typing import List

import numpy as np
import torch

from sched_dfr import SchedDFR
from melt_manager import _apply_mean_broadcast


class CoolManager:
    def __init__(self,
                 down_sample_ratio: float = 2.0,
                 max_compression: int = 4):
        self.down_sample_ratio = float(down_sample_ratio)
        self.max_compression = int(max_compression)
        self._dfr = SchedDFR(self.down_sample_ratio, self.max_compression)

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
        lengths_per_item: List[List[int]] = []
        for b in range(B):
            arr = np.ascontiguousarray(x_np[b].T)
            encoded = self._dfr.optimal_down_sample(arr)
            lengths_per_item.append(list(encoded.encoding_lengths))
        return _apply_mean_broadcast(x, lengths_per_item)
