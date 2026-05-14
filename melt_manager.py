from typing import List, Optional

import numpy as np
import torch

USE_PAPER_D_ENFORCE = False
INCLUDE_SKIP_AS_STEP = True


class MeltManager:
    def __init__(self,
                 max_compression: int = 4,
                 p_tgt: List[float] = [0.1, 0.45, 0.25, 0.2],
                 s_p: int = 100000,
                 concentration_control: float = 30.0,
                 skip_prob: float = 0.5,
                 epsilon: float = 1e-6,
                 use_paper_d_enforce: Optional[bool] = None):
        """Random-rate downsampling curriculum sampler for the CodecSlime Melt stage.

        Args:
            max_compression: U, the largest per-segment downsampling factor.
            p_tgt: target proportions over rates {1..U}. Must sum to 1 (within float tolerance).
            s_p: number of training steps to reach p_tgt (curriculum length).
            concentration_control: Dirichlet concentration scale c.
            skip_prob: probability of no downsampling on a given step.
            epsilon: floor for clipped proportions before forming the Dirichlet alpha.
            use_paper_d_enforce: if None (default), use module-level
                ``USE_PAPER_D_ENFORCE``. If True, enforce sum-to-1 on the *last*
                rate (paper-literal d-vector formula, starts at max randomness).
                If False, enforce on the *first* rate (our fix; starts at min
                randomness and curriculum-anneals to ``p_tgt``).
        """
        self.max_compression = max_compression

        if len(p_tgt) != self.max_compression:
            raise ValueError("Target compression probabilities length doesn't match max compression")
        if abs(sum(p_tgt) - 1.0) > 1e-6:
            raise ValueError("Target compression probabilities must sum to 1")
        self.p_tgt = np.array(p_tgt)

        self.s_p = s_p
        self.concentration_control = concentration_control

        if not (0 <= skip_prob <= 1):
            raise ValueError("Skip probability must be between 0 and 1")
        self.skip_prob = skip_prob
        self.epsilon = epsilon
        self.use_paper_d_enforce = (
            USE_PAPER_D_ENFORCE if use_paper_d_enforce is None else bool(use_paper_d_enforce)
        )

        self._current_training_step: int = 0

    def generate_segment_lenght_propotations(self, increase_step: bool = True) -> Optional[np.ndarray]:
        if np.random.uniform() < self.skip_prob:
            if INCLUDE_SKIP_AS_STEP and increase_step:
                self._current_training_step += 1
            return None

        training_progress = min(self._current_training_step / self.s_p, 1)
        current_prop = training_progress * self.p_tgt
        if self.use_paper_d_enforce:
            current_prop[-1] = 1 - np.sum(current_prop[:-1])
        else:
            current_prop[0] = 1 - np.sum(current_prop[1:])
        current_prop = np.maximum(current_prop, self.epsilon)

        alpha = current_prop * self.concentration_control / (max(1, self._current_training_step / self.s_p) ** 2.5)
        proportion = np.random.dirichlet(alpha)

        if increase_step:
            self._current_training_step += 1

        return proportion

    def _sample_segments_to_T(self, proportion: np.ndarray, T: int) -> List[int]:
        """Sample segment lengths from `proportion` over rates {1..U} until they tile T.

        The final segment is truncated to exactly fit T.
        """
        elements = np.arange(1, self.max_compression + 1)
        total = 0
        segments: List[int] = []
        while total < T:
            s = int(np.random.choice(elements, p=proportion))
            if total + s > T:
                segments.append(T - total)
                break
            total += s
            segments.append(s)
        return segments

    def melt(self, x: torch.Tensor, step: int) -> torch.Tensor:
        """Gradient-preserving Melt apply.

        Args:
            x: float tensor of shape [B, C, T].
            step: the current curriculum step (use Lightning's `global_step` so the
                curriculum is checkpoint-restored on resume).

        Returns:
            A tensor of the same shape as `x`, with each segment along T replaced by
            its per-channel mean (broadcast back to the segment's length). Returns
            `x` unchanged on the skip-prob bypass.
        """
        self._current_training_step = step
        proportion = self.generate_segment_lenght_propotations(increase_step=False)
        if proportion is None:
            return x
        B, _, T = x.shape
        lengths_per_item = [self._sample_segments_to_T(proportion, T) for _ in range(B)]
        return _apply_mean_broadcast(x, lengths_per_item)


def _apply_mean_broadcast(x: torch.Tensor, lengths_per_item: List[List[int]]) -> torch.Tensor:
    """Per-segment mean along T, broadcast back to length T via repeat_interleave.

    Args:
        x: [B, C, T] tensor.
        lengths_per_item: B lists of segment lengths, each summing to T.

    Both `mean` and `repeat_interleave` are differentiable in torch, so gradients
    flow back to `x`. Cool's future manager calls this same helper after computing
    its own (DP-based) `lengths_per_item`.
    """
    out = []
    for b in range(x.shape[0]):
        seg = lengths_per_item[b]
        pieces = torch.split(x[b], seg, dim=-1)
        means = torch.stack([p.mean(dim=-1) for p in pieces], dim=-1)
        lengths = torch.tensor(seg, device=x.device)
        out.append(means.repeat_interleave(lengths, dim=-1))
    return torch.stack(out, dim=0)
