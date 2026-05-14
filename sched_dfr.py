import numpy as np
from dataclasses import dataclass
import torch
from typing import List

@dataclass
class EncodedData:
    encoded_data: np.ndarray
    encoding_lengths: List[int]

class SchedDFR:
    def __init__(self, down_sample_ratio: float, max_compression: int):
        self.down_sample_ratio = down_sample_ratio
        self.max_compression = max_compression

    @staticmethod
    def _difference_loss(raw_output: np.ndarray, encoding_end: int, encoding_length: int) -> float:
        start_idx = encoding_end - encoding_length + 1
        end_idx = encoding_end + 1
        segment = raw_output[start_idx:end_idx]
        segment_mean = np.mean(segment, axis=0)
        l2_norms = np.linalg.norm(segment - segment_mean, ord=2, axis=-1)
        return float(np.sum(l2_norms))

    @staticmethod
    def down_sample(raw_output: np.ndarray, encoding_lengths: List[int]) -> np.ndarray:
        compressed_features = []
        current_start = 0
        for s in encoding_lengths:
            current_end = current_start + s
            segment = raw_output[current_start:current_end]
            segment_mean = np.mean(segment, axis=0)
            compressed_features.append(segment_mean)
            current_start = current_end        

        return np.array(compressed_features)
    
    @staticmethod
    def up_sample(encoded_data: np.ndarray, encoding_lengths: List[int]) -> np.ndarray:
        return np.repeat(encoded_data, encoding_lengths, axis=0)

    @staticmethod
    def _precompute_cost(raw_output: np.ndarray, U: int) -> np.ndarray:
        """Build L[j, s] = sum of per-frame L2 distance from segment mean,
        for the segment raw_output[j-s:j], for j in 1..T and s in 1..U.

        Same math as `_difference_loss(raw_output, j-1, s)` for every (j, s)
        but computed once with a vectorized sliding-window pass. s=1 is
        always 0 (a one-frame segment equals its own mean).
        """
        t = raw_output.shape[0]
        L = np.zeros((t + 1, U + 1), dtype=np.float64)
        for s in range(2, U + 1):
            n_segs = t - s + 1
            if n_segs <= 0:
                continue
            # sliding_window_view(axis=0) returns shape (n_segs, D, s) so the
            # window dim lives last; transpose to (n_segs, s, D) to match the
            # original [s, D] per-segment layout.
            windows = np.lib.stride_tricks.sliding_window_view(
                raw_output, window_shape=s, axis=0
            ).transpose(0, 2, 1)
            means = windows.mean(axis=1, keepdims=True)
            l2 = np.linalg.norm(windows - means, axis=2)  # (n_segs, s)
            L[s:t + 1, s] = l2.sum(axis=1)
        return L

    def optimal_down_sample(self, raw_output: np.ndarray) -> EncodedData:
        with torch.no_grad():
            t, _ = raw_output.shape
            t_tag = int(np.ceil(t / self.down_sample_ratio))
            U = self.max_compression

            # Precompute per-segment cost once; the DP becomes a pure-Python
            # arithmetic loop over the L lookup table. ~100x speedup over the
            # original which recomputed mean+norm inside the innermost loop.
            L = SchedDFR._precompute_cost(raw_output, U)

            optimal_distance = np.full((t + 1, t_tag + 1), -np.inf)
            chosen_s_array = np.zeros((t + 1, t_tag + 1), dtype=np.int32)
            optimal_distance[0, 0] = 0.0
            for i in range(1, t_tag + 1):
                for j in range(i, t + 1):
                    max_s = min(j - i + 1, U)
                    best_val = -np.inf
                    best_s = 0
                    for s in range(1, max_s + 1):
                        val = optimal_distance[j - s, i - 1] - L[j, s]
                        if val >= best_val:
                            best_val = val
                            best_s = s
                    optimal_distance[j, i] = best_val
                    chosen_s_array[j, i] = best_s

            final_s_choices = []
            current_j, current_i = t, t_tag
            while current_j > 0:
                s = int(chosen_s_array[current_j, current_i])
                final_s_choices.append(s)
                current_j -= s
                current_i -= 1
            final_s_choices.reverse()

            assert sum(final_s_choices) == t, f"Invalid encoding ({final_s_choices} vs t={t})"

        return EncodedData(SchedDFR.down_sample(raw_output, final_s_choices), final_s_choices)
    
    def up_sample_encoded(self, encoded_output: EncodedData) -> np.ndarray:
        return SchedDFR.up_sample(encoded_output.encoded_data, encoded_output.encoding_lengths)
