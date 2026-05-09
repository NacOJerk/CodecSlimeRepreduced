import numpy as np
from dataclasses import dataclass
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
    def _down_sample(raw_output: np.ndarray, encoding_lengths: List[int]) -> np.ndarray:
        compressed_features = []
        current_start = 0
        for s in encoding_lengths:
            current_end = current_start + s
            segment = raw_output[current_start:current_end]
            segment_mean = np.mean(segment, axis=0)
            compressed_features.append(segment_mean)
            current_start = current_end        

        return np.array(compressed_features)

    def optimal_down_sample(self, raw_output: np.ndarray) -> EncodedData:
        t, _ = raw_output.shape
        t_tag = np.ceil(t / self.down_sample_ratio).astype(int)
        optiaml_distance = [[-np.inf for _ in range(t_tag + 1)] for _ in range(t + 1)]
        chosen_s_array = [[0 for _ in range(t_tag + 1)] for _ in range(t + 1)]
        optiaml_distance[0][0] = 0
        for i in range(1, t_tag + 1):
            for j in range(1, t + 1):
                max_so_far = -np.inf
                chosen_s = 0
                max_s = min(j - i + 1, self.max_compression)
                for s in range(1, max_s + 1):
                    diff = optiaml_distance[j - s][i - 1] - SchedDFR._difference_loss(raw_output, j - 1, s)
                    if diff >= max_so_far:
                        max_so_far = diff
                        chosen_s = s
                optiaml_distance[j][i] = max_so_far
                chosen_s_array[j][i] = chosen_s


        final_s_choices = []
        current_j = t
        current_i = t_tag
        while current_j > 0:
            final_s_choices.append(chosen_s_array[current_j][current_i])
            current_j -= chosen_s_array[current_j][current_i]
            current_i -= 1
        final_s_choices.reverse()

        assert sum(final_s_choices) == t, f"Invalid encoding ({final_s_choices} vs t={t})"
    
        return EncodedData(SchedDFR._down_sample(raw_output, final_s_choices), final_s_choices)
    
    def up_sample(self, encoded_output: EncodedData) -> np.ndarray:
        return np.repeat(encoded_output.encoded_data, encoded_output.encoding_lengths, axis=0)