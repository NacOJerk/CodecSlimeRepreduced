import numpy as np
from typing import List, Optional

from sched_dfr import SchedDFR

USE_PAPER_D_ENFORCE = False

class MeltManager:
    def __init__(self, 
                 max_compression: int = 4,
                 p_tgt: List[float] = [0.1, 0.45, 0.25, 0.2],
                 s_p: int = 1e5,
                 concentration_control: float = 30.0, 
                 skip_prob: float = 0.5,
                 epsilon: float = 1e-6):
        """
        Initializes the MeltManager with dynamic downsampling parameters.

        Args:
            max_compression (int): The maximum downsampling rate (U). 
                Defines the dimensionality of the rate vectors.
            p_tgt (List[float]): The target mix proportions (p_tgt) over the 
                available rates. Must sum to 1.0.
            s_p (int): The number of training steps (S_p) required to reach 
                the target proportion p_tgt. Typically 10^5 steps.
            concentration_control (float): Controls how sharply the Dirichlet 
                samples cluster (c). This value decays as training progresses.
            skip_prob (float): The probability (rho) of performing no 
                downsampling at all for a given step (typically 0.5).
            epsilon (float): A small constant value (epsilon) used to prevent 
                zero entries in the progress-weighted blend vector (d).
        """
        self.max_compression = max_compression
        
        if len(p_tgt) != self.max_compression:
            raise ValueError("Target compression probabilities length doesn't match max compression")
        if sum(p_tgt) != 1:
            raise ValueError("Target compression probabilities must sum to 1")
        self.p_tgt = np.array(p_tgt)
        
        self.s_p = s_p
        self.concentration_control = concentration_control
        
        if not (0 <= skip_prob <= 1):
            raise ValueError("Skip probability must be between 0 and 1")
        self.skip_prob = skip_prob
        self.epsilon = epsilon
        
        self._current_training_step: int = 0
    
    def generate_segment_lenght_propotations(self, increase_step=True) -> Optional[List[float]]:
        if np.random.uniform() < self.skip_prob:
            return
        
        training_progress = min(self._current_training_step / self.s_p, 1)
        current_prop = training_progress * self.p_tgt
        if USE_PAPER_D_ENFORCE:
            current_prop[-1] = 1 - np.sum(current_prop[:-1]) # The paper had this so the probablity of 4 is the most likely, this feels like a mistake 
        else:
            current_prop[0] = 1 - np.sum(current_prop[1:]) 
        current_prop = np.maximum(current_prop, self.epsilon)

        alpha = current_prop * self.concentration_control / (max(1, self._current_training_step / self.s_p)**2.5)
        propoption = np.random.dirichlet(alpha)

        if increase_step:
            self._current_training_step += 1

        return propoption
    
    def _generate_segment_from_propoption(self, propoption: List[float], size=None) -> int:
        elements = list(range(1, self.max_compression + 1))
        return np.random.choice(elements, size=size, p=propoption)

    def random_down_sample_with_propotion(self, raw: np.ndarray, propoption: List[float]) -> np.ndarray:
        total_entries = raw.shape[0]

        # This can probably be optimized
        total_so_far = 0
        segments = []
        while total_so_far < total_entries:
            segment = int(self._generate_segment_from_propoption(propoption))
            if total_so_far + segment > total_entries:
                segments.append(total_entries - total_so_far)
                break
            total_so_far += segment
            segments.append(segment)
        
        np.random.shuffle(segments)
        return SchedDFR.up_sample(SchedDFR.down_sample(raw, segments), segments)