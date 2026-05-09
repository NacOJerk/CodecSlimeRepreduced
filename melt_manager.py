from typing import List

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
        self.p_tgt = p_tgt
        
        self.s_p = s_p
        self.concentration_control = concentration_control
        
        if not (0 <= skip_prob <= 1):
            raise ValueError("Skip probability must be between 0 and 1")
        self.skip_prob = skip_prob
        self.epsilon = epsilon
        
        self._current_training_step: int = 0