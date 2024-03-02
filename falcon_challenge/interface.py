import abc

import numpy as np

class BCIDecoder:
    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        r"""
            neural_observations: array of shape (n_channels), 10ms binned spike counts
            TODO ideally the action spaces are extracted from task config specified in main package
        """
        pass

    @abc.abstractmethod
    def on_trial_end(self):
        # Optional hook available in H2 (handwriting) to allow periodic test-time adaptation
        pass