from abc import ABC

import numpy as np

class BCIDecoder:
    @ABC.abstractmethod
    def reset(self):
        pass

    @ABC.abstractmethod
    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), 10ms binned spike counts
            TODO ideally the action spaces are extracted from task config specified in main package
        """
        pass