import numpy as np

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder

class RandomDecoder(BCIDecoder):
    r"""
        Proposed flow:
            - Determine the task and find the action space (defining a Gym env, use RLLib, TensorDict?)
            - Sample a random action
    """
    def __init__(self, task_config: FalconConfig):
        self._task_config = task_config

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), binned spike counts
        """
        return np.random.rand(self._task_config.out_dim)