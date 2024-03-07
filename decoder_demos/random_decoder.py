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
        if self._task_config.task == FalconTask.h1:
            return np.random.rand(7)
        elif self._task_config.task == FalconTask.h2:
            return np.random.rand(28) # Or whatever the action space is
        elif self._task_config.task == FalconTask.m1:
            return np.random.rand(2)
        elif self._task_config.task == FalconTask.m2:
            return np.random.rand(2)
        else:
            raise ValueError(f"Unknown task {self._task_config.task}")