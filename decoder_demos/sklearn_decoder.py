r"""
    Load an sklearn decoder.
"""
import pickle
import numpy as np

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder
from data_demos.filtering import apply_exponential_filter, NEURAL_TAU_MS

class SKLearnDecoder(BCIDecoder):
    r"""
        Load an sklearn decoder. Assumes the dimensionality is correct.
    """
    def __init__(self, task_config: FalconConfig, model_path: str):
        self._task_config = task_config
        with open(model_path, 'rb') as f:
            payload = pickle.load(f)
            assert payload['task'] == task_config.task
            self.clf = payload['decoder']
            self.history = payload['history'] + 1
            MAX_HISTORY = int(NEURAL_TAU_MS / 10) * 5 # bin size ms
            self.x_mean = payload['x_mean']
            self.x_std = payload['x_std']
            self.y_mean = payload['y_mean']
            self.y_std = payload['y_std']
            self.raw_history_buffer = np.zeros((MAX_HISTORY, task_config.n_channels))
            self.observation_buffer = np.zeros((self.history, task_config.n_channels))


    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), 10ms binned spike counts
        """
        self.raw_history_buffer = np.roll(self.raw_history_buffer, -1)
        self.raw_history_buffer[-1] = neural_observations
        smth_history = apply_exponential_filter(self.raw_history_buffer, NEURAL_TAU_MS)
        self.observation_buffer = np.roll(self.observation_buffer, -1)
        self.observation_buffer[-1] = (smth_history[-1] - self.x_mean) / self.x_std
        decoder_in = self.observation_buffer[::-1].copy().flatten().reshape(1, -1)
        out = self.clf.predict(decoder_in)[0]
        out = out * self.y_std + self.y_mean
        return out