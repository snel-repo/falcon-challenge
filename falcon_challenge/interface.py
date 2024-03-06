import abc
from pathlib import Path
import numpy as np

class BCIDecoder:
    @abc.abstractmethod
    def reset(self, dataset_tag: str = ""):
        pass

    @staticmethod
    def get_file_tag(filepath: Path):
        pieces = filepath.stem.split('_')
        if pieces[-1] in ['minival', 'calibration', 'eval', 'full']:
            return '_'.join(pieces[:-1])
        return filepath.stem

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