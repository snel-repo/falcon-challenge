import abc
from pathlib import Path
import numpy as np

class BCIDecoder:

    batch_size: int = 1

    @abc.abstractmethod
    def reset(self, dataset_tag: str = ""):
        pass

    # @staticmethod
    # def get_file_tag(filepath: Path, is_dandi=False):
    #     if filepath.name.endswith('behavior+ecephys') # clearly dandi
    #     # if is_dandi:
    #         # dandi-style sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb  
    #         return filepath.stem.split('_')[-2].split('-')[-1]
    #     else:
    #         # other style: L_20120924_held_in_eval.nwb
    #         pieces = filepath.stem.split('_')
    #         if pieces[-1] in ['minival', 'calib', 'calibration', 'eval', 'full']:
    #             return '_'.join(pieces[:-1])
    #     return filepath.stem

    @abc.abstractmethod
    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        r"""
            neural_observations: array of shape (n_channels), binned spike counts
        """
        pass

    def observe(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), binned spike counts
            - for timestamps where we don't want predictions but neural data may be informative (start of trial)
        """
        self.predict(neural_observations)

    @abc.abstractmethod
    def on_trial_end(self):
        # Optional hook available in H2 (handwriting) to allow periodic test-time adaptation
        pass

    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size