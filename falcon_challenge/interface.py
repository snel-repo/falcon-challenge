import abc
from typing import List
from pathlib import Path
import numpy as np

class BCIDecoder:

    r"""
        To accelerate evaluation, we recommend implementing a decoder that can support batched prediction.
        The evaluator reads this batch size to determine how much data to serve at once (will serve 0s to fill to batch size if needed)
        Currently the evaluator batches different session predictions, not different trials within a session; this prevents disruption of neural history.
        User/decoder is currently responsible for setting a batch size that does not OOM.
    """
    batch_size: int = 1

    @abc.abstractmethod
    def reset(self, dataset_tags: List[str] = [""]):
        r"""
            Denote the specific session that is being evaluated is denoted through this function.
            Called when at least one datafile in batch changes.
            When called, dataset_tags will be of length batch_size, with active data hashes as returned from hash_dataset(datafiles).
            
        """
        pass

    @abc.abstractmethod
    def predict(self, neural_observations: np.ndarray) -> np.ndarray:
        r"""
            neural_observations: array of shape (batch_size, n_channels), binned spike counts
        """
        pass

    def observe(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (batch_size, n_channels), binned spike counts
            - for timestamps where we don't want predictions but neural data may be informative (start of trial)
            - provided only if all trials in batch do not need predictions
        """
        self.predict(neural_observations)

    @abc.abstractmethod
    def on_done(self, dones: np.ndarray):
        # Optional hook available in H2 (handwriting) to allow periodic test-time adaptation
        # dones will be one for the trials that just ended
        pass

    def set_batch_size(self, batch_size: int):
        r"""
            May be called by evaluator, but currently this path isn't implemented. Aim for a static batch size
        """
        self.batch_size = batch_size