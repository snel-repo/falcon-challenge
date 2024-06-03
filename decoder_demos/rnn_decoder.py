import numpy as np
# try import torch or warn
try:
    import torch
except ImportError:
    print("Torch not found, please install PyTorch to use this decoder.")
    torch = None

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder


class MyConfig:
    model_path: str = ""

class SimpleRNNDecoder(BCIDecoder):
    def __init__(self, task_config: FalconConfig, decoder_cfg: MyConfig):
        self._task_config = task_config
        self.dnn = torch.load(decoder_cfg.model_path)
        # define which models to load for each file ID (dict of ID -> model path)

    def reset(self, dataset_tag: str = ""):
        # called when the file changes; get the file ID  
        self.neural_history = np.zeros((0, self._task_config.n_channels))
        # load corresponding model for this file ID

    def predict(self, neural_observations: np.ndarray):
        self.neural_history = np.concatenate([self.neural_history, neural_observations[np.newaxis, :]], axis=0)
        return self.dnn(self.neural_history)
        # model inference 
        # apply decoder 
        # return prediction for that timestep 
