r"""
    Load an sklearn decoder.
    To train, for example:
    `python decoder_demos/sklearn_decoder.py --training_dir data/h1/train --calibration_dir data/h1/test_short --mode all`
    To evaluate, for example:
    `python decode_submit.py --evaluation remote/local`
"""
from typing import List
import argparse
import pickle
import numpy as np
from pathlib import Path

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.dataloaders import load_nwb
from falcon_challenge.interface import BCIDecoder

from context_general_bci.config import RootConfig, propagate_config
from context_general_bci.utils import suppress_default_registry
suppress_default_registry()
from context_general_bci.contexts.context_registry import context_registry
from context_general_bci.contexts.context_info import FalconContextInfo, ExperimentalTask
from context_general_bci.model_slim import load_from_checkpoint # TODO

class NDT2Decoder(BCIDecoder):
    r"""
        Load an NDT2 decoder, prepared in:
        https://github.com/joel99/context_general_bci
    """

    def __init__(
            self,
            task_config: FalconConfig,
            model_ckpt_path: str,
            model_cfg_file: str,
        ):
        r"""
            Loading NDT2 requires both weights and model config. Weight loading through a checkpoint is standard.
            Model config is typically stored on wandb, but this is not portable enough. Instead, provide the model config file.
        """
        self._task_config = task_config
        context_registry.register([
            *FalconContextInfo.build_from_dir(
                f'./data/{task_config.task.name}/test',
                task=ExperimentalTask.falcon,
                suffix='eval')])

        # TODO hydra initialize / compose the model config
        try:
            initialize(config_path=(Path(CGB_DIR) / config).relative_to(os.getcwd()), job_name="online_bci_ft", )
        except:
            print('Hydra Initialize failed, assuming this is not the first decoder.')
        config_pieces = Path(config_path).relative_to(Path(CGB_DIR).resolve() / 'config' / 'exp').parts
        override_path = f"+exp/{'/'.join(config_pieces[:-1])}={config_pieces[-1].split('.')[0]}"
        cfg: RootConfig = compose(config_name="config", overrides=[override_path])

        propagate_config(cfg)
        pl.seed_everything(seed=cfg.seed)

        # TODO subject spec somewhere, somehow
        dataset = SpikingDataset(cfg.dataset, override_preprocess_path=True)
        dataset.build_context_index()
        data_attrs = dataset.get_data_attrs()

        # TODO seed everything pl
        model = load_from_checkpoint(model_ckpt_path, cfg=cfg.model, data_attrs=data_attrs)
        model = model.to('cuda:0')
        model.eval()

    def reset(self, dataset: Path = ""):
        # TODO proper serving of MetaKey context
        # TODO inference API, KV cache
        dataset_tag =  self.get_file_tag(dataset)
        self.meta_key = f"{dataset_tag}_meta"
        self.observation_buffer = np.zeros_like(self.observation_buffer)

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), binned spike counts
        """
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = neural_observations
        # TODO batchify
        decoder_in = self.observation_buffer[::-1].copy().flatten().reshape(1, -1)
        out = self.clf.predict(decoder_in)[0]
        return out

if __name__ == "__main__":
    print(f"No train/calibration capabilities in {__file__}, use `context_general_bci` codebase.")