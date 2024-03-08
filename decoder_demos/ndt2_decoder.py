r"""
    Load an sklearn decoder.
    To train, for example:
    `python decoder_demos/sklearn_decoder.py --training_dir data/h1/train --calibration_dir data/h1/test_short --mode all`
    To evaluate, for example:
    `python decode_submit.py --evaluation remote/local`
"""
from typing import List
import os
import numpy as np
from pathlib import Path
import pytorch_lightning as pl

from hydra import compose, initialize_config_module

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.dataloaders import load_nwb
from falcon_challenge.interface import BCIDecoder

from context_general_bci.config import RootConfig, propagate_config, DataKey, MetaKey
from context_general_bci.dataset import DataAttrs, ContextAttrs
from context_general_bci.subjects import SubjectName
from context_general_bci.utils import suppress_default_registry
suppress_default_registry()
from context_general_bci.contexts.context_registry import context_registry
from context_general_bci.contexts.context_info import FalconContextInfo, ExperimentalTask
from context_general_bci.model import load_from_checkpoint
from context_general_bci.model_slim import transfer_model

def format_array_name(subject: str):
    return f'FALCON{subject}-M1'

class NDT2Decoder(BCIDecoder):
    r"""
        Load an NDT2 decoder, prepared in:
        https://github.com/joel99/context_general_bci

        # TODO KV cache - difficult without rotary embeddings
    """

    def __init__(
            self,
            task_config: FalconConfig,
            model_ckpt_path: str,
            model_cfg_path: str,
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

        try:
            initialize_config_module(
                config_module="context_general_bci.config",
                job_name="falcon",
            )
        except:
            print('Hydra Initialize failed, assuming this is not the first decoder.')
        override_path = f"+exp={Path(model_cfg_path).stem}"
        cfg: RootConfig = compose(config_name="config", overrides=[override_path])

        propagate_config(cfg)
        pl.seed_everything(seed=cfg.seed)

        self.subject = getattr(SubjectName, f'falcon_{task_config.task.name}')
        context_idx = {
            MetaKey.array.name: [format_array_name(self.subject)],
            MetaKey.subject.name: [self.subject],
            MetaKey.session.name: sorted([
                self.format_dataset_tag(handle) for handle in task_config.dataset_handles
            ]),
            MetaKey.task.name: [ExperimentalTask.falcon],
        }
        data_attrs = DataAttrs.from_config(cfg, context=ContextAttrs(**context_idx))
        model = load_from_checkpoint(model_ckpt_path, cfg=cfg.model, data_attrs=data_attrs)
        self.model = model.to('cuda:0')
        self.model.eval()

    def format_dataset_tag(self, dataset_stem: str):
        return FalconContextInfo.get_id(self.subject, ExperimentalTask.falcon, FalconContextInfo.get_alias(self.subject, dataset_stem))

    def reset(self, dataset: Path = ""):
        dataset_tag =  self.get_file_tag(dataset) # e.g. stem including _set_N suffix
        self.meta_key = self.format_dataset_tag(dataset_tag)
        self.observation_buffer = np.zeros_like(self.observation_buffer)

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), binned spike counts
        """
        self.observation_buffer = np.roll(self.observation_buffer, -1, axis=0)
        self.observation_buffer[-1] = neural_observations
        decoder_in = self.observation_buffer[::-1].copy().flatten().reshape(1, -1)
        # TODO what other critical ingredients? Format all correctly
        batch = {
            DataKey.spikes: decoder_in,
            MetaKey.session: [self.meta_key],
        }
        out = self.model(batch, last_step_only=True)[0]
        return out

if __name__ == "__main__":
    print(f"No train/calibration capabilities in {__file__}, use `context_general_bci` codebase.")