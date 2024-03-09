r"""
    Load an sklearn decoder.
    To train, for example:
    `python decoder_demos/sklearn_decoder.py --training_dir data/h1/train --calibration_dir data/h1/test_short --mode all`
    To evaluate, for example:
    `python decode_submit.py --evaluation remote/local`
"""
from typing import List
from pathlib import Path
import numpy as np
import torch
import pytorch_lightning as pl
from einops import rearrange

from hydra import compose, initialize_config_module

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.interface import BCIDecoder

from context_general_bci.utils import suppress_default_registry
suppress_default_registry()
from context_general_bci.config import RootConfig, propagate_config, DataKey, MetaKey
from context_general_bci.dataset import DataAttrs, ContextAttrs
from context_general_bci.subjects import SubjectName
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
            model_cfg_stem: str,
            zscore_path: str,
        ):
        r"""
            Loading NDT2 requires both weights and model config. Weight loading through a checkpoint is standard.
            Model config is typically stored on wandb, but this is not portable enough. Instead, directly reference the model config file.
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
                version_base="1.3",
            )
        except:
            print('Hydra Initialize failed, assuming this is not the first decoder.')
        exp_stem, proper_stem = model_cfg_stem.split('/')[:-1], model_cfg_stem.split('/')[-1]
        exp_stem = '/'.join(exp_stem)
        override_path = f"+exp/{exp_stem}={proper_stem}"
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
        data_attrs = DataAttrs.from_config(cfg.dataset, context=ContextAttrs(**context_idx))
        cfg.model.task.decode_normalizer = zscore_path
        model = load_from_checkpoint(model_ckpt_path, cfg=cfg.model, data_attrs=data_attrs)
        model = transfer_model(model, cfg.model, data_attrs)
        self.model = model.to('cuda:0')
        self.model.eval()

        assert task_config.bin_size_ms == cfg.dataset.bin_size_ms, "Bin size mismatch, transform not implemented."
        self.observation_buffer = torch.zeros((cfg.dataset.max_length_ms // task_config.bin_size_ms, task_config.n_channels), dtype=torch.uint8, device='cuda:0')

    def format_dataset_tag(self, dataset_stem: str):
        return FalconContextInfo.get_id(self.subject, ExperimentalTask.falcon, FalconContextInfo.get_alias(self.subject, dataset_stem))

    def reset(self, dataset: Path = ""):
        dataset_tag =  self.get_file_tag(dataset) # e.g. stem including _set_N suffix
        self.set_steps = 0
        self.observation_buffer.zero_()
        self.meta_key = torch.tensor([self.model.data_attrs.context.session.index(self.format_dataset_tag(dataset_tag))], device='cuda:0')

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), binned spike counts
        """
        self.set_steps += 1
        self.observation_buffer = torch.roll(self.observation_buffer, -1, dims=0)
        self.observation_buffer[-1] = torch.as_tensor(neural_observations, dtype=torch.uint8, device='cuda:0')
        decoder_in = rearrange(self.observation_buffer[-self.set_steps:], 't c -> 1 t c 1')
        out = self.model(decoder_in, self.meta_key) # Remove batch dim
        return out[0].cpu().numpy()

if __name__ == "__main__":
    print(f"No train/calibration capabilities in {__file__}, use `context_general_bci` codebase.")