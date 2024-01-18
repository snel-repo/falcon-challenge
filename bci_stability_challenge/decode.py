r"""
    Skeleton class for decoding evaluation under Docker.
    USER IS SUPPOSED TO MODIFY THIS TO IMPORT THEIR DECODE STRATEGY
    Challenge infrastructure according to https://github.com/facebookresearch/habitat-challenge

    Proposed flow: Dockerfile calls this decode on each task.

    Open:
    - Separate eval-fixed abstractions (e.g. the evaluator, interface class) from user-examples (e.g. RandomDecoder)
    - Where do we pipe private data?
"""

import argparse
import os

import numpy as np
import torch

# from config import HabitatChallengeConfigPlugin
from omegaconf import DictConfig

from bci_stability_challenge.config import DecodeConfig
from bci_stability_challenge.interface import BCIDecoder
from bci_stability_challenge.evaluator import Evaluator

class RandomDecoder(BCIDecoder):
    r"""
        TODO: Do we want a streaming API, or to provide all neural data at once? (enforce causality at cost of increased inference?)
        TODO: Should inference be single trial or batched?

        Proposed flow:
            - Determine the task and find the action space (defining a Gym env, use RLLib, TensorDict?)
            - Sample a random action
    """
    def __init__(self, task_config: DecodeConfig):
        self._task_config = task_config

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), 10ms binned spike counts
            TODO ideally the action spaces are extracted from task config specified in main package
        """
        if self._task_config.task == "stability_23_human_7dof":
            return {
                "action": np.random.rand(7),
            }
        elif self._task_config.task == "stability_23_human_handwriting":
            return {
                "action": np.random.rand(28), # Or whatever the action space is
            }
        # etc


# TODO for user to define
class MyConfig:
    model_path: str = ""

class SimpleRNNDecoder(BCIDecoder):
    def __init__(self, task_config: DecodeConfig, decoder_cfg: MyConfig):
        self._task_config = task_config
        self.dnn = torch.load(decoder_cfg.model_path)

    def reset(self):
        self.neural_history = np.zeros((0, self._task_config.n_channels))

    def predict(self, neural_observations: np.ndarray):
        self.neural_history = np.concatenate([self.neural_history, neural_observations[np.newaxis, :]], axis=0)
        return self.dnn(self.neural_history)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=False
    )
    args = parser.parse_args()

    benchmark_config_path = os.environ["CHALLENGE_CONFIG_FILE"]

    # TODO resolve how we specifically get the task config - it shouldn't be complex for us, so we likely don't need this plugin logic, could even just maintain in Dockefiles
    register_hydra_plugin(HabitatChallengeConfigPlugin) # TODO what to do with this.

    config = get_config( # TODO what to do with this?
        benchmark_config_path,
        overrides=[
            "habitat/task/actions=" + args.action_space,
        ],
    )

    decoder_cfg = MyConfig()
    decoder_cfg.model_path = args.model_path

    decoder = RandomDecoder(task_config=config)
    decoder = SimpleRNNDecoder(task_config=config, decoder_cfg=decoder_cfg)
    # TODO see Challenge/Benchmark implementation
    # https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/core/challenge.py
    # https://github.com/facebookresearch/habitat-lab/blob/main/habitat-lab/habitat/core/benchmark.py
    evaluator = Evaluator(eval_remote=args.evaluation == "remote")
    evaluator.evaluate(decoder)


if __name__ == "__main__":
    main()