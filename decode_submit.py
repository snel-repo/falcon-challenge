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
# from omegaconf import DictConfig

from falcon_challenge.config import FalconConfig
from falcon_challenge.evaluator import FalconEvaluator
from falcon_challenge.interface import BCIDecoder

class RandomDecoder(BCIDecoder):
    r"""
        TODO: Do we want a streaming API, or to provide all neural data at once? (enforce causality at cost of increased inference?)
        TODO: Should inference be single trial or batched?

        Proposed flow:
            - Determine the task and find the action space (defining a Gym env, use RLLib, TensorDict?)
            - Sample a random action
    """
    def __init__(self, task_config: FalconConfig):
        self._task_config = task_config

    def predict(self, neural_observations: np.ndarray):
        r"""
            neural_observations: array of shape (n_channels), 10ms binned spike counts
            TODO ideally the action spaces are extracted from task config specified in main package
        """
        if self._task_config.task == "falcon_h1_7d":
            return np.random.rand(7)
        elif self._task_config.task == "falcon_h2":
            return np.random.rand(28) # Or whatever the action space is
        elif self._task_config.task == "falcon_m1":
            return np.random.rand(2)
        elif self._task_config.task == "falcon_m2":
            return np.random.rand(2)
        else:
            raise ValueError(f"Unknown task {self._task_config.task}")

class MyConfig:
    model_path: str = ""

class SimpleRNNDecoder(BCIDecoder):
    def __init__(self, task_config: FalconConfig, decoder_cfg: MyConfig):
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

    # benchmark_config_path = os.environ["CHALLENGE_CONFIG_FILE"]

    # TODO resolve how we specifically get the task config - it shouldn't be complex for us, so we likely don't need this plugin logic, could even just maintain in Dockefiles
    # register_hydra_plugin(HabitatChallengeConfigPlugin) # TODO what to do with this.
    # config = get_config( # TODO what to do with this?
    #     benchmark_config_path,
    #     overrides=[
    #         "habitat/task/actions=" + args.action_space,
    #     ],
    # )
    config = FalconConfig(
        task="falcon_h1_7d",
        n_channels=192,
    )

    decoder = RandomDecoder(task_config=config)

    # decoder_cfg = MyConfig('added_to_docker_cfg.cfg')
    # decoder_cfg.model_path = args.model_path
    # decoder = SimpleRNNDecoder(task_config=config, decoder_cfg=decoder_cfg)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        phase='h1_short')
    evaluator.evaluate(decoder)


if __name__ == "__main__":
    main()