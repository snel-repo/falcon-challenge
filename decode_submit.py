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

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from falcon_challenge.interface import BCIDecoder

from decoder_demos.random_decoder import RandomDecoder
from decoder_demos.rnn_decoder import SimpleRNNDecoder, MyConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=False
    )
    parser.add_argument(
        '--phase', type=str, required=False, default='h1_short'
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
    dataset = args.phase.split('_')[0]
    config = FalconConfig(
        task=getattr(FalconTask, dataset),
        n_channels=192,
    )

    decoder = RandomDecoder(task_config=config)

    # decoder_cfg = MyConfig('added_to_docker_cfg.cfg')
    # decoder_cfg.model_path = args.model_path
    # decoder = SimpleRNNDecoder(task_config=config, decoder_cfg=decoder_cfg)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        phase=args.phase)
    evaluator.evaluate(decoder)


if __name__ == "__main__":
    main()