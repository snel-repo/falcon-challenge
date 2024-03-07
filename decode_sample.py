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
from decoder_demos.sklearn_decoder import SKLearnDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=False
    )
    parser.add_argument(
        '--phase', type=str, required=False, default='h1'
    )
    args = parser.parse_args()

    dataset = args.phase.split('_')[0]
    phase = f'{phase}_short' # TODO remove once terms are removed
    task = getattr(FalconTask, dataset)
    config = FalconConfig(
        task=task,
        n_channels=176,
    )

    # decoder = RandomDecoder(task_config=config)
    decoder = SKLearnDecoder(task_config=config, model_path=f'data/sklearn_{task}.pkl')
    # decoder_cfg = MyConfig('added_to_docker_cfg.cfg')
    # decoder_cfg.model_path = args.model_path
    # decoder = SimpleRNNDecoder(task_config=config, decoder_cfg=decoder_cfg)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        phase=args.phase)
    evaluator.evaluate(decoder)


if __name__ == "__main__":
    main()