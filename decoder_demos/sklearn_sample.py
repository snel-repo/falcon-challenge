r"""
    Sample ridge regression decoder for the Falcon Challenge.
"""

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

import os
print(os.getcwd())
print(os.listdir())

from sklearn_decoder import SKLearnDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=False, default='./local_data/sklearn_FalconTask.h1.pkl'
    )
    parser.add_argument(
        '--phase', type=str, required=False, default='h1'
    )
    args = parser.parse_args()

    dataset = args.phase.split('_')[0]
    phase = args.phase
    task = getattr(FalconTask, dataset)
    config = FalconConfig(
        task=task,
        n_channels=176,
    )

    decoder = SKLearnDecoder(task_config=config, model_path=args.model_path)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        phase=phase)
    evaluator.evaluate(decoder)


if __name__ == "__main__":
    main()