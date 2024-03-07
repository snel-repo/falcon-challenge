r"""
    Sample ridge regression decoder for the Falcon Challenge.
"""

import argparse

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator
from falcon_challenge.interface import BCIDecoder

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
    phase = f'{args.phase}_short' # TODO remove once terms are removed
    task = getattr(FalconTask, dataset)
    config = FalconConfig(
        task=task,
        n_channels=176,
    )

    decoder = SKLearnDecoder(task_config=config, model_path=args.model_path)
    # decoder = RandomDecoder(task_config=config)
    # decoder_cfg = MyConfig('added_to_docker_cfg.cfg')
    # decoder_cfg.model_path = args.model_path
    # decoder = SimpleRNNDecoder(task_config=config, decoder_cfg=decoder_cfg)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        phase=phase)
    evaluator.evaluate(decoder)


if __name__ == "__main__":
    main()