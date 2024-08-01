r"""
    Sample ridge regression decoder for the Falcon Challenge.
"""

import argparse
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from decoder_demos.sklearn_decoder import SKLearnDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"]
    )
    parser.add_argument(
        "--model-path", type=str, required=False, default='./local_data/sklearn_FalconTask.h1.pkl'
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'h2', 'm1', 'm2'], default='h1',
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival'
    )
    parser.add_argument('--batch-size', type=int, help='size of batch for evaluation', default=1)
    args = parser.parse_args()

    task = getattr(FalconTask, args.split)
    config = FalconConfig(
        task=task,
    )

    decoder = SKLearnDecoder(task_config=config, model_path=args.model_path, batch_size=args.batch_size)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()