r"""
    Sample ridge regression decoder for the Falcon Challenge.

    To train a decoder, run `sklearn_decoder.py`.
    Provide the saved model to this script.
    e.g. python decoder_demos/sklearn_sample.py --evaluation local --split m1 --phase test --model-path ./local_data/sklearn_FalconTask.m1.pkl

"""

import argparse
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

from decoder_demos.sklearn_decoder import SKLearnDecoder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation", type=str, required=True, choices=["local", "remote"], help="Local or remote (EvalAI) evaluation."
    )
    parser.add_argument(
        "--model-path", type=str, required=False, default='./local_data/sklearn_FalconTask.h1.pkl', help="Output path of sklearn_decoder.py."
    )
    parser.add_argument(
        '--split', type=str, choices=['h1', 'h2', 'm1', 'm2'], default='h1', help="Which dataset to evaluate on."
    )
    parser.add_argument(
        '--phase', choices=['minival', 'test'], default='minival', help="Which phase to evaluate on (test is only available on remote)"
    )
    args = parser.parse_args()

    task = getattr(FalconTask, args.split)
    config = FalconConfig(
        task=task,
    )

    decoder = SKLearnDecoder(task_config=config, model_path=args.model_path)

    evaluator = FalconEvaluator(
        eval_remote=args.evaluation == "remote",
        split=args.split,
    )
    evaluator.evaluate(decoder, phase=args.phase)


if __name__ == "__main__":
    main()