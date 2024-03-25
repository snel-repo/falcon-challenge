# EvalAI worker stub, dev-testing
import os
import importlib

def main():
    pred_file = os.environ["PREDICTION_PATH"]
    gt_file = os.environ["GT_PATH"]

    challenge_phase = "test"

    CHALLENGE_IMPORT_STRING='falcon_challenge.evaluator'
    challenge_module = importlib.import_module(CHALLENGE_IMPORT_STRING)
    challenge_module.evaluate(
        pred_file,
        gt_file,
        challenge_phase,
        submission_metadata={}
    )

if __name__ == "__main__":
    main()