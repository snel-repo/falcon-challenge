#%%

# Flow on EvalAI is; worker runs the container, which should output predictions at submission/submission.csv
# Worker will compare against ground truth specified in the challenge config repo.
# Answers thus need to be pushed to challenge config repo.
# Need a minival.pkl, and a test.pkl
# needs to be a dict of all split targets
from falcon_challenge.config import FalconTask, FalconConfig
from falcon_challenge.dataloaders import load_nwb
from pathlib import Path
import pickle

r"""
    annotations struct:
    {
        'h1': {
            'hash1': {
                'data': np.array, T x K
                'mask': np.array, T
            }
        }
    }
"""

def assemble_phase_answer_key(phase='minival', answer_key_dir='./data/answer_key'):
    annotations = {}
    for dataset in ['h1', 'h2', 'm1', 'm2']:
        print(f'Loading {dataset} {phase}')
        annotations[dataset] = {}
        dataset_path = Path(answer_key_dir) / dataset / phase
        dataset_files = list(dataset_path.rglob(f'*{phase}*.nwb'))
        task = getattr(FalconTask, dataset)
        config = FalconConfig(task)
        for d in dataset_files:
            neural_data, decoding_targets, trial_change, eval_mask = load_nwb(d, dataset=task)
            if dataset == 'h2':
                eval_targets = decoding_targets
            else:
                eval_targets = decoding_targets[eval_mask]
            annotations[dataset][config.hash_dataset(d.stem)] = {
                'data': eval_targets,
                'mask': eval_mask
            }
        print(annotations[dataset].keys())
    return annotations

minival_annotations = assemble_phase_answer_key('minival')
eval_annotations = assemble_phase_answer_key('eval')
# save these as pickles

with open('./data/answer_key/minival.pkl', 'wb') as f:
    pickle.dump(minival_annotations, f)

with open('./data/answer_key/eval.pkl', 'wb') as f:
    pickle.dump(eval_annotations, f)