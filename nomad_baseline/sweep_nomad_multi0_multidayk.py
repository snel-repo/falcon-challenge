import ray, tempfile, json, shutil, sys, os, subprocess, glob, yaml
import numpy as np
from os import path
from ray import tune

# ========= CONFIGURE THE ALIGNMENT RUN ==========
EXPERIMENT_NAME = '240531_nomad_h1_multi0_multik'
NUM_SAMPLES = 30 # number of samples to take from the search space
RESOURCES_PER_TRIAL = {'cpu': 2, 'gpu': 0.5} # resources to allocate to each process
RUN_SCRIPT = 'train_nomad_multiday0_multidayk.py' # script to use from `run_scripts`
CONFIG_FILE = '/home/bkarpo2/bin/falcon-challenge/nomad_baseline/config/nomad_config.yaml'
run_scripts_dir = path.dirname(path.abspath(__file__))
LOCAL_DIR = '/snel/share/runs/falcon'

# ========== SET UP THE HP SEARCH SPACE ===========
np.random.seed(seed=731) # get consistent HP values for each run

#%%
config = {
    'DAY0_SESS': 'S2',
    'DAY0_LFADS': '/snel/share/runs/falcon/H1_S2_combined_day0',
    'DAYK_SESS': tune.grid_search(['S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12']),
    'RUN_FLAG': 'day0_grid',
    'TRACK': 'H1',
    'CONFIG_PATH': CONFIG_FILE,
    'CFG_UPDATES': {
        'TRAIN.BATCH_SIZE': tune.randint(300, 1100),
        'TRAIN.LR.INIT': tune.loguniform(1e-4, 5e-3),
        'TRAIN.NLL.INCREASE_EPOCH': tune.randint(10, 101),
        'TRAIN.NLL.WEIGHT': tune.loguniform(10., 1000.),
        'TRAIN.KL.INCREASE_EPOCH': tune.randint(10, 101),
        'TRAIN.KL.IC_WEIGHT': tune.loguniform(1.e-5, 1.e-3),
        'TRAIN.KL.CO_WEIGHT': tune.loguniform(1.e-5, 1.e-3)
    }
}


# ========== CLASS THAT MANAGES ALIGNMENT SUBPROCESSES ==========
class tuneAlign(tune.Trainable):
    def _setup(self, config):
        """ Sets up the model for training. """

        config['logdir'] = self.logdir
        self.config = config
        self.run_script = path.join(run_scripts_dir, RUN_SCRIPT)
    
    def _train(self):
        """ Trains the alignment model using a Python 2 subprocess. """

        # Save the alignment HPs to a temporary file
        tfile = tempfile.NamedTemporaryFile()
        with open(tfile.name, 'w') as f:
            json.dump(self.config, f)
        # Create the bash string that will run the alignment script
        sh_str = f"python {self.run_script} {tfile.name}"
        # Log the console output
        stderr = open(os.path.join(self.logdir, 'stderr.txt'), 'w')
        stdout = open(os.path.join(self.logdir, 'stdout.txt'), 'w')
        # Run the alignment script
        subprocess.run(sh_str.split(' '), stderr=stderr, stdout=stdout)
        # Get the results from the alignment script
        output_path = path.join(self.logdir, 'align_out.json')
        with open(output_path, 'r') as f:
            results = json.load(f)
        # Close the output files
        tfile.close()
        stderr.close()
        stdout.close()

        return results

# ========== RUN ALIGNMENT ROUTINE =========
ray.init(address=None) #, num_gpus=8)

analysis = tune.run(
    tuneAlign,
    stop={'training_iteration': 1},
    name=EXPERIMENT_NAME,
    config=config,
    resources_per_trial=RESOURCES_PER_TRIAL,
    num_samples=NUM_SAMPLES,
    local_dir=LOCAL_DIR,
    sync_to_driver='# {source} {target}', # prevents rsync
    verbose=1,
    # resume=True
)