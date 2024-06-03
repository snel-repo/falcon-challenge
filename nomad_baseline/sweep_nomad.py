import ray, tempfile, json, shutil, sys, os, subprocess, glob, yaml
import numpy as np
from os import path
from ray import tune

# ========= CONFIGURE THE ALIGNMENT RUN ==========
EXPERIMENT_NAME = '240527_M2_singlefile_NoMAD_coinspkrem_redo'
NUM_SAMPLES = 30 # number of samples to take from the search space
RESOURCES_PER_TRIAL = {'cpu': 2, 'gpu': 0.5} # resources to allocate to each process
RUN_SCRIPT = 'train_nomad.py' # script to use from `run_scripts`
CONFIG_FILE = '/home/bkarpo2/bin/falcon-challenge/nomad_baseline/config/nomad_config.yaml'
run_scripts_dir = path.dirname(path.abspath(__file__))
LOCAL_DIR = '/snel/share/runs/falcon'

# ========== SET UP THE HP SEARCH SPACE ===========
np.random.seed(seed=731) # get consistent HP values for each run

#%%
# lfads_paths = {
#     'sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb': '/snel/share/runs/falcon/M1_20120924_shorter_pbt',
#     'sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys.nwb': '/snel/share/runs/falcon/M1_sub-MonkeyL-held-in-calib_ses-20120926_behavior+ecephys',
#     'sub-MonkeyL-held-in-calib_ses-20120927_behavior+ecephys.nwb': '/snel/share/runs/falcon/M1_sub-MonkeyL-held-in-calib_ses-20120927_behavior+ecephys',
#     'sub-MonkeyL-held-in-calib_ses-20120928_behavior+ecephys.nwb': '/snel/share/runs/falcon/M1_sub-MonkeyL-held-in-calib_ses-20120928_behavior+ecephys'
# }

config = {
    # 'DAY0_PATH': tune.grid_search([os.path.join('/home/bkarpo2/bin/falcon-challenge/data/m1/sub-MonkeyL-held-in-calib/', x) for x in lfads_paths.keys()]),
    'DAY0_PATH': '/snel/home/bkarpo2/bin/falcon-challenge/data/m2/sub-MonkeyN-held-in-calib/sub-MonkeyN-held-in-calib_ses-2020-10-19-Run2_behavior+ecephys.nwb',
    # 'DAY0_LFADS': tune.sample_from(lambda spec: lfads_paths[spec.config['DAY0_PATH'].split('/')[-1]]),
    'DAY0_LFADS': '/snel/share/runs/falcon/M2_2020-10-19-Run2_coinspkrem',
    'DAYK_PATH': tune.grid_search(glob.glob('/snel/home/bkarpo2/bin/falcon-challenge/data/m2/sub-MonkeyN-held-out-calib/*.nwb')),
    'RUN_FLAG': 'day0_grid',
    'TRACK': 'M2',
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