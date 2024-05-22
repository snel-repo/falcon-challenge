#%% 
import ray, yaml, shutil, sys, os
from ray import tune
from os import path

from tune_tf2.models import create_trainable_class
from tune_tf2.pbt.hps import HyperParam
from tune_tf2.pbt.schedulers import MultiStrategyPBT
from tune_tf2.pbt.trial_executor import SoftPauseExecutor
from lfads_tf2.defaults import DEFAULT_CONFIG_DIR
from lfads_tf2.utils import flatten

# ---------- PBT I/O CONFIGURATION ----------
# the default configuration file for the LFADS model
CFG_PATH = '/home/bkarpo2/bin/stability-benchmark/nomad_baseline/config/config.yaml'
# the directory to save PBT runs (usually '~/ray_results') 
PBT_HOME = '/snel/share/runs/falcon/M1_20120924_shorter_pbt'
if len(sys.argv) > 1:
    PBT_HOME = sys.argv[1]
# the name of this PBT run (run will be stored at {PBT_HOME}/{PBT_NAME})
RUN_NAME = 'pbt_run'
# the dataset to train the PBT model on
DATA_DIR = os.path.join(PBT_HOME, 'input_data')
DATA_PREFIX = 'lfads'
best_worker_path = path.join(PBT_HOME, RUN_NAME, 'model_output')
#%% 
# ---------- PBT RUN CONFIGURATION ----------
# whether to use single machine or cluster
SINGLE_MACHINE = True
# the number of workers to use - make sure machine can handle all
NUM_WORKERS = 16
# the resources to allocate per model
RESOURCES_PER_TRIAL = {"cpu": 2, "gpu": 0.5}
# the hyperparameter space to search
HYPERPARAM_SPACE = {
    'TRAIN.LR.INIT': HyperParam(1e-5, 9e-3, explore_wt=0.3, 
        enforce_limits=True, init=0.003),
    'MODEL.DROPOUT_RATE': HyperParam(0.3, 0.99, explore_wt=0.3,
        enforce_limits=True, sample_fn='uniform', init=0.5),
    # 'MODEL.CD_RATE': HyperParam(0.1, 0.99, explore_wt=0.3,
    #     enforce_limits=True, sample_fn='uniform', init=0.3),
    'TRAIN.L2.GEN_SCALE': HyperParam(1e-6, 1e-3, explore_wt=0.8, enforce_limits=True),
    'TRAIN.L2.CON_SCALE': HyperParam(1e-6, 1e-3, explore_wt=0.8, enforce_limits=True),
    'TRAIN.KL.CO_WEIGHT': HyperParam(1e-5, 1e-3, explore_wt=0.8, enforce_limits=True),
    'TRAIN.KL.IC_WEIGHT': HyperParam(1e-5, 1e-3, explore_wt=0.8, enforce_limits=True),
}
PBT_METRIC='smth_val_nll_heldin'
EPOCHS_PER_GENERATION = 15
MAX_GENERATIONS = 100
# ---------------------------------------------
# setup the data hyperparameters
NUM_CHANNELS = 96 if 'M2' in PBT_HOME else (64 if 'M1' in PBT_HOME else 176)

dataset_info = {
    'MODEL.DATA_DIM': NUM_CHANNELS, 
    'TRAIN.DATA.DIR': DATA_DIR,
    'TRAIN.DATA.PREFIX': DATA_PREFIX
    }
# setup initialization of search hyperparameters
init_space = {name: tune.sample_from(hp.init) 
    for name, hp in HYPERPARAM_SPACE.items()}
# load the configuration as a dictionary and update for this run
flat_cfg_dict = flatten(yaml.full_load(open(CFG_PATH)))
flat_cfg_dict.update(dataset_info)
flat_cfg_dict.update(init_space)
# Set the number of epochs per generation
tuneDimReducedLFADS = create_trainable_class(EPOCHS_PER_GENERATION, model_type='dim_reduced')
# connect to Ray cluster or start on single machine
address = None if SINGLE_MACHINE else 'localhost:10000'
ray.init(address=address)
# create the PBT scheduler
scheduler = MultiStrategyPBT(
    HYPERPARAM_SPACE,
    metric=PBT_METRIC,
    max_generations=MAX_GENERATIONS,
    patience=4, #default = 4 (long - 10)
    min_percent_improvement=0.0005) # default = 0.0005 (long - 0)
# Create the trial executor
executor = SoftPauseExecutor(reuse_actors=True)
# Create the command-line display table
reporter = tune.CLIReporter(metric_columns=['epoch', PBT_METRIC])
# run the tune job, excepting errors
tune.run(
    tuneDimReducedLFADS,
    name=RUN_NAME,
    local_dir=PBT_HOME,
    config=flat_cfg_dict,
    resources_per_trial=RESOURCES_PER_TRIAL,
    num_samples=NUM_WORKERS,
    sync_to_driver='# {source} {target}', # prevents rsync
    scheduler=scheduler,
    progress_reporter=reporter,
    trial_executor=executor,
    verbose=1,
    reuse_actors=True,
)

# load the results dataframe for this run
pbt_dir = path.join(PBT_HOME, RUN_NAME)
df = tune.Analysis(pbt_dir).dataframe()
df = df[df.logdir.apply(lambda path: not 'best_model' in path)]
# find the best model
best_model_logdir = df.loc[df[PBT_METRIC].idxmin()].logdir
best_model_src = path.join(best_model_logdir, 'model_dir')
# copy the best model somewhere easy to find
best_model_dest = best_worker_path
shutil.copytree(best_model_src, best_model_dest)
# perform posterior sampling
from lfads_tf2.subclasses.dimreduced.models import DimReducedLFADS
model = DimReducedLFADS(model_dir=best_model_dest)
model.sample_and_average()