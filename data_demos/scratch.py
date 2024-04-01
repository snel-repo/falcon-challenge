from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pynwb import NWBHDF5IO
from data_demos.styleguide import set_style
set_style()

data_dir = Path('data/h1')
train_files = sorted((data_dir / 'held_in_calib').glob('*calib.nwb'))
test_files = sorted((data_dir / 'held_out_calib').glob('*calib.nwb'))
from falcon_challenge.dataloaders import bin_units, load_nwb
from falcon_challenge.config import FalconTask
BIN_SIZE_S = 0.02

spikes, vel, time, eval_mask = load_nwb(train_files[0], dataset=FalconTask.h1)
breakpoint()