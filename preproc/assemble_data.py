from pathlib import Path
import shutil

root = Path('./data/h1_src')
out = Path('./data/h1')

root = Path('./data/m2/preproc_src')
out = Path('./data/m2')
out.mkdir(exist_ok=True)
# Assemble eval
eval_target = out / 'eval'
eval_target.mkdir(exist_ok=True)
eval_files = list(root.glob('held_in/*eval.nwb')) + list(root.glob('held_out/*eval.nwb'))
# Copy
for ef in eval_files:
    shutil.copy(ef, eval_target / ef.name)

# Assemble minival
minival_target = out / 'minival'
minival_target.mkdir(exist_ok=True)
minival_files = list(root.glob('held_in/*minival.nwb'))
# Copy
for mf in minival_files:
    shutil.copy(mf, minival_target / mf.name)

# Assemble calibration
calib_target = out / 'held_in_calib'
calib_target.mkdir(exist_ok=True)
calib_files = list(root.glob('held_in/*calib.nwb'))
# Copy
for cf in calib_files:
    shutil.copy(cf, calib_target / cf.name)

calib_target = out / 'held_out_calib'
calib_target.mkdir(exist_ok=True)
calib_files = list(root.glob('held_out/*calib.nwb'))
# Copy
for cf in calib_files:
    shutil.copy(cf, calib_target / cf.name)