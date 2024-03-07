import glob
import subprocess
import shlex

""" Run script to make datasets from raw files into NWB format """

base_dir = "/snel/share/share/data/rouse/RTG/"
sessions = glob.glob(base_dir + f"L*AD_Unrect_EMG.mat")

for sess in sessions: 
    idx = sess.split('/')[-1].split('_')[0]
    monkey = idx[0]
    sess = idx[1:]
    cmd = f"python m1_reachgrasp_preprocv2.py {sess} {monkey}"
    subprocess.run(shlex.split(cmd))

