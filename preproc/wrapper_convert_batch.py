import glob
import subprocess
import shlex

""" Run script to make datasets from raw files into NWB format """

base_dir = "/snel/share/share/data/rouse/RTG/"
sessions = glob.glob(base_dir + f"X*AD_Unrect_EMG.mat")

for sess in sessions: 
    idx = sess.split('/')[-1].split('_')[0]
    monkey = idx[0]
    sess = idx[1:]
    cmd = f"python m1b_reachgrasp_preprocv2.py {sess} {monkey}"
    subprocess.run(shlex.split(cmd))

# base_dir = "/snel/share/share/data/bg2/t5_handwriting/CORP_data/stability_benchmark"
# sessions = glob.glob(base_dir + "/*/*.mat")

# for sess in sessions: 
#     ext = sess.split('/')[-2]
#     sess = '.'.join(sess.split('/')[-1].split('.')[1:-1])
#     cmd = f"python h2_preproc.py {sess} {ext}"
#     subprocess.run(shlex.split(cmd))

