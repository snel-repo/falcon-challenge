from pynwb import NWBFile
from pynwb import NWBHDF5IO
import numpy as np
import scipy.signal as signal

from scipy.io import loadmat
import h5py
from datetime import datetime
from dateutil.tz import tzlocal, gettz
import matplotlib.pyplot as plt
from os import path

from pynwb import TimeSeries
from pynwb.misc import Units
from pynwb.behavior import Position
from pynwb.behavior import BehavioralTimeSeries
from pynwb import ProcessingModule
from nwb_convert.nwb_create_utils import (
    create_multichannel_timeseries,
    apply_filt_to_multi_timeseries,
)
from nwb_convert.filtering import (
    apply_notch_filt,
    apply_butter_filt,
    apply_savgol_diff,
    resample_column,
    rectify,
)

import logging
import sys
import yaml

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

save_path = "/snel/share/share/data/rouse/RTG/NWB/"
rouse_base_dir = "/snel/share/share/data/rouse/RTG/raw_files/"
monkey = "L"
exp_date = "20120928"

# emg file
emg_mat_path = path.join(rouse_base_dir, f"{monkey}{exp_date}_AD_Unrect_EMG.mat")
# kinematics file
kin_mat_path = path.join(rouse_base_dir, f"{monkey}_{exp_date}_processed.mat")
# spikes file
spk_mat_path = path.join(rouse_base_dir, f"{monkey}_GHIJKLEF_{exp_date}-data.mat")

# get file string
file_id = f"{monkey}_{exp_date}"

# load mat data
f_emg = loadmat(emg_mat_path)
f_kin = loadmat(kin_mat_path)
f_spk = h5py.File(spk_mat_path, "r")


def convert_datestr_to_datetime(collect_date):
    date_time = datetime.strptime(collect_date, "%Y%m%d").replace(
        tzinfo=gettz("America/Chicago")
    )

    return date_time


date_time = convert_datestr_to_datetime(exp_date)


# === NWBFile Step; create NWB file
logger.info("Creating new NWBFile")
nwbfile = NWBFile(
    session_description="monkey performing reach-to-grasp task",
    identifier=file_id,
    session_start_time=date_time,
    experiment_description="reach-to-grasp center out",
    file_create_date=datetime.now(tzlocal()),
    lab="Rouse",
    institution="University of Kansas",
    experimenter="Dr. Adam Rouse",
)


# === NWBFile Step: add trial info
logger.info("Adding trial info")
# add trial information
nwbfile.add_trial_column(name="gocue_time", description="timing of go cue")
nwbfile.add_trial_column(name="move_onset_time", description="timing of move onset")
nwbfile.add_trial_column(name="contact_time", description="timing of contact onset")
nwbfile.add_trial_column(name="reward_time", description="timing of reward")
nwbfile.add_trial_column(
    name="result", description="outcome of trial (success/failure/abort)"
)
nwbfile.add_trial_column(name="number", description="trial number in rouse dataset")
nwbfile.add_trial_column(name="tgt_loc", description="location of target (angle)")
nwbfile.add_trial_column(name="tgt_obj", description="object to grasp at target")
nwbfile.add_trial_column(
    name="obj_id",
    description="integer id for each object",
)
nwbfile.add_trial_column(
    name="condition_id", description="generic integer id across all loc/obj conditions"
)

n_conds = f_emg['EMGInfo']['trial_id'][0].shape[0]

exp_event_ixs = np.concatenate(f_emg['EMGInfo']['file_event_sample'][0])
exp_event_times = (exp_event_ixs/1000).round(4)
trial_start_times = exp_event_times[:,0] - 1
trial_end_times = exp_event_times[:,-1] + 1
gocue_times = exp_event_times[:,0]
move_onset_times = exp_event_times[:,1]
contact_times = exp_event_times[:,2]
reward_times = exp_event_times[:,3]

generic_cond_ids = np.concatenate([ [i]*f_emg['EMGInfo']['trial_id'][0][i].shape[0] for i in range(n_conds)]).tolist()
object_ids = np.concatenate([ [f_emg['EMGSettings']['objects'][0][0][i][0]]*f_emg['EMGInfo']['trial_id'][0][i].shape[0] for i in range(n_conds)]).tolist()
obj_id_name_map = { 128: 'Coaxial', 64: 'Perpendicular', 8: 'Sphere', 1: 'Button'}
obj_id_num_map = { 128: 1, 64: 2, 8: 3, 1: 4}
obj_names = [*map(obj_id_name_map.get, object_ids)]
object_ids = [*map(obj_id_num_map.get, object_ids)]
locations = np.concatenate([ [f_emg['EMGSettings']['locations'][0][0][i][0]]*f_emg['EMGInfo']['trial_id'][0][i].shape[0] for i in range(n_conds)]).tolist()
# start / gocue / move onset / contact / reward / end
#exp_event_times = np.concatenate((trial_start_times[:,np.newaxis],exp_sample_times,trial_end_times[:,np.newaxis]),axis=1)
exp_trial_ids = np.concatenate(f_emg['EMGInfo']['trial_id'][0]).squeeze()
n_trials = exp_trial_ids.size
trial_order = np.argsort(trial_start_times)

# all trials in dataset are successful (failures have already been excluded)
result = 'R'
for i in range(n_trials):
    ix = trial_order[i]
    number = exp_trial_ids[ix]
    start_time = trial_start_times[ix]
    end_time = trial_end_times[ix]
    gocue_time = gocue_times[ix]
    move_onset_time = move_onset_times[ix]
    contact_time = contact_times[ix]
    reward_time = reward_times[ix]
    tgt_loc = locations[ix]
    tgt_obj = obj_names[ix]
    obj_id = object_ids[ix]
    condition_id = generic_cond_ids[ix]
    nwbfile.add_trial(
        start_time=start_time,
        stop_time=end_time,
        gocue_time=gocue_time,
        move_onset_time=move_onset_time,
        contact_time=contact_time,
        reward_time=reward_time,
        result=result,
        number=number,
        tgt_loc=tgt_loc,
        tgt_obj=tgt_obj,
        obj_id=obj_id,
        condition_id=condition_id,
    )

# === NWBFile Step: add acquisition data
logger.info("Adding acquisition data")

# --- load and process EMG
emg_raw_field = "EMGRawData"
emg_data = f_emg[emg_raw_field]
n_samples = emg_data.shape[0]

def convert_names_to_list(f, name_field):
    names = f[name_field][0][0][0].tolist()
    names = [name[0] for name in names]
    return names

emg_names = convert_names_to_list(f_emg['EMGSettings'], 'ChanNames')

fs_cont = float(f_emg['EMGSettings']['samp_rate'][0][0][0][0])
dt_cont = 1/ fs_cont
#t_cont = np.linspace(0,n_samples*(1/fs_cont), num=n_samples).round(3) # not reliable, floating point precision issues
t_cont = (np.arange(n_samples)/fs_cont).round(4)

#dt_cont = np.unique(np.diff(t_cont)).round(4)[0]
#fs_cont = 1 / dt_cont

emg_mts = create_multichannel_timeseries(
    "emg_raw", emg_names, emg_data, timestamps=t_cont, unit="mV"
)
nwbfile.add_acquisition(emg_mts)
# create processing module
emg_filt = nwbfile.create_processing_module(
    "emg_filtering_module",
    "module to perform emg pre-processing from raw to rectified emg",
)

# extract acquisition data
raw_emg = nwbfile.acquisition["emg_raw"]

notch_cent_freq = [60, 180, 200, 300, 400]
notch_bw_freq = [2, 2, 2, 2, 2]

# high-pass filtering
hp_cutoff_freq = 65  # Hz

# 1) notch filter
notch_emg = apply_filt_to_multi_timeseries(
    raw_emg,
    apply_notch_filt,
    "emg_notch",
    fs_cont,
    notch_cent_freq,
    notch_bw_freq,
)

# 2) high pass filter
hp_emg = apply_filt_to_multi_timeseries(
    notch_emg, apply_butter_filt, "emg_hp", fs_cont, "high", hp_cutoff_freq
)

# 3) rectify
rect_emg = apply_filt_to_multi_timeseries(hp_emg, rectify, "emg")
# add each step to processing module
emg_filt.add_container(notch_emg)
emg_filt.add_container(hp_emg)
emg_filt.add_container(rect_emg)


# --- load and process joint angular kinematics

angle_names = convert_names_to_list(f_kin['AngleInfo'], 'angle_names')

# first we need to construct a continuous array from trialized data
fs_kin = float(f_kin['ViconSettings']['samp_rate'][0][0][0][0])

ds_factor = fs_cont / fs_kin

kin_n_samples = round(n_samples/ds_factor)
t_kin = np.linspace(0,kin_n_samples*(1/fs_kin), num=kin_n_samples).round(4)
n_angles = len(angle_names)
kin_seg_length = f_kin['ViconSettings']['max_samples'][0][0][0][0]

kin_data = np.full((kin_n_samples,n_angles), np.nan)
kin_align_idx = f_kin['ViconInfo']['event_sample'][0][0][0][0]
for i in range(n_conds):
    emg_file_event_sample = f_emg['EMGInfo']['file_event_sample'][0][i]
    kin_file_event_sample = (emg_file_event_sample/ds_factor).round().astype(int) - kin_align_idx
    kin_file_start_sample = kin_file_event_sample[:,0]
    kin_file_end_sample = kin_file_start_sample + kin_seg_length

    n_trials_cond = kin_file_end_sample.size
    for j in range(n_trials_cond):
        kin_file_ix = np.arange(kin_file_start_sample[j],
                                kin_file_end_sample[j])
        n_angles = f_kin['ViconAngles'].shape[0]
        for k in range(n_angles):
            kin_data[kin_file_ix,k] = f_kin['ViconAngles'][k][i][j,:]

kin_mts = create_multichannel_timeseries(
    "joint_kin_raw", angle_names, kin_data, timestamps=t_kin, unit="degrees"
)
nwbfile.add_acquisition(kin_mts)

kin_resample = nwbfile.create_processing_module(
    "kin_resampling_module",
    "module to perform resampling of kinematics data to continuous data sample rate (1KHz)",
)
raw_joint_ang_p = nwbfile.acquisition["joint_kin_raw"]
resample_joint_ang = apply_filt_to_multi_timeseries(
    raw_joint_ang_p, resample_column, "joint_ang_p", fs_cont, fs_kin, timestamps=t_cont
)

resample_data_len = resample_joint_ang[angle_names[0]].data.shape[0]
cont_data_len = resample_joint_ang[angle_names[0]].timestamps.shape[0]
len_diff = cont_data_len - resample_data_len

joint_ang_p = np.full((cont_data_len,len(angle_names)), np.nan)
# trim data
for i, angle_name in enumerate(angle_names):
    if len_diff < 0: # resample length is longer than continuous
        logger.info(f"Trimming kinematics by {len_diff} sample(s) after resampling to match continuous data length")
        joint_ang_p[:,i] = resample_joint_ang[angle_name].data[:cont_data_len]
    elif len_diff > 0: # resample length is shorter than continuous
        logger.info(f"Padding kinematics with {len_diff} sample(s) after resampling to match continuous data length")
        joint_ang_p[:resample_data_len,i] = resample_joint_ang[angle_name].data

resample_joint_ang_p = create_multichannel_timeseries(
    "joint_ang_p", angle_names, joint_ang_p, timestamps=t_cont, unit="degrees"
)

kin_resample.add_container(resample_joint_ang_p)
# pop original data at different sample rate out of NWB
nwbfile.acquisition.pop("joint_kin_raw")


# import spiking data
logger.info("Adding spiking data")

device = nwbfile.create_device(
    name="floating microelectrode_array",
    description="16-electrode array",
    manufacturer="MicroProbes for Life Sciences",
)
E_elec_group = nwbfile.create_electrode_group(
    name="E_electrode_group",
    description="Electrodes in an implanted FMA array labeled E",
    location="Motor Cortex",
    device=device,
)
F_elec_group = nwbfile.create_electrode_group(
    name="F_electrode_group",
    description="Electrodes in an implanted FMA array labeled F",
    location="Motor Cortex",
    device=device,
)
G_elec_group = nwbfile.create_electrode_group(
    name="G_electrode_group",
    description="Electrodes in an implanted FMA array labeled G",
    location="Motor Cortex",
    device=device,
)
H_elec_group = nwbfile.create_electrode_group(
    name="H_electrode_group",
    description="Electrodes in an implanted FMA array labeled H",
    location="Motor Cortex",
    device=device,
)
I_elec_group = nwbfile.create_electrode_group(
    name="I_electrode_group",
    description="Electrodes in an implanted FMA array labeled I",
    location="Motor Cortex",
    device=device,
)
J_elec_group = nwbfile.create_electrode_group(
    name="J_electrode_group",
    description="Electrodes in an implanted FMA array labeled J",
    location="Motor Cortex",
    device=device,
)
K_elec_group = nwbfile.create_electrode_group(
    name="K_electrode_group",
    description="Electrodes in an implanted FMA array labeled K",
    location="Motor Cortex",
    device=device,
)
L_elec_group = nwbfile.create_electrode_group(
    name="L_electrode_group",
    description="Electrodes in an implanted FMA array labeled L",
    location="Motor Cortex",
    device=device,
)

elec_group_map = {
    'E': E_elec_group,
    'F': F_elec_group,
    'G': G_elec_group,
    'H': H_elec_group,
    'I': I_elec_group,
    'J': J_elec_group,
    'K': K_elec_group,
    'L': L_elec_group,
}
t_offset = f_emg['EMGSettings']['analog_start_time'][0][0][0][0]
bin_width = dt_cont
nwbfile.units = Units(
    name="units", description="Sampled at 30 kHz with anti-alias", resolution=bin_width
)

n_units = f_spk['AllSpikeTimes'].shape[0]
array_group_by_chan = f_spk['SpikeSettings']['array_by_channel'][0]
array_group_by_chan = [ chr(array_id) for array_id in array_group_by_chan.tolist()]
elec_id_by_chan = f_spk['SpikeSettings']['unique_channel_num'][0]
orig_elec_ids = np.unique(elec_id_by_chan)
new_elec_ids = np.arange(orig_elec_ids.size)
elec_id_map = { o_ix: n_ix for (o_ix, n_ix) in zip(orig_elec_ids, new_elec_ids)}
elec_id_by_chan = [*map(elec_id_map.get, elec_id_by_chan)]
array_elec_groups = [*map(elec_group_map.get, array_group_by_chan)]
for i in range(n_units):
    elec_id = elec_id_by_chan[i]
    try:
        nwbfile.add_electrode(
            id=int(elec_id),
            x=np.nan,
            y=np.nan,
            z=np.nan,
            imp=np.nan,
            location="M1",
            group=array_elec_groups[i],
            filtering="unknown",
        )
    except ValueError:
        logger.info('Electrode already added to table')
    spike_times = f_spk[f_spk['AllSpikeTimes'][i][0]]
    if spike_times.shape[0] == 1:
        spike_times = spike_times[0]
    else:
        spike_times = np.squeeze(spike_times)

    spike_times = (spike_times - t_offset).round(4)
    # ensure spike times are within bound of continuous data
    keep_mask = (spike_times > t_cont[0]) & (spike_times < t_cont[-1])
    nwbfile.add_unit(
        id=i,
        spike_times=spike_times[keep_mask],
        electrodes=[elec_id],
        obs_intervals=[[t_cont[0], t_cont[-1] + bin_width]],
    )

save_fname = path.join(save_path, file_id + ".nwb")
logger.info(f"Saving NWB file to {save_fname}")
# write processed file
with NWBHDF5IO(save_fname, "w") as io:
    io.write(nwbfile)
