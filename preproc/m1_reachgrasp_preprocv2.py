#%% 

from pynwb import NWBFile
from pynwb import NWBHDF5IO
import numpy as np
import scipy.signal as signal

from scipy.io import loadmat
import h5py, glob, logging, sys, os
from datetime import datetime
from dateutil.tz import tzlocal, gettz
import matplotlib.pyplot as plt
from os import path

from pynwb import TimeSeries
from pynwb.misc import Units
from pynwb.behavior import Position
from pynwb.behavior import BehavioralTimeSeries
from pynwb import ProcessingModule
from nwb_create_utils import (
    create_multichannel_timeseries,
    apply_filt_to_multi_timeseries,
)
from filtering import (
    apply_notch_filt,
    apply_butter_filt,
    apply_savgol_diff,
    apply_clipping,
    apply_scaling,
    resample_column,
    rectify,
)

#%% 
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

SAVE_PATH = "/snel/share/share/derived/rouse/RTG/NWB_FALCON/"
rouse_base_dir = "/snel/share/share/data/rouse/RTG/"
MONKEY = "L"
EXP_DATE = "20120924"

if len(sys.argv) > 1:
    EXP_DATE = sys.argv[1]
    MONKEY = sys.argv[2]

train_dates = ['20120924', '20120926', '20120927', '20120928']
test_dates = ['20121004', '20121017', '20121022', '20121024']

if EXP_DATE in train_dates: 
    IS_TEST_DS = False
if EXP_DATE in test_dates: 
    IS_TEST_DS = True

# emg file
emg_mat_path = path.join(rouse_base_dir, f"{MONKEY}{EXP_DATE}_AD_Unrect_EMG.mat")
# spikes file
spk_mat_path = glob.glob(path.join(rouse_base_dir, f"{MONKEY}_*_{EXP_DATE}-data.mat"))[0]
# get file string
# file_id = f"{monkey}_{exp_date}"
# load mat data
f_emg = loadmat(emg_mat_path)
f_spk = h5py.File(spk_mat_path, "r")

def convert_datestr_to_datetime(collect_date):
    date_time = datetime.strptime(collect_date, "%Y%m%d").replace(
        tzinfo=gettz("America/Chicago")
    )

    return date_time

#%% extracting trial info

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

exp_trial_ids = np.concatenate(f_emg['EMGInfo']['trial_id'][0]).squeeze()
trial_order = np.argsort(trial_start_times)

#%% extracting emg data 

emg_data = f_emg["EMGRawData"]

def convert_names_to_list(f, name_field):
    names = f[name_field][0][0][0].tolist()
    names = [name[0] for name in names]
    return names

emg_names = convert_names_to_list(f_emg['EMGSettings'], 'ChanNames')
fs_cont = float(f_emg['EMGSettings']['samp_rate'][0][0][0][0])
t_offset = f_emg['EMGSettings']['analog_start_time'][0][0][0][0]

#%% 
n_units = np.unique(f_spk['SpikeSettings']['channels'][()]).shape[0]
array_group_by_chan = f_spk['SpikeSettings']['array_by_channel'][0]
array_group_by_chan = [ chr(array_id) for array_id in array_group_by_chan.tolist()]
elec_id_by_chan = f_spk['SpikeSettings']['unique_channel_num'][0]

#%% 

def convert_to_NWB(
    fs_cont, 
    trial_start_times, 
    trial_end_times,
    gocue_times,
    move_onset_times,
    contact_times,
    reward_times, 
    generic_cond_ids,
    object_ids, 
    object_names,
    locations, 
    exp_trial_ids, 
    emg_data, 
    emg_names, 
    t_offset, 
    n_units,
    elec_id_by_chan,
    array_group_by_chan,
    spike_time_thresh=[None, None],
    split_label='in_day_oracle' #calibration, eval
): 
    file_id = f'{MONKEY}_{EXP_DATE}_{split_label}'
    logger.info("Creating new NWBFile")
    nwbfile = NWBFile(
        session_description="monkey performing reach-to-grasp task",
        identifier=file_id,
        session_start_time=convert_datestr_to_datetime(EXP_DATE),
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

    result = 'R'
    for ix in range(exp_trial_ids.size):
        # ix = trial_order[i]
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

    logger.info("Adding acquisition data")

    t_cont = (np.arange(0, emg_data.shape[0])/fs_cont).round(4)

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

    # clipping and quantile normalization of EMG
    CLIP_Q = 0.99
    SCALE_Q = 0.95
    clip_emg = apply_filt_to_multi_timeseries(
        rect_emg, apply_clipping, 'emg_clip', CLIP_Q
    )
    scale_emg = apply_filt_to_multi_timeseries(
        clip_emg, apply_scaling, 'emg_scale', SCALE_Q
    )

    # resample all the processed EMG data to 20ms bins (50Hz) 
    resamp_emg = apply_filt_to_multi_timeseries(
        scale_emg, resample_column, 'emg_resample', 50, fs_cont
    )

    # rectify again 
    rerect_emg = apply_filt_to_multi_timeseries(resamp_emg, rectify, 'emg_rerect')

    # apply low pass filter 
    preprocessed_emg = apply_filt_to_multi_timeseries(
        rerect_emg, apply_butter_filt, 'preprocessed_emg', 50, 'low', 10
    )

    # add each step to processing module
    emg_filt.add_container(notch_emg)
    emg_filt.add_container(hp_emg)
    emg_filt.add_container(rect_emg)
    emg_filt.add_container(clip_emg)
    emg_filt.add_container(scale_emg)
    emg_filt.add_container(resamp_emg)
    emg_filt.add_container(rerect_emg)
    emg_filt.add_container(preprocessed_emg)

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

    nwbfile.units = Units(
        name="units", description="Sampled at 30 kHz with anti-alias", resolution=1/fs_cont
    )
    orig_elec_ids = np.unique(elec_id_by_chan)
    new_elec_ids = np.arange(orig_elec_ids.size)
    elec_id_map = { o_ix: n_ix for (o_ix, n_ix) in zip(orig_elec_ids, new_elec_ids)}
    array_elec_groups = [*map(elec_group_map.get, array_group_by_chan)]

    for i in range(n_units):
        elec_id = orig_elec_ids[i]
        try:
            nwbfile.add_electrode(
                id=int(elec_id_map[elec_id]),
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

        where_elec = np.where(elec_id_by_chan == elec_id)[0]
        all_spike_times = []
        for j in where_elec:
            spike_times = f_spk[f_spk['AllSpikeTimes'][j][0]]
            if spike_times.shape[0] == 1:
                spike_times = spike_times[0]
            else:
                spike_times = np.squeeze(spike_times)
            spike_times = (spike_times - t_offset).round(4)
            all_spike_times.append(spike_times)

        all_spike_times = np.concatenate(all_spike_times)

        start_cutoff, stop_cutoff = spike_time_thresh
        if start_cutoff is None: 
            start_cutoff = t_cont[0]
        if stop_cutoff is None: 
            stop_cutoff = t_cont[-1]
        
        # ensure spike times are within bound of continuous data
        keep_mask = (all_spike_times > start_cutoff) & (all_spike_times < stop_cutoff)
        nwbfile.add_unit(
            id=i,
            spike_times=all_spike_times[keep_mask],
            electrodes=[elec_id_map[elec_id]],
            obs_intervals=[[start_cutoff, stop_cutoff + 1/fs_cont]],
        )
    
    nwb_path = path.join(SAVE_PATH, split_label)
    if not path.exists(nwb_path):
        os.makedirs(nwb_path, mode=0o755)
    save_fname = path.join(nwb_path, file_id + ".nwb")
    logger.info(f"Saving NWB file to {save_fname}")
    # write processed file
    with NWBHDF5IO(save_fname, "w") as io:
        io.write(nwbfile)

#%% 

trial_start_times = trial_start_times[trial_order]
trial_end_times = trial_end_times[trial_order]
gocue_times = gocue_times[trial_order]
move_onset_times = move_onset_times[trial_order]
contact_times = contact_times[trial_order]
reward_times = reward_times[trial_order]
generic_cond_ids = np.array(generic_cond_ids)[trial_order]
object_ids = np.array(object_ids)[trial_order]
object_names = np.array(obj_names)[trial_order]
locations = np.array(locations)[trial_order]
exp_trial_ids = np.array(exp_trial_ids)[trial_order]

#%% 
FEW_SHOT_CALIBRATION_RATIO = 0.2
EVAL_RATIO = 0.4
n_trials = exp_event_times.shape[0]
calibration_num = int(np.ceil(n_trials * FEW_SHOT_CALIBRATION_RATIO))
eval_num = int(n_trials * EVAL_RATIO)

#%%
if IS_TEST_DS:
    logger.info("Creating few-shot calibration split")
    # first calibration num = few shot training 
    calib_end_ind_emg = int(trial_end_times[:calibration_num][-1] * fs_cont)
    convert_to_NWB(
        fs_cont, 
        trial_start_times[:calibration_num], 
        trial_end_times[:calibration_num],
        gocue_times[:calibration_num],
        move_onset_times[:calibration_num],
        contact_times[:calibration_num],
        reward_times[:calibration_num], 
        generic_cond_ids[:calibration_num],
        object_ids[:calibration_num], 
        object_names[:calibration_num],
        locations[:calibration_num], 
        exp_trial_ids[:calibration_num], 
        emg_data[:calib_end_ind_emg, :], 
        emg_names, 
        t_offset, 
        n_units,
        elec_id_by_chan,
        array_group_by_chan,
        spike_time_thresh=[0, trial_end_times[:calibration_num][-1]], 
        split_label='calibration' #'in_day_oracle' #calibration, eval
    )

    logger.info("Creating evaluation split")
    # last eval num = evaluation set
    eval_start_ind_emg = int(trial_start_times[-eval_num:][0] * fs_cont)
    convert_to_NWB(
        fs_cont,
        trial_start_times[-eval_num:],
        trial_end_times[-eval_num:],
        gocue_times[-eval_num:],
        move_onset_times[-eval_num:],
        contact_times[-eval_num:],
        reward_times[-eval_num:],
        generic_cond_ids[-eval_num:],
        object_ids[-eval_num:],
        object_names[-eval_num:],
        locations[-eval_num:],
        exp_trial_ids[-eval_num:],
        emg_data[eval_start_ind_emg:, :],
        emg_names,
        t_offset,
        n_units,
        elec_id_by_chan,
        array_group_by_chan,
        spike_time_thresh=[trial_start_times[-eval_num:][0], emg_data.shape[0]/fs_cont],
        split_label='eval'
    )

    logger.info("Creating in-day oracle split")
    # everything that is not eval set = oracle 
    convert_to_NWB(
        fs_cont, 
        trial_start_times[:-eval_num],
        trial_end_times[:-eval_num],
        gocue_times[:-eval_num],
        move_onset_times[:-eval_num],
        contact_times[:-eval_num],
        reward_times[:-eval_num],
        generic_cond_ids[:-eval_num],
        object_ids[:-eval_num],
        object_names[:-eval_num],
        locations[:-eval_num],
        exp_trial_ids[:-eval_num],
        emg_data[:eval_start_ind_emg, :],
        emg_names,
        t_offset,
        n_units,
        elec_id_by_chan,
        array_group_by_chan,
        spike_time_thresh=[0, trial_start_times[-eval_num:][0]],
        split_label='in_day_oracle'
    )

else: 
    logger.info("Creating full training dataset")
    # need full dataset 
    convert_to_NWB(
        fs_cont,
        trial_start_times,
        trial_end_times,
        gocue_times,
        move_onset_times,
        contact_times,
        reward_times,
        generic_cond_ids,
        object_ids,
        object_names,
        locations,
        exp_trial_ids,
        emg_data,
        emg_names,
        t_offset,
        n_units,
        elec_id_by_chan,
        array_group_by_chan,
        spike_time_thresh=[None, None],
        split_label='full'
    )

    logger.info("Creating minival split")
    # and minival dataset which is last EVAL_RATIO of train data 
    eval_start_ind_emg = int(trial_start_times[-eval_num:][0] * fs_cont)
    convert_to_NWB(
        fs_cont,
        trial_start_times[-eval_num:],
        trial_end_times[-eval_num:],
        gocue_times[-eval_num:],
        move_onset_times[-eval_num:],
        contact_times[-eval_num:],
        reward_times[-eval_num:],
        generic_cond_ids[-eval_num:],
        object_ids[-eval_num:],
        object_names[-eval_num:],
        locations[-eval_num:],
        exp_trial_ids[-eval_num:],
        emg_data[eval_start_ind_emg:, :],
        emg_names,
        t_offset,
        n_units,
        elec_id_by_chan,
        array_group_by_chan,
        spike_time_thresh=[trial_start_times[-eval_num:][0], emg_data.shape[0]/fs_cont],
        split_label='minival'
    )


# %%
