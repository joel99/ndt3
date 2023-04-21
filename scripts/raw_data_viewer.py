#%%
# Scratchpad for raw data processing

from typing import List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from math import ceil

import h5py
from scipy.interpolate import interp1d
from scipy.signal import resample_poly

from einops import rearrange, reduce

from nlb_tools.make_tensors import PARAMS, _prep_mask, make_stacked_array
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import logging

import pynwb
from pynwb import TimeSeries, ProcessingModule, NWBFile, NWBHDF5IO
from pynwb.core import MultiContainerInterface

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.contexts import context_registry, ContextInfo
from context_general_bci.analyze_utils import prep_plt

## Load dataset

dataset_name = 'odoherty_rtt-Loco-20170215_02'
dataset_name = 'odoherty_rtt-Loco-20170227_04'
dataset_name = 'odoherty_rtt-Indy-20161005_06'
context = context_registry.query(alias=dataset_name)

dataset_name = 'odoherty_rtt.*'
contexts = context_registry.query(alias=dataset_name)

dataset_name ='churchland_maze_jenkins-0'

dataset_name = ''
#%%
for context in contexts:
    datapath = context.datapath

    sampling_rate = 1000
    cfg = DatasetConfig()
    cfg.bin_size_ms = 20
    # cfg.bin_size_ms = 5
    def load_bhvr_from_raw(datapath):
        with h5py.File(datapath, 'r') as h5file:
            orig_timestamps = np.squeeze(h5file['t'][:])
            time_span = int((orig_timestamps[-1] - orig_timestamps[0]) * sampling_rate)
            if cfg.odoherty_rtt.load_covariates:
                covariate_sampling = 250 # Hz
                def resample(data):
                    return torch.tensor(
                        resample_poly(data, sampling_rate / covariate_sampling, cfg.bin_size_ms, padtype='line')
                    )
                bhvr_vars = {}
                for bhvr in ['finger_pos']:
                # for bhvr in ['finger_pos', 'cursor_pos', 'target_pos']:
                    bhvr_vars[bhvr] = h5file[bhvr][()].T
                # cursor_vel = np.gradient(cursor_pos[~np.isnan(cursor_pos[:, 0])], axis=0)
                finger_vel = np.gradient(bhvr_vars['finger_pos'], axis=0)
                bhvr_vars[DataKey.bhvr_vel] = torch.tensor(finger_vel)
                for bhvr in bhvr_vars:
                    bhvr_vars[bhvr] = resample(bhvr_vars[bhvr])
        return bhvr_vars

    def load_spikes_from_raw(datapath):
        LOAD_SORTED = True
        with h5py.File(datapath, 'r') as h5file:
            orig_timestamps = np.squeeze(h5file['t'][:])
            time_span = int((orig_timestamps[-1] - orig_timestamps[0]) * sampling_rate)
            int_arrays = [h5file[ref][()][:,0] for ref in h5file['chan_names'][0]]
            make_chan_name = lambda array: ''.join([chr(num) for num in array])
            chan_names = [make_chan_name(array) for array in int_arrays]
            chan_arrays = [cn.split()[0] for cn in chan_names]
            assert (
                len(chan_arrays) == 96 and all([c == 'M1' for c in chan_arrays]) or \
                len(chan_arrays) == 192 and all([c == 'M1' for c in chan_arrays[:96]]) and all([c == 'S1' for c in chan_arrays[96:]])
            ), "Only M1 and S1 arrays in specific format are supported"

            spike_refs = h5file['spikes'][()].T
            if LOAD_SORTED:
                spike_refs = spike_refs[:96] # only M1. Already quite a lot of units to process without, maintains consistency with other datasets. (not exploring multiarea atm)
            channels, units = spike_refs.shape # units >= 1 are sorted, we just want MUA on unit 0
            # print(spike_refs.shape)
            mua_unit = 0

            unit_budget = units if LOAD_SORTED else 1
            spike_arr = torch.zeros((time_span, channels, unit_budget), dtype=torch.uint8)

            min_spike_time = []
            for c in range(channels):
                if h5file[spike_refs[c, mua_unit]].dtype != float:
                    continue
                unit_range = range(units) if LOAD_SORTED else [mua_unit]
                for unit in unit_range:
                    spike_times = h5file[spike_refs[c, unit]][()]
                    if spike_times.shape[0] == 2: # empty unit
                        continue
                    spike_times = spike_times[0]
                    if len(spike_times) < 2: # don't bother
                        continue
                    spike_times = spike_times - orig_timestamps[0]
                    ms_spike_times, ms_spike_cnt = np.unique((spike_times * sampling_rate).round(6).astype(int), return_counts=True)
                    spike_mask = ms_spike_times < spike_arr.shape[0]
                    ms_spike_times = ms_spike_times[spike_mask]
                    ms_spike_cnt = torch.tensor(ms_spike_cnt[spike_mask], dtype=torch.uint8)
                    spike_arr[ms_spike_times, c, unit] = ms_spike_cnt
                    min_spike_time.append(ms_spike_times[0])
            min_spike_time = max(min(min_spike_time), 0) # some spikes come before marked trial start
            spike_arr: torch.Tensor = spike_arr[min_spike_time:, :]

            if LOAD_SORTED:
                # Filter out extremely low FR units, we can't afford to load everything.
                threshold_rate_hz = 0.5 # Less than 0.5Hz (not even one spike every 2 trials)
                threshold_count = threshold_rate_hz * (spike_arr.shape[0] / sampling_rate)
                spike_arr = spike_arr.flatten(1, 2)
                spike_arr = spike_arr[:, (spike_arr.sum(0) > threshold_count).numpy()]

            def compress_vector(vec: torch.Tensor, compression='sum'):
                # vec: at sampling resolution
                full_vec = vec.unfold(0, cfg.odoherty_rtt.chop_size_ms, cfg.odoherty_rtt.chop_size_ms) # Trial x C x chop_size (time)
                return reduce(
                    rearrange(full_vec, 'b c (time bin) -> b time c bin', bin=cfg.bin_size_ms),
                    'b time c bin -> b time c 1', compression
                )
            def chop_vector(vec: torch.Tensor):
                # vec - already at target resolution, just needs chopping
                chops = round(cfg.odoherty_rtt.chop_size_ms / cfg.bin_size_ms)
                return rearrange(
                    vec.unfold(0, chops, chops),
                    'trial hidden time -> trial time hidden'
                ) # Trial x C x chop_size (time)
            full_spikes = compress_vector(spike_arr)
            # print(SubjectArrayRegistry.query_by_array('Indy-M1_all'))
            print(create_spike_payload(full_spikes[0], ['Indy-M1_all'])['Indy-M1_all'].size())
            # print(full_spikes.shape)
        return spike_arr
    spike_arr = load_spikes_from_raw(datapath)

#%%
print(spike_arr.shape)

#%%
ctxs = context_registry.query(task=ExperimentalTask.odoherty_rtt)
session_paths = [ctx.datapath for ctx in ctxs]

def plot_trace(
    ax, bhvr_payload,
    length=2000,
    title: Path | None = None,
    key='finger_pos',
    # key=DataKey.bhvr_vel,
): # check baseline qualitative
    # ax = prep_plt(ax)
    finger_vel = bhvr_payload[key][:length]
    ax.plot(finger_vel[:, 0], finger_vel[:, 1])
    # ax.set_xlim(-0.2, 0.2)
    # ax.set_ylim(-0.2, 0.2)
    if title is not None:
        ax.set_title(title.stem)

# plot all sessions by loading behavior and calling `plot_trace`
fig, axs = plt.subplots(
    ceil(len(session_paths) / 2), 2,
    figsize=(10 * 2, 10 * ceil(len(session_paths) / 2))
)
for i, session_path in enumerate(session_paths):
    bhvr_payload = load_bhvr_from_raw(session_path)
    plot_trace(axs.ravel()[i], bhvr_payload, title=session_path)

#%%
ctxs = context_registry.query(task=ExperimentalTask.gallego_co)
session_paths = [ctx.datapath for ctx in ctxs]
datapath = session_paths[0]
import pyaldata
df: pd.DataFrame = pyaldata.mat2dataframe(datapath, shift_idx_fields=True)

print(df.columns)

#%%
print(torch.tensor(df[f'M1_spikes'][0]).size()) # T x N


#%%
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = prep_plt(ax=ax)
bhvr_payload = load_bhvr_from_raw(session_paths[0])
plot_trace(ax, bhvr_payload, title=session_paths[0])


#%%
# print(nwbfile.fields.keys())
# print(nwbfile.subject)

# Has movement and LFP
# print(nwbfile.processing['behavior'].data_interfaces)
# print(nwbfile.processing['ecephys'].data_interfaces)

# Time intervals: https://pynwb.readthedocs.io/en/stable/tutorials/general/file.html#time-intervals
# print(nwbfile.intervals['trials'])
# Actually, this is the same thing as nwbfile.trials
print(nwbfile.intervals['trials'].colnames)
starts = nwbfile.intervals['trials']['start_time'][:]
print(nwbfile.intervals['trials']['discard_trial'][:][10:])
print(nwbfile.intervals['trials']['go_cue_time'][:])
print(nwbfile.intervals['trials']['move_begins_time'][:])
print(nwbfile.intervals['trials']['move_ends_time'][:])
print(nwbfile.intervals['trials']['discard_trial'][:].sum())
# print(nwbfile.intervals['trials']['stop_time'][:])
# print(len(nwbfile.intervals['trials']['start_time'][:]))
# print(len(nwbfile.intervals['trials']['stop_time'][:]))
plt.plot(nwbfile.intervals['trials']['start_time'][:])

# More time intervals?
# print(nwbfile.trials.id)
#%%
unit = 2
unit = 20

# unit = 1
# print(nwbfile.units.to_dataframe().electrode_group.unique())
# print(nwbfile.units.spike_times)
unit_df = nwbfile.units.to_dataframe()
obs_int = unit_df.obs_intervals.iloc[unit]
print(unit_df.obs_intervals.iloc[unit])
plt.plot(unit_df.obs_intervals.iloc[unit][:,0])
# plt.plot(nwbfile.units.to_dataframe().obs_intervals.iloc[unit][:,1])
# plt.plot(nwbfile.units.to_dataframe().spike_times.iloc[unit])
# print(nwbfile.units.to_dataframe().spike_times.iloc[1])
def unit_stats(unit):
    spikes = unit_df.spike_times.iloc[unit]
    observed_time = (obs_int[:, 1] - obs_int[:, 0]).sum()
    print(f"Unit {unit} avg FR: {(len(spikes) / observed_time):.2f} Hz")
for t in range(20):
    unit_stats(t)

#%%
is_monotonic = lambda spikes: np.all(np.diff(spikes) >= 0)
all_units = nwbfile.units.to_dataframe()
unit_valid = []
for t in range(len(all_units)):
    unit_valid.append(is_monotonic(all_units.spike_times.iloc[t]))
print((np.array(unit_valid) == False).nonzero())

#%%
# Sanity check interval and spike time validity
intervals = nwbfile.units.to_dataframe().obs_intervals.iloc[unit]
print(len(intervals))
#%%
num_contained = []
for t in nwbfile.units.to_dataframe().spike_times.iloc[unit]:
    num_contained.append(np.sum((intervals[:,0] <= t) & (intervals[:,1] >= t)))

print((np.array(num_contained) == 1).all())

# Identify number of overlapping intervals in `intervals`
def num_overlaps(intervals):
    num_overlaps = 0
    for i in range(len(intervals)):
        for j in range(i+1, len(intervals)):
            if (intervals[i,0] <= intervals[j,0] <= intervals[i,1]) or (intervals[j,0] <= intervals[i,0] <= intervals[j,1]):
                num_overlaps += 1
    return num_overlaps
print(num_overlaps(intervals))







#%%
# io.close()
# exit(0)
#%%
# TODO want to add array group info to units

patch_name = 'churchland_reaching'

class NWBDatasetChurchland(NWBDataset):
    def __init__(self, *args, **kwargs):
        kwargs['split_heldout'] = False
        kwargs['skip_fields'] = [
            'Position_Cursor',
            'Position_Eye',
            'Position_Hand',
            'Processed_A001',
            'Processed_A002',
            'Processed_B001',
            'Processed_B002',
        ]
        # Note - currently these fields are dropped due to a slight timing mismatch.
        # If you want them back, you'll need to reduce precision in NWBDataset.load() from 6 digits to 3 digits (which I think is fine)
        # But we currently don't need
        super().__init__(*args, **kwargs)
        self.trial_info = self.trial_info.rename({ # match NLB naming
            'move_begins_time': 'move_onset_time',
            'task_success': 'success',
            'target_presentation_time': 'target_on_time',
            'reaction_time': 'rt',
        }, axis=1)

dataset = NWBDatasetChurchland(exp) #
bin_width = 5
dataset.resample(bin_width)

# make_tensors from NLB can be used on this data with a few patches
# 1. Params are defined in module, rather than taken as an argument. Override this params
# PARAMS[patch_name] = PARAMS['mc_maze']

# 2. Provide a mock heldout spikes field
# I prefer to override the function as below - unclear how to mock heldout spikes
# 3. Add mock trial_split info
dataset.trial_info['split'] = 'train'

def make_input_tensors_simple(dataset, mock_dataset='mc_maze', trial_split=['train'], **kwargs):
    # See `make_train_input_tensors` for documentation
    assert isinstance(dataset, NWBDataset), "`dataset` must be an instance of NWBDataset"

    # Fetch and update params
    params = PARAMS[mock_dataset].copy()
    # unpack params
    spk_field = params['spk_field']
    # hospk_field = params['hospk_field']
    make_params = params['make_params'].copy()

    # Prep mask
    trial_mask = _prep_mask(dataset, trial_split)

    # Make output spiking arrays and put into data_dict
    train_dict = make_stacked_array(dataset, [spk_field], make_params, trial_mask)
    return {
        'train_spikes_heldin': train_dict[spk_field]
    }

import pdb;pdb.set_trace()
spikes = make_input_tensors_simple(
    dataset
)
print(spikes.shape)

# train_dict = make_train_input_tensors(
#     dataset,
#     dataset_name='churchland_reaching',
#     trial_split=['train'],
#     save_file=False
# )

# print(train_dict['train_spikes_heldin'].shape)

