#%%
from nlb_tools.nwb_interface import NWBDataset
import numpy as np
import pandas as pd
from pathlib import Path

import logging

import pynwb
from pynwb import TimeSeries, ProcessingModule, NWBFile, NWBHDF5IO
from pynwb.core import MultiContainerInterface

from nlb_tools.make_tensors import make_train_input_tensors, PARAMS, _prep_mask, make_stacked_array
from context_general_bci.contexts import context_registry

## Load dataset
# From this investigation, it seems like unit 1 is busted
dataset_name = 'churchland_maze_jenkins-0'
# dataset_name = 'churchland_maze_jenkins-3'
dataset_name = 'churchland_maze_nitschke-0'
dataset_name = 'churchland_maze_nitschke-3'
# dataset_name = 'churchland_maze_nitschke-1'
# dataset_name = 'churchland_maze_nitschke-2'

# Gdrive
dataset_name = 'churchland_misc_nitschke-'
dataset_name = 'churchland_misc_jenkins-'
dataset_name = 'churchland_misc_reggie-'
# dataset_name = 'churchland_misc_reggie-' # this seems hdf5
context = context_registry.query(alias=dataset_name)
print(len(context))
# Ok, some are hdf5, some are mat (all masquerade with .mat endings)
path = context[0].datapath
# path = context[1].datapath
path = context[2].datapath
# path = context[3].datapath # h5
# path = context[4].datapath #
# path = context.datapath
print(path)

#%%
# io = NWBHDF5IO(context.datapath, 'r')
# nwbfile = io.read()

# try opening as h5py file
import h5py
f = h5py.File(path, 'r')
#%%
# test = pd.DataFrame(f['R']['handPos'])
data = f['R']
print(data.keys())
# print(data['timeCerebusStart'][10, 0])
# print(data['timeCerebusStart2'][10,0])
# spike crossing start times
start = data[data['timeCerebusStart'][10, 0]][:]
start2 = data[data['timeCerebusStart2'][10, 0]][:]
num_spikes = data[data['numTotalSpikes'][10, 0]][:]
cue_on = data[data['timeCueOn'][10, 0]][:] # this is in trial time...

# print(data['isSuccessful'][0, 0])
ref = data['spikeRaster'][100, 0]
print(ref)
# print(data['spikeRaster'].shape)
# print(data['spikeRaster2'].shape)
# print(data['trialLength'])
# print(data['timeCueOn'])
# print(data[data['timeCueOn'][320, 0]])
spike_data = data[ref]['data'][:]
spike_ir = data[ref]['ir'][:]
spike_jc = data[ref]['jc'][:]
# print(data['spikeRaster'][0])
# test = data[data['spikeRaster'][0, 0]]
# print(test)
print(spike_ir.shape)
print(cue_on.shape)
#%%
test = data[data['timeCueOn'][1, 0]][()]
test = data[data['timeCueOn'][320, 0]][0, 0]
print(test)
#%%
print(spike_ir.shape)
print(spike_jc.shape)
print(spike_data.shape)
import scipy
sps_mtx = scipy.sparse.csc_matrix((spike_data, spike_ir, spike_jc))
sps_mtx = sps_mtx.toarray()
print(sps_mtx.shape)
# print(start)
# print(np.unique(spike_ir, return_counts=True))
# there are 1750 time points at which point spikes have been on
# for each of these timepoints... there are 192 units
# how do we know which ir corresponds to which time point? we can use total spikes...
# print(num_spikes)

# Raster plot for `sps_mtx`
from analyze_utils import prep_plt
import matplotlib.pyplot as plt
import seaborn as sns
def plot_spikes(spikes, ax=None, vert_space=1):

    if ax is None:
        fig, ax = plt.subplots()
    ax = prep_plt(ax)
    sns.despine(ax=ax, left=True, bottom=False)
    spike_t, spike_c = np.where(spikes)
    # prep_plt(axes[_c], big=True)
    time = np.arange(spikes.shape[0])
    ax.scatter(
        time[spike_t], spike_c * vert_space,
        # c=colors,
        marker='|',
        s=10,
        alpha=0.9
        # alpha=0.3
    )
    # time_lim = spikes.shape[0] * 0.001
    # ax.set_xticks(np.linspace(0, spikes.shape[0], 5))
    # ax.set_xticklabels(np.linspace(0, time_lim, 5))
    # ax.set_title("Benchmark Maze (Sorted)")
    # ax.set_title(context.alias)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    return ax
plot_spikes(sps_mtx.T)
print(sps_mtx.T.mean(0))

#%%
print(start.shape)
print(start2.shape)
print(cue_on)

print(spike_data.shape)
print(spike_data.all())
print(spike_ir.shape)
print(spike_ir)
print(spike_ir.min(), spike_ir.max())
print(spike_jc)
print(spike_jc.shape)

#%%
print(test['data'][:].shape)
print(test['ir'])
print(test['jc'])

#%%
import scipy

try:
    mat = scipy.io.loadmat(path, simplify_cells=True)
except NotImplementedError:
    try:
        import mat73
    except ImportError:
        raise ImportError("Must have mat73 installed to load mat73 files.")
    else:
        mat = mat73.loadmat(path)

#%%

#%%
# print((mat['R'][0]['spikeRaster']).toarray().shape)
test = pd.DataFrame(mat['R'])
print(test.columns)
print(test.timeCueOn)
# print(test.spikeRaster)
# print(test.spikeRaster.iloc[0].shape) # Ok, seems to be a sparse raster, need to find the time coords on this (but why do I actually need to? Nah...)
# print(test.spikeRaster.iloc[0])
# print(np.array(test.spikeRaster.iloc[0])[0, )
# print(test.timeCueOn)
# print(test.timeTargetOn)
# # print(test.spikeRaster2)
# # print(test.spikeRaster.iloc[0].sum(0)[0])
# print(test.trialLength[0])
# print(test.trialLength[100])
# print(test.trialLength[500])

#%%

# Nitschke
test = test[test.hasSpikes == 1]
# Mark provided a filtering script, but we won't filter as thoroughly as they do for analysis
# Specifically - we will leave: failure trials, atypical movement, poor units,
# print(len(test.unit)) # length trials
# print(len(test.unit.iloc[0])) # length both arrays (192)
# print(len(test.unit.iloc[10]))
print(len(test.unit.iloc[100][0]['spikeTimes']))
print(test.unit.iloc[100][0]['spikeTimes'])
# Get timebin
print(test.columns)
print(test.trialEndsTime.iloc[100])
print(test.moveBeginsTime.iloc[100])
print(test.commandFlyAppears.iloc[100])
# print(test.hasSpikes)
# print(mat['R'])
#%%
for t in test.iterrows():
    print(t)

#%%
from matplotlib import pyplot as plt
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

