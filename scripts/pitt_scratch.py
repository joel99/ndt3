#%%
r"""
    TODO
    Identify .mat files for relevant trials
    Open .mat
    Find the phases (observation) that are relevant for now
    Find the spikes
    Find the observed kinematic traces

    Batch for other sessions
"""
import pandas as pd
import numpy as np
# import xarray as xr
from pathlib import Path
import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch

data_dir = Path("./data/pitt_co/")
session = 173 # pursuit

data_dir = Path('./data/pitt_misc/mat')
session = 7
# session = 1407 # co

# session_dir = data_dir / f'CRS02bHome.data.{session:05d}'
session_dir = data_dir.glob(f'*{session}*obs.mat').__next__()
if not session_dir.exists():
    print(f'Session {session_dir} not found; Run `prep_all` on the QL .bin files.')
print(session_dir)

def extract_ql_data(ql_data):
    # ql_data: .mat['iData']['QL']['Data']
    # Currently just equipped to extract spike snippets
    # If you want more, look at `icms_modeling/scripts/preprocess_mat`
    # print(ql_data.keys())
    # print(ql_data['TASK_STATE_CONFIG'].keys())
    # print(ql_data['TASK_STATE_CONFIG']['state_num'])
    # print(ql_data['TASK_STATE_CONFIG']['state_name'])
    # print(ql_data['TRIAL_METADATA'])
    def extract_spike_snippets(spike_snippets):
        THRESHOLD_SAMPLE = 12./30000
        return {
            "spikes_source_index": spike_snippets['source_index'], # JY: I think this is NSP box?
            "spikes_channel": spike_snippets['channel'],
            "spikes_source_timestamp": spike_snippets['source_timestamp'] + THRESHOLD_SAMPLE,
            # "spikes_snippets": spike_snippets['snippet'], # for waveform
        }

    return {
        **extract_spike_snippets(ql_data['SPIKE_SNIPPET']['ss'])
    }

def events_to_raster(
    events,
    channels_per_array=128,
):
    """
        Tensorize sparse format.
    """
    events['spikes_channel'] = events['spikes_channel'] + events['spikes_source_index'] * channels_per_array
    bins = np.arange(
        events['spikes_source_timestamp'].min(),
        events['spikes_source_timestamp'].max(),
        0.001
    )
    timebins = np.digitize(events['spikes_source_timestamp'], bins, right=False) - 1
    spikes = torch.zeros((len(bins), 256), dtype=torch.uint8)
    spikes[timebins, events['spikes_channel']] = 1
    return spikes


from context_general_bci.tasks.pitt_co import load_trial
from context_general_bci.analyze_utils import prep_plt
payload = load_trial(session_dir, key='thin_data')

print(payload.keys())
#%%
# Make raster plot
fig, ax = plt.subplots(figsize=(10, 10))

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
    time_lim = spikes.shape[0] * 0.02
    ax.set_xticks(np.linspace(0, spikes.shape[0], 5))
    ax.set_xticklabels(np.linspace(0, time_lim, 5))
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    return ax
plot_spikes(payload['spikes'], ax=ax)

#%%
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
# In pursuit tasks - start of every trial is a hold (maybe half a ms?) and then a mvmt. Some end with a hold.
# Boxcar doesn't look great (super high targets) but it's what pitt folks have been using this whole time so we'll keep it

def get_velocity(position, smooth_time_ms=500):
    # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
    kernel = np.ones((int(smooth_time_ms / 20), 2)) / (smooth_time_ms / 20)
    # print(kernel, position.shape)
    position = convolve(
        position,
        kernel,
        mode='same'
    )
    # return position
    vel = torch.tensor(np.gradient(position, axis=0)).float()
    vel[vel.isnan()] = 0 # extra call to deal with edge values
    return vel

def try_clip_on_trial_boundary(vel, time_thresh_ms=1000, trial_times=None):
    # ! Don't use this. Philosophy: Don't make up new data, just don't use these times.
    # Clip away trial bound jumps in position  - JY inferring these occur when the cursor resets across trials
    trial_bounds = np.where(np.diff(trial_times))[0]
    time_bins = int(time_thresh_ms / 20) # cfg.bin_size_ms
    for tb in trial_bounds:
        vel[tb - time_bins: tb + time_bins, :] = 0
    return vel

# Plot behavior
fig, ax = plt.subplots(figsize=(20, 10))
def plot_behavior(bhvr, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax = prep_plt(ax)
    ax.plot(bhvr[:,0])
    ax.plot(bhvr[:,1])
    # scale xticks labels as each is 20ms
    time_lim = bhvr.shape[0] * 0.02
    ax.set_xticks(np.linspace(0, bhvr.shape[0], 5))
    ax.set_xticklabels(np.linspace(0, time_lim, 5))
    ax.set_xlabel('Time (s)')

def plot_trials(trial_times, ax=None):
    # trial_times is a long vector of the particular trial
    # find when trial changes and draw a vertical demarcation

    if ax is None:
        fig, ax = plt.subplots()
    ax = prep_plt(ax)

    step_times = list(np.where(np.diff(trial_times))[0])
    # print(step_times)
    step_times.append(trial_times.shape[0])
    for step_time in step_times:
        ax.axvline(step_time - 50, linestyle='--', color='k', alpha=0.1)
        ax.axvline(step_time + 25, linestyle='--', color='k', alpha=0.1)
        ax.axvline(step_time, color='k', alpha=0.1)

# vel = get_velocity(payload['position']) # TODO low pri - is this notebook function old? Why is it diff from my PittCOLoader.get_velocity?
# vel = try_clip_on_trial_boundary(vel, trial_times=payload['trial_num'])
# plot_behavior(payload['position'], ax=ax)

from context_general_bci.tasks.pitt_co import PittCOLoader
vel = PittCOLoader.get_velocity(payload['position'])
refit_vel = PittCOLoader.ReFIT(payload['position'], payload['target'])

# plot the times when payload['target'].any(-1) is nan
print(payload['target'].isnan())
# for nan_time in payload['target'].isnan().any(-1).nonzero():
    # ax.axvline(nan_time, color='r', alpha=0.1)

# plot_behavior(vel, ax=ax)
plot_behavior(refit_vel, ax=ax)
plot_trials(payload['trial_num'], ax=ax)

# We're not smooth because these trials are damn long?

#%%