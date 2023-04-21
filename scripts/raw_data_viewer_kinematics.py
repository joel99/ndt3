#%%
from typing import List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from math import ceil

import h5py
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import scipy.signal as signal

from einops import rearrange, reduce
from nlb_tools.make_tensors import PARAMS, _prep_mask, make_stacked_array
from nlb_tools.nwb_interface import NWBDataset

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import logging

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.contexts import context_registry, ContextInfo
from context_general_bci.analyze_utils import prep_plt

## Load dataset

dataset_name = 'odoherty_rtt-Loco-20170215_02'
dataset_name = 'odoherty_rtt-Indy-20161005_06'
context = context_registry.query(alias=dataset_name)
datapath = context.datapath
# Use modes to alternate through a few different codepaths
mode = 'rtt'
# mode = 'pitt'
if mode == 'rtt':
    ctxs = context_registry.query(task=ExperimentalTask.odoherty_rtt)
else:
    # ctxs = context_registry.query(task=ExperimentalTask.observation)
    ctxs = context_registry.query(task=ExperimentalTask.ortho)
    ctxs = context_registry.query(task=ExperimentalTask.fbc)
session_paths = [ctx.datapath for ctx in ctxs]
print(f'Found {len(session_paths)} sessions.')

sampling_rate = 1000
cfg = DatasetConfig()
cfg.bin_size_ms = 20
# cfg.bin_size_ms = 5
def load_bhvr_from_rtt(datapath, sample_strat=None):
    with h5py.File(datapath, 'r') as h5file:
        # orig_timestamps = np.squeeze(h5file['t'][:])
        # time_span = int((orig_timestamps[-1] - orig_timestamps[0]) * sampling_rate)
        if cfg.odoherty_rtt.load_covariates:
            covariate_sampling = 250 # Hz
            def resample(data, sample_strat=sample_strat):
                if not sample_strat:
                    return data
                if sample_strat == 'decimate':
                    downsample_factor = int(covariate_sampling / (1000 / cfg.bin_size_ms))
                    return torch.tensor(
                        signal.decimate(data, downsample_factor, axis=0).copy()
                    )
                elif sample_strat == 'resample':
                    return torch.tensor(
                        signal.resample(data, int(len(data) / covariate_sampling / (cfg.bin_size_ms / 1000)))
                    )
                elif sample_strat == 'resample_poly':
                    return torch.tensor(
                        signal.resample_poly(data, sampling_rate / covariate_sampling, cfg.bin_size_ms, padtype='line')
                    )
            bhvr_vars = {}
            for bhvr in ['finger_pos']:
            # for bhvr in ['finger_pos', 'cursor_pos', 'target_pos']:
                bhvr_vars[bhvr] = h5file[bhvr][()].T
                # bhvr_vars[bhvr] = resample(bhvr_vars[bhvr])
            # cursor_vel = np.gradient(cursor_pos[~np.isnan(cursor_pos[:, 0])], axis=0)
            finger_vel = np.gradient(bhvr_vars['finger_pos'], axis=0)
            bhvr_vars[DataKey.bhvr_vel] = torch.tensor(finger_vel)

            for bhvr in bhvr_vars:
                bhvr_vars[bhvr] = resample(bhvr_vars[bhvr])
    return bhvr_vars

def load_bhvr_from_pitt(datapath, sample_strat=None):
    from tasks.pitt_co import load_trial
    trial_paths = list(datapath.glob("*.mat"))
    payloads = [load_trial(trial_path) for trial_path in trial_paths]
    return payloads

def plot_trace_rtt(
    ax, bhvr_payload,
    length=2000,
    title: Path | None = None,
    key=DataKey.bhvr_vel,
): # check baseline qualitative
    # ax = prep_plt(ax)
    finger_vel = bhvr_payload[key]
    if length:
        finger_vel = finger_vel[:length]
    ax.plot(finger_vel[:, 0], label='x')
    ax.plot(finger_vel[:, 1], label='y')
    # ax.plot(finger_vel[:, 0], finger_vel[:, 1])
    # ax.set_xlim(-0.2, 0.2)
    # ax.set_ylim(-0.2, 0.2)
    ax.legend()
    if title is not None:
        ax.set_title(title.stem)

def plot_trace_pitt(
    ax, bhvr_payload, trial=0,
    title: Path | None = None,
    key = 'position'
):
    # plot 2 traces of x/y profiles, not the 2d trace (since paths are often stereotyped)
    # ax = prep_plt(ax)
    bhvr_payload = bhvr_payload[trial]
    pos = bhvr_payload['position']
    pos = gaussian_filter1d(pos, 2.5, axis=0) # 2.5 bins = 50ms - this (by JY's judgment) gets rid of the system artifacts while relatively preserving most kinematic features
    if key == 'position':
        ax.plot(pos[:, 0], label='x')
        ax.plot(pos[:, 1], label='y')
    elif key == DataKey.bhvr_vel:
        # do velocity
        # apply a gaussian kernel to smooth the readout position, which is spiky due to uninteresting system reasons
        # Without this, the velocity profiles are jagged
        vel = np.gradient(pos, axis=0)
        ax.plot(vel[:, 0], label='x')
        ax.plot(vel[:, 1], label='y')
    ax.legend()
    if title is not None:
        ax.set_title(title.stem)
#%%
tgt_session = session_paths[0]
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = prep_plt(ax=ax)

strat = None
# strat = 'resample'
# strat = 'resample_poly'
# strat = 'decimate'
key = DataKey.bhvr_vel
# key = 'position'

if mode == 'pitt':
    bhvr_payload = load_bhvr_from_pitt(tgt_session, sample_strat=strat)
    plot_trace_pitt(
        ax, bhvr_payload, title=tgt_session, trial=0,
        key=key,
    )
    # print(bhvr_payload[1]['position'][:,0])
    # print(len(bhvr_payload))
    # plot_trace_pitt(ax, bhvr_payload, title=tgt_session, trial=1)

if mode == 'rtt':
    # take 1st 10% of signal
    bhvr_payload = load_bhvr_from_rtt(tgt_session, sample_strat=strat)
    print(bhvr_payload[DataKey.bhvr_vel].max(), bhvr_payload[DataKey.bhvr_vel].min())
    length = int(bhvr_payload['finger_pos'].shape[0] * 0.01)
    plot_trace_rtt(
        ax, bhvr_payload, title=session_paths[0], length=length,
        key='finger_pos' if key == 'position' else DataKey.bhvr_vel,
    )
    title = strat if strat else 'no resample'
    # ax.set_title(tgt_session.stem + ' ' + 'position')
    ax.set_title(f'{title} {key}')

# bin size 5
# `resample` better save for edge artifacts

# bin size 20
# `resample_poly` most similar to raw data, `decimate` slight offset, `resample` has edge artifacts as before
# ? what the heck, where is all this data coming from..
#%%
if mode == 'rtt':
    import seaborn as sns
    # make a 2x2
    f, axs = plt.subplots(2, 2, figsize=(10 * 2, 10 * 2))
    axs = axs.ravel()
    sns.histplot(bhvr_payload[DataKey.bhvr_vel][:,0], label='x', ax=axs[0])
    axs[0].set_title('velocity x')
    sns.histplot(bhvr_payload[DataKey.bhvr_vel][:,1], label='y', ax=axs[1])
    axs[1].set_title('velocity y')
    # do positions
    sns.histplot(bhvr_payload['finger_pos'][:,0], label='x', ax=axs[2])
    axs[2].set_title('position x')
    sns.histplot(bhvr_payload['finger_pos'][:,1], label='y', ax=axs[3])
    axs[3].set_title('position y')
#%%

# plot all sessions by loading behavior and calling `plot_trace`
if mode == 'pitt':
    fig, axs = plt.subplots(
        ceil(len(session_paths) / 2), 2,
        figsize=(10 * 2, 10 * ceil(len(session_paths) / 2))
    )
    for i, session_path in enumerate(session_paths):
        bhvr_payload = load_bhvr_from_pitt(session_path, sample_strat=None)
        plot_trace_pitt(axs.ravel()[i], bhvr_payload, title=session_path, trial=0, key='position')
        # plot_trace_pitt(axs.ravel()[i], bhvr_payload, title=session_path, trial=0, key=DataKey.bhvr_vel)
        # plot_trace_pitt(axs.ravel()[i], bhvr_payload, title=session_path, trial=1)
        # plot_trace_pitt(axs.ravel()[i], bhvr_payload, title=session_path, trial=2)

if mode == 'rtt':
    fig, axs = plt.subplots(
        ceil(len(session_paths) / 2), 2,
        figsize=(10 * 2, 10 * ceil(len(session_paths) / 2))
    )
    for i, session_path in enumerate(session_paths):
        bhvr_payload = load_bhvr_from_rtt(session_path, sample_strat=None)
        plot_trace_rtt(axs.ravel()[i], bhvr_payload, title=session_path, length=10000)
