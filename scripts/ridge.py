#%%
from typing import List
import numpy as np
import pandas as pd
import h5py
import torch
import scipy.signal as signal
import logging
from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from einops import rearrange

# Try ridge regression
# from nlb_tools.evaluation import fit_and_eval_decoder
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, explained_variance_score
from sklearn.model_selection import GridSearchCV

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.config.presets import FlatDataConfig
from context_general_bci.dataset import SpikingDataset

from context_general_bci.analyze_utils import prep_plt, DataManipulator
from context_general_bci.tasks import ExperimentalTask

dataset_name = 'observation_CRS02b_1953_2'
# dataset_name = 'observation_CRS07_150_1'

context = context_registry.query(alias=dataset_name)
if isinstance(context, list):
    context = context[0]
print(context)

default_cfg: DatasetConfig = OmegaConf.create(FlatDataConfig())
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
default_cfg.datasets = [context.alias]
dataset = SpikingDataset(default_cfg)
dataset.build_context_index()
dataset.subset_split() # get train
train, val = dataset.create_tv_datasets()


def smooth_spikes(
    dataset: SpikingDataset, gauss_bw_ms=80
) -> List[torch.Tensor]:
    # Smooth along time axis
    return [DataManipulator.gauss_smooth(
        rearrange(i[DataKey.spikes].float(), 't s c 1 -> 1 t (s c)'),
        bin_size=dataset.cfg.bin_size_ms,
        kernel_sd=gauss_bw_ms,
    ).squeeze(0) for i in dataset]

spike_smth_range = [20, 60, 100, 200, 400, 600]

def sweep_ridge_fit(zero_filt_train=True, zero_filt_eval=True):
    trains = []
    evals = []
    for i in spike_smth_range:
        smth_train = smooth_spikes(train, gauss_bw_ms=i)
        smth_val = smooth_spikes(val, gauss_bw_ms=i)
        train_rates = np.concatenate(smth_train, 0)
        train_behavior = np.concatenate([i[DataKey.bhvr_vel] for i in train], 0)
        eval_rates = np.concatenate(smth_val, 0)
        eval_behavior = np.concatenate([i[DataKey.bhvr_vel] for i in val], 0)
        decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-1, 3, 50)})

        # ? zero filter
        if zero_filt_train:
            train_rates = train_rates[~(np.abs(train_behavior) < 1e-5).all(-1)]
            train_behavior = train_behavior[~(np.abs(train_behavior) < 1e-5).all(-1)]
        if zero_filt_eval:
            eval_rates = eval_rates[~(np.abs(eval_behavior) < 1e-5).all(-1)]
            eval_behavior = eval_behavior[~(np.abs(eval_behavior) < 1e-5).all(-1)]

        decoder.fit(train_rates, train_behavior)
        evals.append(decoder.score(eval_rates, eval_behavior))
        trains.append(decoder.score(train_rates, train_behavior))

    return evals, trains

eval_filt, train_filt = sweep_ridge_fit()
eval_no_filt, train_no_filt = sweep_ridge_fit(zero_filt_train=False, zero_filt_eval=True)

#%%
# Set color palette and line styles
palette = sns.color_palette("Set1", 2)
line_styles = ['-', '--']

# Prepare plot
fig, ax = plt.subplots()
ax = prep_plt(ax)
ax.set_xlabel('(Acausal) Gauss smth kernel ms')
ax.set_ylabel('Decode R2')
ax.axhline(0.8, color='k', linestyle='--', label='reported R2')
ax.legend()

# Plot data
ax.plot(spike_smth_range, train_filt, label='train filtered', color=palette[0], linestyle=line_styles[1])
ax.plot(spike_smth_range, eval_filt, label='eval filtered', color=palette[0], linestyle=line_styles[0])
ax.plot(spike_smth_range, train_no_filt, label='train unfiltered', color=palette[1], linestyle=line_styles[1])
ax.plot(spike_smth_range, eval_no_filt, label='eval unfiltered', color=palette[1], linestyle=line_styles[0])

# Add linestyle and color info to legend
handles, labels = ax.get_legend_handles_labels()
labels = ['{} ({})'.format(label, ls) for label, ls in zip(labels, ['--', '-', '--', '-'])]
legend = ax.legend(handles, labels, loc='best')

#%%
# Debug
trial = 0

ex_train = smooth_spikes(train)[trial]
ex_train_vel = train[trial][DataKey.bhvr_vel]

# Plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(ex_train_vel)
ax[0].set_title('Velocity')
ax[1].plot(ex_train)
ax[1].set_title('Rates')

#%%
