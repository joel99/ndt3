#%%
import numpy as np
import pandas as pd
import torch

import logging
from matplotlib import pyplot as plt
import seaborn as sns

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.tasks import ExperimentalTask

from context_general_bci.analyze_utils import prep_plt, load_wandb_run
from context_general_bci.utils import wandb_query_latest

mode = 'rtt'
# mode = 'pitt'
if mode == 'rtt':
    ctxs = context_registry.query(task=ExperimentalTask.odoherty_rtt)
else:
    ctxs = context_registry.query(task=ExperimentalTask.pitt_co)

context = ctxs[0]
context = context_registry.query(alias='odoherty_rtt-Loco')[0]
print(context)
# datapath = './data/odoherty_rtt/indy_20160407_02.mat'
# context = context_registry.query_by_datapath(datapath)

sample_query = 'base' # just pull the latest run to ensure we're keeping its preproc config
sample_query = '10s_regression'

wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
default_cfg = cfg.dataset
dataset = SpikingDataset(default_cfg)
dataset.build_context_index()
dataset.subset_split()

print(len(dataset))
#%%
trial = 0
# trial = 10
trial = 4000
trial = 3000
trial = 3500
# trial = 3200
# trial = 3100
# trial = 3050
# trial = 3007

trial_name = dataset.meta_df.iloc[trial][MetaKey.unique]
test = torch.load(dataset.meta_df.iloc[trial]['path'])
print(test['cov_mean'], test['cov_min'], test['cov_max'])

trial_cov = dataset[trial][DataKey.bhvr_vel]
print(f'Covariate shape: {trial_cov.shape}')
cov_dims = dataset[trial][DataKey.covariate_labels]
cov_space = dataset[trial].get(DataKey.covariate_space, None)

if DataKey.constraint in dataset[trial]:
    constraints = dataset[trial][DataKey.constraint]
    print(f'Constraint shape: {constraints.shape}')
    f, axes = plt.subplots(2, 1, sharex=True)
else:
    f, axes = plt.subplots(1, 1)
    axes = [axes]

def plot_covs(ax, cov, cov_dims, cov_space: torch.Tensor | None =None):
    ax = prep_plt(ax=ax, big=True)
    if cov_space is None:
        for cov, label in zip(trial_cov.T, cov_dims):
            if cov_dims != 'f':
                cov_pos = cov.cumsum(0)
            else:
                cov_pos = cov
            cov_pos = cov_pos - cov_pos[0]
            ax.plot(cov_pos, label=label)
    else:
        for unique_space in cov_space.unique():
            cov_pos = cov[cov_space == unique_space].cumsum(0)
            cov_pos = cov_pos - cov_pos[0]
            ax.plot(cov_pos, label=cov_dims[unique_space])
    ax.legend()
    ax.set_ylabel('Position')

def plot_constraints(ax):
    ax = prep_plt(ax=ax, big=True)
    ax.plot(constraints)
    ax.set_title('Constraints')

plot_covs(axes[0], trial_cov, cov_dims, cov_space)
if DataKey.constraint in dataset[trial]:
    plot_constraints(axes[1])
axes[0].set_title(trial_name)
#%%
# iterate through trials and print min and max bhvr_vel
min_vel = 0
max_vel = 0
for trial in range(len(dataset)):
    trial_vel = dataset[trial][DataKey.bhvr_vel]
    min_vel = min(min_vel, trial_vel.min())
    max_vel = max(max_vel, trial_vel.max())
print(min_vel, max_vel)

#%%
trial = 10
trial = 26

pop_spikes = dataset[trial][DataKey.spikes]
pop_spikes = pop_spikes[..., 0]
# print diagnostics
# print(pop_spikes[::2].sum(0))
# print(pop_spikes[1::2].sum(0))
# sns.histplot(pop_spikes[::2].sum(0))
# sns.histplot(pop_spikes[1::2].sum(0) - pop_spikes[0::2].sum(0))
print(
    f"Mean: {pop_spikes.float().mean():.2f}, \n"
    f"Std: {pop_spikes.float().std():.2f}, \n"
    f"Max: {pop_spikes.max():.2f}, \n"
    f"Min: {pop_spikes.min():.2f}, \n"
    f"Shape: {pop_spikes.shape}"
)

pop_spikes = pop_spikes.flatten(1, 2)
# pop_spikes = pop_spikes[:, :96]
# wait... 250?
# path_to_old = './data/old_nlb/mc_maze.h5'
# with h5py.File(path_to_old, 'r') as f:
#     print(f.keys())
#     pop_spikes = f['train_data_heldin']
#     pop_spikes = torch.tensor(pop_spikes)
#     print(pop_spikes.shape)
# pop_spikes = pop_spikes[trial]

print(pop_spikes.shape)
# print(pop_spikes.sum(0) / 0.6)
# print(pop_spikes.sum(0))
# Build raster scatter plot of pop_spikes
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
    time_lim = spikes.shape[0] * dataset.cfg.bin_size_ms
    ax.set_xticks(np.linspace(0, spikes.shape[0], 5))
    ax.set_xticklabels(np.linspace(0, time_lim, 5))
    # ax.set_title("Benchmark Maze (Sorted)")
    ax.set_title(context.alias)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    return ax
plot_spikes(pop_spikes)
