#%%
import numpy as np
import pandas as pd
import h5py
import torch

import logging
from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf

from context_general_bci.contexts import context_registry
from context_general_bci.config import RootConfig, DatasetConfig, DataKey, MetaKey
from context_general_bci.data import SpikingDataset
from context_general_bci.config.presets import FlatDataConfig
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.analyze_utils import prep_plt, load_wandb_run
from context_general_bci.utils import wandb_query_latest

# sample_query = 'test' # just pull the latest run
# sample_query = 'human-sweep-simpler_lr_sweep'
# sample_query =

# wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
# _, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
cfg = FlatDataConfig()
cfg.datasets = ['observation_.*']
# cfg.dataset.datasets = ['observation_CRS07Lab_session_82_set_1']
# default_cfg: DatasetConfig = OmegaConf.create(DatasetConfig())
# default_cfg.data_keys = [DataKey.spikes]
cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
dataset = SpikingDataset(cfg)
dataset.build_context_index()
dataset.subset_split()

# import torch
# lengths = []
# for t in range(1000):
#     lengths.append(dataset[t][DataKey.spikes].size(0))
# print(torch.tensor(lengths).max(), torch.tensor(lengths).min())
print(len(dataset))
#%%
print(dataset.list_alias_to_contexts(['observation_CRS07Lab_session_82_set_1_type_obs']))
#%%
from collections import defaultdict
session_stats = defaultdict(list)
for t in range(len(dataset)):
    session_stats[dataset.meta_df.iloc[t][MetaKey.session]].append(dataset[t][DataKey.bhvr_vel])
for session in session_stats:
    session_stats[session] = torch.cat(session_stats[session], 0)

#%%
sessions = list(session_stats.keys())
def summarize(s):
    return s.min().item(), s.max().item(), s.mean().item(), s.std().item(), len(s)
mins, maxes, means, stds, lengths = zip(*[summarize(session_stats[s]) for s in sessions])
# sns.histplot(mins)
# sns.histplot(maxes)
# sns.histplot(stds)
sns.histplot(means)
# sns.histplot(lengths)

# Create blacklist based on statistic cutoffs
blacklist = []
for s in sessions:
    if abs(session_stats[s].mean()) > 0.0002:
        blacklist.append(s)
    elif session_stats[s].std() < 0.001:
        blacklist.append(s)
    elif len(session_stats[s]) < 500: # remove abnormally short sessions (each trial is 100 timesteps)
        blacklist.append(s)
blacklist = [s.split('-')[-1] for s in blacklist]
with open('pitt_obs_blacklist.txt', 'w') as f:
    f.write('\n'.join(blacklist))

#%%
vels = []
for t in range(len(dataset)):
    vels.append(dataset[t][DataKey.bhvr_vel])
vels = torch.cat(vels, 0)
print(vels.shape)
#%%
torch.save({
    'mean': vels.mean(0),
    'std': vels.std(0),
}, 'pitt_obs_zscore.pt')
# print(vels.mean(0), vels.std(0))
# print(vels.min(0), vels.max(0))
# print((vels / vels.std(0)).min(0), (vels / vels.std(0)).max(0))

#%%
# trial = 0
trial = 10
# trial = 30
# trial = 10
trial_vel = dataset[trial][DataKey.bhvr_vel]

# Show kinematic trace by integrating trial_vel
print(trial_vel.shape)
trial_pos = trial_vel.cumsum(0)
trial_pos = trial_pos - trial_pos[0]
# # Plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(trial_vel)
ax[0].set_title('Velocity')
ax[1].plot(trial_pos)
ax[1].set_title('Position')

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
