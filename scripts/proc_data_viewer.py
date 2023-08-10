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
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.config.presets import FlatDataConfig
from context_general_bci.dataset import SpikingDataset

from context_general_bci.analyze_utils import prep_plt
from context_general_bci.tasks import ExperimentalTask
# context_registry.query(task=ExperimentalTask.fbc)

# dataset_name = 'mc_maze_large' # 122 sorted units
# dataset_name = 'mc_maze_medium' # 114 sorted units
# dataset_name = 'mc_maze_small' # 107 sorted units
# dataset_name = 'mc_maze$' # 137 sorted units
# dataset_name = 'churchland_maze_jenkins-0'
dataset_name = 'churchland_maze_jenkins-1'
# dataset_name = 'churchland_maze_jenkins-2'
# dataset_name = 'churchland_maze_jenkins-3'
# dataset_name = 'churchland_maze_nitschke-0'
# dataset_name = 'churchland_maze_nitschke-1'
# dataset_name = 'churchland_maze_nitschke-2'
# dataset_name = 'churchland_maze_nitschke-3'
# dataset_name = 'churchland_maze_nitschke-4'

# dataset_name = 'mc_rtt'
# dataset_name = 'odoherty_rtt-Loco-20170215_02'

# dataset_name = 'odoherty_rtt-Loco-20170216_02'
# dataset_name = 'odoherty_rtt-Loco-20170217_02'
# dataset_name = 'odoherty_rtt-Loco-20170228_02'
# dataset_name = 'odoherty_rtt-Loco-20170301_05'
# dataset_name = 'odoherty_rtt-Loco-20170302_02'
# dataset_name = 'odoherty_rtt-Loco-20170227_04'
dataset_name = 'odoherty_rtt-Loco-20170210_03'

# dataset_name = 'odoherty_rtt-Indy.*'
# dataset_name = 'odoherty_rtt-Indy-20160627_01'
# dataset_name = 'odoherty_rtt-Indy-20160407_02'
# dataset_name = 'odoherty_rtt-Indy-20161005_06'
# dataset_name = 'odoherty_rtt-Indy-20161026_03'
# dataset_name = 'odoherty_rtt-Indy-20170131_02'
# dataset_name = 'odoherty_rtt-Indy-20161005_06'
# dataset_name = 'odoherty_rtt-Indy-20160630_01'
# dataset_name = 'odoherty_rtt-Indy-20160915_01'
# dataset_name = 'odoherty_rtt-Indy-20160916_01'
# dataset_name = 'odoherty_rtt-Indy-20160921_01'
# dataset_name = 'odoherty_rtt-Indy-20160426_01'

# dataset_name = 'dyer_co_chewie_1'
# dataset_name = 'dyer_co_mihi_1'
# dataset_name = 'dyer_co_chewie_2'
# dataset_name = 'dyer_co_mihi_2'
# dataset_name = 'Chewie_CO_20161021'
# dataset_name = 'churchland_misc_nitschke-.*'
# dataset_name = 'churchland_misc_jenkins-10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI'
# dataset_name = 'churchland_misc_reggie-1413W9XGLJ2gma1CCXpg1DRDGpl4-uxkG'
# dataset_name = 'mc_rtt'
# dataset_name = 'gallego_co_Chewie_CO_20160510'
# dataset_name = 'unstructured_CRS02bLab_session_5_set_4_type_free_play'
# dataset_name = 'observation'
# dataset_name = 'ortho'
# dataset_name = 'ortho_CRS07'
# dataset_name = "observation_CRS02bLab_session_1908_set_1"
dataset_name = 'observation_CRS02b_1953_2'
dataset_name = 'observation_CRS08_13_3'
dataset_name = 'observation_CRS08_21_3'
dataset_name = 'observation_CRS08_23_1'
dataset_name = 'observation_CRS08_23_3'
dataset_name = 'observation_CRS08_23_4'

# dataset_name = 'marino_Earl-Del*'
# dataset_name = 'marino_Earl-BCI*'
context = context_registry.query(alias=dataset_name)
if isinstance(context, list):
    context = context[0]
print(context)
# datapath = './data/odoherty_rtt/indy_20160407_02.mat'
# context = context_registry.query_by_datapath(datapath)

default_cfg: DatasetConfig = OmegaConf.create(FlatDataConfig())
default_cfg.odoherty_rtt.include_sorted = False
default_cfg.odoherty_rtt.arrays = ['Indy-M1', 'Loco-M1']
# default_cfg.data_keys = [DataKey.spikes]
# default_cfg.data_keys = [DataKey.spikes]
# default_cfg.data_keys = [DataKey.spikes, DataKey.heldout_spikes, DataKey.bhvr_vel]
default_cfg.data_keys = [DataKey.spikes] # , DataKey.bhvr_vel]
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
default_cfg.bin_size_ms = 20
default_cfg.max_channels = 288
# default_cfg.bin_size_ms = 30
default_cfg.churchland_misc.arrays = ['Jenkins-M1', 'Nitschke-M1', 'Reggie-M1']
default_cfg.max_arrays = min(max(1, len(context.array)), 2)
# default_cfg.max_channels = 250
default_cfg.datasets = [context.alias]
# default_cfg.datasets = ['observation_CRS02bHome.']

dataset = SpikingDataset(default_cfg)
dataset.build_context_index()
dataset.subset_split()
print(context.alias)
print(f'Length: {len(dataset)}')
print()

#%%
lengths = []
for t in range(len(dataset)):
    lengths.append(dataset[t][DataKey.spikes].size(0))
trial = 1
print(f'Spike shape (padded): {dataset[trial][DataKey.spikes].size()}')
print(f'Channels: {dataset[trial]["channel_counts"].sum(1)[0]}')
print(f'Timerange: {min(lengths) * dataset.cfg.bin_size_ms, max(lengths) * dataset.cfg.bin_size_ms}')

#%%
trial = 0
trial = 1
trial = 2
trial = 3
trial = 4
trial = 5
trial = 6
# trial = 10
# trial = 11

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
# Print distribution of velocities etc
vel_min = []
vel_max = []
for t in range(len(dataset)):
    trial_vel = dataset[t][DataKey.bhvr_vel]
    vel_min.append(trial_vel.min().item())
    vel_max.append(trial_vel.max().item())
sns.histplot(vel_min)
sns.histplot(vel_max)

#%%
trial = 0
trial = 1
trial = 2
# trial = 4
# trial = 10

pop_spikes = dataset[trial][DataKey.spikes]
# pop_spikes = dataset[trial][DataKey.heldout_spikes]
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

pop_spikes = pop_spikes.flatten(1)
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

#%%
sample_neurons_per_token = 16
def heatmap_plot(spikes, ax=None):
    # spikes - Time x neurons
    spikes = torch.tensor(spikes)
    spikes = spikes.T # -> neurons x time
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    ax = prep_plt(ax)
    sns.despine(ax=ax, left=True, bottom=False)
    sns.heatmap(spikes, ax=ax, cbar=True, cmap='Greys', yticklabels=False, linewidths=10)
    # for i in range(0, spikes.shape[0] + 1, sample_neurons_per_token):
        # ax.axhline(i, color='black', lw=10)
    # for i in range(spikes.shape[1] + 1):
        # ax.axvline(i, color='white', lw=1)

    ax.set_xticks(np.linspace(0, spikes.shape[1], 5))
    ax.set_xticklabels(np.linspace(0, spikes.shape[1] * dataset.cfg.bin_size_ms, 5).astype(int))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron')
    ax.set_title("RTT Binned (Indy, 2016/06/27)")
    # ax.set_title(context.alias)

    # Rescale cbar to only use 0, 1, 2, 3 for labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 1, 2, 3])
    # label cbar as "spike count"
    cbar.set_label("Spike Count")


    return ax
heatmap_plot(pop_spikes)