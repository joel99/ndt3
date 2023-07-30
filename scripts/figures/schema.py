#%%
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from einops import repeat, rearrange

import logging

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.config.presets import FlatDataConfig
from context_general_bci.dataset import SpikingDataset

from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from context_general_bci.analyze_utils import prep_plt

# dataset_name = 'odoherty_rtt-Indy.*'
dataset_name = 'odoherty_rtt-Indy-20160627_01'
# dataset_name = 'odoherty_rtt-Indy-20161005_06'
# dataset_name = 'odoherty_rtt-Indy-20160630_01'
# dataset_name = 'odoherty_rtt-Indy-20160915_01'
# dataset_name = 'odoherty_rtt-Indy-20160916_01'
# dataset_name = 'odoherty_rtt-Indy-20160921_01'

# dataset_name = 'dyer_co_mihi_2'
# dataset_name = 'dyer_co_chewie_2'
# dataset_name = 'Chewie_CO_20161021'
# dataset_name = 'churchland_misc_nitschke-.*'
# dataset_name = 'churchland_misc_jenkins-10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI'
# dataset_name = 'churchland_misc_reggie-1413W9XGLJ2gma1CCXpg1DRDGpl4-uxkG'
# dataset_name = 'mc_rtt'

context = context_registry.query(alias=dataset_name)
# context = context[0]
print(context)
# datapath = './data/odoherty_rtt/indy_20160407_02.mat'
# context = context_registry.query_by_datapath(datapath)

default_cfg: DatasetConfig = OmegaConf.create(FlatDataConfig())
# default_cfg.data_keys = [DataKey.spikes]
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
default_cfg.odoherty_rtt.include_sorted = False
default_cfg.odoherty_rtt.arrays = ['Indy-M1', 'Loco-M1']
default_cfg.bin_size_ms = 20
# default_cfg.bin_size_ms = 30
default_cfg.max_arrays = min(max(1, len(context.array)), 2)
# default_cfg.max_channels = 250
default_cfg.datasets = [context.alias]
#%%
dataset = SpikingDataset(default_cfg)
dataset.build_context_index()

# import torch
# lengths = []
# for t in range(1000):
#     lengths.append(dataset[t][DataKey.spikes].size(0))
# print(torch.tensor(lengths).max(), torch.tensor(lengths).min())
print(len(dataset))
#%%
trial = 0
# trial = 10
trial_vel = dataset[trial][DataKey.bhvr_vel]

# Show kinematic trace by integrating trial_vel
print(trial_vel.shape)
trial_pos = trial_vel.cumsum(0)
trial_pos = trial_pos - trial_pos[0]
# # Plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(trial_vel)
ax[1].plot(trial_pos)

#%%
trial = 10

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

#%%
# Make schema...
pop_spikes = dataset[trial][DataKey.spikes][..., 0] # T x patch x neuron
pop_spikes = pop_spikes[:, :10]
pop_spikes = pop_spikes.flatten(1, 2)
pop_spikes = pop_spikes.T # C x T
sample_times_per_token = 5
npt = dataset.cfg.neurons_per_token

seed_everything(42)

grid_mask = (torch.rand(pop_spikes.size()) < 0.0)[::npt, ::sample_times_per_token]

grid_mask = (torch.rand(pop_spikes.size()) < 0.5)[::npt, ::sample_times_per_token]
grid_mask = (torch.rand(pop_spikes.size(1)) < 0.5)[::sample_times_per_token].unsqueeze(0).repeat(pop_spikes.size(0), 1)
# grid_mask = (torch.rand(pop_spikes.size()) < 0.5)[::npt, ::sample_times_per_token] * 0
print(grid_mask.size())

def heatmap_plot(spikes, ax=None):
    # spikes - Neurons x Time
    # spikes = torch.tensor(spikes)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    # ax = prep_plt(ax)
    # sns.despine(ax=ax, left=True, bottom=False)
    sns.heatmap(
        spikes,
        ax=ax,
        cbar=True,
        cmap='Greys',
        # cmap='crest',
        yticklabels=False,
    )

    for neuron in range(grid_mask.size(0)):
        ax.axhline(neuron * npt, color='white', lw=1.0)
        for time in range(grid_mask.size(1)):
            if neuron == 0:
                ax.axvline(time * sample_times_per_token, color='white', lw=1.0)
            if grid_mask[neuron, time]:
                ax.axvspan(
                    time * sample_times_per_token,
                    (time + 1) * sample_times_per_token,
                    (neuron) / grid_mask.size(0), # for some reason this is fractional, not pixel
                    (neuron+1) / grid_mask.size(0),
                    color='grey',
                    # alpha=0.8
                )

    ax.set_xticks([])
    # ax.set_xticks(np.linspace(0, spikes.shape[1], 5))
    # ax.set_xticklabels(np.linspace(0, spikes.shape[1] * dataset.cfg.bin_size_ms, 5).astype(int))

    ax.set_xlabel('Time', fontsize=20)
    # add rightwards arrow
    ax.annotate(
        '', xy=(0.65, -0.035), xytext=(0.58, -0.035),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", color='black')
    )

    ax.set_ylabel('Neuron', fontsize=20)
    # add upwards arrow
    ax.annotate(
        '', xy=(-0.035, 0.7), xytext=(-0.035, 0.6),
        xycoords='axes fraction', textcoords='axes fraction',
        arrowprops=dict(arrowstyle="->", color='black')
    )


    # ax.set_title("RTT Binned (Indy, 2016/06/27)")
    ax.set_title("Single-trial spiking activity", fontsize=20)
    # ax.set_title("Input (Single-trial spiking activity)", fontsize=16)
    # ax.set_title("Target", fontsize=16)
    # ax.set_title(context.alias)

    # Rescale cbar to only use 0, 1, 2, 3 for labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 1, 2, 3])
    # label cbar as "spike count"
    cbar.set_label("Spike Count")

    # remove cbar
    cbar.remove()

    return ax
heatmap_plot(pop_spikes)

def extract_grid(spikes, grid_mask):
    # Nope
    # spikes - Neurons x Time
    # grid_mask - Neurons x Time
    # spikes = torch.tensor(spikes)
    # grid_mask = torch.tensor(grid_mask)
    grid_spikes = rearrange(spikes, '(patch neurons) (pt time) -> patch pt neurons time', patch=grid_mask.size(0), pt=grid_mask.size(1))
    grid_spikes = grid_spikes[grid_mask]

    num_plots = 5
    f, axes = plt.subplots(min(num_plots, len(grid_spikes)), 1, figsize=(2, 2))
    for i in range(num_plots):
        sns.heatmap(grid_spikes[i], cbar=False, cmap='Greys', yticklabels=False, ax=axes[i])




# print(extract_grid(pop_spikes, grid_mask))