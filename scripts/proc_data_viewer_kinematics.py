#%%
r""" What does raw data look like? (Preprocessing devbook) """
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

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
from context_general_bci.utils import wandb_query_latest, unflatten

sample_query = 'small_40m_dense-72yzibgz'
# wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
wandb_run = wandb_query_latest(sample_query, allow_running=True, use_display=True)[0]

# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss', load_model=False)
run_cfg = cfg.dataset
cfg.dataset.eval_datasets = []
cfg.dataset.exclude_datasets = []

run_cfg.datasets = [
    # 'pitt_broad_pitt_co_BMI01Lab_231.*',
    # 'pitt_broad_pitt_co_BMI01Lab_1_.*',
    # 'pitt_broad_pitt_co_BMI01Lab_97_16.*',
    # 'pitt_broad_pitt_co_BMI01Lab_296.*',
    # FBC Helicopter

    # Helicopter session
    # 'pitt_broad_pitt_co_CRS02bLab_1942.*',
    # 'pitt_broad_pitt_co_CRS02bLab_1942_1',
    # 'pitt_broad_pitt_co_CRS02bLab_1942_2',
    # 'pitt_broad_pitt_co_CRS02bLab_1942_3',

    # 'pitt_broad_pitt_co_CRS02bLab_1942_6',

    'pitt_broad_pitt_co_CRS02bLab_1942_7',
    # 'pitt_broad_pitt_co_CRS02bLab_1942_8',

    # Force
    # 'pitt_test_pitt_co_CRS07Home_108_1',
    # 'pitt_test_pitt_co_CRS07Lab_95_6',
    # 'pitt_test_pitt_co_CRS07Lab_78_10',
    # 'pitt_return_pitt_co_CRS07Home_108_1',

    # Force
    # 'pitt_broad_pitt_co_CRS07Home_108_1',
    # 'pitt_broad_pitt_co_CRS07Home_108_3',
    # More helicopter rescue
    # 'pitt_broad_pitt_co_CRS07Home_118_1',
    # 'pitt_broad_pitt_co_CRS07Home_118_5',

    # Archival
    # 'pitt_broad_pitt_co_CRS02bLab_100_1',

#     'pitt_broad_pitt_co_CRS07Home_32',
#     'pitt_broad_pitt_co_CRS07Home_33',
#     'pitt_broad_pitt_co_CRS07Home_34',
#     'pitt_broad_pitt_co_CRS07Home_35',
#     'pitt_broad_pitt_co_CRS07Home_52', # mislabel
#   # - pitt_broad_pitt_co_CRS07Lab_52
#     'pitt_broad_pitt_co_CRS07Home_53',
#     'pitt_broad_pitt_co_CRS07Home_49',
#   # - pitt_broad_pitt_co_CRS07Home_57 # ! Not found
#     'pitt_broad_pitt_co_CRS07Home_69',
#     'pitt_broad_pitt_co_CRS07Home_71',
#     'pitt_broad_pitt_co_CRS07Home_83',
#     'pitt_broad_pitt_co_CRS07Home_88',
#     'pitt_broad_pitt_co_CRS07Home_61',
#     'pitt_broad_pitt_co_CRS07Home_108',

    # 'pitt_broad_pitt_co_CRS02bLab_1776_1.*'
    # 'pitt_broad_pitt_co_CRS02bLab_245_12.*'
    # 'pitt_broad_pitt_co_CRS08Lab_9_.*',
    # 'pitt_broad_pitt_co_CRS07Home_108_.*',

    # 'dyer_co.*',
    # 'gallego_.*', # Gallego
    # 'odoherty_rtt-Indy-20160627_01.*',
    # 'churchland_maze_jenkins.*',
    # 'churchland_misc_jenkins.*',
    # 'churchland_misc_jenkins-10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI',
    # 'churchland_maze_nitschke.*',
    # 'churchland_misc_reggie-1413.*',
    # 'churchland_misc_reggie-151n.*',
    # 'churchland_misc_reggie-19eu.*', # clear
    # 'churchland_misc_reggie-1TFV.*', # sus
    # 'churchland_misc_reggie-1eeP.*', # sus-ish. Magnitudes are so small!
    # 'delay.*'
]

dataset = SpikingDataset(run_cfg)
dataset.build_context_index()
dataset.subset_split()

print(len(dataset))
names = context_registry.query(alias=dataset.cfg.datasets[0])
if isinstance(names, list):
    for n in names:
        print(n.alias)
else:
    print(names.alias)
#%%
has_brain_control = {}
dimensions = {}
for session in dataset.meta_df[MetaKey.session].unique():
    session_df = dataset.meta_df[dataset.meta_df[MetaKey.session] == session]
    all_constraints = []
    for trial in session_df.index:
        if DataKey.constraint in dataset[trial]:
            constraints = dataset[trial][DataKey.constraint]
            all_constraints.append(constraints)
        if session not in dimensions:
            dimensions[session] = dataset[trial][DataKey.covariate_labels]
            break
    # all_constraints = torch.cat(all_constraints, dim=0)
    # print(f'Session {session}: {all_constraints.shape}')
    # has_brain_control[session] = (all_constraints[:, 0] < 1).any()
from pprint import pprint
pprint(dimensions)

# for session in has_brain_control:
#     if not has_brain_control[session]:
#         print(session)


#%%
from pathlib import Path

# trial_indices = range(40)
# trial_indices = np.arange(0, 36, 9)
trial_indices = np.arange(0, 27, 9)
# trial_indices = np.arange(8)
# trial_indices = np.arange(12)+24
# trial_indices = np.arange(12)+24+12
# trial_indices = np.arange(5)
# trial_indices = np.arange(3)+24+26
# trial_indices = range(4)
# trial_indices = range(2)
USE_CONSTRAINT = False
USE_RETURN = False
# USE_RETURN = True
def plot_covs(
        ax,
        trial_cov,
        cov_dims,
        cov_space: torch.Tensor | None =None,
        fontsize=18,
    ):
    # print(trial_cov.shape)
    ax = prep_plt(ax=ax, size='huge')
    if cov_space is None:
        for cov, label in zip(trial_cov.T, cov_dims):
            if label != 'f':
                cov_pos = cov # Avoid cumsum to avoid visual discontinuity across trials
                # cov_pos = cov.cumsum(0)
            else:
                cov_pos = cov
            cov_pos = cov_pos - cov_pos[0]
            ax.plot(cov_pos, label=label)
    else:
        for i, unique_space in enumerate(cov_space.unique()):
            label = cov_dims[i]
            cov_pos = trial_cov[cov_space == unique_space]
            # if label not in ['rx', 'ry']:
                # continue
            if label != 'f':
                # cov_pos = cov_pos.cumsum(0)
                cov_pos = cov_pos # Avoid cumsum to avoid visual discontinuity across trials
            # cov_pos = cov_pos - cov_pos[0]
            ax.plot(cov_pos, label=cov_dims[i])
    ax.set_ylim(-1, 1)
    # Set major y-ticks
    ax.set_yticks([-1, 0, 1])

    # Set minor y-ticks
    ax.set_yticks(np.arange(-1, 1.1, 0.25), minor=True)

    # Enable minor grid lines
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.3)

    ax.set_ylabel('Velocity (au)')
    # Convert xticks to 20x ms
    xticks = ax.get_xticks()
    xticks = np.arange(0, xticks.max(), 50)
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks / 50).astype(int))
    # Set all fontsizes huge


def plot_spikes(ax, spikes, spike_times, spike_positions):
    r"""
        Make raster plot.
        spikes: (n_tokens, n_neurons_patch)
        spike_times: (n_tokens, 1)
        spike_positions: (n_tokens, 1)
    """
    ax = prep_plt(ax=ax, size='huge')
    spike_population = unflatten(spikes.unsqueeze(0), spike_times.unsqueeze(0), spike_positions.unsqueeze(0)).squeeze(0)
    # Do imshow instead
    ax.imshow(spike_population.T, aspect='auto', cmap='gray_r')
    ax.set_ylabel("Channels")
    # Raster plot
    # spike_time, spike_space = spike_population.nonzero().T
    # ax.scatter(spike_time, spike_space, s=5, marker='|', color='k')

def plot_constraints(ax_row, dataset, col, trial):
    constraints = dataset[trial][DataKey.constraint]
    times = dataset[trial][DataKey.constraint_time]
    space = dataset[trial][DataKey.constraint_space]
    # print('Constraint:', constraints.shape)
    prep_plt(ax_row[col], size="huge")
    # print(constraints.shape, times)
    # If off axis, that means it's on
    space_offset = constraints / 2 + constraints / 24 * space.unsqueeze(-1)
    for i, pos in enumerate(space.unique()):
        ax_row[col].scatter(times[space == pos], space_offset[:, 0][space == pos], label='fbc-lock', marker='x', s=500, color=palette[i], alpha=0.5)
        ax_row[col].scatter(times[space == pos], space_offset[:, 1][space == pos] + 1, label='active', marker='+', s=500, color=palette[i], alpha=0.5)
        ax_row[col].scatter(times[space == pos], space_offset[:, 2][space == pos] + 2, label='passive', marker='1', s=500, color=palette[i], alpha=0.5)
    # Draw a hline at 0.4
    ax_row[col].axhline(y=0.9, color='k', linestyle='--')
    ax_row[col].axhline(y=1.9, color='k', linestyle='--')

    # ax_row[col].set_ylim(-0.1, 3.1)
    # plot_constraints(axes[1, col], constraints, times)

def plot_rewards(ax_row, dataset, col, trial):
    returns = dataset[trial][DataKey.task_return]
    rewards = dataset[trial][DataKey.task_reward]
    times = dataset[trial][DataKey.task_return_time]
    prep_plt(ax_row[col], size="huge")
    ax_row[col].scatter(times, returns[:, 0], label='return', marker='^', s=1000)
    ax_row[col].scatter(times, rewards[:, 0] + 0.25, label='reward', marker='^', s=1000) # Offset for visibility
    # ax_row[col].set_ylim(-0.1, 8.1)
    # Put minor gridlines every 1
    ax_row[col].yaxis.set_minor_locator(plt.MultipleLocator(1))
    # Turn on grid
    ax_row[col].grid(which='minor', axis='y', linestyle='--')

def plot_multiple_trials(trial_indices, dataset):
    # Determine the number of rows needed for constraints
    use_constraint = USE_CONSTRAINT and any(DataKey.constraint in dataset[trial] for trial in trial_indices)
    use_return = USE_RETURN and any(DataKey.task_return in dataset[trial] for trial in trial_indices)
    n_rows = 1 + 1 + int(use_constraint) + int(use_return)
    height_ratios = [1] * n_rows
    height_ratios[1] = 2  # 2nd row will be 3 times larger
    # height_ratios[1] = 3  # 2nd row will be 3 times larger

    f, axes = plt.subplots(
        n_rows, len(trial_indices),
        sharex=True, sharey='row',
        figsize=(12, 10),
        # figsize=(len(trial_indices) * 8, 8),
        gridspec_kw={'height_ratios': height_ratios},
    )
    f.suptitle(dataset.cfg.datasets)

    cur_trial_name = ""
    palette = sns.color_palette(n_colors=10)

    for col, trial in enumerate(trial_indices):
        # trial_name = dataset.meta_df.iloc[trial][MetaKey.unique]
        # test = torch.load(dataset.meta_df.iloc[trial]['path'])
        trial_name = '_'.join(Path(dataset.meta_df.iloc[trial]['path']).stem.split('_')[-4:-1])
        # print(dataset.meta_df.iloc[trial]['path'])
        if trial_name != cur_trial_name:
            cur_trial_name = trial_name
            # Plot vertical, annotate with rotated trial name
            axes[0, col].axvline(x=0, color='k', linestyle='--')
            axes[0, col].text(0.1, 0.5, trial_name, rotation=90, transform=axes[0, col].transAxes, fontsize=8)

        # Extract trial data
        trial_cov = dataset[trial][DataKey.bhvr_vel]
        print(trial_cov.shape)
        cov_dims = dataset[trial][DataKey.covariate_labels]
        cov_space = dataset[trial].get(DataKey.covariate_space, None)
        # Plot covariates
        plot_covs(axes[0, col], trial_cov, cov_dims, cov_space)

        # Plot spikes
        plot_spikes(
            axes[1, col],
            dataset[trial][DataKey.spikes][..., 0],
            dataset[trial][DataKey.time],
            dataset[trial][DataKey.position]
        )

        if use_constraint and DataKey.constraint in dataset[trial]:
            plot_constraints(axes[2], dataset, col, trial)

        if use_return and DataKey.task_return in dataset[trial]:
            plot_rewards(axes[2 + int(use_constraint)], dataset, col, trial)

    for final_ax in axes[:, -1]:
        final_ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    for i in range(axes.shape[1]):
        axes[-1, i].set_xlabel('Time (s)')

plot_multiple_trials(trial_indices, dataset)

#%%
# Pull the raw file, if you can
from pathlib import Path
from context_general_bci.tasks.pitt_co import load_trial, PittCOLoader, interpolate_nan
from context_general_bci.config import DEFAULT_KIN_LABELS
from torch.nn import functional as F
datapath = Path('data') / '/'.join(Path(dataset.meta_df.iloc[trial]['path']).parts[2:-1])
print(datapath, datapath.exists())
payload = load_trial(datapath, key='thin_data', limit_dims=run_cfg.pitt_co.limit_kin_dims)
covariates_smth = PittCOLoader.get_velocity(payload['position'])
# covariates_smth = PittCOLoader.get_velocity(payload['position'], kernel=np.ones((5, 1)) / 5)
covariates_raw = PittCOLoader.get_velocity(payload['position'], kernel=np.ones((1, 1)))
brain_control = payload.get('brain_control', None)
active_assist = payload.get('active_assist', None)
passive_assist = payload.get('passive_assist', None)

print(brain_control.shape)
print(covariates_smth.shape)
print(payload['position'].shape)
# constraints = payload[DataKey.constraint]
# constraint_time = payload[DataKey.constraint_time]
# print(constraints)
# print(covariates[:,2].max(), covariates[:,2].min())
# plt.plot(covariates[:, 2], label='vz')
# plt.plot(covariates[:, 6], label='vgx')
# plt.plot(payload['position'][3150:3200, 0], label='x')
ax = prep_plt()
raw_dims = [0]
raw_dims = [1, 2]
raw_dims = [1, 2, 6]
# raw_dims = [1, 2, 6]
# raw_dims = [6]
raw_dims = [6]
# raw_dims = [8]
# xlim = [500, 600]
# xlim = [0, 200]
raw_dims = []
xlim = []
# xlim = [1000, 2000]
# xlim = [1000, 4000]
palette = sns.color_palette(n_colors=len(raw_dims) + 2)
for i, r in enumerate(raw_dims):
    # ax.plot(covariates_smth[:, r], label=f'{DEFAULT_KIN_LABELS[r]} smth', color=palette[i])
    # ax.plot(covariates_raw[:, r], label=f'{DEFAULT_KIN_LABELS[r]} raw', color=palette[i], linestyle='--')
    ax.plot(payload['position'][:, r], label=f'{DEFAULT_KIN_LABELS[r]} raw', color=palette[i], linestyle='--')

ax.plot(payload['force'], label='Force')
ax.set_yscale('log')
ax.set_ylim(1e-2, 2e3)
# ax.plot(brain_control[:, 0] * 0.01, label='fbc-lock')
# ax.plot(active_assist[:, 0] * 0.01, label='active p', color=palette[-2])
# ax.plot(active_assist[:, 1] * 0.02, label='active r')
# ax.plot(active_assist[:, 2] * 0.03, label='active g', color=palette[-1])
# ax.plot(passive_assist[:, 0] * 0.01, label='passiv')

ax.legend()
if xlim:
    ax.set_xlim(xlim)
# *20 to convert xticks to ms
xticks = ax.get_xticks()
ax.set_xticks(xticks)
ax.set_xticklabels((xticks / 50).astype(int))
# ax.set_xticklabels((xticks * 20).astype(int))
ax.set_xlabel('s')
ax.set_title(f'Velocity {datapath.stem}')

#%%
passed = payload['passed']
print(payload['trial_num'])