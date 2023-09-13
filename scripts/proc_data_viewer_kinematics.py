#%%
r""" What does raw data look like? (Preprocessing devbook) """
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

sample_query = 'base' # just pull the latest run to ensure we're keeping its preproc config
sample_query = '10s_loco_regression'

# Return run
sample_query = 'sparse'

wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
run_cfg = cfg.dataset
# run_cfg.datasets = ['pitt_broad_pitt_co_CRS02bLab_1776_13.*']
# run_cfg.datasets = ['pitt_broad_pitt_co_CRS02bLab_1965.*']
# run_cfg.datasets = ['pitt_broad_pitt_co_CRS02bLab_1789_2.*']
run_cfg.datasets = [
    'pitt_return_pitt_co_CRS07Lab_97_13',
    'pitt_return_pitt_co_CRS07Lab_97_15',
    'pitt_return_pitt_co_CRS07Lab_97_17',
] # 13, 15, 17 are all FBC

run_cfg.datasets = [
    # 'pitt_broad_pitt_co_CRS02bLab_1942.*',
    # 'pitt_broad_pitt_co_CRS02bLab_1942_3',
    'pitt_broad_pitt_co_CRS02bLab_1942_6',
]

dataset = SpikingDataset(run_cfg)
dataset.build_context_index()
dataset.subset_split()

print(len(dataset))
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
# print(has_brain_control)
pprint(dimensions)
# for session in has_brain_control:
#     if not has_brain_control[session]:
#         print(session)

#%%
from pathlib import Path
def plot_covs(ax, trial_cov, cov_dims, cov_space: torch.Tensor | None =None):
    # print(trial_cov.shape)
    ax = prep_plt(ax=ax, big=True)
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
    # ax.set_ylabel('Position')
    # Convert xticks to 20x ms
    # xticks = ax.get_xticks()
    # ax.set_xticklabels((xticks * 20).astype(int))
    # ax.set_xlabel('Time (ms)')


def plot_multiple_trials(trial_indices, dataset):
    n_trials = len(trial_indices)

    # Determine the number of rows needed for constraints
    n_constraint_rows = 1 if any(DataKey.constraint in dataset[trial] for trial in trial_indices) else 0
    n_return_rows = 1 if any(DataKey.task_return in dataset[trial] for trial in trial_indices) else 0
    # Create subplots
    f, axes = plt.subplots(
        1 + n_constraint_rows + n_return_rows, len(trial_indices),
        sharex=True, sharey='row',
        figsize=(n_trials * 8, 8),
    )

    cur_trial_name = ""
    for col, trial in enumerate(trial_indices):
        # trial_name = dataset.meta_df.iloc[trial][MetaKey.unique]
        # test = torch.load(dataset.meta_df.iloc[trial]['path'])
        trial_name = '_'.join(Path(dataset.meta_df.iloc[trial]['path']).stem.split('_')[-4:-1])
        if trial_name != cur_trial_name:
            # print(trial_name)
            cur_trial_name = trial_name
            # Plot vertical
            axes[0, col].axvline(x=0, color='k', linestyle='--')
            # annotate with rotated trial name
            axes[0, col].text(0.1, 0.5, trial_name, rotation=90, transform=axes[0, col].transAxes, fontsize=8)
        # print('Mean: ', test['cov_mean'])
        # print('Min: ', test['cov_min'])
        # print('Max: ', test['cov_max'])

        # trial_cov = dataset[trial][DataKey.bhvr_vel]
        # print(f'Covariate shape: {trial_cov.shape}')

        # Extract trial data
        trial_cov = dataset[trial][DataKey.bhvr_vel]
        cov_dims = dataset[trial][DataKey.covariate_labels]
        cov_space = dataset[trial].get(DataKey.covariate_space, None)

        # Plot covariates
        plot_covs(axes[0, col], trial_cov, cov_dims, cov_space)

        # Plot constraints if available
        if DataKey.constraint in dataset[trial]:
            constraints = dataset[trial][DataKey.constraint]
            times = dataset[trial][DataKey.constraint_time]
            # print(constraints)
            prep_plt(axes[1, col], big=True)
            # print(constraints.shape, times)
            axes[1, col].scatter(times, constraints[:, 0], label='fbc-lock', marker='|')
            # axes[1, col].scatter(times, constraints[:, 1], label='active', marker='|')
            # axes[1, col].scatter(times, constraints[:, 2], label='passive', marker='|')
            # axes[1 + n_constraint_rows, col].set_ylim(-0.1, 3.1)
            # plot_constraints(axes[1, col], constraints, times)

        if DataKey.task_return in dataset[trial]:
            returns = dataset[trial][DataKey.task_return]
            rewards = dataset[trial][DataKey.task_reward]
            # print(returns)
            # print(returns.shape)
            times = dataset[trial][DataKey.task_return_time]
            prep_plt(axes[1 + n_constraint_rows, col], big=True)
            axes[1 + n_constraint_rows, col].scatter(times, returns[:, 0], label='return', marker='|')
            axes[1 + n_constraint_rows, col].scatter(times, rewards[:, 0], label='reward', marker='|')
            # axes[1 + n_constraint_rows, col].set_ylim(-0.1, 8.1)
            # Put minor gridlines every 1
            axes[1 + n_constraint_rows, col].yaxis.set_minor_locator(plt.MultipleLocator(1))
            # Turn on grid
            axes[1 + n_constraint_rows, col].grid(which='minor', axis='y', linestyle='--')
    axes[0, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2, col].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Usage
trial_indices = [0, 1, 2]  # Add the trial indices you want to plot
trial_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Add the trial indices you want to plot
# trial_indices = range(40)
trial_indices = range(12)
trial_indices = range(2)
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

covariates = PittCOLoader.get_velocity(payload['position'])
# print(covariates[:,2].max(), covariates[:,2].min())
# plt.plot(covariates[:, 2], label='vz')
# plt.plot(covariates[:, 6], label='vgx')
# plt.plot(payload['position'][3150:3200, 0], label='x')
print(covariates.shape)
print(covariates[498:502, 2])
ax = prep_plt()
raw_dims = [0]
raw_dims = [2]
# raw_dims = [1, 2, 6]
xlim = [0, 1000]
for i in raw_dims:
    ax.plot(covariates[:, i], label=DEFAULT_KIN_LABELS[i])
ax.legend()
ax.set_xlim(xlim)
# *20 to convert xticks to ms
ax.set_xticklabels((ax.get_xticks() / 50).astype(int))
ax.set_xlabel('s')
ax.set_title(f'Velocity {datapath.stem}')