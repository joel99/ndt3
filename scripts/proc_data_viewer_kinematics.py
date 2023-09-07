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

context = context_registry.query(alias='odoherty_rtt-Loco')[0]
print(context)
# datapath = './data/odoherty_rtt/indy_20160407_02.mat'
# context = context_registry.query_by_datapath(datapath)

sample_query = 'base' # just pull the latest run to ensure we're keeping its preproc config
# sample_query = '10s_regression'

wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
default_cfg = cfg.dataset
default_cfg.pitt_co.chop_size_ms = 10000
dataset = SpikingDataset(default_cfg)
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
    all_constraints = torch.cat(all_constraints, dim=0)
    # print(f'Session {session}: {all_constraints.shape}')
    has_brain_control[session] = (all_constraints[:, 0] < 1).any()

# print(has_brain_control)
# print(dimensions)
# print those without brain control
for session in has_brain_control:
    if not has_brain_control[session]:
        print(session)
#%%
trial = 0
# trial = 10
# trial = 4000
# trial = 3000
# trial = 3500
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

use_constraint = False and DataKey.constraint in dataset[trial]
if use_constraint:
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
            if label != 'f':
                cov_pos = cov.cumsum(0)
            else:
                cov_pos = cov
            cov_pos = cov_pos - cov_pos[0]
            ax.plot(cov_pos, label=label)
    else:
        for i, unique_space in enumerate(cov_space.unique()):
            label = cov_dims[i]
            cov_pos = cov[cov_space == unique_space]
            if label != 'f':
                cov_pos = cov_pos.cumsum(0)
            cov_pos = cov_pos - cov_pos[0]
            ax.plot(cov_pos, label=cov_dims[i])
    # plot legend off side
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylabel('Position')
    # Convert xticks to 20x ms
    xticks = ax.get_xticks()
    ax.set_xticklabels((xticks * 20).astype(int))
    ax.set_xlabel('Time (ms)')

def plot_constraints(ax):
    ax = prep_plt(ax=ax, big=True)
    ax.plot(constraints)
    ax.set_title('Constraints')

plot_covs(axes[0], trial_cov, cov_dims, cov_space)
if use_constraint:
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