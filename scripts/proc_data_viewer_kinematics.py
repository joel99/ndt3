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

# context = context_registry.query(alias='odoherty_rtt-Loco')[0]
# datapath = './data/odoherty_rtt/indy_20160407_02.mat'
# context = context_registry.query_by_datapath(datapath)
# print(context)

sample_query = 'base' # just pull the latest run to ensure we're keeping its preproc config
# sample_query = '10s_regression'

wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
run_cfg = cfg.dataset
run_cfg.datasets = ['pitt_broad_pitt_co_CRS02bLab_1776_13.*']
# run_cfg.datasets = ['pitt_broad_pitt_co_CRS02bLab_1965.*']
# run_cfg.datasets = ['pitt_broad_pitt_co_CRS02bLab_1789_2.*']
# run_cfg.datasets = ['pitt_broad_pitt_co_CRS07Home_53_4.*']
run_cfg.pitt_co.chop_size_ms = 12000
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
trial = 0
# trial = 1
# trial = 2
# trial = 10
# trial = 100
# trial = 4000
# trial = 3000
# trial = 3500
# trial = 3200
# trial = 3100
# trial = 3050
# trial = 3007

trial_name = dataset.meta_df.iloc[trial][MetaKey.unique]
test = torch.load(dataset.meta_df.iloc[trial]['path'])
print(dataset.meta_df.iloc[trial]['path'])
print('Mean: ', test['cov_mean'])
print('Min: ', test['cov_min'])
print('Max: ', test['cov_max'])

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
            # cov_pos = cov_pos - cov_pos[0]
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
# Pull the raw file, if you can
from pathlib import Path
from context_general_bci.tasks.pitt_co import load_trial, PittCOLoader, interpolate_nan
from torch.nn import functional as F
datapath = Path('data') / '/'.join(Path(dataset.meta_df.iloc[trial]['path']).parts[2:-1])
print(datapath, datapath.exists())
payload = load_trial(datapath, key='thin_data', limit_dims=run_cfg.pitt_co.limit_kin_dims)

def mock_velocity(position, kernel=np.ones((int(180 / 20), 1))/ (180 / 20)):
    position = interpolate_nan(position)
    position = position - position[0] # zero out initial position
    position = F.conv1d(position.T.unsqueeze(1), torch.tensor(kernel).float().T.unsqueeze(1), padding='same')[:,0].T
    vel = torch.as_tensor(np.gradient(position.numpy(), axis=0)).float() # note gradient preserves shape
    vel = interpolate_nan(vel) # extra call to deal with edge values
    return vel
# print(payload['force'].shape)
# print(payload['position'].shape)
# plt.plot(payload['force'], label='f')
position = payload['position'][:500,:1]
# plt.plot(position[:, 0], label='x')
kernel = np.ones((int(180 / 20), 1))/ (180 / 20)
position = F.conv1d(position.T.unsqueeze(1), torch.tensor(kernel).float().T.unsqueeze(1), padding='same')[:,0].T
# plt.plot(payload['position'][:, 0], label='x')
# plt.plot(position[:, 0], label='x smth')

# plt.plot(np.gradient(payload['position'][:, 0]), label='x')
# plt.plot(payload['position'][:, 1], label='y')
# plt.plot(PittCOLoader.smooth(payload['position'][:, 2:3]), label='z')
plt.legend()
# plt.plot(payload['position'][:, 0], label='ry')
# plt.plot(payload['position'][:, 6], label='gx')
# print(payload['position'][:, 0].max(), payload['position'][:, 0].min())
# print(payload['position'][:, 6].max(), payload['position'][:, 6].min())

covariates = PittCOLoader.get_velocity(payload['position'])
# print(covariates[:,2].max(), covariates[:,2].min())
# plt.plot(covariates[:, 2], label='vz')
# plt.plot(covariates[:, 6], label='vgx')
# plt.plot(payload['position'][3150:3200, 0], label='x')
plt.plot(covariates[:, 0], label='vx')
# %%
