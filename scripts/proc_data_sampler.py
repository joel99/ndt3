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
from context_general_bci.config import DatasetConfig, DataKey, MetaKey, DEFAULT_KIN_LABELS
from context_general_bci.dataset import SpikingDataset
from context_general_bci.tasks import ExperimentalTask

from context_general_bci.analyze_utils import prep_plt, load_wandb_run
from context_general_bci.utils import wandb_query_latest

sample_query = 'base' # just pull the latest run to ensure we're keeping its preproc config
sample_query = '10s_loco_regression'

# Return run
sample_query = 'sparse'
sample_query = 'h512_l6_return'
sample_query = 'base'
sample_query = 'bhvr_12l_512'
wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss', load_model=False)
run_cfg = cfg.dataset
run_cfg.datasets = ['pitt_broad.*']
run_cfg.datasets = ['pitt_test_pitt_co_CRS07Home_108_1'] # dummy accelerate load

# run_cfg.data_keys = [*run_cfg.data_keys, 'cov_min', 'cov_max', 'cov_mean'] # Load these directly, for some reason they're cast
dataset = SpikingDataset(run_cfg)
dataset.build_context_index()
dataset.subset_split()

#%%
from tqdm import tqdm
set_stats = {} # No set level granularity, will have to build from labels later
session_stats = {}
trial_stats = [] # No tracking...
dimensions = {}

def process_session(meta_session):
    session_stats = {}
    trial_stats = []
    session_df = dataset.meta_df[dataset.meta_df[MetaKey.session] == meta_session]
    # breakpoint() # get the session
    all_constraints = []
    session_length = 0
    dim = None
    for trial in session_df.index: # Oh, we don't have set level granularity...
        if DataKey.constraint in dataset[trial]:
            constraints = dataset[trial][DataKey.constraint]
            all_constraints.append(constraints)
        if dim is None:
            dim = dataset[trial][DataKey.covariate_labels]
        mode = torch.mode(dataset[trial][DataKey.bhvr_vel].flatten())
        mode_value = mode.values[0] if len(mode.values.shape) >= 1 else mode.values
        mode_count = (dataset[trial][DataKey.bhvr_vel].flatten() == mode_value).sum()
        total_count = dataset[trial][DataKey.bhvr_vel].flatten().shape[0]
        trial_stats.append({
            'length': dataset[trial][DataKey.time].max(),
            'mode': mode_value.item(),
            'mode_count': mode_count.item(),
            'total_count': total_count,
            'max_return': dataset[trial][DataKey.task_return].max().item(),
        })
        session_length += trial_stats[-1]['length']
    payload = torch.load(session_df.iloc[-1].path)
    for k in ['cov_min', 'cov_max', 'cov_mean']:
        if k not in payload or payload[k] == None:
            payload[k] = torch.zeros(dataset.cfg.behavior_dim)
        elif payload[k].shape[0] == 8: # No force
            payload[k] = torch.cat([payload[k], torch.zeros(1)])
    subject, session, set = meta_session.split('_')[-3:]
    session_stats[meta_session] = {
        "subject": subject,
        "session": int(session),
        "set": int(set),
        "length": session_df.shape[0],
        "dimensions": dim,
        "cov_min": payload["cov_min"],
        "cov_max": payload["cov_max"],
        "cov_mean": payload["cov_mean"],
        "session_length": session_length.item(),
        "has_brain_control": (torch.cat(all_constraints)[:, 0] < 1).any().item(),
    }
    # if len(session_stats) > 10:
        # break # trial
    return session_stats, trial_stats
global_session = {}
global_trial = []
unique_sessions = dataset.meta_df[MetaKey.session].unique()
total_sessions = len(unique_sessions)

results = []
for meta_session in tqdm(unique_sessions):
    results.append(process_session(meta_session))
    # if len(results) > 2:
        # break
combined_trial_stats = []
combined_session_stats = {}
for session_stat, trial_stat in results:
    combined_trial_stats.extend(trial_stat)
    combined_session_stats.update(session_stat)
# for meta_session in tqdm(dataset.meta_df[MetaKey.session].unique()):
    # session_stats[meta_session], trial_stats = process_session(meta_session)
torch.save({
    'session': combined_session_stats,
    'trial': combined_trial_stats,
}, 'scripts/proc_data_sampler.pt')
from pprint import pprint
# pprint(dimensions)
# for session in has_brain_control:
#     if not has_brain_control[session]:
#         print(session)
#%%
payload = torch.load('scripts/proc_data_sampler.pt')
session_stats = payload['session']
trial_stats = payload['trial']
print(len(trial_stats))

sessions = pd.DataFrame.from_dict(session_stats, orient='index')
print(sessions.columns)

# * Count overview
plt.figure(figsize=(8, 5))
sns.countplot(x='has_brain_control', hue='subject', data=sessions)
plt.title('Sessions with Brain Control to No Brain Control')
plt.xlabel('Has Brain Control')
plt.ylabel('Count')
plt.show()
# * Heavy skew to CRS02bLab.
#%%
# * Session lengths, roughly
fig, axs = plt.subplots(2, 1, figsize=(16, 10))

# Plot the histogram for 'length'
sns.histplot(sessions, x='length', bins=100, label='pseudotrials', color='blue', kde=True, alpha=0.5, ax=axs[0])
axs[0].set_title('Pseudotrials count per session/set')
axs[0].set_xlabel('Length (Pseudotrials)')
axs[0].set_ylabel('Frequency')
axs[0].legend()

# Plot the histogram for 'session_length'
sns.histplot(sessions, x='session_length', bins=100, label='timesteps (20ms)', color='red', kde=True, alpha=0.5, ax=axs[1])
axs[1].set_title('Total bins per session/set')
axs[1].set_xlabel('Length (Timebins 20ms)')
axs[1].set_ylabel('Frequency')
axs[1].legend()

plt.tight_layout()
plt.show()


#%%
plt.figure(figsize=(12, 5))
ax = plt.gca()
ax = prep_plt(ax)
sns.ecdfplot(sessions, x='session_length', label='timesteps (20ms)', color='blue')
plt.title('CDF of Session Lengths')
plt.xlabel('Length')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.show()

#%%
melted_sessions = pd.melt(sessions, value_vars=['session_length', 'length'], var_name='Type', value_name='Length')

# Plot the histogram
plt.figure(figsize=(12, 5))
sns.histplot(melted_sessions, x='Length', hue='Type', bins=1000, kde=True, alpha=0.5)
plt.title('Histogram of Session and Trial Lengths')
plt.xlabel('Length')
plt.ylabel('Frequency')

#%%
# * Dimensions
print(sessions.columns)
print(sessions['dimensions'])
# Net occurrences
# Flatten the list of dimensions
flattened_dimensions = [dim for sublist in sessions['dimensions'].dropna() for dim in sublist]

# Create a DataFrame from the flattened list
dimensions_df = pd.DataFrame(flattened_dimensions, columns=['Dimension'])

# Plot the barplot for dimension occurrences
plt.figure(figsize=(10, 5))
sns.countplot(data=dimensions_df, x='Dimension', order=dimensions_df['Dimension'].value_counts().index)
plt.title('Occurrences of Each Dimension')
plt.xlabel('Dimension')
plt.ylabel('Set Count')
plt.show()

#%%
# Dimensionality of tasks
# Compute the number of dimensions for each row
sessions['num_dimensions'] = sessions['dimensions'].apply(lambda x: len(x) if x is not None else 0)

# Plot the histogram
plt.figure(figsize=(10, 5))
sns.histplot(sessions, x='num_dimensions', bins=np.arange(sessions['num_dimensions'].min(), sessions['num_dimensions'].max() + 1) - 0.5, kde=False)
plt.title('Histogram of Number of Dimensions per Row')
plt.xlabel('Number of Dimensions')
plt.ylabel('Frequency')
plt.show()

#%%
# Pull session with 7 dimensions
print(sessions[sessions['num_dimensions'] == 7][['subject', 'session', 'set', 'has_brain_control']])

#%%
# View cov min and max
# flattened_cov_min = [torch.cat(sublist) for sublist in sessions['cov_min'].dropna()]
#%%
print(sessions['cov_min'][0].shape)
print(sessions['cov_min'][3].shape)
print(sessions['dimensions'][0])
#%%
sessions['dim_subset_mask'] = sessions['dimensions'].apply(lambda x: np.array([DEFAULT_KIN_LABELS.index(i) for i in x]))
# print([len(i) for i in dim_subset_mask.tolist()])
# print([len(i) for i in sessions['cov_min'].tolist()])
# assert all([len(i) <= len(j) for i, j in zip(dim_subset_mask.tolist(), sessions['cov_min'].tolist())])
# for i, j in zip(dim_subset_mask.tolist(), cov_min_ext.tolist()):
#     print(j.shape)
#     print(i)
#     print(j[i])
#%%
flattened_cov_min = torch.cat(sessions.apply(lambda x: torch.cat([x.cov_min, torch.zeros(1)])[x.dim_subset_mask], axis=1).tolist())
flattened_cov_max = torch.cat(sessions.apply(lambda x: torch.cat([x.cov_max, torch.zeros(1)])[x.dim_subset_mask], axis=1).tolist())
flattened_cov_mean = torch.cat(sessions.apply(lambda x: torch.cat([x.cov_mean, torch.zeros(1)])[x.dim_subset_mask], axis=1).tolist())

cov_df = pd.DataFrame({
    'min': flattened_cov_min,
    'max': flattened_cov_max,
    'mean': flattened_cov_mean,
    'dim': flattened_dimensions,
})

# Make a square figure
# f = plt.figure(figsize=(10, 10))
# ax = plt.gca()
# ax = prep_plt(ax)
# sub_df = cov_df[cov_df['dim'] != 'f']
# ax = sns.scatterplot(data=sub_df, x='min', y='max', hue='dim', ax=ax)

# Exclude rows where 'dim' is 'f'
sub_df = cov_df[cov_df['dim'] != 'null']

# Create FacetGrid
g = sns.FacetGrid(sub_df, col="dim", col_wrap=3, sharex=False, sharey=False)
g.map(sns.scatterplot, "min", "max", s=10, alpha=0.5)
g.set_axis_labels("Min", "Max")
g.set_titles(col_template="{col_name}")
g.fig.suptitle('Min Max of Covariates across sets')

#%%
trials = pd.DataFrame.from_dict(trial_stats)
print(trials.columns)
# all lengths are tensors, cast to item
trials['length'] = trials['length'].apply(lambda x: x.item())
ax = sns.histplot(trials, x='length', bins=100, label='pseudotrials', color='blue', kde=True, alpha=0.5)
ax.set_yscale("log")
ax.set_ylim(1, 5e6)
# print(trials['length'])

#%%
# Plot max return
print(trials.max_return)
print(trials.max_return.max()) # Max return set to 13. So it's not a max return issue - something about min reward or something like that... then.
# A reward of 13 in 15s - totally plausible, in observation. I literally achieve it in my pursuit tasks.
ax = sns.histplot(trials, x='max_return', bins=100, label='pseudotrials', color='blue', kde=True, alpha=0.5)