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
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config.presets import FlatDataConfig, ScaleHistoryDatasetConfig
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.analyze_utils import prep_plt, load_wandb_run
from context_general_bci.utils import wandb_query_latest

# sample_query = 'test' # just pull the latest run
# sample_query = 'human-sweep-simpler_lr_sweep'
# sample_query =

# wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
# _, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
cfg = ScaleHistoryDatasetConfig()
cfg.datasets = ['pitt_broad_pitt_co_CRS02bLab_1965_1*']
print(context_registry.query(alias='pitt_broad_pitt_co_CRS02bLab_1965')[0].alias)

# cfg.dataset.datasets = ['observation_CRS07Lab_session_82_set_1']
# default_cfg: DatasetConfig = OmegaConf.create(DatasetConfig())
# default_cfg.data_keys = [DataKey.spikes]
cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
dataset = SpikingDataset(cfg)
dataset.build_context_index() # Train/val isn't going to bleed in 2 floats.
# dataset.subset_split()

# import torch
# lengths = []
# for t in range(1000):
#     lengths.append(dataset[t][DataKey.spikes].size(0))
# print(torch.tensor(lengths).max(), torch.tensor(lengths).min())
print(len(dataset))
#%%
from collections import defaultdict
session_stats = defaultdict(list)
debug = None
for t in range(len(dataset)):
    if dataset.meta_df.iloc[t][MetaKey.session] == "ExperimentalTask.pitt_co-CRS02b-1365-pitt_test_pitt_co_CRS02bLab_1365_20":
        debug = dataset[t][DataKey.bhvr_vel]
    else:
        pass
    print(dataset.meta_df.iloc[t][MetaKey.unique])
    # session_stats[dataset.meta_df.iloc[t][MetaKey.session]].append(dataset[t][DataKey.bhvr_vel])
# for session in session_stats:
    # session_stats[session] = torch.cat(session_stats[session], 0)
#%%
plt.plot(debug[:,1])
# torch.save(session_stats, 'pitt_obs_session_stats.pt')
#%%
session_stats = torch.load('pitt_obs_session_stats.pt')
sessions = list(session_stats.keys())
def summarize(s):
    return s.min().item(), s.max().item(), s.mean().item(), s.std().item(), len(s)
mins, maxes, means, stds, lengths = zip(*[summarize(session_stats[s]) for s in sessions])


#%%
print(torch.tensor(maxes).sort(descending=True).indices)
print(torch.tensor(mins).sort().indices)
print(mins[0])
print((torch.tensor(mins) < -10).sum())
print((torch.tensor(maxes) > 10).sum())
print(len(mins))
for s in sessions:
    if summarize(session_stats[s])[1] > 1000:
        print(s)
#%%
# sns.boxplot(mins)
# sns.boxplot(maxes)
# sns.boxplot(np.array(maxes) - np.array(mins))

# print quantiles of maxes


# sns.boxplot(means)

# sns.histplot(mins)
# sns.histplot(maxes)
# sns.histplot(stds)
# sns.histplot(means)
# sns.histplot(lengths)
print(session_stats[sessions[0]].max(0))

