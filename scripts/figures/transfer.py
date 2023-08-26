#%%
import logging
import sys
from typing import List
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
from model import transfer_model, logger

from analyze_utils import get_run_config, load_wandb_run, wandb_query_latest, wandb_query_experiment
from analyze_utils import prep_plt

from matplotlib.colors import LogNorm, Normalize

# pull experiment
experiment = [
    'multi_vs_single/single_unsup_sort',
    'multi_vs_single/multi_unsup_sort',
    'multi_vs_single/cross_task_sort',
]

# call wandb api to pull all runs with this experiment tag

runs = wandb_query_experiment(
    experiment,
    state={"$in": ['finished', 'crashed']},
    duration={"$gt": 300},
)

def get_train_dataset_size(wandb_run):
    cfg = get_run_config(wandb_run, tag='val_loss')
    dataset = SpikingDataset(cfg.dataset)
    dataset.subset_split()
    dataset.subset_scale(
        limit_per_session=cfg.dataset.scale_limit_per_session,
        limit_per_eval_session=cfg.dataset.scale_limit_per_eval_session,
        keep_index=True
    )
    return len(dataset)

def get_run_dict(run):
    df = run.history(
        samples=1000, # unfortunately heavy since we need log scale
        keys=[
            'trainer/global_step',
            'eval_loss',
            'val_loss',
            'epoch',
        ]
    )
    out = {}
    # out['test_loss'] = [df.loc[df['eval_loss'].idxmin()]['eval_loss']]
    out['test_loss'] = [df.loc[df['val_loss'].idxmin()]['eval_loss']]
    # Be wary of the different stories the above items tell.
    out['id'] = [run.id]
    out['tag'] = [run.config['tag']]
    out['dataset_size'] = [len(run.config['dataset']['datasets']) * run.config['dataset']['scale_limit_per_session']]
    if out['dataset_size'] == [0]:
        out['dataset_size'] = [get_train_dataset_size(run)]
    out = pd.DataFrame(out)
    return out

def get_run_df(runs, labels):
    return pd.concat([
        get_run_dict(run) for run in runs if run.config['tag'] in labels
    ])

#%%
title = 'Test loss against dataset size'
run_labels = [
    'single_100', 'single_200', 'single_400', 'single_800', 'single_1600',
    'loco_200', 'loco_400', 'loco_800', 'loco_6400',
    'multi_200', 'multi_400', 'multi_800', 'multi_1600', 'multi_all',
    'reach_7000', 'reach_80000',
]

groups = {
    'single_100': 'single',
    'single_200': 'single',
    'single_400': 'single',
    'single_800': 'single',
    'single_1600': 'single',
    'loco_200': 'subject',
    'loco_400': 'subject',
    'loco_800': 'subject',
    'loco_6400': 'subject',
    'multi_200': 'session',
    'multi_400': 'session',
    'multi_800': 'session',
    'multi_1600': 'session',
    'multi_all': 'session',
    'reach_7000': 'task',
    'reach_80000': 'task',
}

df = get_run_df(runs, run_labels)
df['group'] = df['tag'].map(groups)
#%%

ax = prep_plt()
sns.lineplot(
    data=df,
    x='dataset_size',
    y='test_loss',
    hue='group',
    # style='tag',
    ax=ax,
    alpha=0.7,
    legend=True
)

# Relabel legend
# relabel = {
#     'scale_stitch': 'NDT1 + Stitch',
#     'scale_nonflat': 'NDT1 (Time)',
#     'scale1': 'NDT2 (4-Neuron)',
#     'scale1_t32': 'NDT2 (32-Neuron)',
#     'scale1_t144': 'NDT2 (128-Neuron)',
# }
# # offsets = {
# #     'scale_stitch': (-50, -10),
# #     'scale_nonflat': (0, 30) if mode == 'step' else (-30, 30),
# #     'scale1': (-40, 30) if mode == 'step' else (-60, -10),
# #     'scale1_t32': (0, 30) if mode == 'step' else (-30, 30),
# #     'scale1_t144': (0, 30) if mode == 'step' else (-30, 30),
# # }
# handles, labels = ax.get_legend_handles_labels()
# labels = [relabel[label] for label in labels]
# ax.legend(handles, labels, frameon=False)

ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylim(0.25, 0.33)
ax.set_xlabel('Dataset Size (Training Samples)')
ax.set_ylabel('Test loss')

# Only mark min and max on y-axis
ax.set_xticks([100, 400, 1600, 6400, 25600, 102400])
ax.set_xticklabels([100, 400, 1600, 6400, 25600, 102400])
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax.yaxis.set_minor_locator(ticker.NullLocator())

# rotate ytick labels by 90 degrees
# do not use scientific notation
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.tick_params(axis='y', rotation=90)

#%%
title = "Test loss against FT size"
experiment = [
    'multi_vs_single/single_unsup_sort',
    'pt_vs_ft',
]

runs = wandb_query_experiment(
    experiment,
    state={"$in": ['finished', 'crashed']},
    duration={"$gt": 300},
)

run_labels = [
    'ft_800_100', 'ft_800_200', 'ft_800_400', 'ft_800_800', 'ft_800_1600',
    'ft_1600_100', 'ft_1600_200', 'ft_1600_400', 'ft_1600_800', 'ft_1600_1600',
    'ft_80k_100', 'ft_80k_200', 'ft_80k_400', 'ft_80k_800', 'ft_80k_1600',
    'single_100', 'single_200', 'single_400', 'single_800', 'single_1600',
]

groups = lambda x: x.split('_')[-2]

df = get_run_df(runs, run_labels)
df['group'] = df['tag'].map(groups)

#%%

ax = prep_plt()
sns.lineplot(
    data=df,
    x='dataset_size',
    y='test_loss',
    hue='group',
    # style='tag',
    ax=ax,
    alpha=0.7,
    legend=True
)

ax.set_xscale('log')
ax.set_yscale('log')
# ax.set_ylim(0.25, 0.33)
ax.set_xlabel('In-session Training Samples')
ax.set_ylabel('Test loss')

ax.set_title('Finetune vs from scratch')