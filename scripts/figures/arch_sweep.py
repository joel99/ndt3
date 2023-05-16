#%%
# Corresponding to exp/arch
import logging
import sys
from typing import List
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import seaborn as sns
import torch
import pytorch_lightning as pl
from einops import rearrange

from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs

from context_general_bci.analyze_utils import get_run_config, load_wandb_run, prep_plt
from context_general_bci.utils import wandb_query_latest, wandb_query_experiment

from matplotlib.colors import LogNorm, Normalize

def get_run_dict(run):
    keys = [
        'trainer/global_step',
        'eval_loss',
        'val_loss',
        'epoch',
    ]
    df = run.history(
        samples=1000, # unfortunately heavy since we need log scale
        keys=keys
    )
    out = {}
    out['val_loss'] = [df.loc[df['val_loss'].idxmin()]['val_loss']]
    out['test_loss'] = [df.loc[df['val_loss'].idxmin()]['eval_loss']]
    out['id'] = [run.id]
    out['tag'] = [run.config['tag']]
    out['experiment'] = [run.config['experiment_set']]
    out['dropout'] = [run.config['model']['dropout']]
    out['weight_decay'] = [run.config['model']['weight_decay']]
    out['hidden_size'] = [run.config['model']['hidden_size']]
    out = pd.DataFrame(out)
    return out

def get_run_df(runs):
    return pd.concat([
        get_run_dict(run) for run in runs
    ])


#%%
SORTED = False
SORTED = True
title = f'Sample HP sweep for Indy 06/27/2016 ({"Sorted" if SORTED else "Unsorted"})'

# pull experiment
if SORTED:
    experiment = [
        'arch/tune_hp',
    ]
else:
    experiment = [
        'arch/tune_hp_unsort',
    ]

runs = wandb_query_experiment(
    experiment,
    state={"$in": ['finished']},
    duration={"$gt": 300},
)


df = get_run_df(runs)
#%%
crop_labels = lambda x: x.split('-')[0]
df['label'] = df['tag'].apply(crop_labels)
CAMERA = {
    'time': 'NDT1',
    'stitch': 'NDT1 + Stitch',
    # 'single_f8': 'NDT2 - Intra (Patch 8)', # Honestly just confusing to include
    'f32': 'NDT2 (Patch 32)',
}
sub_df = df[df['label'].isin(CAMERA.keys())]
sub_df['label'] = sub_df['label'].apply(lambda x: CAMERA[x])
ax = prep_plt()
ax = sns.stripplot(
    data=sub_df,
    x='label',
    y='test_loss',
    # hue='weight_decay',
    hue='dropout',
    # style='hidden_size',
    ax=ax,
    order=['NDT2 (Patch 32)', 'NDT1 + Stitch', 'NDT1'],
)

ax.set_ylabel("Poisson NLL ($\downarrow$)")
ax.set_xlabel('')
# rotate all xlabels by 45 degrees
for item in ax.get_xticklabels():
    item.set_rotation(45)
    # increase fontsize
    item.set_fontsize(16)
ax.set_title(title, fontsize=18)