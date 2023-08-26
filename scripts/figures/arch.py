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
import lightning.pytorch as pl
from einops import rearrange

from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
from model import transfer_model

from analyze_utils import load_wandb_run
from utils import wandb_query_experiment
from analyze_utils import prep_plt, get_dataloader, stack_batch

from matplotlib.colors import LogNorm, Normalize

def eval_run(run, trials, eval_datasets=[]):
    trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
    src_model, cfg, run = load_wandb_run(run, tag='val_loss')
    if eval_datasets:
        cfg.dataset.eval_datasets = eval_datasets
    dataset = SpikingDataset(cfg.dataset)
    if cfg.dataset.eval_datasets:
        dataset.subset_split(splits=['eval'])
    else:
        dataset.subset_split()
    dataset.build_context_index()
    if not cfg.dataset.eval_datasets:
        _, dataset = dataset.create_tv_datasets()
    data_attrs = dataset.get_data_attrs()
    model = transfer_model(src_model, cfg.model, data_attrs)
    dataloader = get_dataloader(dataset, batch_size=32)
    evals = []
    for i in range(trials):
        pl.seed_everything(i)
        heldin_metrics = stack_batch(trainer.test(model, dataloader, verbose=False))
        test_nll = heldin_metrics['test_infill_loss'] if 'test_infill_loss' in heldin_metrics else heldin_metrics['test_shuffle_infill_loss']
        evals.append(test_nll.mean().item())
    return torch.tensor(evals).mean()

def get_run_dict(
    run,
    eval_trials=3, # if 0, take from wandb. else run N trials
    # eval_trials=0, # if 0, take from wandb. else run N trials
):
    keys = [
        'trainer/global_step',
        'eval_loss',
        'val_loss',
        'epoch',
    ]
    if str(ModelTask.kinematic_decoding) in run.config['model']['task']['tasks']:
        keys.append('eval_kinematic_r2')
    df = run.history(
        samples=1000, # unfortunately heavy since we need log scale
        keys=keys
    )
    out = {}
    # out['test_loss'] = [df.loc[df['eval_loss'].idxmin()]['eval_loss']]
    out['val_loss'] = [df.loc[df['val_loss'].idxmin()]['val_loss']]
    if eval_trials == 0 or 'eval_kinematic_r2' in keys: # ignore eval if doing decoding, that's deterministic
        out['test_loss'] = [df.loc[df['val_loss'].idxmin()]['eval_loss']]
    else:
        out['test_loss'] = [eval_run(run, eval_trials)]
    # Be wary of the different stories the above items tell. Using latter to be correct, former is usually what eye sees on wandb.
    out['id'] = [run.id]
    out['tag'] = [run.config['tag']]
    out['experiment'] = [run.config['experiment_set']]
    if str(ModelTask.kinematic_decoding) in run.config['model']['task']['tasks']:
        out['test_kinematic_r2'] = [df.loc[df['val_loss'].idxmin()]['eval_kinematic_r2']]
    # if out['tag'] == ['single_f8'] and 'eval_kinematic_r2' in df.columns:
        # print(run, len(df['val_loss']))
        # print(df['val_loss'].idxmin())
        # print(df['eval_kinematic_r2'].max())
    out = pd.DataFrame(out)
    return out

def get_run_df(runs, labels):
    concat = pd.concat([
        get_run_dict(run) for run in runs if run.config['tag'] in labels
    ])
    # Dedup, take lowest val loss
    concat = concat.sort_values('val_loss').drop_duplicates(['tag', 'experiment'], keep='first')
    return concat

tag_label = {
    'f8': 'NDT-2.8',
    'f32': 'NDT-2.32',
    'f128': 'NDT-2.128',
    'single_f8': 'NDT-2.8 Single',
    'single_f32': 'NDT-2.32 Single',
    'single_time': 'NDT Single',
    'time': 'NDT',
    'stitch': 'NDT-Stitch',
    'task_f32': 'NDT-2.32 Task',
    'task_f8': 'NDT-2.8 Task',
    'subject_loco_f32': 'NDT-2.32 Subject',
    'subject_loco_f8': 'NDT-2.8 Subject',
    'task_stitch': 'NDT-Stitch Task',
    'subject_loco_stitch': 'NDT-Stitch Subject',
}

#%%
title = 'Arch Unsup and Sup'

# pull experiment
experiment = [
    'arch/base',
    'arch/base/probe',
    'arch/cross',
    'arch/cross/probe',
]

runs = wandb_query_experiment(
    experiment,
    state={"$in": ['finished', 'failed', 'crashed', 'running']},
    duration={"$gt": 300},
)

run_labels = [
    'f8',
    'f32',
    'f128',
    'single_f8',
    'single_f32',
    'single_time',
    'stitch',
    'time',
    'subject_loco_f32',
    'task_f32',
    'subject_loco_stitch',
    'task_stitch',
    # 'time_8l', # Inspected in wandb to be not better than `time`
]

df = get_run_df(runs, run_labels)

# Collapse test r2 into tags
print(df.experiment)
# df[df['experiment'] == 'arch/base'].sort_values('tag')['test_kinematic_r2'] = df[df['experiment'] == 'arch/base/sup'].sort_values('tag')['test_kinematic_r2']

# Apply the r2s to the unsupervised runs
ordered_r2 = df[df['experiment'].isin(['arch/base/probe', 'arch/cross/probe'])].sort_values('tag')['test_kinematic_r2'].values
ordered_targets = df[df['experiment'].isin(['arch/base', 'arch/cross'])].sort_values('tag')['tag'].values

# use order of `ordered_targets` to apply r2s
for target, r2 in zip(ordered_targets, ordered_r2):
    df.loc[df['tag'] == target, 'test_kinematic_r2'] = r2
df = df[df['experiment'].isin(['arch/base', 'arch/cross'])]
#%%
variant_label = {
    'f8': 'session',
    'f32': 'session',
    'f128': 'session',
    'single_f8': 'single',
    'single_f32': 'single',
    'single_time': 'single',
    'time': 'session',
    'stitch': 'session',
    'task_f32': 'task',
    'subject_loco_f32': 'subject',
    'task_stitch': 'task',
    'subject_loco_stitch': 'subject',
}

df['variant'] = df['tag'].map(variant_label)

ax = prep_plt()
sns.scatterplot(
    data=df,
    x='test_loss',
    y='test_kinematic_r2',
    hue='variant',
    ax=ax,
)

ax.set_ylabel('Velocity $R^2$')
ax.set_xlabel('Test NLL')

tag_label = {
    'f8': 'NDT-2.8',
    'f32': 'NDT-2.32',
    'f128': 'NDT-2.128',
    'single_f8': 'NDT-2.8 Single',
    'single_f32': 'NDT-2.32 Single',
    'single_time': 'NDT Single',
    'time': 'NDT',
    'stitch': 'NDT-Stitch',
    'task_f32': 'NDT-2.32 Task',
    'subject_loco_f32': 'NDT-2.32 Subject',
    'task_stitch': 'NDT-Stitch Task',
    'subject_loco_stitch': 'NDT-Stitch Subject',
}

# Annotate each of the points with their tag
for i, row in df.iterrows():
    ax.annotate(
        # row['tag'], # Will figure out cosmetics later
        tag_label[row['tag']],
        (row['test_loss'], row['test_kinematic_r2']),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center'
    )


#%%

# Unsorted
title = 'Arch Unsup and Sup'

# pull experiment
experiment = [
    'arch/base_unsort',
    'arch/base_unsort/probe',
    'arch/cross_unsort',
    'arch/cross_unsort/probe',
]

runs = wandb_query_experiment(
    experiment,
    state={"$in": ['finished', 'failed', 'crashed', 'running']},
    duration={"$gt": 300},
)

run_labels = [
    'f8',
    'f32',
    'f128',
    'single_f8',
    # 'single_f32',
    'single_time',
    'stitch',
    'time',
    'subject_loco_f8',
    # 'subject_loco_f32',
    'task_f8',
    # 'task_f32',
    'subject_loco_stitch',
    'task_stitch',
    # 'time_8l', # Inspected in wandb to be not better than `time`
]

df = get_run_df(runs, run_labels)

# Collapse test r2 into tags
print(df.experiment)
# df[df['experiment'] == 'arch/base'].sort_values('tag')['test_kinematic_r2'] = df[df['experiment'] == 'arch/base/sup'].sort_values('tag')['test_kinematic_r2']

# Apply the r2s to the unsupervised runs
ordered_r2 = df[df['experiment'].isin(['arch/base_unsort/probe', 'arch/cross_unsort/probe'])].sort_values('tag')['test_kinematic_r2'].values
ordered_targets = df[df['experiment'].isin(['arch/base_unsort', 'arch/cross_unsort'])].sort_values('tag')['tag'].values

# use order of `ordered_targets` to apply r2s
for target, r2 in zip(ordered_targets, ordered_r2):
    df.loc[df['tag'] == target, 'test_kinematic_r2'] = r2
df = df[df['experiment'].isin(['arch/base_unsort', 'arch/cross_unsort'])]
#%%
variant_map = {
    'task': 'task',
    'subject': 'subject'
}
variant_label = lambda x: variant_map.get(x.split('_')[0], 'session')

df['variant'] = df['tag'].map(variant_label)

ax = prep_plt()
sns.scatterplot(
    data=df,
    x='test_loss',
    # y='test_loss',
    y='test_kinematic_r2',
    hue='variant',
    ax=ax,
)

ax.set_ylabel('Velocity $R^2$')
ax.set_xlabel('Test NLL')

# Annotate each of the points with their tag
for i, row in df.iterrows():
    ax.annotate(
        # row['tag'], # Will figure out cosmetics later
        tag_label[row['tag']],
        # (row['test_loss'], row['test_loss']),
        (row['test_loss'], row['test_kinematic_r2']),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center'
    )

