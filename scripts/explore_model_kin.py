#%%
r"""
    What do model behavioral predictions look like? (Devbook)
"""
# restrict cuda to gpu 1
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import logging
import sys
import itertools
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.model import transfer_model, logger
from context_general_bci.contexts import context_registry

from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt
from context_general_bci.utils import get_wandb_run, wandb_query_latest


mode = 'train'
# mode = 'test'

query = 'base-i9ix8t7q'
query = 'no_blacklist_no_session-xn7zeqpo'
query = 'no_sem-73nhkuba'
# query = 'grasp_icl-0rrikpqe'

query = '10s_loco_regression'
# query = '10s_indy_regression'

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

# cfg.model.task.metrics = [Metric.kinematic_r2]
cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]

# target_dataset = 'pitt_broad_CRS02bLab_1925_3' # 0.97 click acc
target_dataset = 'pitt_broad_pitt_co_CRS02bLab_1918' # 0.95 click acc
# target_dataset = 'pitt_broad_pitt_co_CRS02bLab_1965'
# target_dataset = 'pitt_broad_pitt_co_CRS02bLab_1993'

target_dataset = 'odoherty_rtt-Loco-20170210_03'
target_dataset = 'odoherty_rtt-Loco-20170213_02'
# target_dataset = 'odoherty_rtt-Indy-20161026_03'
# target_dataset = None

dataset = SpikingDataset(cfg.dataset)
print("Original length: ", len(dataset))
if target_dataset:
    ctx = dataset.list_alias_to_contexts([target_dataset])[0]

if cfg.dataset.eval_datasets and mode == 'test':
    dataset.subset_split(splits=['eval'])
else:
    # Mock training procedure to identify val data
    dataset.subset_split() # remove test data
    train, val = dataset.create_tv_datasets()
    dataset = train
    dataset = val

if target_dataset:
    dataset.subset_by_key([ctx.id], MetaKey.session)
    print("Subset length: ", len(dataset))


data_attrs = dataset.get_data_attrs()
print(data_attrs)

model = transfer_model(src_model, cfg.model, data_attrs)

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
# def get_dataloader(dataset: SpikingDataset, batch_size=300, num_workers=1, **kwargs) -> DataLoader:
def get_dataloader(dataset: SpikingDataset, batch_size=16, num_workers=1, **kwargs) -> DataLoader:
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.tokenized_collater
    )

dataloader = get_dataloader(dataset)
heldin_outputs = stack_batch(trainer.predict(model, dataloader))
heldin_metrics = stack_batch(trainer.test(model, dataloader))
# A note on fullbatch R2 calculation - in my experience by bsz 128 the minibatch R2 ~ fullbatch R2 (within 0.01); for convenience we use minibatch R2

#%%
from context_general_bci.config import DEFAULT_KIN_LABELS
pred = heldin_outputs[Output.behavior_pred]
true = heldin_outputs[Output.behavior]
positions = heldin_outputs[f'{DataKey.covariate_space}_target']
padding = heldin_outputs[f'covariate_{DataKey.padding}_target']

print(pred[0].shape)
print(true[0].shape)
print(heldin_outputs[f'{DataKey.covariate_space}_target'].shape)
print(heldin_outputs[f'{DataKey.covariate_space}_target'].unique())
# print(heldin_outputs[DataKey.covariate_labels])

def flatten(arr):
    return np.concatenate(arr) if isinstance(arr, list) else arr.flatten()
flat_padding = flatten(padding)

if model.data_attrs.semantic_covariates:
    flat_space = flatten(positions)
    flat_space = flat_space[~flat_padding]
    coords = [DEFAULT_KIN_LABELS[i] for i in flat_space]
else:
    # remap position to global space
    coords = []
    labels = heldin_outputs[DataKey.covariate_labels]
    for i, trial_position in enumerate(positions):
        coords.extend(np.array(labels[i])[trial_position])
    coords = np.array(coords)
    coords = coords[~flat_padding]

df = pd.DataFrame({
    'pred': flatten(pred)[~flat_padding],
    'true': flatten(true)[~flat_padding],
    'coord': coords,
})
# plot marginals
subdf = df
# subdf = df[df['coord'].isin(['y'])]

g = sns.jointplot(x='true', y='pred', hue='coord', data=subdf, s=3, alpha=0.4)

# set title
g.fig.suptitle(f'{query} {mode} {target_dataset} Velocity R2: {heldin_metrics["test_kinematic_r2"]:.2f}')
#%%
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)
trials = 4
trials = 1
trials = min(trials, len(heldin_outputs[Output.behavior_pred]))
trials = range(trials)

colors = sns.color_palette('colorblind', len(trials))
def plot_trial(trial, ax, color, label=False):
    vel_true = heldin_outputs[Output.behavior][trial]
    vel_pred = heldin_outputs[Output.behavior_pred][trial]
    dims = heldin_outputs[f'{DataKey.covariate_space}_target'][trial]
    pad = heldin_outputs[f'covariate_{DataKey.padding}_target'][trial]
    vel_true = vel_true[~pad]
    vel_pred = vel_pred[~pad]
    dims = dims[~pad]
    for i, dim in enumerate(dims.unique()):
        dim_mask = dims == dim
        true_dim = vel_true[dim_mask]
        pred_dim = vel_pred[dim_mask]
        dim_label = DEFAULT_KIN_LABELS[dim] if model.data_attrs.semantic_covariates else heldin_outputs[DataKey.covariate_labels][trial][dim]
        if dim_label != 'f':
            true_dim = true_dim.cumsum(0)
            pred_dim = pred_dim.cumsum(0)
        ax.plot(true_dim, label=f'{dim_label} true' if label else None, linestyle='-', color=color)
        ax.plot(pred_dim, label=f'{dim_label} pred' if label else None, linestyle='--', color=color)

    # ax.plot(pos_true[:,0], pos_true[:,1], label='true' if label else '', linestyle='-', color=color)
    # ax.plot(pos_pred[:,0], pos_pred[:,1], label='pred' if label else '', linestyle='--', color=color)
    # ax.set_xlabel('X-pos')
    # ax.set_ylabel('Y-pos')
    # make limits square
    # ax.set_aspect('equal', 'box')


for i, trial in enumerate(trials):
    plot_trial(trial, ax, colors[i], label=i==0)
ax.legend()
ax.set_title(f'{mode} {target_dataset} Trajectories')
ax.set_ylabel(f'Force (minmax normalized)')
# xticks - 1 bin is 20ms. Express in seconds
ax.set_xticklabels(ax.get_xticks() * cfg.dataset.bin_size_ms / 1000)
# express in seconds
ax.set_xlabel('Time (s)')

#%%
# Look for the raw data
mins = []
maxes = []
for i in dataset.meta_df[MetaKey.session].unique():
    # sample a trial
    trial = dataset.meta_df[dataset.meta_df[MetaKey.session] == i].iloc[0]
    print(trial.path)
    # Open the processed payload, print minmax
    payload = torch.load(trial.path)
    print(payload['cov_min'])
    print(payload['cov_max'])
    # append and plot
    mins.extend(payload['cov_min'].numpy())
    maxes.extend(payload['cov_max'].numpy())
    # open the original payload
    # og_path = trial.path.parent.parent / 'original' / trial.path.name
ax = prep_plt()
ax.set_title(f'{query} Raw MinMax bounds')
ax.scatter(mins, maxes)
ax.set_xlabel('Min')
ax.set_ylabel('Max')
# ax.plot(mins, label='min')
# ax.plot(maxes, label='max')
# ax.legend()