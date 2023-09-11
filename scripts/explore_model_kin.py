#%%
r"""
    What do model behavioral predictions look like? (Devbook)
"""
# restrict cuda to gpu 1
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
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

# query = '10s_loco_regression'
# query = '10s_indy_regression'

query = '10s_exclude-vpqsnnam'
query = '10s_regression_exclude-hio7q3x6'
query = '10s_exclude-xfcwrte7'

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

# cfg.model.task.metrics = [Metric.kinematic_r2]
cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]

# target = 'pitt_broad_CRS02bLab_1925_3' # 0.97 click acc
target = 'pitt_broad_pitt_co_CRS02bLab_1918' # 0.95 click acc
# target = 'pitt_broad_pitt_co_CRS02bLab_1965'
# target = 'pitt_broad_pitt_co_CRS02bLab_1993'

target = 'odoherty_rtt-Loco-20170210_03'
target = 'odoherty_rtt-Loco-20170213_02'
# target = [
#     'odoherty_rtt-Indy-20160407_02',
#     'odoherty_rtt-Indy-20160627_01',
#     'odoherty_rtt-Indy-20161005_06',
#     'odoherty_rtt-Indy-20161026_03',
#     'odoherty_rtt-Indy-20170131_02',
# ]

# target = 'odoherty_rtt-Indy-20161026_03'
target = None

dataset = SpikingDataset(cfg.dataset)
print("Original length: ", len(dataset))
if target == None:
    target = []
if not isinstance(target, list):
    target = [target]
if target:
    ctx = dataset.list_alias_to_contexts(target)[0]

if cfg.dataset.eval_datasets and mode == 'test':
    dataset.subset_split(splits=['eval'])
else:
    # Mock training procedure to identify val data
    dataset.subset_split() # remove test data
    train, val = dataset.create_tv_datasets()
    dataset = train
    dataset = val

if target:
    dataset.subset_by_key([ctx.id], MetaKey.session)
    if len(dataset) == 0: # Context wasn't in, reset
        cfg.dataset.datasets = target
        cfg.dataset.exclude_datasets = []
        dataset = SpikingDataset(cfg.dataset)
        print("Reset to novel context: ", len(dataset))
    else:
        print("Subset length: ", len(dataset))


data_attrs = dataset.get_data_attrs()
print(data_attrs)

model = transfer_model(src_model, cfg.model, data_attrs)

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
# def get_dataloader(dataset: SpikingDataset, batch_size=300, num_workers=1, **kwargs) -> DataLoader:
def get_dataloader(dataset: SpikingDataset, batch_size=32, num_workers=1, **kwargs) -> DataLoader:
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
ICL_CROP = 2 * 50 * 2 # Quick hack to eval only a certain portion of data. 2s x 50 bins/s x 2 dims
# ICL_CROP = 0

from context_general_bci.config import DEFAULT_KIN_LABELS
pred = heldin_outputs[Output.behavior_pred]
true = heldin_outputs[Output.behavior]
positions = heldin_outputs[f'{DataKey.covariate_space}_target']
padding = heldin_outputs[f'covariate_{DataKey.padding}_target']

if ICL_CROP:
    pred = pred[:, -ICL_CROP:]
    true = true[:, -ICL_CROP:]
    positions = positions[:,-ICL_CROP:]
    padding = padding[:, -ICL_CROP:]

print(pred[0].shape)
print(true[0].shape)
print(positions.shape)
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
# Recompute R2 between pred / true
from sklearn.metrics import r2_score
r2 = r2_score(subdf['true'], subdf['pred'])
mse = np.mean((subdf['true'] - subdf['pred'])**2)
# set title
g.fig.suptitle(f'{query} {mode} {str(target)[:20]} Velocity R2: {r2:.2f}, MSE: {mse:.4f}')
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
ax.set_title(f'{mode} {str(target)[:20]} Trajectories')
ax.set_ylabel(f'Force (minmax normalized)')
# xticks - 1 bin is 20ms. Express in seconds
ax.set_xticklabels(ax.get_xticks() * cfg.dataset.bin_size_ms / 1000)
# express in seconds
ax.set_xlabel('Time (s)')

#%%
# Look for the raw data
from pathlib import Path
from context_general_bci.tasks.rtt import ODohertyRTTLoader
mins = []
maxes = []
raw_mins = []
raw_maxes = []
bhvr_vels = []
bhvr_pos = []
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
    path_pieces = Path(trial.path).parts
    og_path = Path(path_pieces[0], *path_pieces[2:-1])
    spike_arr, bhvr_raw, _ = ODohertyRTTLoader.load_raw(og_path, cfg.dataset, ['Indy-M1', 'Loco-M1'])
    bhvr_vel = bhvr_raw[DataKey.bhvr_vel].flatten()
    bhvr_vels.append(bhvr_vel)
    # bhvr_pos.append(bhvr_raw['position'])
    raw_mins.append(bhvr_vel.min().item())
    raw_maxes.append(bhvr_vel.max().item())
ax = prep_plt()
ax.set_title(f'{query} Raw MinMax bounds')
ax.scatter(mins, maxes)
ax.scatter(raw_mins, raw_maxes)
ax.set_xlabel('Min')
ax.set_ylabel('Max')
# ax.plot(mins, label='min')
# ax.plot(maxes, label='max')
# ax.legend()
#%%
print(bhvr_pos[0][:,1:3].shape)
# plt.plot(bhvr_pos[0][:, 1:3])
# plt.plot(bhvr_vels[3])
# plt.plot(bhvr_vels[2])
# plt.plot(bhvr_vels[1])
# plt.plot(bhvr_vels[0])
import scipy.signal as signal
def resample(data):
    covariate_rate = cfg.dataset.odoherty_rtt.covariate_sampling_rate
    base_rate = int(1000 / cfg.dataset.bin_size_ms)
    # print(base_rate, covariate_rate, base_rate / covariate_rate)
    return torch.tensor(
        # signal.resample(data, int(len(data) / cfg.dataset.odoherty_rtt.covariate_sampling_rate / (cfg.dataset.bin_size_ms / 1000))) # This produces an edge artifact
        signal.resample_poly(data, base_rate, covariate_rate, padtype='line')
    )
# 250Hz to 5Hz - > 2000
# plt.plot(bhvr_pos[0][:, 1:3])
plt.plot(resample(bhvr_pos[0][:, 1:3]))