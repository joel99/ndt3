#%%
r"""
    What do model behavioral predictions look like? (Devbook)
"""
# restrict cuda to gpu 1
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import logging
import sys
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
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model, logger
from context_general_bci.contexts import context_registry

from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt
from context_general_bci.utils import get_wandb_run, wandb_query_latest


mode = 'train'
# mode = 'test'

query = 'base-i9ix8t7q'

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

cfg.model.task.metrics = [Metric.kinematic_r2]
cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]

# target_dataset = 'pitt_broad_CRS02bLab_1925_3' # 0.97 click acc
target_dataset = 'pitt_broad_pitt_co_CRS02bLab_1918' # 0.95 click acc
target_dataset = 'pitt_broad_pitt_co_CRS02bLab_1965'


dataset = SpikingDataset(cfg.dataset)
print("Original length: ", len(dataset))
ctx = dataset.list_alias_to_contexts([target_dataset])[0]
# dataset.subset_by_key([ctx.id], MetaKey.session)
# print(len(dataset))
if cfg.dataset.eval_datasets and mode == 'test':
    dataset.subset_split(splits=['eval'])
else:
    # Mock training procedure to identify val data
    dataset.subset_split() # remove test data
    train, val = dataset.create_tv_datasets()
    dataset = train
    dataset = val


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

print(pred[0].shape)
print(true[0].shape)
print(heldin_outputs[f'{DataKey.covariate_space}_target'].shape)
#%%
def flatten(arr):
    return np.concatenate(arr) if isinstance(arr, list) else arr.flatten()
flat_pred = flatten(pred)
flat_true = flatten(true)
flat_space = flatten(heldin_outputs[f'{DataKey.covariate_space}_target'])
flat_padding = flatten(heldin_outputs[f'covariate_{DataKey.padding}_target'])
flat_pred = flat_pred[~flat_padding]
flat_true = flat_true[~flat_padding]
flat_space = flat_space[~flat_padding]

assert dataset.cfg.semantic_positions, 'Need to specify semantic covariates to plot kinematics (covariate_label path not implemented)'

df = pd.DataFrame({
    'pred': flat_pred.flatten(),
    'true': flat_true.flatten(),
    'coord': [DEFAULT_KIN_LABELS[i] for i in flat_space],
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
        dim_label = DEFAULT_KIN_LABELS[dim]
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
# print(heldin_outputs[Output.rates].max(), heldin_outputs[Output.rates].mean())
# test = heldin_outputs[Output.heldout_rates]
rates = heldin_outputs[Output.rates] # b t c


spikes = [rearrange(x, 't a c -> t (a c)') for x in heldin_outputs[Output.spikes]]
ax = prep_plt()

num = 20
# channel = 5
# channel = 10
# channel = 18
# channel = 19
# channel = 20
# channel = 80

colors = sns.color_palette("husl", num)

# for trial in range(num):
#     ax.plot(rates[trial][:,channel], color=colors[trial])

y_lim = ax.get_ylim()[1]
# plot spike raster
# for trial in range(num):
#     spike_times = spikes[trial,:,channel].nonzero()
#     y_height = y_lim * (trial+1) / num
#     ax.scatter(spike_times, torch.ones_like(spike_times)*y_height, color=colors[trial], s=10, marker='|')

trial = 10
trial = 15
# trial = 17
# trial = 18
# trial = 80
# trial = 85
for channel in range(num):
    # ax.scatter(np.arange(test.shape[1]), test[0,:,channel], color=colors[channel], s=1)
    ax.plot(rates[trial][:,channel * 2], color=colors[channel])
    # ax.plot(rates[trial][:,channel * 3], color=colors[channel])

    # smooth the signal with a gaussian kernel

# from scipy import signal
# peaks, _ = signal.find_peaks(test[trial,:,2], distance=4)
# print(peaks)
# print(len(peaks))
# for p in peaks:
#     ax.axvline(p, color='k', linestyle='--')



ax.set_ylabel('FR (Hz)')
ax.set_yticklabels((ax.get_yticks() * 1000 / cfg.dataset.bin_size_ms).round())
# relabel xtick unit from 5ms to ms
ax.set_xlim(0, 50)
ax.set_xticklabels(ax.get_xticks() * cfg.dataset.bin_size_ms)
ax.set_xlabel('Time (ms)')
# plt.plot(test[0,:,0])
ax.set_title(f'FR Inference: {query}')

#%%
# Debugging (for mc_maze dataset)
pl.seed_everything(0)
example_batch = next(iter(dataloader))
print(example_batch[DataKey.spikes].size())
print(example_batch[DataKey.spikes].sum())
# print(example_batch[DataKey.spikes][0,:,0,:,0].nonzero())
# First 10 timesteps, channel 8 fires 3x
print(example_batch[DataKey.spikes][0,:,0,:,0][:10, 8])
# Now, do masking manually

# No masking
backbone_feats = model(example_batch)
example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=False)
print(example_out[Output.logrates].size())
print(example_out[Output.logrates][0, :, 0, :][:10, 8]) # extremely spiky prediction

# # With masking
# example_batch[DataKey.spikes][0, :, 0, :, 0][:10] = 0
# backbone_feats = model(example_batch)
# example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=False)
# print(example_out[Output.logrates].size())
# print(example_out[Output.logrates][0, :, 0, :][:10, 8]) # unspiked prediction.
# OK - well if true mask occurs, model appropriately doesn't predict high spike.

# Key symptom - whether or not a spike occurs at a timestep is affecting its own prediction
# example_batch[DataKey.spikes][0, :, 0, :, 0][1] = 0
# backbone_feats = model(example_batch)
# example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=False)
# print(example_out[Output.logrates].size())
# print(example_out[Output.logrates][0, :, 0, :][:10, 8]) # unspiked prediction.


# Masking through model update_batch also seems to work
model.task_pipelines[ModelTask.infill.value].update_batch(example_batch)
print(example_batch['is_masked'][0].nonzero())
backbone_feats = model(example_batch)
example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=True)
# example_out = model.task_pipelines[ModelTask.infill.value](example_batch, backbone_feats, compute_metrics=False)
print(example_out[Metric.bps])
print(example_out[Output.logrates].size())
print(example_out[Output.logrates][0, :, 0, :][:10, 8]) # unspiked prediction.


# Ok - so the model is correctly predicting unspiked for masked timesteps.
# Then why is test time evaluation so spiky? Even when we mask?
# Let's check again...