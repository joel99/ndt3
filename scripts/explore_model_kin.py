#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
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
mode = 'test'

# query = "human-sweep-simpler_lr_sweep-dgnx7mn9"
# query = "human_only-5a62oz96"
# query = 'session_cross_noctx-wc24ulkl'

# query = 'human_sum_contrast-3inx20gs'
# query = 'human_sum_contrast-l2zg9sju'
# query = 'human_sum_contrast-lheohfzn'
# query = 'sup_3200-frag-odoherty_rtt-Indy-20160627_01-sweep-simpler_lr_sweep-iy1xf1bc'
query = 'human_obs_m5_lr1e5-2ixlx1c9'
# query = 'human_aug-nxy3te61'
query = 'online_obs_tune-nr2b51yd'
# query = 'online_obs_tune-0ee5xlel'

query = 'human_crs08_tune-mip3powm'
# query = 'online_human_pt-5ytq361d'
query = 'online_human_pt-cfzx5heb'
query = 'online_human_pt-w3rnnpj4'

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

cfg.model.task.metrics = [Metric.kinematic_r2]
cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]

# Joint transfer exp
pipeline_model = ""

if pipeline_model:
    pipeline_model = load_wandb_run(wandb_query_latest(pipeline_model, allow_running=True, use_display=True)[0], tag='val_loss')[0]
    cfg.model.task = pipeline_model.cfg.task

# target_dataset = 'observation_CRS02bLab_session_1908_set_1'
# target_dataset = 'observation_CRS02bLab_session_1913_set_1'
# target_dataset = 'observation_CRS02bLab_session_1827.*'
# target_dataset = 'observation_CRS02bLab_session_1922_set_6'
target_dataset = 'observation_CRS02b_1904_6'
# target_dataset = 'observation_CRS02bLab_session_1925_set_3'
# target_dataset = 'observation_CRS07_150_1_2d_cursor_center_out'

# cfg.dataset.datasets = ["observation_CRS02bLab_session_1908_set_1"]
# cfg.dataset.eval_datasets = ["observation_CRS02bLab_session_1908_set_1"]
target_dataset = 'observation_CRS08_10_2'
target_dataset = 'observation_CRS08_11_16'

# target_dataset = 'odoherty_rtt-Indy-20160627_01'
# cfg.dataset.datasets = ["odoherty_rtt-Indy-20160627_01"]
# cfg.dataset.eval_datasets = ["odoherty_rtt-Indy-20160627_01"]

# cfg.dataset.datasets = [target_dataset]
# cfg.dataset.eval_datasets = [target_dataset]
# cfg.dataset.eval_datasets = []
# cfg.dataset.eval_ratio = 0.5
# cfg.model.task.decode_normalizer = 'pitt_obs_zscore.pt'

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
print(f'{len(dataset)} examples')
cfg.model.task.tasks = [ModelTask.kinematic_decoding]

model = transfer_model(src_model, cfg.model, data_attrs)

if pipeline_model:
    pipeline_model = transfer_model(pipeline_model, cfg.model, data_attrs)
    model.task_pipelines[ModelTask.kinematic_decoding.value] = pipeline_model.task_pipelines[ModelTask.kinematic_decoding.value]
    # model.transfer_io(pipeline_model)


trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
# def get_dataloader(dataset: SpikingDataset, batch_size=300, num_workers=1, **kwargs) -> DataLoader:
def get_dataloader(dataset: SpikingDataset, batch_size=16, num_workers=1, **kwargs) -> DataLoader:
# def get_dataloader(dataset: SpikingDataset, batch_size=128, num_workers=1, **kwargs) -> DataLoader:
# def get_dataloader(dataset: SpikingDataset, batch_size=200, num_workers=1, **kwargs) -> DataLoader:
    # Defaults set for evaluation on 1 GPU.
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collater_factory()
    )

dataloader = get_dataloader(dataset)
heldin_metrics = stack_batch(trainer.test(model, dataloader))
heldin_outputs = stack_batch(trainer.predict(model, dataloader))
# A note on fullbatch R2 calculation - in my experience by bsz 128 the minibatch R2 ~ fullbatch R2 (within 0.01); for convenience we use minibatch R2

offset_bins = model.task_pipelines[ModelTask.kinematic_decoding.value].bhvr_lag_bins

#%%
pred = heldin_outputs[Output.behavior_pred]
pred = [p[offset_bins:] for p in pred]
true = heldin_outputs[Output.behavior]
true = [t[offset_bins:] for t in true]

flat_pred = np.concatenate(pred) if isinstance(pred, list) else pred.flatten()
flat_true = np.concatenate(true) if isinstance(true, list) else true.flatten()
flat_pred = flat_pred[(flat_true != model.data_attrs.pad_token).any(-1)]
flat_true = flat_true[(flat_true != model.data_attrs.pad_token).any(-1)]

df = pd.DataFrame({
    'pred': flat_pred.flatten(),
    'true': flat_true.flatten(),
    'coord': flat_pred.shape[0] * ['x'] + flat_pred.shape[0] * ['y'],
})
# plot marginals
g = sns.jointplot(x='true', y='pred', hue='coord', data=df, s=3, alpha=0.4)

# set title
g.fig.suptitle(f'{query} {mode} {target_dataset} Velocity R2: {heldin_metrics["test_kinematic_r2"]:.2f}')
#%%
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca())
trials = range(4)
# trials = range(2)
# trials = torch.arange(0, 12, 2)

colors = sns.color_palette('colorblind', len(trials))
def plot_trial(trial, ax, color, label=False):
    vel_true = heldin_outputs[Output.behavior][trial][offset_bins:]
    vel_pred = heldin_outputs[Output.behavior_pred][trial][offset_bins:]
    # remove padding
    pad_index = np.where(vel_true[:,0] == 5)[0]
    if len(pad_index) > 0:
        # assume sequential
        vel_pred = vel_pred[:pad_index[0]]
        vel_true = vel_true[:pad_index[0]]

    pos_true = vel_true.cumsum(0)
    pos_pred = vel_pred.cumsum(0)
    ax.plot(pos_true[:,0], pos_true[:,1], label='true' if label else '', linestyle='-', color=color)
    ax.plot(pos_pred[:,0], pos_pred[:,1], label='pred' if label else '', linestyle='--', color=color)
    ax.set_xlabel('X-pos')
    ax.set_ylabel('Y-pos')
    # make limits square
    ax.set_aspect('equal', 'box')


for i, trial in enumerate(trials):
    plot_trial(trial, ax, colors[i], label=i == 0)
ax.legend()
ax.set_title(f'{mode} {target_dataset} Trajectories')

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