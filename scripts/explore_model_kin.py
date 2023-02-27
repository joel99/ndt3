#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
from model import transfer_model, logger
from contexts import context_registry

from analyze_utils import stack_batch, get_wandb_run, load_wandb_run, wandb_query_latest
from analyze_utils import prep_plt

query = "indy_base_decode"
# query = "pitt_obs_decode"
query = "pitt_obs_decode_scratch"
query = "test_overfit"
query = "pitt_obs_decode_scratch-sweep-small_wide-7ycit09t"
# query = "pitt_obs_decode_scratch-sweep-small_wide-t1h7knvd"
# query = "rtt_single-35cqqwnl"
# query = "rtt_single-sweep-ft_reg-yni1txy2"
query = "rtt_flat_indy-a25iab76"
query = "indy_causal-stmn13ew"
query = "indy_causal-xj392pjd"
query = "indy_causal_v2-3w1f6vzx"
# query = "loco_causal-rppp73zx"
query = "mc_rtt_joint_tune_800-162hvyl4"
query = "indy_causal_joint_0s-tne69igz"
query = "robust_joint_unsup_tune_800-t1dtvj2p"
# query = "mc_rtt_unsup_800-k0m5uklu"

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='co_bps')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='bps')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

# cfg.dataset.datasets = cfg.dataset.datasets[:1]
# cfg.model.task.tasks = [ModelTask.infill]
cfg.model.task.metrics = [Metric.kinematic_r2]
# cfg.model.task.metrics = [Metric.bps, Metric.all_loss]
cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]
# cfg.dataset.datasets = cfg.dataset.datasets[-1:]

# cfg.dataset.datasets = ['odoherty_rtt-Indy-201606.*']
# cfg.dataset.datasets = ['odoherty_rtt-Indy-20160627_01']
# cfg.dataset.eval_datasets = []

# Joint transfer exp
pipeline_model = ""
# pipeline_model = "indy_causal_joint_0s-tne69igz"
# pipeline_model = "indy_causal_freeze_enc-j2lxcf6a"

if pipeline_model:
    pipeline_model = load_wandb_run(wandb_query_latest(pipeline_model, allow_running=True, use_display=True)[0], tag='val_loss')[0]
    cfg.model.task = pipeline_model.cfg.task

# Ahmadi 21 eval set sanity ballpark
# TARGET_ALIASES = ['odoherty_rtt-Indy-20161005_06']
# TARGET_ALIASES = ['odoherty_rtt-Indy-20160627_01']

# PSID-RNN eval set sanity ballpark
# TARGET_ALIASES = ['odoherty_rtt-Indy-201606.*', 'odoherty_rtt-Indy-20160915.*', 'odoherty_rtt-Indy-20160916.*', 'odoherty_rtt-Indy-20160921.*']
# TARGET_ALIASES = ['odoherty_rtt-Indy.*']

# TARGET_ALIASES = ['odoherty_rtt-Loco-20170215_02']
# TARGET_ALIASES = ['odoherty_rtt-Loco.*']
TARGET_ALIASES = []

TARGET_DATASETS = [context_registry.query(alias=td) for td in TARGET_ALIASES]

FLAT_TARGET_DATASETS = []

# cfg.dataset.datasets = ['mc_rtt']
# cfg.dataset.eval_datasets = ['mc_rtt']

for td in TARGET_DATASETS:
    if td == None:
        continue
    if isinstance(td, list):
        FLAT_TARGET_DATASETS.extend(td)
    else:
        FLAT_TARGET_DATASETS.append(td)
TARGET_DATASETS = [td.id for td in FLAT_TARGET_DATASETS]

if cfg.dataset.datasets == ['mc_rtt']:
    cfg.dataset.datasets = TARGET_ALIASES
    cfg.dataset.eval_datasets = []

dataset = SpikingDataset(cfg.dataset)
if cfg.dataset.eval_datasets and not TARGET_DATASETS:
    dataset.subset_split(splits=['eval'])
else:
    # Mock training procedure to identify val data
    dataset.subset_split() # remove test data
    train, val = dataset.create_tv_datasets()
    dataset = val

if query == "indy_causal_v2-3w1f6vzx": # peculiar split logic on crc cluster not yet patched, we imported the split used
    dataset = torch.load('val_dataset.pth') # transferred from crc cluster
if TARGET_DATASETS:
    dataset.subset_by_key(TARGET_DATASETS, key=MetaKey.session)

data_attrs = dataset.get_data_attrs()
print(data_attrs)
print(f'{len(dataset)} examples')

model = transfer_model(src_model, cfg.model, data_attrs)

if pipeline_model:
    pipeline_model = transfer_model(pipeline_model, cfg.model, data_attrs)
    model.task_pipelines[ModelTask.kinematic_decoding.value] = pipeline_model.task_pipelines[ModelTask.kinematic_decoding.value]
    # model.transfer_io(pipeline_model)


trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
# def get_dataloader(dataset: SpikingDataset, batch_size=300, num_workers=1, **kwargs) -> DataLoader:
def get_dataloader(dataset: SpikingDataset, batch_size=128, num_workers=1, **kwargs) -> DataLoader:
# def get_dataloader(dataset: SpikingDataset, batch_size=200, num_workers=1, **kwargs) -> DataLoader:
    # Defaults set for evaluation on 1 GPU.
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collater_factory()
    )

dataloader = get_dataloader(dataset)
#%%
heldin_metrics = stack_batch(trainer.test(model, dataloader))
heldin_outputs = stack_batch(trainer.predict(model, dataloader))

# A note on fullbatch R2 calculation - in my experience by bsz 128 the minibatch R2 ~ fullbatch R2 (within 0.01); for convenience we use minibatch R2
# from sklearn.metrics import r2_score
# print(r2_score(heldin_outputs[Output.behavior[:,offset_bins:]].flatten(0, 1), heldin_outputs[Output.behavior_pred[:,offset_bins:]].flatten(0, 1)), multioutput='raw_values')

#%%
import pandas as pd
offset_bins = model.task_pipelines[ModelTask.kinematic_decoding.value].bhvr_lag_bins
# B T H
# The predictions here are definitely not correct...
# So why do we have low error?
# ?
trials = 1
# print(heldin_outputs[Output.behavior][0,:10,0])
# print(heldin_outputs[Output.behavior_pred][0,:10,0])
# print(heldin_outputs[Output.behavior].mean())
# print(heldin_outputs[Output.behavior].max())
# print(heldin_outputs[Output.behavior].min())
# sns.histplot(heldin_outputs[Output.behavior].flatten())
# sns.histplot(heldin_outputs[Output.behavior_pred].flatten())
pred = heldin_outputs[Output.behavior_pred][:,offset_bins:]
true = heldin_outputs[Output.behavior][:,offset_bins:]
df = pd.DataFrame({
    'pred': pred.flatten(),
    'true': true.flatten(),
    'coord': (pred.shape[0] * pred.shape[1]) * ['x'] + (pred.shape[0] * pred.shape[1]) * ['y'],
})
# ax = prep_plt()
# sns.scatterplot(x='true', y='pred', hue='coord', data=df, ax=ax, s=3, alpha=0.4)

# plot marginals
sns.jointplot(x='true', y='pred', hue='coord', data=df, s=3, alpha=0.4)
# sns.jointplot(x='true', y='pred', hue='coord', data=df, kind='hist') # Too slow


#%%
ax = prep_plt()
# sns.histplot(heldin_outputs[Output.behavior_pred].flatten(), ax=ax, bins=20)
# ax.set_title('Distribution of velocity predictions')

sns.histplot(heldin_outputs[Output.behavior].flatten(), ax=ax, bins=20)
ax.set_title('Distribution of velocity targets')
ax.set_yscale('log')
#%%
ax = prep_plt()
# trials = range(4)
trials = torch.arange(10)

colors = sns.color_palette('colorblind', len(trials))
def plot_trial(trial, ax, color, label=False):
    vel_true = heldin_outputs[Output.behavior][trial][6:]
    vel_pred = heldin_outputs[Output.behavior_pred][trial][6:]
    pos_true = vel_true.cumsum(0)
    pos_pred = vel_pred.cumsum(0)
    ax.plot(pos_true[:,0], pos_true[:,1], label='true' if label else '', linestyle='-', color=color)
    ax.plot(pos_pred[:,0], pos_pred[:,1], label='pred' if label else '', linestyle='--', color=color)

for i, trial in enumerate(trials):
    plot_trial(trial, ax, colors[i], label=i == 0)
ax.legend()

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