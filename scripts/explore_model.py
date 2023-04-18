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
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.data import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model, logger

from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt
from context_general_bci.utils import  get_wandb_run, wandb_query_latest

pl.seed_everything(0)

# wandb_run = get_wandb_run("maze_med-1j0loymb")
query = "indy_causal-4i8yc4bc"
query = "loco_causal-ctkuwqpl"
query = "indy_single-bw25v0ci"
query = "rtt_f32_v2-7alicf5z"
query = "med_f32_b-vozm3zip"
query = "maze_large-wuawcvls"
query = "maze_med-5v08a4oy"
query = "maze_small-hr4prtoe"
# query = "rtt-1maz3ea5"

# query = "ndt2_128_maze_med_2a-kvuo6q15" # Minor drop, qualitatively smooth.
# query = "ndt2_128_maze_small-tnnvmdkv"
# query = "ndt2_32_rtt-05dqi05j"

# query = "m3_150k_large-sweep-simple_lr_sweep-axzvm22s"
# query = "m3_150k_med-sweep-simple_lr_sweep-81fvl2ws"
query = "m3_150k_small-sweep-simple_lr_sweep-rpqx4bpq"
# query = "m3_150k_rtt-sweep-simple_lr_sweep-81saz29y"

# query = "m3_150k_proj-7vt5c4dr"
# query = "maze_nlb_ref_03-rypos2lo"
# query = "maze_nlb_ref_14d-h1p5a86g"
# query = "m3_150k_proj_lpft_unsup-swnrji4l"
# query = "m3_150k-03eat4mn"
# query = "m5_150k_b-xia8dehq"
# query = "m3_150k-hodo53b1"
# query = "drop2-5ap51v0v"
# query = "drop4-d1qombxf"
query = "m3_150k_proj_lpft-cu3pjkjx"
query = "single_time_nlb-nnnow3uw"
query = "single_time-mvrzxhxs"
# query = "single_time_nlb_r300-vj268v03" # 0.6 drop # gold standard
# query = "single_time_nlb_r300-hu9vxleg" # 2x, worse, but still very smooth
# query = "single_time_nlb_r300-qyhfrt06"
# query = "single_time_nlb_r300_full-2iwg032a"
# query = "f32-wi0xe1mn"
# query = "single_ndt2_nlb_r300-a0tv97b3"
# query = "f32_10x-824eizh4"

# query = "m3_150k-91lyoz89"
# query = "m3_150k-hodo53b1"
# query = "m3_150k_small-km6s6yuw"

query = "f32_5ho-tsntbsci"

# query = "f32_nlb-b4rz44ou"
# query = "mc_rtt-i0n8o24x"

query = 'test-tzz4zm2r'
# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
# wandb_run = wandb_query_latest(query, exact=True, allow_running=True)[0]
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='bps')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='co-bps')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
# cfg.dataset.datasets = cfg.dataset.datasets[:1]
# cfg.model.task.tasks = [ModelTask.infill]
# cfg.model.task.metrics = [Metric.bps]
# cfg.model.task.metrics = [Metric.bps, Metric.all_loss]
cfg.model.task.outputs = [
    Output.logrates,
    # Output.heldout_logrates,
    Output.spikes
]
print(cfg.dataset.datasets)
# cfg.dataset.datasets = ['odoherty_rtt-Indy-20161005_06']
# cfg.dataset.datasets = cfg.dataset.datasets[-1:]
# cfg.dataset.datasets = ['mc_maze$']
# cfg.dataset.datasets = ['mc_maze_large']
# cfg.dataset.datasets = ['mc_maze_medium']
# cfg.dataset.datasets = ['mc_maze_small']
# cfg.dataset.datasets = ['churchland_misc_reggie-1413W9XGLJ2gma1CCXpg1DRDGpl4-uxkG']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170215_02']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170214_02']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170213_02']

# cfg.dataset.datasets = ['mc_rtt']
# if 'rtt' in query:
#     cfg.dataset.datasets = ['odoherty_rtt-Indy-20161005_06']
#     # cfg.dataset.datasets = ['odoherty_rtt-Indy-20161014_04']
# if 'gallego' in query:
#     cfg.dataset.datasets = ['Chewie_CO_20150313']
#     cfg.dataset.datasets = ['Mihili_CO_20140304']
# if 'nitschke' in query:
#     cfg.dataset.datasets = ['churchland_misc_nitschke-1D8KYfy5IwMmEZaKOEv-7U6-4s-7cKINK']
# if 'pitt' in query:
#     cfg.dataset.datasets = ['CRS02bHome.data.00437']
# cfg.dataset.eval_datasets = []
# cfg.dataset.eval_datasets = [
#     'odoherty_rtt-Indy-20160407_02'
# ]

print(cfg.dataset.datasets)

dataset = SpikingDataset(cfg.dataset)
if cfg.dataset.eval_datasets:
    dataset.subset_split(splits=['eval'])
else:
    # dataset.subset_by_key(['test'], key='split')
    dataset.subset_split() # remove data-native test trials etc
dataset.build_context_index()
if not cfg.dataset.eval_datasets:
    train, val = dataset.create_tv_datasets()
    dataset = val

data_attrs = dataset.get_data_attrs()
print(data_attrs)
# data_attrs.context.session = ['ExperimentalTask.odoherty_rtt-Indy-20161014_04'] # definitely using..
model = transfer_model(src_model, cfg.model, data_attrs)
print(f'{len(dataset)} examples')
trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
# print(context_registry.query(alias='Mihi'))
# model.cfg.task.outputs = [Output.heldout_logrates]
# model.cfg.task.metrics = [Metric.bps]
def get_dataloader(dataset: SpikingDataset, batch_size=100, num_workers=1, **kwargs) -> DataLoader:
    # Defaults set for evaluation on 1 GPU.
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collater_factory()
    )

dataloader = get_dataloader(dataset)
# dataloader = get_dataloader(dataset, batch_size=16)
# dataloader = get_dataloader(dataset, batch_size=4)

print(query)
heldin_metrics = stack_batch(trainer.test(model, dataloader))
# import pdb;pdb.set_trace()
#%%
heldin_outputs = stack_batch(trainer.predict(model, dataloader))
# load submission.h5
# import h5py

# payload = h5py.File('submission.h5', 'r')
# test_rates = payload['mc_maze_small_20']
# heldin_rates = test_rates['eval_rates_heldin']
# heldout_rates = test_rates['eval_rates_heldout']

heldin_rates = heldin_outputs[Output.rates] # b t c
heldout_rates = heldin_outputs[Output.heldout_rates] if Output.heldout_rates in heldin_outputs else None

# print(rates.size())

if not data_attrs.serve_tokens_flat:
    spikes = [rearrange(x, 't a c -> t (a c)') for x in heldin_outputs[Output.spikes]]
# ax = prep_plt()

num = 20
num = 5
# channel = 5

colors = sns.color_palette("husl", num)

# for trial in range(num):
#     ax.plot(rates[trial][:,channel], color=colors[trial])

# y_lim = ax.get_ylim()[1]

# trial = 10
# trial = 15
# trial = 17
# trial = 18
# trial = 80
# trial = 85
# trials = [0, 1, 2]
trials = [3, 4, 5]
# trials = [5, 6, 7]

def plot_trial(rates, trial, ax):
    ax = prep_plt(ax)
    for channel in range(num):
        # ax.scatter(np.arange(test.shape[1]), test[0,:,channel], color=colors[channel], s=1)
        # ax.plot(rates[trial][:,channel * 2], color=colors[channel])
        ax.plot(rates[trial][:,channel * 3], color=colors[channel])

        # smooth the signal with a gaussian kernel

    # from scipy import signal
    # peaks, _ = signal.find_peaks(test[trial,:,2], distance=4)
    # print(peaks)
    # print(len(peaks))
    # for p in peaks:
    #     ax.axvline(p, color='k', linestyle='--')
    # print(rates[trial].size())


    ax.set_ylabel('FR (Hz)')
    ax.set_yticklabels((ax.get_yticks() * 1000 / cfg.dataset.bin_size_ms).round())
    # relabel xtick unit from 5ms to ms
    # ax.set_xlim(0, 50)
    # print(ax.get_xtick)
    # ax.set_xticklabels(ax.get_xticks() * cfg.dataset.bin_size_ms)
    # ax.set_xlabel('Time (ms)')
    # ax.set_title()
size = (len(trials), 1 if heldout_rates is None else 2)
f, axes = plt.subplots(*size, figsize=(5 if heldout_rates is None else 10, 7.5), sharex=True)
for i, trial in enumerate(trials):
    if heldout_rates is None:
        plot_trial(heldin_rates, trial, axes[i])
    else:
        plot_trial(heldin_rates, trial, axes[i, 0])
        plot_trial(heldout_rates, trial, axes[i, 1])
# axes[-1, 0].set_xticklabels(axes[-1, 0].get_xticks() * cfg.dataset.bin_size_ms)
# axes[-1, 1].set_xticklabels(axes[-1, 0].get_xticks() * cfg.dataset.bin_size_ms)

plt.suptitle(
    f'{query} ({heldin_metrics["test_loss"].item():.3f}) {"(All enc)" if data_attrs.serve_tokens_flat else ""}'
    # f'Out ({heldin_metrics["test_co-bps"].item():.4f}) : {query} {"(All enc)" if data_attrs.serve_tokens_flat else ""}'
)
plt.tight_layout()
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