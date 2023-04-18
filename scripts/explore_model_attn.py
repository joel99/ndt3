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

from context_general_bci.analyze_utils import stack_batch, get_wandb_run, load_wandb_run, wandb_query_latest
from context_general_bci.analyze_utils import prep_plt

# wandb_run = get_wandb_run("maze_med-1j0loymb")
query = "maze_small"
query = "maze_med"
# query = "maze_large"
query = "maze_nlb"
# query = "maze_med_ft"
# query = "maze_small_ft"
# query = "maze_large_ft"
# query = "maze_all_256"
# query = "maze_all"

query = "maze_jenkins_stitch"
# query = "maze_nlb_stitch_out"
# query = "maze_nlb"

# query = 'maze_med_20'

# query = "rtt_all"
# query = "rtt_all_256"
# query = "rtt_nlb_infill_only"
# query = 'rtt_nlb_07'
# query = 'rtt_nlb_pt'

# query = "rtt_indy_nlb"
# query = "rtt_indy_nlb_stitch"
# query = "rtt_indy1"
# query = "rtt_indy2"
# query = "rtt_indy2_noembed"
# query = "rtt_all_sans_add"
# query = "rtt_indy_sans_256_d01"
# query = "rtt_indy_stitch"
query = "rtt_all_256"
# query = "rtt_indy_ablate"
# query = "rtt_all_512"
query = "rtt_token_padded_nostitch"

# query = "rtt_loco"
# query = "rtt_loco1"
# query = "rtt_loco2"
# query = "rtt_loco_test1"
# query = "rtt_loco_test2"
# query = "rtt_loco_test3"
# query = "rtt_loco_test4"
# query = 'rtt_indy_256_linear'
# query = 'test'

# query = 'rtt_loco_d2'
# query = 'rtt_loco_512'
# query = 'rtt_loco_256_stitch'
# query = 'test'

# query = 'nitschke_token'
# query = 'nitschke'
# query = 'nitschke_single'

# query = 'gallego_token'
# query = 'gallego'
# query = 'gallego_chewie'
# query = 'gallego_chewie_single'

# query = 'pitt_single'
# query = 'pitt'
# query = 'pitt_obs'
# query = 'pitt_20'
# query = 'pitt_obs_20'

# query = 'ks'
# query = 'ks_ctx'

# wandb_run = wandb_query_latest(query, exact=True, allow_running=False)[0]
wandb_run = wandb_query_latest(query, exact=True, allow_running=True)[0]
print(wandb_run.id)

# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='co_bps')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='bps')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
# cfg.dataset.datasets = cfg.dataset.datasets[:1]
cfg.model.task.tasks = [ModelTask.infill]
cfg.model.task.metrics = [Metric.bps, Metric.all_loss]
cfg.model.task.outputs = [Output.logrates, Output.spikes]
print(cfg.dataset.datasets)
# cfg.dataset.datasets = cfg.dataset.datasets[-1:]
# cfg.dataset.datasets = ['mc_maze$']
# cfg.dataset.datasets = ['mc_maze_large']
# cfg.dataset.datasets = ['mc_maze_medium']
# cfg.dataset.datasets = ['mc_maze_small']
# cfg.dataset.datasets = ['churchland_maze_jenkins-1']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170215_02']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170214_02']
# cfg.dataset.datasets = ['odoherty_rtt-Loco-20170213_02']

# cfg.dataset.datasets = ['mc_rtt']
if 'rtt' in query:
    cfg.dataset.datasets = ['odoherty_rtt-Indy-20161005_06']
    # cfg.dataset.datasets = ['odoherty_rtt-Indy-20161014_04']
if 'gallego' in query:
    cfg.dataset.datasets = ['Chewie_CO_20150313']
    cfg.dataset.datasets = ['Mihili_CO_20140304']
if 'nitschke' in query:
    cfg.dataset.datasets = ['churchland_misc_nitschke-1D8KYfy5IwMmEZaKOEv-7U6-4s-7cKINK']
if 'pitt' in query:
    cfg.dataset.datasets = ['CRS02bHome.data.00329']

# cfg.dataset.eval_datasets = []
print(cfg.dataset.datasets)
dataset = SpikingDataset(cfg.dataset)
if cfg.dataset.eval_datasets:
    dataset.subset_split(splits=['eval'])
else:
    dataset.subset_split()
dataset.build_context_index()
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
#%%
from analyze_utils import SaveOutput, patch_attention
save_output = SaveOutput()
layer_tgt = model.backbone.encoder.layers[0]
# layer_tgt = model.backbone.encoder.layers[1]
# layer_tgt = model.backbone.encoder.layers[2]
# layer_tgt = model.backbone.encoder.layers[1]
patch_attention(layer_tgt.self_attn)
# 100 trials, 4 heads, 200 tokens, 200 tokens
hook_handle = layer_tgt.self_attn.register_forward_hook(save_output)

print(query)
heldin_metrics = stack_batch(trainer.test(model, dataloader))
# heldin_outputs = stack_batch(trainer.predict(model, dataloader))

print(save_output.outputs[0].shape)
print(save_output.outputs[1].shape)
# print(save_output.outputs[2].shape)
# print(save_output.outputs[3].shape)

print(dataset[0][DataKey.spikes].shape)
print(data_attrs.max_channel_count)
print(data_attrs.max_arrays)
#%%
# Plot the attention heatmap
import seaborn as sns
from matplotlib.colors import LogNorm
import pandas as pd
attn_trial = save_output.outputs[0][0].cpu()
# print(attn_head.sum(0)) # how much weight did each token get
# print(attn_head.sum(1)) # how much weight did each token give (should be 1)

# attn_head = attn_trial[3] # attending x attended
# attn_head = attn_trial[1] # attending x attended
# attn_head = attn_trial[2] # attending x attended
# make 4 subplots
f, axs = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)
def plot_attn(ax, attn_head, crop_ctx=True):
    ax = prep_plt(ax)
    # turn 2d matrix into a long dataframe with columns 'target' and 'src'
    if crop_ctx:
        attn_head = attn_head[:-2, :-2]
    attn_head = attn_head[:, 1::2]

    # Turn into DF so we can get marginals
    # attn_head = attn_head[1::2, ::2]
    # df = pd.DataFrame(attn_head)
    # df = df.melt()
    # print(df)

    sns.heatmap(attn_head, ax=ax)
    # sns.heatmap(attn_head, ax=ax, vmax=0.1)
    # ax.set_xlabel('Target')
    # ax.set_ylabel('Src')

for head in range(attn_trial.shape[0]):
    plot_attn(axs[head//2, head%2], attn_trial[head])
    # add marginals
plt.xlabel('Target')
plt.ylabel('Src')
