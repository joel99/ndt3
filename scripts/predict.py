#%%
from collections import defaultdict
from typing import Dict, List
from omegaconf import OmegaConf

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from copy import deepcopy

from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nlb_tools.make_tensors import save_to_h5

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.model import BrainBertInterface
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.contexts import context_registry

from context_general_bci.analyze_utils import get_wandb_run, load_wandb_run

# dataset_name = 'mc_rtt'
# dataset_name = 'mc_maze$'

wandb_id = "maze_all_small_ft-23vu306p"

ids = [
    # Parity check (joint NDT1 settings)
    # "rtt-1maz3ea5",
    # "maze_small-hr4prtoe",
    # "maze_med-5v08a4oy",
    # "maze_large-wuawcvls",

    # Not-quite parity NDT2 check (cf heldin)
    # "ndt2_32_rtt-05dqi05j",
    # "ndt2_128_maze_small-tnnvmdkv",
    # "ndt2_128_maze_med_2a-kvuo6q15",
    # "ndt2_128_maze_large_1a-irc57zhv",

    # Scale init
    "m3_150k_large-sweep-simple_lr_sweep-axzvm22s",
    "m3_150k_med-sweep-simple_lr_sweep-81fvl2ws",
    "m3_150k_small-sweep-simple_lr_sweep-rpqx4bpq",
    "m3_150k_rtt-sweep-simple_lr_sweep-81saz29y",
]
# wandb_run = get_wandb_run(wandb_id)
# heldout_model, cfg, data_attrs = load_wandb_run(wandb_run, tag='val-')
#%%
def get_dataloader(dataset: SpikingDataset, batch_size=100, num_workers=1, **kwargs) -> DataLoader:
    # Defaults set for evaluation on 1 GPU.
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collater_factory()
    )

def stack_batch(batch_out: List[Dict[str, torch.Tensor]]):
    out = defaultdict(list)
    for batch in batch_out:
        for k, v in batch.items():
            out[k].append(v)
    for k, v in out.items():
        out[k] = torch.cat(v)
    return out

SPOOFS = { # heldout neuron shapes
# https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb
    'mc_rtt': [30, 32, 1],
    'mc_maze_large': [35, 40, 1],
    'mc_maze_medium': [35, 38, 1],
    'mc_maze_small': [35, 35, 1],
}
HELDIN = {
    'mc_rtt': [30, 98, 1],
    'mc_maze_large': [35, 122, 1],
    'mc_maze_medium': [35, 114, 1],
    'mc_maze_small': [35, 107, 1],
}
def create_submission_dict(wandb_run):
    print(f"creating submission for {wandb_run.id}")
    heldout_model, cfg, data_attrs = load_wandb_run(wandb_run, tag='co-bps')
    # heldout_model, cfg, data_attrs = load_wandb_run(wandb_run, tag='val_loss')
    heldout_model.cfg.task.outputs = [Output.heldout_logrates, Output.spikes]
    cfg.dataset.data_keys = [DataKey.spikes, DataKey.heldout_spikes]
    cfg.dataset.heldout_key_spoof_shape = SPOOFS[cfg.dataset.datasets[0]]

    dataset = SpikingDataset(cfg.dataset)
    test_dataset = deepcopy(dataset)
    dataset.subset_split()
    dataset.build_context_index()
    test_dataset.subset_by_key(['test'], key='split')
    test_dataset.build_context_index()
    if cfg.init_from_id:
        base_run = get_wandb_run(cfg.init_from_id)
        heldin_model, *_ = load_wandb_run(base_run, tag='val_loss')
        heldin_model.cfg.task.outputs = [Output.logrates, Output.spikes]
        # TODO make sure that the heldin model data attrs are transferred
        # raise NotImplementedError
    else:
        heldin_model = heldout_model
        heldin_model.cfg.task.outputs = [Output.logrates, Output.heldout_logrates, Output.spikes]

    dataloader = get_dataloader(dataset)
    test_dataloader = get_dataloader(test_dataset)

    trainer = pl.Trainer(gpus=1, default_root_dir='./data/tmp')
    heldin_outputs = stack_batch(trainer.predict(heldin_model, dataloader))
    test_heldin_outputs = stack_batch(trainer.predict(heldin_model, test_dataloader))
    if cfg.init_from_id:
        heldout_outputs = stack_batch(trainer.predict(heldout_model, dataloader))
        test_heldout_outputs = stack_batch(trainer.predict(heldout_model, test_dataloader))
    else:
        heldout_outputs = heldin_outputs
        test_heldout_outputs = test_heldin_outputs

    # Crop heldout neurons
    heldout_count = SPOOFS[cfg.dataset.datasets[0]][1]
    heldout_outputs[Output.heldout_rates] = heldout_outputs[Output.heldout_rates][...,:heldout_count]
    test_heldout_outputs[Output.heldout_rates] = test_heldout_outputs[Output.heldout_rates][...,:heldout_count]
    heldin_count = HELDIN[cfg.dataset.datasets[0]][1]
    heldin_outputs[Output.rates] = heldin_outputs[Output.rates][...,:heldin_count]
    test_heldin_outputs[Output.rates] = test_heldin_outputs[Output.rates][...,:heldin_count]
    return dataset.cfg.datasets[0], {
        'train_rates_heldin': heldin_outputs[Output.rates].squeeze(2).numpy(),
        'train_rates_heldout': heldout_outputs[Output.heldout_rates].numpy(),
        'eval_rates_heldin': test_heldin_outputs[Output.rates].squeeze(2).numpy(),
        'eval_rates_heldout': test_heldout_outputs[Output.heldout_rates].numpy(),
    }

# #%%
# wandb_runs = [get_wandb_run(wandb_id) for wandb_id in ids]
# submit_dict = create_submission_dict(wandb_runs[0])
# # test = heldin_outputs[Output.rates].squeeze(2).numpy()
# # test = test_heldin_outputs[Output.rates].squeeze(2).numpy()
# test = submit_dict['eval_rates_heldout']
# for trial in range(len(test)):
#     plt.plot(test[trial,:,20])
#     # plt.plot(test[trial,:,10])
#     if trial > 5:
#         break
# # plt.plot(test[0,:,0])
# print("done")
#%%
wandb_runs = [get_wandb_run(wandb_id) for wandb_id in ids]
# Create spikes for NLB submission https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb
suffix = '' # no suffix needed for 5ms submissions
suffix = '_20'
output_dict = {}
for r in wandb_runs:
    dataset_name, payload = create_submission_dict(r)
    if dataset_name == "mc_maze_med":
        dataset_name = "mc_maze_medium"
    output_dict[dataset_name + suffix] = payload

print(output_dict.keys())
print(output_dict[dataset_name + suffix].keys()) # sanity check
# for rtt, expected shapes are 1080 / 272, 120, 98 / 32
print(output_dict[dataset_name + suffix]['train_rates_heldin'].shape) # should be trial x time x neuron
# print(output_dict[dataset_name + suffix]['train_rates_heldout'])

# remove file if exists
if os.path.exists("submission.h5"):
    os.remove("submission.h5")
save_to_h5(output_dict, "submission.h5")
#%%
print(output_dict[dataset_name+suffix]['train_rates_heldin'].sum())