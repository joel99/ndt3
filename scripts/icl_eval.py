#%%
# Basic script to probe for ICL capabilities
from typing import Dict
import itertools
from matplotlib import pyplot as plt
import numpy as np
import torch
import seaborn as sns
import lightning.pytorch as pl

from einops import rearrange, pack, unpack
from sklearn.metrics import r2_score

from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model
from context_general_bci.task_io import create_token_padding_mask
from context_general_bci.contexts import context_registry
from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt
from context_general_bci.utils import get_wandb_run, wandb_query_latest

# Exactly matched to training
CONTEXT_S_SUITE = [0, 3, 12, 27]
PROBE_S_SUITE = [3] # Mostly an efficiency param, matched to train for simplicity
STRIDE_S_SUITE = [3] # Shorter should yield higher perf
BINS_PER_S = 50

EVAL_DATASET_SUITE = [
    "odoherty_rtt-Indy-20160407_02",
    "odoherty_rtt-Indy-20160627_01",
    "odoherty_rtt-Indy-20161005_06",
    "odoherty_rtt-Indy-20161026_03",
    "odoherty_rtt-Indy-20170131_02"
]

wandb_run = wandb_query_latest('no_embed', allow_running=True)[0]
# wandb_run = wandb_query_latest('30s_no_embed')[0]
print(f'ICL Eval for: {wandb_run.id}')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run)
# cfg.model.task.outputs = [Output.behavior, Output.behavior_pred] # Don't actually need this, just need the metric
cfg.dataset.exclude_datasets = []

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')

def compute_icl_eval(data_id, context_s=27, probe_s=3, stride_s=3, batch_size=16):
    cfg.dataset.datasets = [data_id]
    dataset = SpikingDataset(cfg.dataset)
    data_attrs = dataset.get_data_attrs()
    model = transfer_model(src_model, cfg.model, data_attrs)
    full_payload: Dict[DataKey, torch.Tensor] = {}
    # I need non-trialized data to scan. Concatenate the payload (RTT preprocessing is just chopping)
    for k in [
        DataKey.spikes, DataKey.time, DataKey.position,
        DataKey.bhvr_vel, DataKey.covariate_time, DataKey.covariate_space,
    ]:
        full_payload[k] = torch.cat([i[k] for i in dataset], dim=0)
    # plt.plot(full_payload[DataKey.bhvr_vel][full_payload[DataKey.covariate_space] == 0])
    context_tokens = context_s * BINS_PER_S
    probe_tokens = probe_s * BINS_PER_S
    stride_tokens = stride_s * BINS_PER_S
    assert probe_tokens == stride_tokens, "Stride must be equal to probe for now"
    # Following predict only makes sense for an infilling model.
    print(context_tokens + probe_tokens)
    scan_payload = {
        k: v.unfold(0, context_tokens + probe_tokens, stride_tokens) for k, v in full_payload.items()
    } # Intra-fold time is last, pseudo-trial is first.

    all_bhvr = []
    all_bhvr_tgt = []
    for trial in range(0, scan_payload[DataKey.spikes].size(0), batch_size):
        batch = {
            k: v[trial:trial+batch_size, ...].to('cuda:0') for k, v in scan_payload.items()
        }
        batch[DataKey.padding] = create_token_padding_mask(batch[DataKey.spikes], batch)
        # Crop neural context
        model.task_pipelines[ModelTask.shuffle_infill].crop_batch(
            mask_ratio=probe_s / (context_s + probe_s),
            batch=batch,
            eval_mode=False,
            shuffle=False # Keep order when cropping
        )
        # Crop behavioral context
        model.task_pipelines[ModelTask.kinematic_decoding].crop_batch(
            mask_ratio=probe_s / (context_s + probe_s),
            batch=batch,
            eval_mode=False,
            shuffle=False # Keep order when cropping
        )
        enc, enc_time, enc_space, enc_pad = model(batch)
        # TODO call task io modules as per usual? And ensure we can retrieve those outputs?
        # Since we're square, no need to worry about padding
        # No trial context
        bhvr = model.task_pipelines[ModelTask.kinematic_decoding].get_cov_pred(
            batch, enc, enc_time, enc_space, enc_pad, eval_mode=False
        )
        bhvr_tgt = model.task_pipelines[ModelTask.kinematic_decoding].get_target(batch)
        all_bhvr.append(bhvr)
        all_bhvr_tgt.append(bhvr_tgt)
    return r2_score(torch.cat(all_bhvr_tgt, dim=0).cpu().numpy(), torch.cat(all_bhvr, dim=0).cpu().numpy())
results = []
for data_id, context_s, probe_s, stride_s in itertools.product(EVAL_DATASET_SUITE, CONTEXT_S_SUITE, PROBE_S_SUITE, STRIDE_S_SUITE):
    results.append({
        'data_id': data_id,
        'context_s': context_s,
        'probe_s': probe_s,
        'stride_s': stride_s,
        'icl': compute_icl_eval(data_id, context_s, probe_s, stride_s)    })
    break