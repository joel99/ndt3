#%%
# Qualitative plot
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model, logger, BrainBertInterface

from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt, get_dataloader
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest

pl.seed_everything(0)

EVAL_DATASETS = [
    'observation_CRS02b_19.*',
    # 'observation_CRS07_15.*',
    # 'observation_CRS07_16.*',
]
# expand by querying alias
EVAL_DATASETS = SpikingDataset.list_alias_to_contexts(EVAL_DATASETS)
EVAL_ALIASES = [x.alias for x in EVAL_DATASETS]

EXPERIMENTS_KIN = [
    f'pitt_v3/probe_01_cross',
]

queries = [
    # 'human_obs_limit',
    # 'human_obs_m5',
    # 'human_obs_m5_lr1e5', # note this LR is infeasibly slow for RT. Takes ~46 minutes.
    # 'human_m5',
    'human_m5_lr1e5',
    # 'human_unsup',
]

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed']},
})
print(f'Found {len(runs_kin)} runs. Evaluating on {len(EVAL_ALIASES)} datasets.')

TARGET_ALIAS = 'observation_CRS07_150_1_2d_cursor_center_out'
TARGET_ALIAS = 'observation_CRS02b_1925_11_2d_cursor_center_out'
#%%
USE_THRESH = False
# USE_THRESH = True
def get_single_payload(cfg: RootConfig, src_model, run, experiment_set, mode='nll', dataset=None):
    if dataset is None:
        dataset = SpikingDataset(cfg.dataset)
        dataset.subset_split(splits=['eval'])
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    set_limit = run.config['dataset']['scale_limit_per_eval_session']
    cfg.model.task.tasks = [ModelTask.kinematic_decoding] # remove stochastic shuffle
    if USE_THRESH:
        cfg.model.task.metrics = [Metric.kinematic_r2, Metric.kinematic_r2_thresh]
        cfg.model.task.behavior_fit_thresh = 0.1
    model = transfer_model(src_model, cfg.model, data_attrs)
    dataloader = get_dataloader(dataset)
    # the dataset name is of the for {type}_{subject}_session_{session}_set_{set}_....mat
    # parse out the variables
    _, subject, session, set_num, *_ = dataset.cfg.eval_datasets[0].split('_')

    payload = {
        'limit': set_limit,
        'variant': run.name.split('-')[0],
        'series': experiment_set,
        'data_id': f"{subject}_{session}_{set_num}",
        'subject': subject,
        'session': int(session),
        'set': int(set_num),
        'lr': run.config['model']['lr_init'], # swept
    }

    model.cfg.task.outputs = [Output.behavior, Output.behavior_pred]
    sanity = trainer.test(model, dataloader)
    heldin_outputs = stack_batch(trainer.predict(model, dataloader))
    offset_bins = model.task_pipelines[ModelTask.kinematic_decoding.value].bhvr_lag_bins
    pred = heldin_outputs[Output.behavior_pred]
    # preserve trial structure
    assert isinstance(pred, torch.Tensor)
    # * We should mark a blip in the stitch..
    pred = pred[:,offset_bins:]
    true = heldin_outputs[Output.behavior][:, offset_bins:]
    # Don't flatten yet - we need to mark train, val
    pad_per_trial = (true != model.data_attrs.pad_token).any(-1).sum(-1)
    pred = pred[(true != model.data_attrs.pad_token).any(-1)]
    true = true[(true != model.data_attrs.pad_token).any(-1)]
    payload = {
        **payload,
        'pad': pad_per_trial,
        'pred': pred,
        'true': true,
    }
    return payload

def build_df(runs, mode='nll'):
    df = []
    seen_set = {}
    for run in runs:
        variant, _frag, *rest = run.name.split('-')
        if variant not in queries:
            continue
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
        experiment_set = run.config['experiment_set']
        pl.seed_everything(seed=cfg.seed)
        # Mirror run.py dataset construction
        ref_df = SpikingDataset(cfg.dataset)
        tv_ref = deepcopy(ref_df)
        eval_ref = deepcopy(ref_df)
        eval_ref.subset_split(splits=['eval'])
        tv_ref.subset_split()
        train_ref, val_ref = tv_ref.create_tv_datasets()
        for i, alias in enumerate(EVAL_ALIASES):
            if alias != TARGET_ALIAS:
                continue
            inst_df = deepcopy(ref_df)
            inst_df.cfg.eval_datasets = [alias]
            inst_df.cfg.datasets = [alias]
            inst_df.subset_by_key([EVAL_DATASETS[i].id], key=MetaKey.session)
            # ! Ok... so we want to identify whether each trial belongs to train, val, or eval.
            valid_keys = list(val_ref.meta_df[
                (val_ref.meta_df[MetaKey.session] == EVAL_DATASETS[i].id)
            ][MetaKey.unique])
            eval_keys = list(eval_ref.meta_df[
                (eval_ref.meta_df[MetaKey.session] == EVAL_DATASETS[i].id)
            ][MetaKey.unique])
            train_keys = list(train_ref.meta_df[
                (train_ref.meta_df[MetaKey.session] == EVAL_DATASETS[i].id)
            ][MetaKey.unique])
            def map_unique_to_split(unique):
                if unique in valid_keys:
                    return 'val'
                elif unique in eval_keys:
                    return 'eval'
                elif unique in train_keys:
                    return 'train'
                else:
                    raise ValueError(f'Could not find {unique} in any split')
            inst_key_splits = [
                map_unique_to_split(unique) for unique in inst_df.meta_df[MetaKey.unique]
            ]
            if (
                variant,
                alias,
                run.config['model']['lr_init'],
                experiment_set
            ) in seen_set:
                continue

            # Generate a full run of predictions
            payload = get_single_payload(cfg, src_model, run, experiment_set, mode=mode, dataset=inst_df)
            payload['trial_split'] = inst_key_splits
            df.append(payload)
            seen_set[(variant, alias, run.config['model']['lr_init']), experiment_set] = True
    return pd.DataFrame(df)
kin_df = build_df(runs_kin, mode='kin_r2')

#%%
predictions_session = kin_df['pred'][0] # T x 2
true_session = kin_df['true'][0]
palette = sns.color_palette('colorblind', predictions_session.shape[-1])

# We want to label trial split according to whether it was in the train, val or eval split
trial_lengths = kin_df['pad'][0]
trial_labels = kin_df['trial_split'][0]
palette_split = {'train': 'tab:blue', 'val': 'tab:orange', 'eval': 'tab:green'}


# Plot these over time
# Label ax as x and y decode
fig, ax = plt.subplots(2, 1, figsize=(24, 8), sharex=True)
for dim in range(2):
    ax = prep_plt(ax)
    ax[dim].plot(predictions_session[:, dim], label=f'pred {dim}', linestyle='--', color=palette[dim])
    ax[dim].plot(true_session[:, dim], label=f'true {dim}', color=palette[dim])
    ax[dim].legend(frameon=False)
    ax[dim].set_ylabel(f'Dim {dim}')

    # Add colored backgrounds for each trial split
    for i, trial in enumerate(trial_labels):
        start = sum(trial_lengths[:i])
        end = sum(trial_lengths[:i+1])
        ax[dim].axvspan(start, end, alpha=0.3, color=palette_split[trial])

    # Add custom legend entry for trial split background
    handles, labels = ax[dim].get_legend_handles_labels()
    custom_handles = [plt.Rectangle((0,0),1,1, alpha=0.3, color=palette_split[key]) for key in palette_split]
    custom_labels = ['Train', 'Val', 'Eval']
    handles += custom_handles
    labels += custom_labels
    ax[dim].legend(handles, labels, loc='upper left', frameon=False)

# each step is 20ms, change xlabels to ms, only 5 marks

ax[1].set_xticks(np.linspace(0, predictions_session.shape[0], 5))
ax[1].set_xticklabels(np.linspace(0, predictions_session.shape[0] * 20 / 1000, 5).round())
ax[1].set_xlabel('Time (s)')


# title with alias
fig.suptitle(f'Vel decode: {TARGET_ALIAS}')
