#%%
# Probably the main point of this minor plot is to demonstrate that there _is_ scaling; it's almost impossible to tell whether the trend is slowing at this point.
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model, logger

from context_general_bci.analyze_utils import (
    stack_batch, load_wandb_run, prep_plt, get_dataloader
)
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest

pl.seed_everything(0)

UNSORT = True
# UNSORT = False

ROBUST_RUN = 'session_cross_noctx-wc24ulkl'
DATASET_WHITELIST = [
    "odoherty_rtt-Indy-20160407_02",
    "odoherty_rtt-Indy-20170131_02",
    "odoherty_rtt-Indy-20160627_01",
]

EXPERIMENTS_KIN = [
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/probe',
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/decode',
    f'scale_decode/probe',
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/probe_s2',
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/decode_s2',
    f'scale_decode/probe_s2',
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/probe_s3',
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/decode_s3',
    f'scale_decode/probe_s3',
    # f'scale_decode/probe/mix',
]

queries = [
    's100',
    's200',
    's400',
    's800',
    's1600',
    's3200',
    'sup_20',
    'sup_100',
    'sup_200',
    'sup_800',
    'sup_3200',
    'unsup_20',
    'unsup_100',
    'unsup_200',
    'unsup_800',
    'unsup_3200',
]

merge_queries = [
    f'{q}-frag-{d}' for q in queries for d in DATASET_WHITELIST
]

trainer = pl.Trainer(accelerator='cpu', devices=1, default_root_dir='./data/tmp')
# trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished', 'crashed']},
    "config.dataset.odoherty_rtt.include_sorted": not UNSORT,
})
runs_kin = [r for r in runs_kin if r.config['dataset']['datasets'][0] in DATASET_WHITELIST and r.name.split('-')[0] in queries]

runs_kin.append(get_wandb_run(ROBUST_RUN))

print(f'Found {len(runs_kin)} runs')
#%%
def get_evals(model, dataloader, runs=8, mode='nll'):
    evals = []
    for i in range(runs):
        pl.seed_everything(i)
        heldin_metrics = stack_batch(trainer.test(model, dataloader, verbose=False))
        if mode == 'nll':
            test = heldin_metrics['test_infill_loss'] if 'test_infill_loss' in heldin_metrics else heldin_metrics['test_shuffle_infill_loss']
        else:
            test = heldin_metrics['test_kinematic_r2']
        test = test.mean().item()
        evals.append({
            'seed': i,
            mode: test,
        })
    return pd.DataFrame(evals)[mode].mean()
    # return evals

def get_single_payload(cfg: RootConfig, src_model, run, experiment_set, mode='nll'):
    dataset = SpikingDataset(cfg.dataset)
    set_limit = run.config['dataset']['scale_limit_per_eval_session']
    # if set_limit == 0:
        # train_dev_dataset = SpikingDataset(cfg.dataset)
        # train_dev_dataset.subset_split()
        # set_limit = len(train_dev_dataset)
    dataset.subset_split(splits=['eval'])
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    cfg.model.task.tasks = [ModelTask.kinematic_decoding] # remove stochastic shuffle
    model = transfer_model(src_model, cfg.model, data_attrs)
    dataloader = get_dataloader(dataset)

    payload = {
        'limit': set_limit,
        'variant': run.name.split('-')[0],
        'series': experiment_set,
        'dataset': cfg.dataset.datasets[0],
        'seed': run.config['seed'],
        'lr': run.config['model']['lr_init'], # swept
    }
    payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
    return payload

def build_df(runs, mode='nll'):
    df = []
    seen_set = {}
    for run in runs:
        if 'frag' not in run.name and run.name != ROBUST_RUN:
            continue
        variant, _frag, *rest = run.name.split('-')
        print(run.name, run.config['experiment_set'])
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
        dataset_name = cfg.dataset.datasets[0] # drop wandb ID
        if dataset_name not in DATASET_WHITELIST and run.name != ROBUST_RUN:
            continue
        if run.name == ROBUST_RUN:
            # Special patch for robust run - inject 3 evaluations
            for dataset_name in DATASET_WHITELIST:
                cfg.dataset.datasets = [dataset_name]
                cfg.dataset.exclude_datasets = []
                payload = get_single_payload(cfg, src_model, run, 'session_robust', mode=mode)
                df.append(payload)
            continue

        experiment_set = run.config['experiment_set']
        if variant.startswith('sup') or variant.startswith('unsup'):
            experiment_set = experiment_set + '_' + variant.split('_')[0]
        if (
            variant,
            dataset_name,
            run.config['model']['lr_init'],
            run.config['seed'],
            experiment_set
        ) in seen_set:
            continue
        payload = get_single_payload(cfg, src_model, run, experiment_set, mode=mode)
        df.append(payload)
        seen_set[(variant, dataset_name, run.config['model']['lr_init']), run.config['seed'], experiment_set] = True
    return pd.DataFrame(df)

kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'dataset', 'series', 'seed'])

df = kin_df
import os
os.makedirs('data/tmp', exist_ok=True)
torch.save(df, f'data/tmp/{__file__.split("/")[-1]}.pt')

#%%
df = torch.load(f'data/tmp/{__file__.split("/")[-1]}.pt')

series_remap = {
    'scale_decode/probe_s2_sup': 'scale_decode/probe_sup',
    'scale_decode/probe_s2_unsup': 'scale_decode/probe_unsup',
    'scale_v3/intra_unsort/probe_s2': 'scale_v3/intra_unsort/probe',
    'scale_v3/intra_unsort/decode_s2': 'scale_v3/intra_unsort/decode',
    'scale_decode/probe_s3_sup': 'scale_decode/probe_sup',
    'scale_decode/probe_s3_unsup': 'scale_decode/probe_unsup',
    'scale_v3/intra_unsort/probe_s3': 'scale_v3/intra_unsort/probe',
    'scale_v3/intra_unsort/decode_s3': 'scale_v3/intra_unsort/decode',
    'scale_decode/probe_sup': 'scale_decode/probe_sup',
    'scale_decode/probe_unsup': 'scale_decode/probe_unsup',
    'scale_v3/intra_unsort/probe': 'scale_v3/intra_unsort/probe',
    'scale_v3/intra_unsort/decode': 'scale_v3/intra_unsort/decode',
    'session_robust': 'session_robust',
}
df['series'] = df['series'].map(series_remap)

prescribed_limits = {
    's3200': 3190,
    'unsup_3200': 3190,
    'unsup_800': 770,
    'unsup_200': 200,
    'unsup_100': 100,
    'unsup_20': 20,
    'sup_3200': 3190,
    'sup_800': 770,
    'sup_200': 200,
    'sup_100': 100,
    'sup_20': 20,
    's1600': 1600,
    's800': 770, # relevant for the 2 limited datasets
    's400': 400,
    's200': 200,
    's100': 100,
}
# override `limit` with `prescribed_limits` based on `variant` for `scale_v3/intra_unsort/probe` series
df.loc[df['variant'].isin(prescribed_limits.keys()) & (df['series'] == 'scale_v3/intra_unsort/probe'), 'limit'] = df.loc[df['variant'].isin(prescribed_limits.keys()) & (df['series'] == 'scale_v3/intra_unsort/probe'), 'variant'].map(prescribed_limits)

#%%
print(df[(df['series'] == 'scale_v3/intra_unsort/probe') & (df['variant'] == 's3200')])

#%%
# from context_general_bci.analyze_utils import STYLEGUIDE
sans_robust_df = df[df['series'] != 'session_robust']
palette = sns.color_palette('colorblind', n_colors=len(sans_robust_df['series'].unique()))
hue_order = sans_robust_df['series'].unique()

g = sns.relplot(
    x='limit',
    y='kin_r2',
    style='series',
    hue='series',
    hue_order=hue_order,
    data=sans_robust_df,
    palette=palette,
    errorbar='se',
    # kind='scatter',
    # markers={
        # STYLEGUIDE['markers']
    # }, Nay, this is a different set, can't bring under styleguide.
    markers=True,
    kind='line',
    facet_kws={'sharex': False, 'sharey': False},
    col='dataset',
    # row='dataset',
)
def deco(data, use_title=True, **kws):
    ax = plt.gca()
    ax = prep_plt(ax)
    ax.set_xscale('log')
    ax.set_xlabel('Target session trials')
    ax.set_ylabel('Vel R2')
    # ax.set_yscale('log')

    # Identify the kin r2 of the robust run which has the same dataset
    dataset = data['dataset'].values[0]
    robust_kin_r2 = df[(df['series'] == 'session_robust') & (df['dataset'] == dataset)]['kin_r2'].values[0]
    ax.axhline(robust_kin_r2, color='k', linestyle='--', linewidth=1)
    # Annotate as 'session robust'
    ax.text(18, robust_kin_r2 - 0.01, 'Pretrained (0-Shot)', va='top', ha='left', fontsize=16)
    if not use_title:
        ax.set_title('')

relabel = {
    'scale_v3/intra_unsort/probe': 'Scratch\n(100 Supervised Trials)',
    'scale_v3/intra_unsort/decode': 'Scratch',
    'scale_decode/probe_sup': 'Supervised tune',
    'scale_decode/probe_unsup': 'Unsupervised tune',
    'session_robust': 'Session Robust',
}
g._legend.set_title('Variant')
for t, l in zip(g._legend.texts, hue_order):
    t.set_text(relabel[l])


g.map_dataframe(deco)
g.fig.suptitle(f'Tuning a Decoder ({"Unsorted" if UNSORT else "Sorted"})', y=1.05, fontsize=28)

#%%
# Single panel blowout
# Like the above, but just the middle panel
middle_data = df[df['dataset'] == 'odoherty_rtt-Indy-20160627_01']

# Plot the middle panel data
plt.rc('axes', labelsize=18)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
fig = plt.figure(figsize=(6, 6))
ax = prep_plt(fig.gca())

middle_plot = sns.lineplot(
# middle_plot = sns.relplot(
    x='limit',
    y='kin_r2',
    hue='series',
    style='series',
    hue_order=hue_order,
    data=middle_data,
    palette=palette,
    markers=True,
    errorbar='se',
    # kind='line',
    ax=ax
)

# Customize the middle panel plot
deco(middle_data, use_title=False)
# middle_plot.fig.suptitle(f'Middle Panel: Unsup. Transfer Scaling (100 Trial Calibration)', y=1.05, fontsize=20)
middle_plot.get_legend().set_title('Series')
# middle_plot._legend.set_title('Series')
for t, l in zip(middle_plot.get_legend().texts, middle_plot.get_legend().legendHandles):
# for t, l in zip(middle_plot._legend.texts, middle_plot._legend.legendHandles):
    t.set_text(relabel.get(t.get_text(), t.get_text()))
    t.set_fontsize(16)
    l.set_markersize(10)
    # Exclude Session robust
    if t.get_text() == 'Session Robust':
        l.set_visible(False)
        t.set_visible(False)
# middle_plot._legend.remove()
# Reposition legend to the bottom right

middle_plot.get_legend().set_bbox_to_anchor((0.4, 0.35))
# Turn off frame
# middle_plot.get_legend().get_frame().set_linewidth(0.0)
middle_plot.get_legend().get_frame().set_visible(False)

# drop title
middle_plot.get_legend().set_title('')

middle_plot.set_ylabel('Velocity $R^2$')

# middle_plot.legend.set_bbox_to_anchor((0.6, 0.3))
# save fig as svg
fig.savefig('intra_decode.svg', bbox_inches='tight')