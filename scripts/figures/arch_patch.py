#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

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

PLOT_DECODE = False
USE_SORTED = True
# USE_SORTED = False

MODE = 'patch'
MODE = 'mask_ratio'

exp_tag = f'robust{"" if USE_SORTED else "_unsort"}'
EXPERIMENTS_KIN = [
    f'arch/{exp_tag}/probe',
]
EXPERIMENTS_NLL = [
    f'arch/{exp_tag}',
    f'arch/{exp_tag}/tune',
]

if MODE == 'patch':
    queries = [
        'single_f8',
        'f8',
        'f8_nopool',
        'subject_f8',
        'task_f8',
        'single_time',
        'single_f32',
        'f32',
        'f32_nopool',
        'subject_f32',
        'task_f32',
    ]
elif MODE ==  'mask_ratio':
    queries = [
        # 'f32',
        # 'f32_m25',
        # 'f32_m75',
        # 'stitch',
        # 'stitch_m25',
        'stitch_m75'
    ]


trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "config.dataset.scale_limit_per_eval_session": 300,
    # "config.dataset.odoherty_rtt.include_sorted": USE_SORTED,
    "state": {"$in": ['finished']},
})
runs_nll = wandb_query_experiment(EXPERIMENTS_NLL, order='created_at', **{
    "config.dataset.scale_limit_per_eval_session": 300,
    "config.dataset.odoherty_rtt.include_sorted": USE_SORTED,
    "state": {"$in": ['finished']},
})
runs_nll = [r for r in runs_nll if r.name.split('-')[0] in queries]
for r in runs_nll:
    print(r.name)
runs_kin = [r for r in runs_kin if r.name.split('-')[0] in queries]
print(len(runs_nll))
print(len(runs_kin)) # 4 * 5 * 3
# runs_kin = runs_kin[:10]
# runs_nll = runs_nll[:10]
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

def build_df(runs, mode='nll'):
    df = []
    seen_set = {}
    for run in runs:
        if 'frag' not in run.name:
            continue
        variant, _frag, *rest = run.name.split('-')
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
        dataset_name = cfg.dataset.datasets[0] # drop wandb ID
        if (variant, dataset_name, run.config['model']['lr_init']) in seen_set:
            continue
        dataset = SpikingDataset(cfg.dataset)
        dataset.subset_split(splits=['eval'])
        dataset.build_context_index()
        data_attrs = dataset.get_data_attrs()
        model = transfer_model(src_model, cfg.model, data_attrs)
        dataloader = get_dataloader(dataset, batch_size=16 if cfg.model.neurons_per_token == 8 else 100)
        payload = {
            'variant': variant,
            'dataset': dataset_name,
            'chunk': run.config['model']['neurons_per_token'],
            'lr': run.config['model']['lr_init'], # swept
        }
        payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
        df.append(payload)
        seen_set[(variant, dataset_name, run.config['model']['lr_init'])] = True
    return pd.DataFrame(df)
kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'dataset'])

nll_df = build_df(runs_nll, mode='nll')
# find f8 and f32 and clone to f8_nopool and f32_nopool
if 'f8_nopool' in queries:
    f8_df = nll_df[nll_df.variant == 'f8'].copy()
    f8_df['variant'] = 'f8_nopool'
    nll_df = pd.concat([nll_df, f8_df])
if 'f32_nopool' in queries:
    f32_df = nll_df[nll_df.variant == 'f32'].copy()
    f32_df['variant'] = 'f32_nopool'
    nll_df = pd.concat([nll_df, f32_df])

# merge on variant and dataset, filling empty with 0s
df = pd.merge(kin_df, nll_df, on=['variant', 'dataset'], how='outer').fillna(0)

#%%
# Meta-annotation

source_map = {
    'single_f8': 'single',
    'f8': 'session',
    'f8_nopool': 'session',
    'subject_f8': 'subject',
    'task_f8': 'task',
    'task_f32': 'task',
    'task_stitch': 'task',
    'subject_f32': 'subject',
    'subject_stitch': 'subject',
    'f32': 'session',
    'f32_nopool': 'session',
    'stitch': 'session',
    'single_f32': 'single',
    'single_time': 'single',
    'time': 'session',
    'stitch_96': 'session',

    'f32_m25': 'session',
    'f32_m75': 'session',
}
arch_map = {
    'single_f8': 'f8',
    'f8': 'f8',
    'f8_nopool': 'f8_nopool',
    'subject_f8': 'f8',
    'task_f8': 'f8',
    'task_f32': 'f32',
    'task_stitch': 'stitch',
    'subject_f32': 'f32',
    'f32_nopool': 'f32_nopool',
    'subject_stitch': 'stitch',
    'f32': 'f32',
    'stitch': 'stitch',
    'single_f32': 'f32',
    'single_time': 'time',
    'time': 'time',
    'stitch_96': 'stitch',
}

if MODE == 'mask_ratio':
    arch_map = {
        'f32': '50%',
        'f32_m25': '25%',
        'f32_m75': '75%',
    }

df['source'] = df['variant'].apply(lambda x: source_map[x])
df['arch'] = df['variant'].apply(lambda x: arch_map[x])

# https://docs.google.com/spreadsheets/d/1WpmhgttDJY09IxHzZqfHh5e8vozeI9c42o8xLfibEng/edit?usp=sharing
eRFH_baseline_kin = {
    'odoherty_rtt-Indy-20160407_02': 0.64575,
    'odoherty_rtt-Indy-20160627_01': 0.53125,
    'odoherty_rtt-Indy-20161005_06': 0.484,
    'odoherty_rtt-Indy-20161026_03': 0.5955,
    'odoherty_rtt-Indy-20170131_02': 0.5113,
}


#%%
ax = prep_plt()
aggr_variant = df.groupby(['variant', 'source', 'arch']).mean().reset_index()
print(len(aggr_variant.arch.unique()))
palette = sns.color_palette('colorblind', len(aggr_variant.arch.unique()))
hue_order = list(aggr_variant.arch.unique())
ax = sns.scatterplot(
    x='nll',
    y='kin_r2',
    hue='arch',
    hue_order=hue_order,
    # hue='variant',
    style='source',
    s=100,
    data=aggr_variant,
    palette=palette,
    legend=True,
)

# Annotate the individual datapoints
# for i, row in aggr_variant.iterrows():
#     ax.text(
#         row.nll, row.kin_r2 + 0.005, row.variant, color=palette[i], ha='center', va='bottom',
#         fontsize=14,
#     )

mean_baseline = np.mean([eRFH_baseline_kin[k] for k in eRFH_baseline_kin if k in df.dataset.unique()])
ax.axhline(mean_baseline, ls='--', color='black')# , label='mean across variants')
# Annotate the horizontal line

# ax.set_title(f'Vel R2 vs NLL ({"Sorted" if USE_SORTED else "Unsorted"})')
# Annotate with Sorted or Unsorted in bottom left of plot

# Velocity decoding R2 for y, with latex
ax.set_ylabel("Velocity $R^2$ ($\\uparrow$)")
ax.set_xlabel('Test NLL ($\downarrow$)')

# Reduce major xticks for clarity
ax.xaxis.set_major_locator(ticker.LinearLocator(3))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.3f}"))
ax.xaxis.set_minor_locator(ticker.LinearLocator(5))
# ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.5, 0.7, 3)))
ax.yaxis.set_minor_locator(ticker.FixedLocator(np.linspace(0.5, 0.7, 5)))
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
# Update the legend - kill arch, we can show those color coded.

# Only for this panel
camera_ready_arch_remap = {
    'f8': 'Patch 8 (Pool)',
    'f32': 'Patch 32 (Pool)',
    'f32_nopool': 'Patch 32 (No pool)',
    'f8_nopool': 'Patch 8 (No pool)',
    'time': 'NDT1',
    # 'stitch': 'NDT1-Stitch'
}

if MODE == 'mask_ratio':
    camera_ready_arch_remap = {
        '50%': '50% Mask',
        '25%': '25% Mask',
        '75%': '75% Mask',
    }

# Annotate the data with their variant. Skip once variant has been marked
# marked_variants = set()
# for i, row in aggr_variant.iterrows():
#     if row.arch in marked_variants:
#         continue
#     ax.text(
#         row.nll + 0.0005, row.kin_r2 + 0.005,
#         camera_ready_arch_remap[row.arch],
#         color=palette[hue_order.index(row.arch)], ha='left', va='bottom',
#         fontsize=14,
#     )
#     marked_variants.add(row.arch)

# No.... this doesn't look good. Just put them under legend. We'll adjust in post.
# Add text for three archictectures on the right side of the plot


# Only keep the labels from source, onwards
if MODE == 'mask_ratio':
    handles, labels = ax.get_legend_handles_labels()
    # keep only 2-4
    handles = handles[1:4]
    labels = labels[1:4]
    ax.legend(
        handles, labels, fontsize=14, ncol=1, frameon=False,
        title='Mask Ratio', title_fontsize=18,
    )
    ax.text(
        ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.5,
        ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
        'Sorted' if USE_SORTED else 'Unsorted', color='black',
        ha='right', va='top',
        fontsize=18, fontweight='bold'
    )
else:
    ax.text(
        ax.get_xlim()[1] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.35,
        ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
        'Sorted' if USE_SORTED else 'Unsorted', color='black',
        ha='right', va='top',
        fontsize=18, fontweight='bold'
    )
    for arch, y in zip(hue_order, np.arange(len(hue_order)) * -0.09 + 0.46):
        ax.text(
            1.25,
            y,
            camera_ready_arch_remap[arch],
            color=palette[hue_order.index(arch)], ha='center', va='top',
            fontsize=16,
            # in axes coords
            transform=ax.transAxes,
        )
    handles, labels = ax.get_legend_handles_labels()
    print(labels)
    source_idx = labels.index('source')
    labels = labels[source_idx:]
    order = ['single', 'session', 'subject', 'task']
    remap = {
        'source': 'Source',
        'single': 'Intra-session',
        'session': 'Cross-Session',
        'subject': 'Cross-Subject',
        'task': 'Cross-Task',
    }
    reorder_idx = [labels.index(o) for o in order]
    labels = np.array([remap[l] for l in labels])[reorder_idx]
    handles = np.array(handles[source_idx:])[reorder_idx]

    lgd = ax.legend(
        handles, labels, loc='upper right', fontsize=14, ncol=1, frameon=False,
        # title='Data Source', title_fontsize=14,
        bbox_to_anchor=(1.52, 0.92),
    )
    for handle in lgd.legendHandles:
        handle._sizes = [80]

