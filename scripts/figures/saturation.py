#%%
import logging
import sys
from typing import List
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from data import SpikingDataset, DataAttrs
from model import transfer_model, logger

from analyze_utils import get_run_config, load_wandb_run, wandb_query_latest
from analyze_utils import prep_plt

from matplotlib.colors import LogNorm, Normalize
import wandb

# pull experiment
experiment = ['saturation', 'saturation/loco_failfast']

# Quickly pull a run to get full dataset size
def get_train_dataset_size(sample='scale1'):
    wandb_run = wandb_query_latest(sample, exact=False, allow_running=True)[0]
    cfg = get_run_config(wandb_run, tag='val_loss')
    dataset = SpikingDataset(cfg.dataset)
    dataset.subset_split()
    train, _ = dataset.create_tv_datasets()
    return len(train)
train_size = get_train_dataset_size()

# call wandb api to pull all runs with this experiment tag

def wandb_query_experiment(
    experiment: str | List[str],
    wandb_user="joelye9",
    wandb_project="context_general_bci",
):
    if not isinstance(experiment, list):
        experiment = [experiment]
    api = wandb.Api()
    filters = {
        'config.experiment_set': {"$in": experiment},
    }
    runs = api.runs(f"{wandb_user}/{wandb_project}", filters=filters)
    return runs

runs = wandb_query_experiment(experiment)

def get_run_dict(run):
    out = run.history(
        samples=1000, # unfortunately heavy since we need log scale
        keys=[
            'trainer/global_step',
            'eval_loss',
            'epoch',
        ]
    )
    out['id'] = run.id
    out['tag'] = run.config['tag']
    out['scale_factor'] = run.config['dataset']['scale_ratio']
    if run.config['model']['session_embed_strategy'] == str(EmbedStrat.none):
        out['context_tokens'] = 0
    else:
        out['context_tokens'] = run.config['model']['session_embed_token_count']
    out['token_size'] = run.config['model']['neurons_per_token']
    out['trials'] = round(out['scale_factor'] * train_size)
    out['n_layers'] = run.config['model']['transformer']['n_layers']
    return out

def get_run_df(runs, labels):
    return pd.concat([
        get_run_dict(run) for run in runs if run.config['tag'] in labels
    ])

# run_dicts = [get_run_dict(run) for run in runs]

#%%
title = 'General sufficiency of spatial patches'
run_labels = ['scale1', 'scale1_t32', 'scale1_t144', 'scale_nonflat', 'scale_stitch'] # t144 is a typo, it's 128-neuron

mode = 'step'
mode = 'epoch'

df = get_run_df(runs, run_labels)
# Thin out the plot for legibility
df = df.groupby("tag").sample(n=100, random_state=1, replace=False)

ax = prep_plt()
sns.lineplot(
    data=df,
    x='trainer/global_step' if mode == 'step' else 'epoch',
    y='eval_loss',
    hue='tag',
    style='tag',
    ax=ax,
    alpha=0.7,
    legend=True
)

# Relabel legend
relabel = {
    'scale_stitch': 'NDT1 + Stitch',
    'scale_nonflat': 'NDT1 (Time)',
    'scale1': 'NDT2 (4-Neuron)',
    'scale1_t32': 'NDT2 (32-Neuron)',
    'scale1_t144': 'NDT2 (128-Neuron)',
}
# offsets = {
#     'scale_stitch': (-50, -10),
#     'scale_nonflat': (0, 30) if mode == 'step' else (-30, 30),
#     'scale1': (-40, 30) if mode == 'step' else (-60, -10),
#     'scale1_t32': (0, 30) if mode == 'step' else (-30, 30),
#     'scale1_t144': (0, 30) if mode == 'step' else (-30, 30),
# }
handles, labels = ax.get_legend_handles_labels()
labels = [relabel[label] for label in labels]
ax.legend(handles, labels, frameon=False)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e2 if mode == 'step' else 1e1, ax.get_xlim()[1])
ax.set_ylim(0.25, 0.33)
ax.set_xlabel('Training Steps' if mode == 'step' else 'Epochs')
ax.set_ylabel('Test loss')

# Only mark min and max on y-axis
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax.yaxis.set_minor_locator(ticker.NullLocator())

# rotate ytick labels by 90 degrees
# do not use scientific notation
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.tick_params(axis='y', rotation=90)


order = ['scale_stitch', 'scale_nonflat', 'scale1_t144', 'scale1_t32', 'scale1', 'scale1_t32']
# Convert legend to annotations
# for i, line in enumerate(ax.lines):
#     # Get the source tag associated with line
#     tag = order[i]
#     x = line.get_xdata()[-1]
#     y = line.get_ydata()[-1]
#     ax.annotate(
#         relabel[tag],
#         xy=(x, y),
#         xytext=offsets[tag],
#         textcoords='offset points',
#         ha='center',
#         va='center',
#         color=line.get_color(),
#         fontsize=12,
#     )

#%%

title = "[RTT Indy] Spatial tokens degrade gracefully"
run_labels = [
    'scale8_stitch',
    'scale_stitch',
    # 'scale16_t32',
    # 'scale4_t32',
    # 'scale1_t32',
    'scale16',
    'scale4',
    'scale1',
]


df = get_run_df(runs, run_labels)
# Thin out the plot for legibility
df = df.groupby("tag").sample(n=100, random_state=1, replace=False)
def variant(x):
    if 'stitch' in x.tag:
        return "Stitch"
    return f"{x.token_size}-token"
df['Model'] = df.apply(variant, axis=1)
df['Scaledown'] = df['scale_factor'].apply(lambda x: f"{int(1/x)}x")

ax = prep_plt()
sns.lineplot(
    data=df,
    x='trainer/global_step' if mode == 'step' else 'epoch',
    y='eval_loss',
    hue='Model',
    # style='variant',
    size='Scaledown',
    size_order=['1x', '4x', '8x', '16x'],
    ax=ax,
    alpha=0.7,
    legend=True
)

# # Relabel legend
# relabel = {
#     'scale_stitch': 'NDT1 + Stitch',
#     'scale_nonflat': 'NDT1 (Time)',
#     'scale1': 'NDT2 (4-Neuron)',
#     'scale1_t32': 'NDT2 (32-Neuron)',
#     'scale1_t144': 'NDT2 (128-Neuron)',
# }
handles, labels = ax.get_legend_handles_labels()
# labels = [relabel[label] for label in labels]
ax.legend(handles, labels, frameon=False, loc='upper right')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e2 if mode == 'step' else 1e1, ax.get_xlim()[1])
ax.set_ylim(0.25, 0.33)
ax.set_xlabel('Training Steps' if mode == 'step' else 'Epochs')
ax.set_ylabel('Test loss')

# Only mark min and max on y-axis
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax.yaxis.set_minor_locator(ticker.NullLocator())

# rotate ytick labels by 90 degrees
# do not use scientific notation
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.tick_params(axis='y', rotation=90)


#%%
# Loco transfer
title = "[RTT Indy] Loco transfer"

run_labels = [
    'loco_stitch',
    'loco_t128',
    'loco_t32',
    'loco_t4',
    'scale1'
]


df = get_run_df(runs, run_labels)
# Thin out the plot for legibility
df = df.groupby("tag").sample(n=100, random_state=1, replace=False)

ax = prep_plt()
sns.lineplot(
    data=df,
    x='trainer/global_step' if mode == 'step' else 'epoch',
    y='eval_loss',
    hue='tag',
    # style='variant',
    ax=ax,
    alpha=0.7,
    legend=True
)

# Relabel legend
relabel = {
    'loco_stitch': 'S2->S1 Stitch',
    'loco_t128': 'S2->S1 (128-Neuron)',
    'loco_t32': 'S2->S1 (32-Neuron)',
    'loco_t4': 'S2->S1 (4-Neuron)',
    'scale1': 'S1 (4-Neuron)',
}
handles, labels = ax.get_legend_handles_labels()
labels = [relabel[label] for label in labels]
ax.legend(handles, labels, frameon=False, loc='upper right')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e2 if mode == 'step' else 1e1, ax.get_xlim()[1])
ax.set_ylim(0.25, 0.33)
ax.set_xlabel('Training Steps' if mode == 'step' else 'Epochs')
ax.set_ylabel('Test loss')

# Only mark min and max on y-axis
ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
ax.yaxis.set_minor_locator(ticker.NullLocator())

# rotate ytick labels by 90 degrees
# do not use scientific notation
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
ax.tick_params(axis='y', rotation=90)

#%%

title = '[RTT Indy] Power law saturation'
run_labels = ['scale1', 'scale2', 'scale4', 'scale8', 'scale16', 'scale32']

df = get_run_df(runs, run_labels)

cmap = sns.color_palette('viridis_r', as_cmap=True)

ax = prep_plt()
sns.lineplot(
    data=df,
    x='trainer/global_step',
    y='eval_loss',
    hue='trials',
    palette=cmap,
    ax=ax,
    alpha=1.0,
    legend=False
)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e2, ax.get_xlim()[1])
ax.set_xlabel('Training Steps')
ax.set_ylabel('Test loss')

# Plot trials as colorbar
norm = LogNorm(df['trials'].min(), df['trials'].max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Trials')

# For each scale factor, find the training step corresponding to the minimum test loss
min_steps = []
min_losses = []
for scale, group in df.groupby('scale_factor'):
    min_steps.append(group.loc[group['eval_loss'].idxmin()]['trainer/global_step'])
    min_losses.append(group['eval_loss'].min())

# Scatter the minimum test loss vs. training steps, on top of the existing curves
ax.scatter(min_steps, min_losses, color='k', s=40, zorder=10)

# Fit power law to the minimum test loss vs. training steps
from scipy.optimize import curve_fit
def power_law(x, a, b):
    return a * x**b
popt, pcov = curve_fit(power_law, min_steps[:4], min_losses[:4])
x = torch.linspace(1e2, 1e6, 100)
y = power_law(x, *popt)
ax.plot(x, y, color='k', linestyle='--', label='Power law fit')

# Annotate the power law fit in axis coordinates, with latex
ax.text(
    0.2, 0.45,
    rf'$L={popt[0]:.2f}S^{{{popt[1]:.2f}}}$',
    transform=ax.transAxes,
    horizontalalignment='center',
)


#%%
# JY: This plot is actually pretty unsatisfying, cf https://wandb.ai/joelye9/context_general_bci/reports/-Saturation---VmlldzozNjAyOTE3
# - we should consider building nicer wandb plots; static plots are dumb.

title = "Increased context capacity speeds convergence"
run_labels = [
    "scale2", "scale2_0s", "scale2_8s",
    "scale8", "scale8_0s", "scale8_8s",
    "scale32", "scale32_0s", "scale32_8s",
]

df = get_run_df(runs, run_labels)

cmap = sns.color_palette('viridis_r', as_cmap=True)
# Make a facet grid with one row per scale factor
g = sns.FacetGrid(
    data=df,
    row='trials',
    row_order=sorted(df['trials'].unique()),
    sharex=True,
    sharey=True,
    height=3,
    aspect=2,
    gridspec_kws={'hspace': 0.05},
)
g.map_dataframe(
    sns.lineplot,
    x='trainer/global_step',
    y='eval_loss',
    hue='context_tokens',
    style='context_tokens',
    # style_order=reversed(sorted(df['context_tokens'].unique())),
    palette=cmap,
    alpha=0.8,
    legend=True,
    # legend=False,
)

g.add_legend(title='Context Tokens', bbox_to_anchor=(0.8, 0.4), loc='lower right')


labels = sorted(df['trials'].unique())
for i, ax in enumerate(g.axes.flat):
    ax = prep_plt(ax=ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(4e2, ax.get_xlim()[1])
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Test loss')
    ax.set_title('')
    # Annotate with label on top right of plot
    ax.text(
        0.9, 0.8,
        # only use 3 sigfigs
        rf'$S={labels[i]:.2g}$ trials',
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=15,
    )

    # Only mark min and max on y-axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    # rotate ytick labels by 90 degrees
    # do not use scientific notation
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.tick_params(axis='y', rotation=90)



#%%
title = 'Not bottlenecked by model size'
# TODO
