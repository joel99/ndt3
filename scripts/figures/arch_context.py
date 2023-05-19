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
import pytorch_lightning as pl

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs

from context_general_bci.analyze_utils import (
    get_run_config, load_wandb_run, prep_plt
)
from context_general_bci.utils import wandb_query_latest, wandb_query_experiment

from matplotlib.colors import LogNorm, Normalize

# pull experiment

PLOT_VERSION = "supplement_multiscale"
PLOT_VERSION = "multiseed"
# PLOT_VERSION = "multiseed_subject"
if PLOT_VERSION == "multiseed":
    experiment = ['arch/context_s3', 'arch/context_s2', 'arch/context']
elif PLOT_VERSION == "multiseed_subject":
    experiment = ['arch/context_subject_s3', 'arch/context_subject_s2', 'arch/context_subject']
    experiment = ['arch/context_subject_s2', 'arch/context_subject'] # s3 not ready for main paper deadline
else:
    experiment = ['arch/context']

# Quickly pull a run to get full dataset size
# def get_train_dataset_size(sample='scale1_1s'):
#     wandb_run = wandb_query_latest(sample, exact=False, allow_running=True)[0]
#     cfg = get_run_config(wandb_run, tag='val_loss')
#     dataset = SpikingDataset(cfg.dataset)
#     dataset.subset_split()
#     train, _ = dataset.create_tv_datasets()
#     return len(train)
# train_size = get_train_dataset_size()
train_size = 26000 # est, no need to repull every time. Will regenerate data if so, breaking other ongoing runs.

# call wandb api to pull all runs with this experiment tag

runs = wandb_query_experiment(
    experiment,
    state={"$in": ['finished', 'failed', 'crashed', 'running']},
    duration={"$gt": 300},
)

def get_run_dict(run):
    history = run.scan_history(
    # out = run.history(
        # samples=5000, # unfortunately heavy since we need log scale
        keys=[
            'trainer/global_step',
            'eval_loss',
            'epoch',
        ]
    )
    out = pd.DataFrame(history)
    out['id'] = run.id
    out['tag'] = run.config['tag']
    out['seed'] = run.config['seed']
    out['scale_factor'] = run.config['dataset']['scale_ratio']
    if 'subject' in PLOT_VERSION:
        if run.config['model']['subject_embed_strategy'] == str(EmbedStrat.none):
            out['context_tokens'] = 0
        else:
            out['context_tokens'] = run.config['model']['subject_embed_token_count']
    else:
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

# filter to keep only the first of each variant
def hash_run(run):
    return (run.config['tag'], run.config['experiment_set'])
seen_variants = {}
for run in runs:
    print(run.config['tag'], run.config['experiment_set'])
    if hash_run(run) not in seen_variants:
        seen_variants[hash_run(run)] = run
runs = list(seen_variants.values())
print(len(runs))
# run_dicts = [get_run_dict(run) for run in runs]

#%%
def plot_multiscale():
    title = "Increased context capacity speeds convergence"
    run_labels = [
        "scale1_0s", "scale1_1s", "scale1_8s",
        "scale4_0s", "scale4_1s", "scale4_8s",
        "scale16_0s", "scale16_1s", "scale16_8s",
    ]

    df = get_run_df(runs, run_labels)

    cmap = sns.color_palette('viridis_r', n_colors=3)
    # cmap = sns.color_palette('viridis_r', as_cmap=True)
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
        # ax.set_xlim(4e2, ax.get_xlim()[1])
        ax.set_ylim(ax.get_ylim()[0], 0.6)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Test loss')
        ax.set_title('')
        # Annotate with label on top right of plot
        ax.text(
            0.9, 0.6,
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

def plot_multiseed():
    run_labels = [
        "scale1_0s", "scale1_1s", "scale1_8s",
    ]
    df = get_run_df(runs, run_labels)
    df = df[df['trainer/global_step'] % 10 == 0]

    cmap = sns.color_palette('viridis_r', n_colors=3)

    # Filter dataframe for runs with "scale1" in their names

    # Create the plot
    fig = plt.figure(figsize=(6, 4))
    ax = prep_plt(ax=fig.gca())
    ax = sns.lineplot(
        data=df,
        x='trainer/global_step',
        y='eval_loss',
        hue='context_tokens',
        style='context_tokens',
        palette=cmap,
        alpha=0.8,
        errorbar='se',
        ax=ax
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ax.get_ylim()[0], 0.4)  # Set max ylim to 0.45
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Test loss')
    ax.set_title('')

    leg = ax.legend(
        title=f'{"Subject" if "subject" in PLOT_VERSION else "Session"} Tokens', loc='lower left', frameon=False,
        fontsize=14,
        title_fontsize=14,
    )
    for line in leg.get_lines():
        line.set_linewidth(3.0)

    # Only mark min and max on y-axis
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    # Rotate ytick labels by 90 degrees
    # Do not use scientific notation
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.tick_params(axis='y', rotation=90)



    # Create the inset axes
    ax_inset = fig.add_axes([0.65, 0.5, 0.2, 0.35])  # Adjust the position and size of the inset

    # Zoomed-in plot in the inset axes
    sns.lineplot(
        data=df,
        x='trainer/global_step',
        y='eval_loss',
        hue='context_tokens',
        style='context_tokens',
        palette=cmap,
        alpha=0.8,
        ax=ax_inset,
        legend=False,
    )

    ax_inset.set_yscale('linear')  # Use linear scale for the y-axis in the inset
    ax_inset.set_xscale('log')
    if 'subject' in PLOT_VERSION:
        ax_inset.set_xlim(1e4, 2.5e4)
        ax_inset.set_ylim(0.32, 0.325)
    else:
        ax_inset.set_xlim(1e4, 4.e4)  # Adjust the x-axis limits for the zoomed-in view
        ax_inset.set_ylim(0.31, 0.32)  # Adjust the y-axis limits for the zoomed-in view
        ax_inset.set_yticks([0.31, 0.32])
    ax_inset.get_xaxis().set_visible(False)
    # ax_inset.set_xticks([])
    # ax_inset.set_xticklabels([])
    # ax_inset.set_xticks([5e3, 2e4])
    ax_inset.spines['top'].set_alpha(0.2)
    ax_inset.spines['right'].set_alpha(0.2)
    ax_inset.spines['bottom'].set_alpha(0.2)
    ax_inset.yaxis.set_major_locator(ticker.MaxNLocator(2))
    ax_inset.yaxis.set_minor_locator(ticker.MaxNLocator(4))
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')
    ax_inset.set_title('')  # Add a title to the inset plot
    # turn on grid for both major and minor ticks
    ax_inset.grid(True, which='both', alpha=0.1)

    inset_box = ax.indicate_inset_zoom(ax_inset, edgecolor='black', alpha=0.5, linewidth=2)

    fig.tight_layout()
    fig.savefig('arch_context.png', bbox_inches='tight')


if PLOT_VERSION in ['multiseed', 'multiseed_subject']:
    plot_multiseed()
elif PLOT_VERSION == 'supplement_multiscale':
    plot_multiscale()