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
    stack_batch, load_wandb_run,
    prep_plt, get_dataloader,
)
from context_general_bci.utils import (
    wandb_query_experiment, get_wandb_run, wandb_query_latest
)

pl.seed_everything(0)

UNSORT = True
# UNSORT = False

DATASET_WHITELIST = [
    # "odoherty_rtt-Indy-20160407_02",
    # "odoherty_rtt-Indy-20170131_02",
    "odoherty_rtt-Indy-20160627_01",
]

EXPERIMENTS_NLL = [
    f'scale_v3/intra{"_unsort" if UNSORT else ""}',
    f'scale_v3/session{"_unsort" if UNSORT else ""}/tune_intra',
    f'scale_v3/task{"_unsort" if UNSORT else ""}/tune_intra',
]

queries = [
    's100',
    's200',
    's400',
    's800',
    's1600',
    's3200',
]

merge_queries = [
    f'{q}-frag-{d}' for q in queries for d in DATASET_WHITELIST
]

trainer = pl.Trainer(accelerator='cpu', devices=1, default_root_dir='./data/tmp')
runs_nll = wandb_query_experiment(EXPERIMENTS_NLL, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed']},
    "config.tag": {"$in": merge_queries},
    "config.dataset.odoherty_rtt.include_sorted": not UNSORT,
})
print(f'Found {len(runs_nll)} NLL runs')

#%%
def extract_exp(exp_str: str):
    # if ends with '/probe' or '/tune', remove it
    if exp_str.endswith('/probe'):
        exp_str = exp_str[:-6]
    if exp_str.endswith('/tune'):
        exp_str = exp_str[:-5]
    return exp_str
    # return exp_str.split('/')[-2:]

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
        if dataset_name not in DATASET_WHITELIST:
            continue
        series = extract_exp(run.config['experiment_set'])
        if (variant, dataset_name, series, run.config['model']['lr_init']) in seen_set:
            continue
        dataset = SpikingDataset(cfg.dataset)
        set_limit = run.config['dataset']['scale_limit_per_eval_session']
        if set_limit == 0:
            train_dev_dataset = SpikingDataset(cfg.dataset)
            train_dev_dataset.subset_split()
            set_limit = len(train_dev_dataset)
        dataset.subset_split(splits=['eval'])
        dataset.build_context_index()
        data_attrs = dataset.get_data_attrs()
        model = transfer_model(src_model, cfg.model, data_attrs)
        dataloader = get_dataloader(dataset)
        payload = {
            'limit': set_limit,
            'variant': variant,
            'series': extract_exp(run.config['experiment_set']),
            'dataset': dataset_name,
            'lr': run.config['model']['lr_init'], # swept
        }
        payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
        df.append(payload)
        seen_set[(variant, dataset_name, series, run.config['model']['lr_init'])] = True
    return pd.DataFrame(df)

nll_df = build_df(runs_nll, mode='nll')
df = nll_df
#%%
print(nll_df[nll_df['variant'] == 's1600'])
# Fix these labels, for some reason limit was cast to 770
nll_df.loc[nll_df['variant'] == 's1600', 'limit'] = 1600
# drop redundant runs
df = nll_df.drop_duplicates(['limit', 'dataset', 'series', 'lr'])
# print(df[df['limit'] == 770])
# print(df['limit'].value_counts())
#%%
# Show just NLL in logscale
palette = sns.color_palette('colorblind', n_colors=len(df['dataset'].unique()))
dataset_order = df.groupby(['dataset']).mean().sort_values('nll').index
ax = prep_plt()
ax = sns.scatterplot(
    x='limit',
    y='nll',
    hue='dataset',
    style='series',
    hue_order=dataset_order,
    data=df,
    palette=palette,
    ax=ax,
    legend=False
)
ax.set_xscale('log')
# ax.set_yscale('log')

# Fit power law to the minimum test loss vs. training steps
from scipy.optimize import curve_fit
def power_law(x, a, b):
    return a * x**b

def plot_dataset_power_law(sub_df, ax, **kwargs):
    popt, pcov = curve_fit(power_law, sub_df['limit'], sub_df['nll'])
    x = torch.linspace(sub_df['limit'].min(), sub_df['limit'].max(), 100)
    y = power_law(x, *popt)
    ax.plot(x, y, linestyle='--', **kwargs)
    # annotate with power law
    # ax.annotate(f'{popt[1]:.4f}', xy=(x[0], y[0]), xytext=(x[0] + 10, y[0]), **kwargs)


for i, dataset in enumerate(dataset_order):
    sub_df = df[df['dataset'] == dataset]
    plot_dataset_power_law(sub_df, ax, color=palette[i])

ax.set_title(f'Intra-session scaling ({"unsorted" if UNSORT else "sorted"})')


#%%

relabel = {
    'scale_v3/session_unsort/tune_intra': 'Pretrain Session',
    'scale_v3/task_unsort/tune_intra': 'Pretrain Task',
    'scale_v3/intra_unsort': 'Scratch (Single-session)',
}
palette = sns.color_palette('colorblind', n_colors=len(df['series'].unique()))
# hue_order = list(df.groupby(['series']).mean().sort_values('nll').index)
hue_order = [
    'scale_v3/intra_unsort',
    'scale_v3/session_unsort/tune_intra',
    'scale_v3/task_unsort/tune_intra',
]
dataset_order = sorted(df['dataset'].unique())
g = sns.relplot(
    x='limit',
    y='nll',
    hue='series',
    style='series',
    hue_order=hue_order,
    data=df,
    palette=palette,
    kind='scatter',
    facet_kws={'sharex': False, 'sharey': False},
    col='dataset',
    col_order=dataset_order,
)

# retitle subplots
title_remap = {
    'odoherty_rtt-Indy-20170131_02': 'Final',
    'odoherty_rtt-Indy-20160407_02': 'Initial',
    'odoherty_rtt-Indy-20160627_01': 'Middle',
}

def deco(data, **kws):
    ax = plt.gca()
    ax = prep_plt(ax)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.set_ylabel('Test NLL')
    ax.set_xlabel('Target context trials')
    for i, series in enumerate(hue_order):
        sub_df = data[data['series'] == series]
        plot_dataset_power_law(sub_df, ax, color=palette[i])

    alias = ax.get_title().split('=')[1].strip()
    ax.set_title(title_remap.get(alias, alias), fontsize=20)

# df['series_relabel'] = df['series'].map(relabel)
# relabel legend
g._legend.set_title('Pretraining')
for t, l in zip(g._legend.texts, hue_order):
    t.set_text(relabel[l])

g.map_dataframe(deco)
g.fig.suptitle(f'Pretrain-Positive Regimes', y=1.05, fontsize=28)
# g.fig.suptitle(f'Convergence of Pretraining  ({"Unsorted" if UNSORT else "Sorted"})', y=1.05, fontsize=28)


#%%
from context_general_bci.analyze_utils import STYLEGUIDE
from matplotlib import ticker
# Single zoom
relabel = {
    'scale_v3/session_unsort/tune_intra': 'Session',
    'scale_v3/task_unsort/tune_intra': 'Task',
    'scale_v3/intra_unsort': 'Single-session (Scratch)',
}
palette = sns.color_palette('colorblind', n_colors=len(df['series'].unique()))
# hue_order = list(df.groupby(['series']).mean().sort_values('nll').index)
hue_order = [
    'scale_v3/intra_unsort',
    'scale_v3/session_unsort/tune_intra',
    'scale_v3/task_unsort/tune_intra',
]
dataset_order = sorted(df['dataset'].unique())
g = sns.relplot(
    x='limit',
    y='nll',
    hue='series',
    style='series',
    hue_order=hue_order,
    data=df,
    palette=palette,
    kind='scatter',
    facet_kws={'sharex': False, 'sharey': False},
    col='dataset',
    col_order=dataset_order,
    markers={
        'scale_v3/intra_unsort': STYLEGUIDE['markers']['single'],
        'scale_v3/session_unsort/tune_intra': STYLEGUIDE['markers']['session'],
        'scale_v3/task_unsort/tune_intra': STYLEGUIDE['markers']['task'],
        # 'task_unsort': STYLEGUIDE['markers']['task'],
        # STYLEGUIDE['markers']
    },
    legend=False,
)

# retitle subplots
title_remap = {
    'odoherty_rtt-Indy-20170131_02': 'Final',
    'odoherty_rtt-Indy-20160407_02': 'Initial',
    'odoherty_rtt-Indy-20160627_01': 'Middle',
}

def deco(data, **kws):
    ax = plt.gca()
    ax = prep_plt(ax)
    ax.set_xscale('log')
    # ax.set_yscale('log')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_minor_locator(ticker.MaxNLocator(5))
    ax.set_xlabel('Target context trials')
    for i, series in enumerate(hue_order):
        sub_df = data[data['series'] == series]
        plot_dataset_power_law(sub_df, ax, color=palette[i])

        # Add text annotations below the first point in each series
        x = min(sub_df['limit'])#.iloc[0]
        y = max(sub_df['nll']) #.iloc[0]
        label = relabel[series]
        color = palette[i]
        # ax.text(x, y-0.003, label, color=color, ha='left', va='top', fontsize=18)

    alias = ax.get_title().split('=')[1].strip()
    ax.set_title(title_remap.get(alias, alias), fontsize=20)
    ax.set_title("")
    ax.set_ylabel('Test NLL')

# relabel legend
# g._legend.remove()
g.map_dataframe(deco)

# Export as svg
g.fig.savefig('convergence.svg', bbox_inches='tight')