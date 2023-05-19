#%%
# Probably the main point of this minor plot is to demonstrate that there _is_ scaling; it's almost impossible to tell whether the trend is slowing at this point.
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
from matplotlib import ticker
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
    prep_plt, get_dataloader
)
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest

pl.seed_everything(0)

UNSORT = True
# UNSORT = False

DATASET_WHITELIST = [
    "odoherty_rtt-Indy-20160407_02",
    "odoherty_rtt-Indy-20170131_02",
    "odoherty_rtt-Indy-20160627_01",
]

EXPERIMENTS_NLL = [
    f'scale_v3/intra{"_unsort" if UNSORT else ""}',
    f'scale_v3/session{"_unsort" if UNSORT else ""}/tune',
    f'scale_v3/subject{"_unsort" if UNSORT else ""}/tune',
    f'scale_v3/task{"_unsort" if UNSORT else ""}/tune',
]
EXPERIMENTS_KIN = [
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/probe',
    f'scale_v3/session{"_unsort" if UNSORT else ""}/probe',
    f'scale_v3/subject{"_unsort" if UNSORT else ""}/probe',
    f'scale_v3/task{"_unsort" if UNSORT else ""}/probe',
]

queries = [
    's100',
    's200',
    's400',
    's800',
    's1600',
    's3200',
    's800_h128',
    's4k_h192',
    'f32',
    'subject_f32',
    'task_f32',
    's90k_l8',
    's130k_l12',
    's270k_l16'
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
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed']},
    "config.dataset.odoherty_rtt.include_sorted": not UNSORT,
})
runs_kin = [r for r in runs_kin if r.config['dataset']['datasets'][0] in DATASET_WHITELIST and r.name.split('-')[0] in queries]

print(f'Found {len(runs_nll)} NLL runs and {len(runs_kin)} kin runs')

#%%
def extract_exp(exp_str: str):
    # if ends with '/probe' or '/tune', remove it
    if exp_str.endswith('/probe'):
        exp_str = exp_str[:-6]
    if exp_str.endswith('/tune'):
        exp_str = exp_str[:-5]
    return exp_str.split('/')[-1]

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

kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'dataset', 'series'])

nll_df = build_df(runs_nll, mode='nll')

# merge on variant and dataset, filling empty with 0s

# df = build_df(runs_nll, mode='nll')
df = pd.merge(kin_df, nll_df, on=['variant', 'dataset', 'series'], how='outer').fillna(0)
#%%
# Recast variant - if s100, s200, s400, s800, s1600, s3200, label as "single"
df['limit'] = df['limit_y']
df['variant_remap'] = df['variant'].apply(
    lambda x: 'single' if x in [
        's100', 's200', 's400', 's800', 's1600', 's3200'
    ] else x
)
inferred_limits = {
    's4k_h192': 4000,
    's800_h128': 800,
    'f32': 23308, # pulled from slurm log https://wandb.ai/joelye9/context_general_bci/runs/zxpveqo1/overview?workspace=user-joelye9
    'subject_f32': 20265, # from https://wandb.ai/joelye9/context_general_bci/runs/ltd2ms0d/overview?workspace=user-joelye9
    'task_f32': 18304, # roughly calculated from https://wandb.ai/joelye9/context_general_bci/runs/0574z9md/overview?workspace=user-joelye9
    's90k_l8': 85218,
    's130k_l12': 126764,
    's270k_l16': 270000,
}
df['inferred_limit'] = df.apply(
    lambda x: inferred_limits.get(x['variant'], x['limit']),
    axis=1
)


#%%
# Show just NLL in logscale
palette = sns.color_palette('colorblind', n_colors=len(df['dataset'].unique()))
dataset_order = df.groupby(['dataset']).mean().sort_values('nll').index
df = df[df['inferred_limit'] != 126764] # PM's specific data restriction

ax = prep_plt()
ax = sns.scatterplot(
    x='inferred_limit',
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
ax.set_yscale('log')

# Fit power law to the minimum test loss vs. training steps
from scipy.optimize import curve_fit
def power_law(x, a, b):
    return a * x**b

def plot_dataset_power_law(sub_df, ax, **kwargs):
    popt, pcov = curve_fit(power_law, sub_df['inferred_limit'], sub_df['nll'])
    x = torch.linspace(sub_df['inferred_limit'].min(), sub_df['inferred_limit'].max(), 100)
    y = power_law(x, *popt)
    ax.plot(x, y, linestyle='--', **kwargs)
    # annotate with power law
    # ax.annotate(f'{popt[1]:.4f}', xy=(x[0], y[0]), xytext=(x[0] + 10, y[0]), **kwargs)


for i, dataset in enumerate(dataset_order):
    sub_df = df[df['dataset'] == dataset]
    plot_dataset_power_law(sub_df, ax, color=palette[i])

ax.set_title(f'Intra-session scaling ({"unsorted" if UNSORT else "sorted"})')


#%%
from context_general_bci.analyze_utils import STYLEGUIDE
hue_order = [
    'intra_unsort',
    'session_unsort',
    'task_unsort', # flip for consistency with third plot
    'subject_unsort',
]

palette = sns.color_palette('colorblind', n_colors=len(df['series'].unique()))
# hue_order = list(df.groupby(['series']).mean().sort_values('nll').index)
dataset_order = sorted(df['dataset'].unique())
g = sns.relplot(
    x='inferred_limit',
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

def deco(data, use_title=True, **kws):
    ax = plt.gca()
    ax = prep_plt(ax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pretraining trials')
    ax.set_ylabel('Test NLL')
    # Only use 3 xticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_minor_locator(ticker.MaxNLocator(3))

    if use_title:
        alias = ax.get_title().split('=')[1].strip()
        ax.set_title(title_remap.get(alias, alias), fontsize=20)
    else:
        ax.set_title('')

    for i, series in enumerate(hue_order):
        sub_df = data[data['series'] == series]
        plot_dataset_power_law(sub_df, ax, color=palette[i])

# relabel legend
label_remap = {
    'intra_unsort': 'Intra',
    'session_unsort': 'Session',
    'subject_unsort': 'Subject',
    'task_unsort': 'Task',
}
g._legend.set_title('Series')
for t, l in zip(g._legend.texts, g._legend.legendHandles):
    t.set_text(label_remap.get(t.get_text(), t.get_text()))


g.map_dataframe(deco)
g.fig.suptitle(f'Unsup. Transfer Scaling (100 Trial Calibration)', y=1.05, fontsize=28)
# g.fig.suptitle(f'Unsupervised Transfer ({"Unsorted" if UNSORT else "Sorted"})', y=1.05, fontsize=28)

#%%
fig = plt.figure(figsize=(6, 6))
ax = prep_plt(fig.add_subplot(111))
# Like the above, but just the middle panel
middle_data = df[df['dataset'] == 'odoherty_rtt-Indy-20160627_01']

print(dataset_order)

ax.set_xscale('log')
ax.set_yscale('log')

# Plot the middle panel data
middle_data = middle_data[middle_data['inferred_limit'] != 126764] # PM's specific data restriction

middle_plot = sns.scatterplot(
# middle_plot = sns.relplot(
    x='inferred_limit',
    y='nll',
    hue='series',
    style='series',
    hue_order=hue_order,
    data=middle_data,
    palette=STYLEGUIDE['palette'],
    markers={
        'intra_unsort': STYLEGUIDE['markers']['single'],
        'session_unsort': STYLEGUIDE['markers']['session'],
        'subject_unsort': STYLEGUIDE['markers']['subject'],
        'task_unsort': STYLEGUIDE['markers']['task'],
        # STYLEGUIDE['markers']
    },
    # kind='scatter',
    legend=True,
    ax=ax
)

ax.set_xlabel('Pretraining trials')
ax.set_ylabel('Test NLL')
# Only use 3 xticks
ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
ax.yaxis.set_minor_locator(ticker.MaxNLocator(3))

# if use_title:
#     alias = ax.get_title().split('=')[1].strip()
#     ax.set_title(title_remap.get(alias, alias), fontsize=20)
# else:
#     ax.set_title('')

for i, series in enumerate(hue_order):
    sub_df = middle_data[middle_data['series'] == series]
    plot_dataset_power_law(sub_df, ax, color=palette[i])

# Customize the middle panel plot
# deco(middle_data, use_title=False)



# middle_plot.fig.suptitle(f'Middle Panel: Unsup. Transfer Scaling (100 Trial Calibration)', y=1.05, fontsize=20)
# ax.get_legend().set_title('Series')
# middle_plot._legend.set_title('Series')
# for t, l in zip(middle_plot._legend.texts, middle_plot._legend.legendHandles):
    # t.set_text(label_remap.get(t.get_text(), t.get_text()))

# export as svg
fig.savefig('transfer_scale_nll.svg', bbox_inches='tight')

#%%

# Supervised (kin_r2)
dataset_order = df.groupby(['dataset']).mean().sort_values('kin_r2').index
ax = prep_plt()
ax = sns.scatterplot(
    x='inferred_limit',
    y='kin_r2',
    hue='dataset',
    style='variant_remap',
    hue_order=dataset_order,
    data=df,
    palette=palette,
    ax=ax,
    legend=False
)
ax.set_xscale('log')
ax.set_xlabel('Pretraining set size')
ax.set_title('Supervised probe (100 trials)')
ax.set_ylabel('Vel R2')
# ax.set_yscale('log')

#%%
# convert to relplot
palette = sns.color_palette('colorblind', n_colors=len(df['series'].unique()))
hue_order = df.groupby(['series']).mean().sort_values('nll').index
hue_order = [
    'intra_unsort',
    'session_unsort',
    'task_unsort', # flip for consistency with third plot
    'subject_unsort',
]
g = sns.relplot(
    x='inferred_limit',
    y='kin_r2',
    style='series',
    hue='series',
    hue_order=hue_order,
    data=df,
    palette=palette,
    facet_kws={'sharex': False, 'sharey': False},
    col='dataset',
    col_order=dataset_order,
    kind='line',
    markers=True,
)

def deco(data, use_title=True, **kws):
    ax = plt.gca()
    ax = prep_plt(ax)
    ax.set_xscale('log')
    ax.set_xlabel('Pretraining trials')
    ax.set_ylabel(r'Vel $R^2$')
    # Only use 3 xticks
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.yaxis.set_minor_locator(ticker.MaxNLocator(5))
    if use_title and '=' in ax.get_title():
        alias = ax.get_title().split('=')[1].strip()
        ax.set_title(title_remap.get(alias, alias), fontsize=20)
    else:
        ax.set_title('')


g.map_dataframe(deco)

label_remap = {
    'intra_unsort': 'Intra',
    'session_unsort': 'Session',
    'subject_unsort': 'Subject',
    'task_unsort': 'Task',
}
g._legend.set_title('Series')
for t, l in zip(g._legend.texts, g._legend.legendHandles):
    t.set_text(label_remap.get(t.get_text(), t.get_text()))


g.map_dataframe(deco)
g.fig.suptitle(f'Sup. Transfer Scaling (100 Trial Calibration)', y=1.05, fontsize=28)
# g.fig.suptitle(f'100 Trial Transfer Vel R2 ({"Unsorted" if UNSORT else "Sorted"})', y=1.05, fontsize=28)

#%%
fig = plt.figure(figsize=(6, 6))
ax = prep_plt(fig.add_subplot(111))
# Like the above, but just the middle panel
middle_data = df[df['dataset'] == 'odoherty_rtt-Indy-20160627_01']

# Plot the middle panel data
middle_plot = sns.lineplot(
# middle_plot = sns.relplot(
    x='inferred_limit',
    y='kin_r2',
    hue='series',
    style='series',
    hue_order=hue_order,
    data=middle_data,
    palette=palette,
    # kind='line',
    markers={
        'intra_unsort': STYLEGUIDE['markers']['single'],
        'session_unsort': STYLEGUIDE['markers']['session'],
        'subject_unsort': STYLEGUIDE['markers']['subject'],
        'task_unsort': STYLEGUIDE['markers']['task'],
        # STYLEGUIDE['markers']
    },
    ax=ax,
    # legend=True,
    legend=False,
)

# Customize the middle panel plot
deco(middle_data, use_title=False)
ax.set_ylabel('Velocity $R^2$')
# middle_plot.fig.suptitle(f'Middle Panel: Unsup. Transfer Scaling (100 Trial Calibration)', y=1.05, fontsize=20)
# middle_plot._legend.set_title('Series')
# for t, l in zip(middle_plot._legend.texts, middle_plot._legend.legendHandles):
    # t.set_text(label_remap.get(t.get_text(), t.get_text()))

# middle_plot._legend.remove()