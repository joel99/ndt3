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

from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt, get_dataloader
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
]
EXPERIMENTS_KIN = [
    f'scale_v3/intra{"_unsort" if UNSORT else ""}/probe',
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
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed']},
    "config.dataset.odoherty_rtt.include_sorted": not UNSORT,
})
runs_kin = [r for r in runs_kin if r.config['dataset']['datasets'][0] in DATASET_WHITELIST and r.name.split('-')[0] in queries]

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
        if dataset_name not in DATASET_WHITELIST:
            continue
        if (variant, dataset_name, run.config['model']['lr_init']) in seen_set:
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
            'dataset': dataset_name,
            'lr': run.config['model']['lr_init'], # swept
        }
        payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
        df.append(payload)
        seen_set[(variant, dataset_name, run.config['model']['lr_init'])] = True
    return pd.DataFrame(df)

kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'dataset'])

nll_df = build_df(runs_nll, mode='nll')

# merge on variant and dataset, filling empty with 0s
df = pd.merge(kin_df, nll_df, on=['variant', 'dataset'], how='outer').fillna(0)

# df = build_df(runs_nll, mode='nll')
df['limit'] = df['limit_y']
#%%
# Show just NLL in logscale
palette = sns.color_palette('colorblind', n_colors=len(df['dataset'].unique()))
dataset_order = df.groupby(['dataset']).mean().sort_values('nll').index
ax = prep_plt()
ax = sns.scatterplot(
    x='limit',
    y='nll',
    hue='dataset',
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

def plot_dataset_power_law(df, dataset, ax, **kwargs):
    sub_df = df[df['dataset'] == dataset]
    popt, pcov = curve_fit(power_law, sub_df['limit'], sub_df['nll'])
    x = torch.linspace(sub_df['limit'].min(), sub_df['limit'].max(), 100)
    y = power_law(x, *popt)
    ax.plot(x, y, linestyle='--', **kwargs)
    # annotate with power law
    ax.annotate(f'{popt[1]:.4f}', xy=(x[0], y[0]), xytext=(x[0] + 10, y[0]), **kwargs)


for i, dataset in enumerate(dataset_order):
    plot_dataset_power_law(df, dataset, ax, color=palette[i])

ax.set_title(f'Intra-session scaling ({"unsorted" if UNSORT else "sorted"})')

#%%

# Supervised (kin_r2)
dataset_order = df.groupby(['dataset']).mean().sort_values('kin_r2').index
ax = prep_plt()
ax = sns.scatterplot(
    x='limit',
    y='kin_r2',
    hue='dataset',
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
