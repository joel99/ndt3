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
# USE_SORTED = True
USE_SORTED = False

exp_tag = f'robust{"" if USE_SORTED else "_unsort"}'
EXPERIMENTS_KIN = [
    f'arch/{exp_tag}/probe',
]
EXPERIMENTS_NLL = [
    f'arch/{exp_tag}',
    f'arch/{exp_tag}/tune',
]
# am missing for s2 tune
# 1 task_f32
# 2 stitch
# 1 f32

queries = [
    'single_f32_rEFH_parity',
    'single_f32',
]
if not USE_SORTED:
    queries.append('time')

trainer = pl.Trainer(accelerator='cpu', devices=1, default_root_dir='./data/tmp')
# trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "config.dataset.scale_limit_per_eval_session": 300,
    "config.dataset.odoherty_rtt.include_sorted": USE_SORTED,
    "state": {"$in": ['finished']},
})
runs_nll = wandb_query_experiment(EXPERIMENTS_NLL, order='created_at', **{
    "config.dataset.scale_limit_per_eval_session": 300,
    "config.dataset.odoherty_rtt.include_sorted": USE_SORTED,
    "state": {"$in": ['finished']},
})
runs_kin = [r for r in runs_kin if r.name.split('-')[0] in queries]
runs_nll = [r for r in runs_nll if r.name.split('-')[0] in queries]
from collections import defaultdict
from pprint import pprint
hash_set = defaultdict(lambda: 0)
for r in runs_kin:
    print(r.config['tag'].split('-')[0], r.config['model']['lr_init'], r.config['dataset']['datasets'][0].split('-')[-1])
    hash_set[(r.config['tag'].split('-')[0], r.config['model']['lr_init'], r.config['dataset']['datasets'][0].split('-')[-1])] += 1
pprint(hash_set)

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

def hash_cfg(cfg, dataset_name, variant):
    return (variant, dataset_name, cfg['model']['lr_init'], cfg['seed'], cfg['experiment_set'])

def build_df(runs, mode='nll'):
    df = []
    seen_set = {}
    for run in runs:
        if 'frag' not in run.name:
            continue
        variant, _frag, *rest = run.name.split('-')
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
        dataset_name = cfg.dataset.datasets[0] # drop wandb ID
        if hash_cfg(cfg, dataset_name, variant) in seen_set:
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
            'experiment_set': run.config['experiment_set'],
            'seed': run.config['seed'],
        }
        payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
        df.append(payload)
        seen_set[(variant, dataset_name, run.config['model']['lr_init'])] = True
    return pd.DataFrame(df)
kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'dataset', 'seed', 'experiment_set'])

nll_df = build_df(runs_nll, mode='nll')

# merge on variant and dataset, filling empty with 0s
df = pd.merge(kin_df, nll_df, on=['variant', 'dataset', 'seed', 'experiment_set'], how='outer')
# df = pd.merge(kin_df, nll_df, on=['variant', 'dataset', 'seed'], how='outer').fillna(0)

# cache at <name_of_this_file>.pt
import os
os.makedirs('data/tmp', exist_ok=True)
torch.save(df, f'data/tmp/{USE_SORTED}_{__file__.split("/")[-1]}.pt')


#%%
df = torch.load(f'data/tmp/{USE_SORTED}_{__file__.split("/")[-1]}.pt')
print(df)
# df = df.fillna(0)
# df = df.dropna() # If we lost a few LR sweeps... oh well
# Meta-annotation

source_map = {
    'single_f8': 'single',
    'f8': 'session',
    'subject_f8': 'subject',
    'task_f8': 'task',
    'task_f32': 'task',
    'task_stitch': 'task',
    'subject_f32': 'subject',
    'subject_stitch': 'subject',
    'f32': 'session',
    'stitch': 'session',
    'single_f32': 'single',
    'single_time': 'single',
    'time': 'session',
    'stitch_96': 'session', # TODO check which one is better and just report that
}
arch_map = {
    'single_f8': 'f8',
    'f8': 'f8',
    'subject_f8': 'f8',
    'task_f8': 'f8',
    'task_f32': 'f32',
    'task_stitch': 'stitch',
    'subject_f32': 'f32',
    'subject_stitch': 'stitch',
    'f32': 'f32',
    'stitch': 'stitch',
    'single_f32': 'f32',
    'single_time': 'time',
    'time': 'time',
    'stitch_96': 'stitch',
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

# print(kin_df.columns)
# get unique counts - are all the runs done?
# print(df)
# df.groupby(['variant']).count()
df = df.drop_duplicates(['variant', 'dataset', 'seed', 'kin_r2'])
print(len(df))
seed_counts = df.groupby(['variant', 'dataset']).count()['seed']
print(seed_counts[seed_counts < 3])
#%%
# * Barplots
PLOT = 'nll'
PLOT = 'kin_r2'
df['arch_group'] = df['arch'].apply(lambda x: 'NDT2' if x == 'f32' else 'NDT')

order = ['time', 'stitch', 'f32']
hue_order = ['single', 'session', 'subject', 'task']
source_rename = {
    'single': 'Intra',
    'session': 'Session',
    'subject': 'Subject',
    'task': 'Task',
}
order = ['NDT', 'NDT2']

palette = sns.color_palette(n_colors=len(hue_order))
mean_df = df.groupby(['arch_group', 'source']).mean().reset_index()
ax = prep_plt()
ax = sns.barplot(
    # x='dataset',
    # hue='variant',
    x=PLOT,
    y='arch_group',
    hue='source',
    order=order,
    hue_order=hue_order,
    palette=palette,
    data=mean_df,
    width=0.9,
    ax=ax,
)

# label poisson NLL with latex down arrow
if PLOT == 'nll':
    if USE_SORTED:
        ax.set_xlim(0.288, 0.298)
    ax.set_xlabel('Poisson NLL ($\downarrow$)')
else:
    ax.set_xlabel("Velocity $R^2$ ($\\uparrow$)")
ax.set_ylabel('')
ax.set_yticklabels(['NDT', 'NDT2'])
# ax.set_yticklabels(['NDT', 'NDT-Stitch', 'NDT2'])

# Remove the legend
ax.get_legend().remove()

# Get the 'source' values corresponding to the bars
sources = mean_df['source'].values

# Iterate over the bars and the sources, and add text
for container, source in zip(ax.containers, hue_order):
    for bar in container:
        # skip nans
        if np.isnan(bar.get_y()) or np.isnan(bar.get_width()):
            continue
        ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                source_rename[source],
                ha='left',
                va='center',
                color='white',
                fontsize=14,
        )

#%%
aggr_variant = df.groupby(['variant', 'source', 'arch', 'seed']).mean().reset_index()
print(aggr_variant)

# Currently we have two experiment sets, trim the probes...

from scipy.stats import sem
from context_general_bci.analyze_utils import STYLEGUIDE
fig = plt.figure(figsize=(6, 6))
ax = prep_plt(fig.gca())

# Calculate means and standard error of the mean for each group
means = aggr_variant.groupby(['variant', 'source', 'arch']).mean().reset_index()
sems = aggr_variant.groupby(['variant', 'source', 'arch']).agg(sem).reset_index()

palette = sns.color_palette('colorblind', len(aggr_variant))
hue_order = list(aggr_variant.arch.unique())
ax = sns.scatterplot(
    x='nll',
    y='kin_r2',
    hue='arch',
    hue_order=hue_order,
    # hue='variant',
    style='source',
    s=125,
    data=means,
    palette=STYLEGUIDE['palette'],
    legend=True,
    markers=STYLEGUIDE['markers'],
    facecolors=None,
    # markers=['o', 's', 'D', 'P'],
)

# Add error bars with matplotlib
for i in range(len(means)):
    plt.errorbar(
        x=means['nll'].iloc[i],
        y=means['kin_r2'].iloc[i],
        xerr=sems['nll'].iloc[i],
        yerr=sems['kin_r2'].iloc[i],
        fmt='o',
        capsize=2,
        markersize=0,
        # marker='<',
        color=palette[hue_order.index(means['arch'].iloc[i])]
    )

# Add some lineplot to connect single to each of the other sources
def annotate():
    xs = np.array([
        means[means['source'] == 'single']['nll'].iloc[0],
        means[means['source'] == 'session']['nll'].iloc[0]
    ]) + 0.0005
    ys = np.array([
        means[means['source'] == 'single']['kin_r2'].iloc[0],
        means[means['source'] == 'session']['kin_r2'].iloc[0]
    ]) + 0.005
    ax.plot(
        xs, ys,
        color='k',
        # color=palette[hue_order.index('f32')],
        linestyle='--',
        linewidth=1,
        # arrow head
        marker='>',
        markevery=[-1],  # Remove the marker at one end of the line
        alpha=0.5,
    )

    ax.text(
        xs.mean(), ys.mean() + 0.005,
        'NDT2 (Ours)', color='k',
        rotation=0,
        alpha=0.8,
        # 'NDT2', color=palette[hue_order.index('f32')],
        ha='left', va='bottom',
        fontsize=18,
    )
    mean_baseline = np.mean([eRFH_baseline_kin[k] for k in eRFH_baseline_kin if k in df.dataset.unique()])
    ax.axhline(
        mean_baseline, ls='--', color='black',
        alpha=0.5
    )# , label='mean across variants')
    # Annotate the horizontal line
    ax.text(
        ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015,
        mean_baseline + 0.003,
        'rEFH (Makin 18)', color='black', ha='left', va='bottom',
        fontsize=14,
    )

annotate()

# Annotate the individual datapoints
# for i, row in aggr_variant.iterrows():
#     ax.text(
#         row.nll, row.kin_r2 + 0.005, row.variant, color=palette[i], ha='center', va='bottom',
#         fontsize=14,
#     )

# ax.set_title(f'Vel R2 vs NLL ({"Sorted" if USE_SORTED else "Unsorted"})')
# Annotate with Sorted or Unsorted in bottom left of plot
ax.text(
    ax.get_xlim()[1] - (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.015,
    ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
    'Sorted' if USE_SORTED else 'Unsorted', color='black',
    ha='right', va='top',
    fontsize=18, fontweight='bold'
)
# Velocity decoding R2 for y, with latex
ax.set_ylabel("Velocity $R^2$ ($\\longrightarrow$)", fontsize=18)
ax.set_xlabel('Test NLL ($\\longleftarrow$)', fontsize=18)

# Reduce major xticks for clarity
ax.xaxis.set_major_locator(ticker.LinearLocator(3))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x:.3f}"))
ax.xaxis.set_minor_locator(ticker.LinearLocator(5))
ax.yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.5, 0.7, 3)))
ax.yaxis.set_minor_locator(ticker.FixedLocator(np.linspace(0.5, 0.7, 9)))
ax.grid(which='both')
ax.grid(which='minor', alpha=0.1)
ax.grid(which='major', alpha=0.4)
# increase label font size
ax.tick_params(axis='both', which='major', labelsize=16)

# Update the legend - kill arch, we can show those color coded.
# Only for this panel
camera_ready_arch_remap = {
    'f32': 'NDT2',
    'time': 'NDT1',
    'stitch': 'NDT1-Stitch'
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
for arch, y in zip(hue_order, np.array([0.44, 0.35, 0.26]) + -0.08):
    ax.text(
        1.25,
        # 0.85,
        y,
        camera_ready_arch_remap[arch],
        color=palette[hue_order.index(arch)], ha='center', va='top',
        fontsize=16,
        # in axes coords
        transform=ax.transAxes,
    )

handles, labels = ax.get_legend_handles_labels()
print(labels)
# Only keep the labels from source, onwards
source_idx = labels.index('source')
labels = labels[source_idx:]
order = ['single', 'session', 'subject', 'task']
remap = {
    'source': "$\mathbf{Data\ Source}$",
    'single': 'Single-session\n(Scratch)',
    'session': 'Multi-Session',
    'subject': 'Multi-Subject',
    'task': 'Multi-Task',
}
reorder_idx = [labels.index(o) for o in order]
labels = np.array([remap[l] for l in labels])[reorder_idx]
handles = np.array(handles[source_idx:])[reorder_idx]

lgd = ax.legend(
    handles, labels, loc='upper right', fontsize=14, ncol=1, frameon=False,
    title='Data Source', title_fontsize=16,
    # title='Data Source', title_fontsize=14,
    bbox_to_anchor=(1.7, 0.92),
    # bbox_to_anchor=(1.02, 0.92),
)
for handle in lgd.legendHandles:
    handle._sizes = [80]

# save as svg
plt.savefig(f'arch_robust_{"sorted" if USE_SORTED else "unsorted"}.svg', bbox_inches='tight')

#%%
# make facet grid with model cali
sorted_datasets = sorted(df.variant.unique())

# Hmm... essentially no aggregation should happen here, except across seeds
# Calculate means and standard error of the mean for each group
means = df.groupby(['variant', 'source', 'arch', 'dataset']).mean().reset_index()
sems = df.groupby(['variant', 'source', 'arch', 'dataset']).agg(sem).reset_index()

palette = sns.color_palette('colorblind', len(aggr_variant))
hue_order = list(aggr_variant.arch.unique())
# ax = sns.scatterplot(
#     x='nll',
#     y='kin_r2',
#     hue='arch',
#     hue_order=hue_order,
#     # hue='variant',
#     style='source',
#     s=125,
#     data=means,
#     palette=palette,
#     legend=True,
#     markers={
#         'single': 'o',
#         'session': 'D',
#         'subject': 's',
#         'task': 'X',
#     },
#     facecolors=None,
#     # markers=['o', 's', 'D', 'P'],
# )

# Add error bars with matplotlib
# for i in range(len(means)):
#     plt.errorbar(
#         x=means['nll'].iloc[i],
#         y=means['kin_r2'].iloc[i],
#         xerr=sems['nll'].iloc[i],
#         yerr=sems['kin_r2'].iloc[i],
#         fmt='o',
#         capsize=2,
#         markersize=0,
#         # marker='<',
#         color=palette[hue_order.index(means['arch'].iloc[i])]
#     )


g = sns.relplot(
    data=means,
    col='dataset',
    x='nll',
    y='kin_r2',
    # hue='variant',
    hue='arch',
    # style='variant',
    style='source',
    s=150,
    col_wrap=3,
    facet_kws={'sharey': False, 'sharex': False}
)
# increase facet legend font size
legend = g._legend

# Modify the font size of the legend
# legend.set_title('Legend Title', fontsize=12)
remap = {
    'arch': 'Architecture',
    'source': 'Transfer Source',
    'f32': 'NDT2 (Patch 32)',
    'time': 'NDT1',
    'stitch': 'NDT1-Stitch',
    'single': 'Scratch (Single context)',
    'session': 'Session',
    'subject': 'Subject',
    'task': 'Task',
}
for t in legend.texts:
    t.set_text(remap[t.get_text()])
    t.set_fontsize(12)  # Update font size
# legend.get_texts()[0].set_fontsize(10)  # Set font size for legend labels


def deco(data, **kws):
    ax = plt.gca()
    ax = prep_plt(ax)

    # Annotate the horizontal line
    mean_baseline = eRFH_baseline_kin[data.dataset.unique()[0]]
    ax.axhline(mean_baseline, ls='--', color='black', label='mean across variants')
    # annotate rEFH position
    ax.text(
        ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01, mean_baseline, 'rEFH', color='black', ha='left', va='bottom',
        fontsize=14,
    )
    ax.set_ylabel('Velocity $R^2$')
    ax.set_xlabel('Test NLL')
g.map_dataframe(deco)
g.fig.suptitle(f'Archictecture comparisons - Velocity decode vs. NLL ({"Sorted" if USE_SORTED else "Unsorted"})', y=1.05, fontsize=28)
sns.move_legend(g, "upper left", bbox_to_anchor=(.7, .5), fontsize=20)
