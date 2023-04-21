#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from einops import rearrange
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.analyze_utils import prep_plt
import pandas as pd
import pytorch_lightning as pl


#%%
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

EXPERIMENTS_KIN = [
    f'scale_decode',
]

queries = [
    'session_cross',
]

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished']},
    "config.dataset.odoherty_rtt.include_sorted": not UNSORT,
})
runs_kin = [r for r in runs_kin if r.name.split('-')[0] in queries]
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

def build_df(runs, mode='nll'):
    df = []
    seen_set = {}
    for run in runs:
        variant, _frag, *rest = run.name.split('-')
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
        dataset_name = cfg.dataset.datasets[0] # drop wandb ID
        dataset_name = "odoherty_rtt-Indy-20160627_01"
        if dataset_name not in DATASET_WHITELIST:
            continue
        experiment_set = run.config['experiment_set']
        if (variant, dataset_name, run.config['model']['lr_init'], experiment_set) in seen_set:
            continue
        dataset = SpikingDataset(cfg.dataset)
        set_limit = run.config['dataset']['scale_limit_per_eval_session']
        # if set_limit == 0:
            # train_dev_dataset = SpikingDataset(cfg.dataset)
            # train_dev_dataset.subset_split()
            # set_limit = len(train_dev_dataset)
        dataset.subset_split(splits=['eval'])
        dataset.build_context_index()
        data_attrs = dataset.get_data_attrs()
        model = transfer_model(src_model, cfg.model, data_attrs)

        dataloader = get_dataloader(dataset)
        payload = {
            'limit': set_limit,
            'variant': variant,
            'series': experiment_set,
            'dataset': dataset_name,
            'lr': run.config['model']['lr_init'], # swept
        }
        payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
        df.append(payload)
        seen_set[(variant, dataset_name, run.config['model']['lr_init']), experiment_set] = True
    return pd.DataFrame(df)

kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'dataset', 'series'])

df = kin_df


#%%

# Temperature stuff
palette = sns.color_palette('colorblind', 2)
from torch.distributions import poisson

lambdas = torch.arange(0, 30, 2)
def change_temp(probs, temperature):
    return (probs / temperature).exp()  / (probs / temperature).exp().sum()
for l in lambdas:
    dist = poisson.Poisson(l)
    plt.plot(dist.log_prob(torch.arange(0, 20)).exp(), color=palette[0])
    plt.plot(change_temp(dist.log_prob(torch.arange(0, 20)).exp(), 0.01), color=palette[1])

#%%
batch = torch.load('valid.pth')
# batch = torch.load('debug_batch.pth')

#%%
print(batch['tgt'].size())
sns.histplot(batch['tgt'].cpu().numpy().flatten())
#%%
print(batch[DataKey.spikes].size())
print(batch[DataKey.bhvr_vel].size())

trial = 0
# trial = 1
# trial = 2
# trial = 3
# trial = 4
# trial = 5

trial_vel = batch[DataKey.bhvr_vel][trial].cpu()
trial_spikes = batch[DataKey.spikes][trial].cpu()

def plot_vel(vel, ax):
    ax = prep_plt(ax=ax)
    ax.plot(vel)
def plot_raster(spikes, ax, vert_space=0.1, bin_size_ms=5):
    ax = prep_plt(ax)
    spikes = rearrange(spikes, 't a c h -> t (a c h)')
    sns.despine(ax=ax, left=True, bottom=False)
    spike_t, spike_c = np.where(spikes)
    # prep_plt(axes[_c], big=True)
    time = np.arange(spikes.shape[0])
    ax.scatter(
        time[spike_t], spike_c * vert_space,
        # c=colors,
        marker='|',
        s=10,
        alpha=0.9
        # alpha=0.3
    )
    time_lim = spikes.shape[0] * bin_size_ms
    ax.set_xticks(np.linspace(0, spikes.shape[0], 5))
    ax.set_xticklabels(np.linspace(0, time_lim, 5))
    # ax.set_title("Benchmark Maze (Sorted)")
    # ax.set_title(context.alias)
    ax.set_xlabel('Time (ms)')
    ax.set_yticks([])
    return ax


f = plt.figure(figsize=(10, 10))
plot_vel(trial_vel, f.add_subplot(2, 1, 1))
plot_raster(trial_spikes, f.add_subplot(2, 1, 2), bin_size_ms=20)



#%%
# Draw a 3d scatterplot of several random point clouds in space
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

colors = sns.color_palette('colorblind', 10)
fig = plt.figure()
ax = fig.add_subplot(111)
# ax = fig.add_subplot(111, projection='3d')

# Generate random data
n = 100
xs = np.random.rand(n)
ys = np.random.rand(n)
zs = np.random.rand(n)

def plot_3d_cluster(mu, std, color):
    xs, ys, zs = generate_3d_cluster(mu, std)
    ax.scatter(xs, ys, zs, c=color, marker='o')

def generate_2d_cluster(mu, std):
    # Generate random vector for array inputs mu, std
    n = 100
    xs = np.random.normal(mu[0], std[0], n)
    ys = np.random.normal(mu[1], std[1], n)
    return xs, ys

def plot_2d_cluster(mu, std, color):
    xs, ys = generate_2d_cluster(mu, std)
    ax.scatter(xs, ys, c=color, marker='o')
# Plot the points
# ax.scatter(xs, ys, zs, c='b', marker='o')
mus = np.random.rand(2, 10) * 10
stds = np.random.rand(2, 10)
for i in range(5):
    plot_2d_cluster(mus[:, i], stds[:, i], color=colors[i])
