#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model, logger

from context_general_bci.analyze_utils import stack_batch, get_wandb_run, load_wandb_run, wandb_query_latest
from context_general_bci.analyze_utils import prep_plt, get_dataloader

pl.seed_everything(0)

# query = "f32_5ho-tsntbsci"

queries = {
    # ('mc_rtt', 'single_125'): "mc_rtt-acvg1rx4",
    # ('odoherty_rtt-Indy-20160407_02', 'single_125'): 'v20160407_02-bd2mecm7',
    # ('odoherty_rtt-Indy-20160627_01', 'single_125'): 'v20160627_01-f6uausza',
    # ('odoherty_rtt-Indy-20161005_06', 'single_125'): 'v20161005_06-y7boucqc',
    # ('odoherty_rtt-Indy-20161026_03', 'single_125'): 'v20161026_03-jhqdpzfn',
    # ('odoherty_rtt-Indy-20170131_02', 'single_125'): 'v20170131_02-mzob9ju8',

    ('mc_rtt', 'single_300'): "mc_rtt-i0n8o24x",
    # ('odoherty_rtt-Indy-20160407_02', 'single_300'): 'v20160407_02-f4amdka7',
    # ('odoherty_rtt-Indy-20160627_01', 'single_300'): 'v20160627_01-s3ayyo36',
    # ('odoherty_rtt-Indy-20161005_06', 'single_300'): 'v20161005_06-46q14zbe',
    # ('odoherty_rtt-Indy-20161026_03', 'single_300'): 'v20161026_03-yc2110p0',
    # ('odoherty_rtt-Indy-20170131_02', 'single_300'): 'v20170131_02-qfhl1ckj',

    ('mc_rtt', 'sess30_125'): "f32_5ho_125-d01ki8cu",
    # ('odoherty_rtt-Indy-20160407_02', 'sess30_125'): 'f32_5ho_125-f4psegt5',
    # ('odoherty_rtt-Indy-20160627_01', 'sess30_125'): 'f32_5ho_125-f4psegt5',
    # ('odoherty_rtt-Indy-20161005_06', 'sess30_125'): 'f32_5ho_125-f4psegt5',
    # ('odoherty_rtt-Indy-20161026_03', 'sess30_125'): 'f32_5ho_125-f4psegt5',
    # ('odoherty_rtt-Indy-20170131_02', 'sess30_125'): 'f32_5ho_125-f4psegt5',

    ('mc_rtt', 'sess30_300'): "f32_nlb-b4rz44ou",
    # ('odoherty_rtt-Indy-20160407_02', 'sess30_300'): 'f32_5ho-tsntbsci',
    # ('odoherty_rtt-Indy-20160627_01', 'sess30_300'): 'f32_5ho-tsntbsci',
    # ('odoherty_rtt-Indy-20161005_06', 'sess30_300'): 'f32_5ho-tsntbsci',
    # ('odoherty_rtt-Indy-20161026_03', 'sess30_300'): 'f32_5ho-tsntbsci',
    # ('odoherty_rtt-Indy-20170131_02', 'sess30_300'): 'f32_5ho-tsntbsci',

    ('mc_rtt', 'ks150_125'): "ks_150-y5dlcktw",
    ('mc_rtt', 'ks150_300'): "ks_150-39avas64",
    # ('odoherty_rtt-Indy-20160407_02', 'ks150_300'): 'm3_150k_5ho_cross-gu7m10s8',
    # ('odoherty_rtt-Indy-20160627_01', 'ks150_300'): 'm3_150k_5ho_cross-gu7m10s8',
    # ('odoherty_rtt-Indy-20161005_06', 'ks150_300'): 'm3_150k_5ho_cross-gu7m10s8',
    # ('odoherty_rtt-Indy-20161026_03', 'ks150_300'): 'm3_150k_5ho_cross-gu7m10s8',
    # ('odoherty_rtt-Indy-20170131_02', 'ks150_300'): 'm3_150k_5ho_cross-gu7m10s8',
}

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')

def get_model_and_dataloader(query, eval_datasets=[]):
    wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]

    src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
    cfg.model.task.outputs = [
        Output.logrates,
        Output.spikes
    ]
    print(cfg.dataset.datasets)
    cfg.dataset.eval_datasets = eval_datasets
    # cfg.dataset.eval_datasets = [
    #     'odoherty_rtt-Indy-20160407_02'
    # ]

    dataset = SpikingDataset(cfg.dataset)
    if cfg.dataset.eval_datasets:
        dataset.subset_split(splits=['eval'])
    else:
        dataset.subset_split() # remove data-native test trials etc
    dataset.build_context_index()
    if not cfg.dataset.eval_datasets:
        train, val = dataset.create_tv_datasets()
        dataset = val

    data_attrs = dataset.get_data_attrs()
    # print(data_attrs)
    model = transfer_model(src_model, cfg.model, data_attrs)
    # print(f'{len(dataset)} examples')
    dataloader = get_dataloader(dataset)
    return model, dataloader

#%%
def get_evals(model, dataloader, runs=10):
    evals = []
    for i in range(runs):
        pl.seed_everything(i)
        heldin_metrics = stack_batch(trainer.test(model, dataloader, verbose=False))
        test_nll = heldin_metrics['test_infill_loss'] if 'test_infill_loss' in heldin_metrics else heldin_metrics['test_shuffle_infill_loss']
        test_nll = test_nll.mean().item()
        evals.append({
            'seed': i,
            'test_nll': test_nll,
        })
    return evals

all_evals = []
for data_model in queries:
    model, dataloader = get_model_and_dataloader(queries[data_model], eval_datasets=[data_model[0]])
    evals = get_evals(model, dataloader)
    evals = pd.DataFrame(evals)
    evals['dataset'] = data_model[0]
    model_tag, cali_trials = data_model[1].split('_')
    evals['model_type'] = model_tag
    evals['model_cali'] = cali_trials
    all_evals.append(evals)
all_evals = pd.concat(all_evals)

#%%
# make facet grid with model cali
g = sns.FacetGrid(
    all_evals,
    col='dataset',
    col_wrap=3,
    sharey=False,
    sharex=False,
    height=3,
    aspect=1.5,
)


# # plot test nll
g.map_dataframe(
    sns.stripplot,
    x='model_cali',
    hue='model_type',
    y='test_nll',
    dodge=True,
)

# add legend
g.add_legend()
# Set xlabel to calibation trials
g.set_axis_labels('Calibration trials', 'Test NLL')

# ax = prep_plt()

# ax = sns.scatterplot(
# ax = sns.stripplot(
    # x='dataset',
    # x='model_cali',
    # hue='model_type',
    # y='test_nll',
    # dodge=True,
    # data=all_evals,
    # ax=ax
# )
# ax.set_title('Test NLL across eval seeds')

# rotate xlabels
# for item in ax.get_xticklabels():
    # item.set_rotation(45)



#%%
heldin_outputs = stack_batch(trainer.predict(model, dataloader))
# load submission.h5
# import h5py

# payload = h5py.File('submission.h5', 'r')
# test_rates = payload['mc_maze_small_20']
# heldin_rates = test_rates['eval_rates_heldin']
# heldout_rates = test_rates['eval_rates_heldout']

heldin_rates = heldin_outputs[Output.rates] # b t c
heldout_rates = heldin_outputs[Output.heldout_rates] if Output.heldout_rates in heldin_outputs else None

# print(rates.size())

if not data_attrs.serve_tokens_flat:
    spikes = [rearrange(x, 't a c -> t (a c)') for x in heldin_outputs[Output.spikes]]
# ax = prep_plt()

num = 20
num = 5
# channel = 5

colors = sns.color_palette("husl", num)

# for trial in range(num):
#     ax.plot(rates[trial][:,channel], color=colors[trial])

# y_lim = ax.get_ylim()[1]

# trial = 10
# trial = 15
# trial = 17
# trial = 18
# trial = 80
# trial = 85
# trials = [0, 1, 2]
trials = [3, 4, 5]
# trials = [5, 6, 7]

def plot_trial(rates, trial, ax):
    ax = prep_plt(ax)
    for channel in range(num):
        # ax.scatter(np.arange(test.shape[1]), test[0,:,channel], color=colors[channel], s=1)
        # ax.plot(rates[trial][:,channel * 2], color=colors[channel])
        ax.plot(rates[trial][:,channel * 3], color=colors[channel])

        # smooth the signal with a gaussian kernel

    # from scipy import signal
    # peaks, _ = signal.find_peaks(test[trial,:,2], distance=4)
    # print(peaks)
    # print(len(peaks))
    # for p in peaks:
    #     ax.axvline(p, color='k', linestyle='--')
    # print(rates[trial].size())


    ax.set_ylabel('FR (Hz)')
    ax.set_yticklabels((ax.get_yticks() * 1000 / cfg.dataset.bin_size_ms).round())
    # relabel xtick unit from 5ms to ms
    # ax.set_xlim(0, 50)
    # print(ax.get_xtick)
    # ax.set_xticklabels(ax.get_xticks() * cfg.dataset.bin_size_ms)
    # ax.set_xlabel('Time (ms)')
    # ax.set_title()
size = (len(trials), 1 if heldout_rates is None else 2)
f, axes = plt.subplots(*size, figsize=(5 if heldout_rates is None else 10, 7.5), sharex=True)
for i, trial in enumerate(trials):
    if heldout_rates is None:
        plot_trial(heldin_rates, trial, axes[i])
    else:
        plot_trial(heldin_rates, trial, axes[i, 0])
        plot_trial(heldout_rates, trial, axes[i, 1])
# axes[-1, 0].set_xticklabels(axes[-1, 0].get_xticks() * cfg.dataset.bin_size_ms)
# axes[-1, 1].set_xticklabels(axes[-1, 0].get_xticks() * cfg.dataset.bin_size_ms)

plt.suptitle(
    f'{query} ({heldin_metrics["test_loss"].item():.3f}) {"(All enc)" if data_attrs.serve_tokens_flat else ""}'
    # f'Out ({heldin_metrics["test_co-bps"].item():.4f}) : {query} {"(All enc)" if data_attrs.serve_tokens_flat else ""}'
)
plt.tight_layout()
