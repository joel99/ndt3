#%%
from typing import List
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from einops import rearrange
import lightning.pytorch as pl
from sklearn.linear_model import Ridge # note default metric is r2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.config.presets import FlatDataConfig
from context_general_bci.dataset import SpikingDataset
from context_general_bci.subjects import SubjectInfo, create_spike_payload

from context_general_bci.analyze_utils import prep_plt, DataManipulator
from context_general_bci.tasks.pitt_co import load_trial, PittCOLoader

USE_RAW = False
# USE_RAW = True
USE_CAT_SPIKES = False
USE_CAT_SPIKES = True
LAG_MS = 0

# dataset_name = 'observation_CRS02b_1953_2'
# dataset_name = 'observation_CRS02b_0_0'
dataset_name = 'observation_CRS07'

context = context_registry.query(alias=dataset_name)
if isinstance(context, list):
    context = context[0]

default_cfg: DatasetConfig = OmegaConf.create(FlatDataConfig())
default_cfg.data_keys = [DataKey.spikes, DataKey.bhvr_vel]
default_cfg.datasets = [context.alias]


if USE_RAW:
    # Skip the trial sectioning, padding
    raw_data = load_trial(context.datapath, key='thin_data')
    spikes = raw_data['spikes']
    subject = 'CRS02b' if 'CRS02b' in context.alias else 'CRS07'
    spikes = create_spike_payload(spikes, [
        f'{subject}-lateral_m1', f'{subject}-medial_m1',
    ])
    all_spikes = torch.cat([spikes[k] for k in spikes if 'm1' in k], 1)[None, ..., 0]
    all_bhvr = PittCOLoader.get_velocity(raw_data['position'])
    all_trials = raw_data['trial_num']
else:
    dataset = SpikingDataset(default_cfg)
    dataset.build_context_index()
    dataset.subset_split() # get train
    train, val = dataset.create_tv_datasets()
    all_spikes = rearrange(torch.as_tensor(np.concatenate([i[DataKey.spikes] for i in dataset], 0)), 't s c 1 -> 1 t (s c)')
    all_bhvr = torch.as_tensor(np.concatenate([i[DataKey.bhvr_vel] for i in dataset], 0))
    all_trials = torch.cat([torch.full(n[DataKey.bhvr_vel].shape[:1], i) for i, n in enumerate(dataset)])

print(
    f'RAW? {USE_RAW}, spikes: {all_spikes.shape}, bhvr: {all_bhvr.shape}',
)
all_bhvr = all_bhvr.roll(LAG_MS // 20, 0)
all_bhvr[:LAG_MS // 20] = 0
print(
    all_spikes.sum(), all_bhvr.max(), all_bhvr.min()
)

# make a really wide figure
plt.figure(figsize=(20, 10))
# plt.plot(all_spikes[0,:,0])
# plt.plot(all_spikes[0,:,1])
# plt.plot(all_spikes[0,:,3])
plt.plot(all_bhvr[:,0])
plt.plot(all_bhvr[:,1])
#%%
pl.seed_everything(0)

def smooth_spikes(
    dataset: SpikingDataset, kernel_sd=80
) -> List[torch.Tensor]:
    # Smooth along time axis
    return [DataManipulator.gauss_smooth(
        rearrange(i[DataKey.spikes].float(), 't s c 1 -> 1 t (s c)'),
        bin_size=dataset.cfg.bin_size_ms,
        kernel_sd=kernel_sd,
    ).squeeze(0) for i in dataset]

spike_smth_range = [100, 200, 400, 600]
# spike_smth_range = [20, 60, 100, 200, 400, 600]

def sweep_ridge_fit(zero_filt_train=True, zero_filt_eval=True):
    trains = []
    evals = []

    if USE_RAW or USE_CAT_SPIKES:
        train_split, test_split = train_test_split(
            all_trials.unique(),
            test_size=0.2, random_state=42
        )

        train_split = torch.isin(all_trials, train_split)
        test_split = torch.isin(all_trials, test_split)
        train_behavior = all_bhvr[train_split]
        eval_behavior = all_bhvr[test_split]
    else:
        train_behavior = np.concatenate([i[DataKey.bhvr_vel] for i in train], 0)
        eval_behavior = np.concatenate([i[DataKey.bhvr_vel] for i in val], 0)
    decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-1, 3, 50)})

    def single_score(smth, train_bhvr, val_bhvr):
        if USE_RAW or USE_CAT_SPIKES:
            # Split into train and validation sets
            breakpoint()
            smth_spikes = DataManipulator.gauss_smooth(
                all_spikes,
                bin_size=dataset.cfg.bin_size_ms,
                kernel_sd=smth,
            )[0]
            train_rates, eval_rates = smth_spikes[train_split], smth_spikes[test_split]
            # print(smth_spikes.shape, train_rates.shape, eval_rates.shape)
        else:
            smth_train = smooth_spikes(train, kernel_sd=smth)
            smth_val = smooth_spikes(val, kernel_sd=smth)
            train_rates = np.concatenate(smth_train, 0)
            eval_rates = np.concatenate(smth_val, 0)

        if zero_filt_train:
            train_rates = train_rates[~(np.abs(train_bhvr) < 1e-5).all(-1)]
            train_bhvr = train_bhvr[~(np.abs(train_bhvr) < 1e-5).all(-1)]
        if zero_filt_eval:
            eval_rates = eval_rates[~(np.abs(val_bhvr) < 1e-5).all(-1)]
            val_bhvr = val_bhvr[~(np.abs(val_bhvr) < 1e-5).all(-1)]
        # print(train_rates.shape, train_bhvr.shape, eval_rates.shape, val_bhvr.shape)
        decoder.fit(train_rates, train_bhvr)
        return decoder.score(eval_rates, val_bhvr), decoder.score(train_rates, train_bhvr)

    for i in spike_smth_range:
        val_score, train_score = single_score(i, train_behavior, eval_behavior)
        evals.append(val_score)
        trains.append(train_score)
    return evals, trains

eval_filt, train_filt = sweep_ridge_fit()
eval_no_filt, train_no_filt = sweep_ridge_fit(zero_filt_train=False, zero_filt_eval=True)


# Set color palette and line styles
palette = sns.color_palette("Set1", 2)
line_styles = ['-', '--']

ref_scores = {
    'observation_CRS02b_0_0': 0.5,
    'observation_CRS02b_1953_2': 0.67,
    'observation_CRS07_150_1': 0.8,
}
# Prepare plot
fig, ax = plt.subplots()
ax = prep_plt(ax)
ax.set_xlabel('(Acausal) Gauss smth kernel ms')
ax.set_ylabel('Decode R2')
if dataset_name in ref_scores:
    ax.axhline(
        ref_scores[dataset_name],
        color='k', linestyle='--', label='reported R2'
    )

# Plot data
ax.plot(spike_smth_range, train_filt, label='train filtered', color=palette[0], linestyle=line_styles[1])
ax.plot(spike_smth_range, eval_filt, label='eval filtered', color=palette[0], linestyle=line_styles[0])
ax.plot(spike_smth_range, train_no_filt, label='train unfiltered', color=palette[1], linestyle=line_styles[1])
ax.plot(spike_smth_range, eval_no_filt, label='eval unfiltered', color=palette[1], linestyle=line_styles[0])

ax.legend()
ax.set_title(f'{dataset_name} - Ridge regression Raw: {USE_RAW}, Cat: {USE_CAT_SPIKES}')
# Add linestyle and color info to legend
# handles, labels = ax.get_legend_handles_labels()
# labels = ['{} ({})'.format(label, ls) for label, ls in zip(labels, ['--', '-', '--', '-'])]
# legend = ax.legend(handles, labels, loc='best')

#%%
# Debug
trial = 0

ex_train = smooth_spikes(train)[trial]
ex_train_vel = train[trial][DataKey.bhvr_vel]

# Plot
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(ex_train_vel)
ax[0].set_title('Velocity')
ax[1].plot(ex_train)
ax[1].set_title('Rates')
