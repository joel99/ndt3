#%%
# Autoregressive inference procedure, for generalist model
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
from datetime import datetime
from pytz import timezone

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from sklearn.metrics import r2_score

from context_general_bci.model import transfer_model
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import RootConfig, ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.contexts import context_registry

from context_general_bci.utils import wandb_query_latest
from context_general_bci.analyze_utils import (
    stack_batch, load_wandb_run, prep_plt, rolling_time_since_student, get_dataloader, DIMS,
    data_label_to_target
)


# query = 'data_min-jkohlswe'
# query = 'data_indy-jt456lfs'
# query = 'neural_data_monkey-pitt_800-33jazjoo'

# query = 'neural_data_monkey-pitt_100-glcgd2x0'
query = 'pitt_monkey-92bj8iw0'
# query = 'pitt_monkey_16k-4rm2fxnq'
query = 'pitt_monkey_16k-sq9jr9d0'
# query = 'pitt-rku9o9ve'


# Rouse tuned
# query = 'pitt_monkey_160-qj087lns'

# CRS08 tuned
# query = 'pitt_monkey-yv2du2y1'
query = 'pitt_monkey-hedfeq5w'


wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_kinematic_r2')

cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]


# data_label ='indy'
data_label = ''
data_label = 'crs08_grasp'
data_label = ''
if data_label:
    target = data_label_to_target(data_label)
else:
    target = [
        # 'rouse.*',
        'pitt_broad_pitt_co_CRS07Home_108_.*',
        # 'pitt_broad_pitt_co_CRS08Lab_9_.*',

        # 'miller_Jango-Jango_20150730_001',
        # 'dyer_co_chewie_2',
        # 'gallego_co_Chewie_CO_20160510',
        # 'churchland_misc_jenkins-10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI',
        # 'churchland_maze_jenkins.*'

        # 'odoherty_rtt-Indy-20160627_01', # Robust ref - goal 0.7

        # 'odoherty_rtt-Indy-20160407_02',
        # 'odoherty_rtt-Indy-20160627_01',
        # 'odoherty_rtt-Indy-20161005_06',
        # 'odoherty_rtt-Indy-20161026_03',
        # 'odoherty_rtt-Indy-20170131_02',

        # 'odoherty_rtt-Loco-20170210_03',
        # 'odoherty_rtt-Loco-20170213_02',
        # 'odoherty_rtt-Loco-20170214_02',

        # 'odoherty_rtt-Loco-20170215_02',
        # 'odoherty_rtt-Loco-20170216_02',
        # 'odoherty_rtt-Loco-20170217_02'
        # 'pitt_broad_pitt_co_CRS02bLab_1899', # Some error here. But this is 2d, so leaving for now...
        # 'pitt_broad_pitt_co_CRS02bLab_1761',
        # 'pitt_broad_pitt_co_CRS07Home_32',
        # 'pitt_broad_pitt_co_CRS07Home_88',
        # 'pitt_broad_pitt_co_CRS02bLab_1776_1.*'
    ]
    # data_label = [i for i in DIMS.keys() if dataset.cfg.datasets[0].startswith(i)][0]
    data_label = 'grasp'
    print(f'Assuming: {data_label}')

# Note: This won't preserve train val split, try to make sure eval datasets were held out
print(cfg.dataset.eval_ratio)
if cfg.dataset.eval_ratio > 0 and cfg.dataset.eval_ratio < 1: # i.e. brand new dataset, not monitored during training
    # Not super robust... we probably want to make this more like... expand datasets and compute whether overlapped
    dataset = SpikingDataset(cfg.dataset) # Make as original
    dataset.subset_split(splits=['eval'], keep_index=True)
    TARGET_DATASETS = [context_registry.query(alias=td) for td in target]
    FLAT_TARGET_DATASETS = []
    for td in TARGET_DATASETS:
        if td == None:
            continue
        if isinstance(td, list):
            FLAT_TARGET_DATASETS.extend(td)
        else:
            FLAT_TARGET_DATASETS.append(td)
    TARGET_DATASETS = [td.id for td in FLAT_TARGET_DATASETS]
    dataset.subset_by_key(TARGET_DATASETS, key=MetaKey.session)
else:
    cfg.dataset.datasets = target
    cfg.dataset.exclude_datasets = []
    cfg.dataset.eval_datasets = []
    dataset = SpikingDataset(cfg.dataset)
pl.seed_everything(0)
print("Eval length: ", len(dataset))
data_attrs = dataset.get_data_attrs()
print(data_attrs)

model = transfer_model(src_model, cfg.model, data_attrs)

# model.cfg.eval.teacher_timesteps = int(50 * 13.) # 0.5s
# model.cfg.eval.teacher_timesteps = int(50 * 9.)
# model.cfg.eval.teacher_timesteps = int(50 * 4.) # 0.5s
model.cfg.eval.teacher_timesteps = int(50 * 1.) # 0.5s
# model.cfg.eval.student_gap = int(50 * 0.)
model.cfg.eval.student_gap = int(50 * 1.)

trainer = pl.Trainer(
    accelerator='gpu', devices=1, default_root_dir='./data/tmp',
    precision='bf16-mixed',
)
dataloader = get_dataloader(dataset, batch_size=16, num_workers=16)
# dataloader = get_dataloader(dataset, batch_size=128, num_workers=16)
heldin_outputs = stack_batch(trainer.predict(model, dataloader))
#%%
print(heldin_outputs[Output.behavior_pred].shape)
print(heldin_outputs[Output.behavior].shape)

prediction = heldin_outputs[Output.behavior_pred]
target = heldin_outputs[Output.behavior]
is_student = heldin_outputs[Output.behavior_query_mask]
# Compute R2
r2 = r2_score(target, prediction)
# r2_student = r2_score(target[is_student], prediction[is_student])
is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
valid = is_student_rolling > model.cfg.eval.student_gap
mse = torch.mean((target[valid] - prediction[valid])**2, dim=0)
r2_student = r2_score(target[valid], prediction[valid])

print(f'R2: {r2:.4f}')
print(f'R2 Student: {r2_student:.4f}')
print(model.cfg.eval)
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)
palette = sns.color_palette(n_colors=2)
colors = [palette[0] if is_student[i] else palette[1] for i in range(len(is_student))]
ax.scatter(target, prediction, s=3, alpha=0.4, color=colors)
# target_student = target[is_student]
# prediction_student = prediction[is_student]
# target_student = target_student[prediction_student.abs() < 0.8]
# prediction_student = prediction_student[prediction_student.abs() < 0.8]
# robust_r2_student = r2_score(target_student, prediction_student)
ax.set_xlabel('True')
ax.set_ylabel('Pred')
ax.set_title(f'{query} {data_label} R2 Student: {r2_student:.2f}')
#%%
palette = sns.color_palette(n_colors=2)
camera_label = {
    'x': 'Vel X',
    'y': 'Vel Y',
    'z': 'Vel Z',
    'EMG_FCU': 'FCU',
    'EMG_ECRl': 'ECRl',
    'EMG_FDP': 'FDP',
    'EMG_FCR': 'FCR',
    'EMG_ECRb': 'ECRb',
    'EMG_EDCr': 'EDCr',
}
# xlim = [0, 1000]
xlim = [0, 5000]
subset_cov = []
# subset_cov = ['EMG_FCU', 'EMG_ECRl']

def plot_prediction_spans(ax, is_student, prediction, color, model_label):
    # Convert boolean tensor to numpy for easier manipulation
    is_student_np = is_student.cpu().numpy()

    # Find the changes in the boolean array
    change_points = np.where(is_student_np[:-1] != is_student_np[1:])[0] + 1

    # Include the start and end points for complete spans
    change_points = np.concatenate(([0], change_points, [len(is_student_np)]))

    # Initialize a variable to keep track of whether the first line is plotted
    first_line = True

    # Plot the lines
    for start, end in zip(change_points[:-1], change_points[1:]):
        if is_student_np[start]:  # Check if the span is True
            label = model_label if first_line else None  # Label only the first line
            ax.plot(
                np.arange(start, end),
                prediction[start:end],
                color=color,
                label=label,
                alpha=1.,
                linestyle='-',
                linewidth=2,
            )
            first_line = False  # Update the flag as the first line is plotted

def plot_target_pred_overlay(
        target,
        prediction,
        is_student,
        label,
        model_label="Pred",
        ax=None,
        palette=palette,
        plot_xlabel=False,
        xlim=None,
):
    ax = prep_plt(ax, big=True)
    palette[0] = 'k'

    if xlim:
        target = target[xlim[0]:xlim[1]]
        prediction = prediction[xlim[0]:xlim[1]]
        is_student = is_student[xlim[0]:xlim[1]]
    # Plot true and predicted values
    ax.plot(target, label=f'True', linestyle='-', alpha=0.2, color=palette[0])
    # ax.plot(prediction, label=f'pred', linestyle='--', alpha=0.75)

    # ax.scatter(
    #     is_student.nonzero(),
    #     prediction[is_student],
    #     label=f'Pred',
    #     alpha=0.5,
    #     color=palette[1],
    #     s=5,
    # )
    model_label = f'{model_label} ({r2_student:.2f})'
    plot_prediction_spans(
        ax, is_student, prediction, palette[1], model_label
    )
    if xlim is not None:
        ax.set_xlim(0, xlim[1] - xlim[0])
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks * cfg.dataset.bin_size_ms / 1000)
    if plot_xlabel:
        ax.set_xlabel('Time (s)')

    ax.set_yticks([-1, 0, 1])
    # Set minor y-ticks
    ax.set_yticks(np.arange(-1, 1.1, 0.25), minor=True)
    # Enable minor grid lines
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.3)
    ax.set_ylabel(f'{camera_label.get(label, label)} (au)')

    legend = ax.legend(
        loc='upper center',  # Positions the legend at the top center
        bbox_to_anchor=(0.8, 1.1),  # Adjusts the box anchor to the top center
        ncol=len(palette),  # Sets the number of columns equal to the length of the palette to display horizontally
        frameon=False,
        fontsize=20
    )
    # Make text in legend colored accordingly
    for color, text in zip(palette, legend.get_texts()):
        text.set_color(color)

    # ax.get_legend().remove()

labels = DIMS[data_label]
num_dims = len(labels)
if subset_cov:
    subset_dims = [i for i in range(num_dims) if labels[i] in subset_cov]
    labels = [labels[i] for i in subset_dims]
else:
    subset_dims = range(num_dims)
fig, axs = plt.subplots(
    len(subset_dims), 1, figsize=(8, 2.5 * len(subset_dims)),
    sharex=True, sharey=True
)

for i, dim in enumerate(subset_dims):
    plot_target_pred_overlay(
        target[dim::num_dims],
        prediction[dim::num_dims],
        is_student[dim::num_dims],
        label=labels[i],
        ax=axs[i],
        plot_xlabel=i == subset_dims[-1], xlim=xlim
    )

plt.tight_layout()


data_label_camera = {
    'odoherty': "O'Doherty",
    'miller': 'IsoEMG',
}
fig.suptitle(
    f'{data_label_camera.get(data_label, data_label)} 0-Shot $R^2$ ($\\uparrow$)',
    fontsize=20,
    # offset
    x=0.35,
    y=0.99,
)
# fig.suptitle(f'{query}: {data_label_camera.get(data_label, data_label)} Velocity $R^2$ ($\\uparrow$): {r2_student:.2f}')

#%%
# ICL_CROP = 2 * 50 * 2 # Quick hack to eval only a certain portion of data. 2s x 50 bins/s x 2 dims
# ICL_CROP = 3 * 50 * 2 # Quick hack to eval only a certain portion of data. 3s x 50 bins/s x 2 dims
# ICL_CROP = 0

# from context_general_bci.config import DEFAULT_KIN_LABELS
# pred = heldin_outputs[Output.behavior_pred]
# true = heldin_outputs[Output.behavior]
# positions = heldin_outputs[f'{DataKey.covariate_space}_target']
# padding = heldin_outputs[f'covariate_{DataKey.padding}_target']

# if ICL_CROP:
#     if isinstance(pred, torch.Tensor):
#         pred = pred[:, -ICL_CROP:]
#         true = true[:, -ICL_CROP:]
#         positions = positions[:,-ICL_CROP:]
#         padding = padding[:, -ICL_CROP:]
#     else:
#         print(pred[0].shape)
#         pred = [p[-ICL_CROP:] for p in pred]
#         print(pred[0].shape)
#         true = [t[-ICL_CROP:] for t in true]
#         positions = [p[-ICL_CROP:] for p in positions]
#         padding = [p[-ICL_CROP:] for p in padding]

# # print(heldin_outputs[f'{DataKey.covariate_space}_target'].unique())
# # print(heldin_outputs[DataKey.covariate_labels])

# def flatten(arr):
#     return np.concatenate(arr) if isinstance(arr, list) else arr.flatten()
# flat_padding = flatten(padding)

# if model.data_attrs.semantic_covariates:
#     flat_space = flatten(positions)
#     flat_space = flat_space[~flat_padding]
#     coords = [DEFAULT_KIN_LABELS[i] for i in flat_space]
# else:
#     # remap position to global space
#     coords = []
#     labels = heldin_outputs[DataKey.covariate_labels]
#     for i, trial_position in enumerate(positions):
#         coords.extend(np.array(labels[i])[trial_position])
#     coords = np.array(coords)
#     coords = coords[~flat_padding]

# df = pd.DataFrame({
#     'pred': flatten(pred)[~flat_padding].flatten(), # Extra flatten - in list of tensors path, there's an extra singleton dimension
#     'true': flatten(true)[~flat_padding].flatten(),
#     'coord': coords,
# })
# # plot marginals
# subdf = df
# # subdf = df[df['coord'].isin(['y'])]

# g = sns.jointplot(x='true', y='pred', hue='coord', data=subdf, s=3, alpha=0.4)
# # Recompute R2 between pred / true
# from sklearn.metrics import r2_score
# r2 = r2_score(subdf['true'], subdf['pred'])
# mse = np.mean((subdf['true'] - subdf['pred'])**2)
# # set title
# g.fig.suptitle(f'{query} {mode} {str(target)[:20]} Velocity R2: {r2:.2f}, MSE: {mse:.4f}')

#%%
# f = plt.figure(figsize=(10, 10))
# ax = prep_plt(f.gca(), big=True)
# trials = 4
# trials = 1
# trials = min(trials, len(heldin_outputs[Output.behavior_pred]))
# trials = range(trials)

# colors = sns.color_palette('colorblind', df.coord.nunique())
# label_unique = list(df.coord.unique())
# # print(label_unique)
# def plot_trial(trial, ax, color, label=False):
#     vel_true = heldin_outputs[Output.behavior][trial]
#     vel_pred = heldin_outputs[Output.behavior_pred][trial]
#     dims = heldin_outputs[f'{DataKey.covariate_space}_target'][trial]
#     pad = heldin_outputs[f'covariate_{DataKey.padding}_target'][trial]
#     vel_true = vel_true[~pad]
#     vel_pred = vel_pred[~pad]
#     dims = dims[~pad]
#     for i, dim in enumerate(dims.unique()):
#         dim_mask = dims == dim
#         true_dim = vel_true[dim_mask]
#         pred_dim = vel_pred[dim_mask]
#         dim_label = DEFAULT_KIN_LABELS[dim] if model.data_attrs.semantic_covariates else heldin_outputs[DataKey.covariate_labels][trial][dim]
#         if dim_label != 'f':
#             true_dim = true_dim.cumsum(0)
#             pred_dim = pred_dim.cumsum(0)
#         color = colors[label_unique.index(dim_label)]
#         ax.plot(true_dim, label=f'{dim_label} true' if label else None, linestyle='-', color=color)
#         ax.plot(pred_dim, label=f'{dim_label} pred' if label else None, linestyle='--', color=color)

#     # ax.plot(pos_true[:,0], pos_true[:,1], label='true' if label else '', linestyle='-', color=color)
#     # ax.plot(pos_pred[:,0], pos_pred[:,1], label='pred' if label else '', linestyle='--', color=color)
#     # ax.set_xlabel('X-pos')
#     # ax.set_ylabel('Y-pos')
#     # make limits square
#     # ax.set_aspect('equal', 'box')


# for i, trial in enumerate(trials):
#     plot_trial(trial, ax, colors[i], label=i==0)
# ax.legend()
# ax.set_title(f'{mode} {str(target)[:20]} Trajectories')
# # ax.set_ylabel(f'Force (minmax normalized)')
# # xticks - 1 bin is 20ms. Express in seconds
# ax.set_xticklabels(ax.get_xticks() * cfg.dataset.bin_size_ms / 1000)
# # express in seconds
# ax.set_xlabel('Time (s)')

# #%%
# # Look for the raw data
# from pathlib import Path
# from context_general_bci.tasks.rtt import ODohertyRTTLoader
# mins = []
# maxes = []
# raw_mins = []
# raw_maxes = []
# bhvr_vels = []
# bhvr_pos = []
# for i in dataset.meta_df[MetaKey.session].unique():
#     # sample a trial
#     trial = dataset.meta_df[dataset.meta_df[MetaKey.session] == i].iloc[0]
#     print(trial.path)
#     # Open the processed payload, print minmax
#     payload = torch.load(trial.path)
#     print(payload['cov_min'])
#     print(payload['cov_max'])
#     # append and plot
#     mins.extend(payload['cov_min'].numpy())
#     maxes.extend(payload['cov_max'].numpy())
#     # open the original payload
#     path_pieces = Path(trial.path).parts
#     og_path = Path(path_pieces[0], *path_pieces[2:-1])
#     spike_arr, bhvr_raw, _ = ODohertyRTTLoader.load_raw(og_path, cfg.dataset, ['Indy-M1', 'Loco-M1'])
#     bhvr_vel = bhvr_raw[DataKey.bhvr_vel].flatten()
#     bhvr_vels.append(bhvr_vel)
#     # bhvr_pos.append(bhvr_raw['position'])
#     raw_mins.append(bhvr_vel.min().item())
#     raw_maxes.append(bhvr_vel.max().item())
# ax = prep_plt()
# ax.set_title(f'{query} Raw MinMax bounds')
# ax.scatter(mins, maxes)
# ax.scatter(raw_mins, raw_maxes)
# ax.set_xlabel('Min')
# ax.set_ylabel('Max')
# # ax.plot(mins, label='min')
# # ax.plot(maxes, label='max')
# # ax.legend()
# #%%
# print(bhvr_pos[0][:,1:3].shape)
# # plt.plot(bhvr_pos[0][:, 1:3])
# # plt.plot(bhvr_vels[3])
# # plt.plot(bhvr_vels[2])
# # plt.plot(bhvr_vels[1])
# # plt.plot(bhvr_vels[0])
# import scipy.signal as signal
# def resample(data):
#     covariate_rate = cfg.dataset.odoherty_rtt.covariate_sampling_rate
#     base_rate = int(1000 / cfg.dataset.bin_size_ms)
#     # print(base_rate, covariate_rate, base_rate / covariate_rate)
#     return torch.tensor(
#         # signal.resample(data, int(len(data) / cfg.dataset.odoherty_rtt.covariate_sampling_rate / (cfg.dataset.bin_size_ms / 1000))) # This produces an edge artifact
#         signal.resample_poly(data, base_rate, covariate_rate, padtype='line')
#     )
# # 250Hz to 5Hz - > 2000
# # plt.plot(bhvr_pos[0][:, 1:3])
# plt.plot(resample(bhvr_pos[0][:, 1:3]))
