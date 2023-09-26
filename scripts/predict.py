#%%
# Autoregressive inference procedure, for generalist model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
import seaborn as sns
import torch
torch.set_warn_always(False) # Turn off warnings, we get cast spam otherwise
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from context_general_bci.model import transfer_model
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.contexts import context_registry

from context_general_bci.analyze_utils import stack_batch, load_wandb_run, prep_plt
from context_general_bci.utils import get_wandb_run, wandb_query_latest

query = 'rtt-gvgaiv76'
# query = 'bhvr_12l_512-ijdvhprq'
# query = 'base-2dvz5mgm'
# query = 'rtt_c512-fr4nb2hw'
# query = 'rtt_c512_m5-83uzsg6w'
# query = 'base-1hw7fz9e' # Miller
query = 'rtt_c512_bsz_256-ji1wdvmg'
# query = 'rtt_c512_m8_bsz_256-c7ldp7xm'
# query = 'rtt_c512_km8_bsz_256-2vx7pj8e'
query = 'monkey_c512_km8_bsz_256-x5y1sfpa'
query = 'bhvr_12l_512_km8_c512-abij2xtx' # currently -0.36, lol.
query = 'monkey_trialized-5qp70fgs'
# query = 'bhvr_12l_1024_km8_c512-6p6h9m7l'
query = 'monkey_trialized_6l_1024-22lwlmk7'
query = 'monkey_trialized_6l_1024-zgsjsog0'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

# cfg.model.task.metrics = [Metric.kinematic_r2]
cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]

target = [
    # 'odoherty_rtt-Indy-20160407_02',
    # 'odoherty_rtt-Indy-20160627_01',
    # 'odoherty_rtt-Indy-20161005_06',
    # 'odoherty_rtt-Indy-20161026_03',
    # 'odoherty_rtt-Indy-20170131_02',
    # 'odoherty_rtt-Indy-20160627_01',
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
    # 'miller_Jango-Jango_20150730_001',

    'dyer_co_.*',
    # 'dyer_co_mihi_1',
    # 'gallego_co_Chewie_CO_20160510',
    # 'churchland_misc_jenkins-10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI'
    # 'churchland_maze_jenkins.*'
]

# Note: This won't preserve train val split, try to make sure eval datasets were held out
cfg.dataset.datasets = target
dataset = SpikingDataset(cfg.dataset)
pl.seed_everything(0)
# Quick cheese - IDR how to subset by length, so use "val" to get 20% quickly
dataset.subset_scale(limit_per_session=48)
# dataset.subset_scale(limit_per_session=4)
# train, val = dataset.create_tv_datasets()
# dataset = val
print("Eval length: ", len(dataset))
data_attrs = dataset.get_data_attrs()
print(data_attrs)

# Subset dataset to 16 trials.


model = transfer_model(src_model, cfg.model, data_attrs)

model.cfg.eval_teacher_timesteps = 25
# model.cfg.eval_teacher_timesteps = 10
# model.cfg.eval_teacher_timesteps = 50
# model.cfg.eval_teacher_timesteps = 100
# model.cfg.eval_teacher_timesteps = 9 * 50 # 9s RTT
# model.cfg.eval_teacher_timesteps = 400

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
def get_dataloader(dataset: SpikingDataset, batch_size=8, num_workers=1, **kwargs) -> DataLoader:
# def get_dataloader(dataset: SpikingDataset, batch_size=16, num_workers=1, **kwargs) -> DataLoader:
# def get_dataloader(dataset: SpikingDataset, batch_size=32, num_workers=1, **kwargs) -> DataLoader:
# def get_dataloader(dataset: SpikingDataset, batch_size=64, num_workers=1, **kwargs) -> DataLoader:
# def get_dataloader(dataset: SpikingDataset, batch_size=128, num_workers=1, **kwargs) -> DataLoader:
    return DataLoader(dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.tokenized_collater,
    )

dataloader = get_dataloader(dataset)
heldin_outputs = stack_batch(trainer.predict(model, dataloader))
#%%
print(heldin_outputs[Output.behavior_pred].shape)
print(heldin_outputs[Output.behavior].shape)

prediction = heldin_outputs[Output.behavior_pred]
target = heldin_outputs[Output.behavior]
is_student = heldin_outputs[Output.behavior_query_mask]
# Compute R2
from sklearn.metrics import r2_score
r2 = r2_score(target, prediction)
r2_student = r2_score(target[is_student], prediction[is_student])
print(f'R2: {r2:.4f}')
print(f'R2 Student: {r2_student:.4f}')

f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)
palette = sns.color_palette(n_colors=2)
colors = [palette[0] if is_student[i] else palette[1] for i in range(len(is_student))]
ax.scatter(target, prediction, s=3, alpha=0.4, color=colors)
ax.set_xlabel('True')
ax.set_ylabel('Pred')
ax.set_title(f'{query} {str(target)[:20]} R2 Student: {r2_student:.2f}')
#%%
from context_general_bci.config import REACH_DEFAULT_KIN_LABELS, EMG_CANON_LABELS
MILLER_LABELS = [*REACH_DEFAULT_KIN_LABELS, *EMG_CANON_LABELS]
def plot_target_pred_overlay(target, prediction, is_student, label='x', ax=None):
    # Prepare the plot
    ax = prep_plt(ax)

    # Plot true and predicted values
    ax.plot(target, label=f'true {label}', linestyle='-', alpha=0.75)
    ax.plot(prediction, label=f'pred {label}', linestyle='--', alpha=0.75)

    # Overlay student-guided traces
    ax.scatter(
        is_student.nonzero(),
        target[is_student],
        label=f'student true {label}',
        color='k',
        alpha=0.5,
    )
    ax.scatter(
        is_student.nonzero(),
        prediction[is_student],
        label=f'student pred {label}',
        alpha=0.5,
        color='red'
    )

    ax.set_xlim(0, 1000)
    # ax.set_xlim(0, 5000)
    ax.legend()
    ax.set_title(f'Dim {MILLER_LABELS[label]}')

NUM_DIMS = 2
NUM_DIMS = 4
# NUM_DIMS = 9
fig, axs = plt.subplots(NUM_DIMS, 1, figsize=(20, 5 * NUM_DIMS), sharex=True)
print(target.shape)

for i in range(NUM_DIMS):
    plot_target_pred_overlay(target[i::NUM_DIMS], prediction[i::NUM_DIMS], is_student[i::NUM_DIMS], label=i, ax=axs[i])

plt.tight_layout()
fig.suptitle(f'{query}: {str(target)[:20]} Velocity R2 Stud: {r2_student:.2f}')

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
