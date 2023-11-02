#%%
# Autoregressive inference procedure, for generalist model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import lightning.pytorch as pl

from sklearn.metrics import r2_score

from context_general_bci.model import transfer_model
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import RootConfig, ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.contexts import context_registry

from context_general_bci.utils import wandb_query_latest
from context_general_bci.analyze_utils import (
    stack_batch, load_wandb_run, prep_plt, rolling_time_since_student, get_dataloader,
    data_label_to_target
)

query = 'pitt_monkey_cond_return-b77cdokm'
query = 'pitt_monkey_return-2qihb67w'
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_kinematic_r2')

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.return_logits,
    Output.return_target
]


# data_label ='indy'
data_label = ''
data_label = 'crs08_grasp'
# data_label = 'miller'
data_label = ''
if data_label:
    target = data_label_to_target(data_label)
else:
    target = [

        # 'pitt_broad_pitt_co_CRS02bLab_1942.*',
        # 'pitt_broad_pitt_co_CRS02bLab_1942_1',
        # 'pitt_broad_pitt_co_CRS02bLab_1942_2',
        # 'pitt_broad_pitt_co_CRS02bLab_1942_3',

        # 'pitt_broad_pitt_co_CRS02bLab_1942_1', # OL
        # 'pitt_broad_pitt_co_CRS02bLab_1942_4', # OL

        # 'pitt_broad_pitt_co_CRS02bLab_1942_2', # Ortho
        # 'pitt_broad_pitt_co_CRS02bLab_1942_5', # Ortho
        'pitt_broad_pitt_co_CRS02bLab_1942_3', # FBC
        'pitt_broad_pitt_co_CRS02bLab_1942_6', # FBC
        # 'pitt_broad_pitt_co_CRS02bLab_1942_7', # Free play
        # 'pitt_broad_pitt_co_CRS02bLab_1942_8', # Free play

        # 'rouse.*',
        # 'pitt_broad_pitt_co_CRS02bLab_1942_1',
        # 'pitt_broad_pitt_co_CRS07Home_108_.*',
        # 'pitt_broad_pitt_co_CRS08Lab_36_.*',
        # 'pitt_broad_pitt_co_CRS08Lab_10_.*',

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
    # data_label = 'grasp'
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

CUE_LENGTH_S = 1
CUE_LENGTH_S = 3
# CUE_LENGTH_S = 9
# CUE_LENGTH_S = 30

EVAL_GAP_S = 45 - CUE_LENGTH_S - 6 # TAIL
EVAL_GAP_S = 45 - CUE_LENGTH_S - 40 # TAIL

model.cfg.eval.teacher_timesteps = int(CUE_LENGTH_S * 1000 / cfg.dataset.bin_size_ms)
model.cfg.eval.student_gap = int(EVAL_GAP_S * 1000 / cfg.dataset.bin_size_ms)
model.cfg.eval.use_student = True
model.cfg.eval.use_student = False

# Interventions
modifier = 'zero_reward'
modifier = 'three_return'
modifier = ''
if modifier == 'zero_reward':
    model.cfg.eval.zero_reward = True
elif modifier == 'three_return':
    model.cfg.eval.const_return = 3 # Counterfactually high - the natural range is 1-2
else:
    model.cfg.eval.zero_reward = False

trainer = pl.Trainer(
    accelerator='gpu', devices=1, default_root_dir='./data/tmp',
    precision='bf16-mixed',
)
dataloader = get_dataloader(dataset, batch_size=16, num_workers=16)
# dataloader = get_dataloader(dataset, batch_size=128, num_workers=16)
outputs = stack_batch(trainer.predict(model, dataloader))#%%
print(outputs[Output.behavior_pred].shape)
print(outputs[Output.behavior].shape)
print(outputs[DataKey.covariate_labels.name])
#%%
print(outputs.keys())

return_logits = outputs[Output.return_logits].to(dtype=float)#.numpy() # Shape N x K
return_target = outputs[Output.return_target].to(dtype=float)#.numpy() # Shape N

return_logits = return_logits[:, 1:5]
# Assuming 'return_logits' is a PyTorch tensor of shape (N, K)
THRESHOLD = 0.0  # Set your threshold probability here.
# We simply set a nucleus of 0.3
# THRESHOLD = 0.3 # enough to get rid of absurd returns on a spot check.
# OK - note that not all of these are actually _returns_ some of these are just reward tokens, no?
# Hmm...
# THRESHOLD = 0.0  # Set your threshold probability here

# Apply softmax to convert logits to probabilities
probabilities = torch.softmax(return_logits, dim=1)

# Convert to numpy for visualization
probabilities_np = probabilities.cpu().detach().numpy()

# Normalize the probabilities
probabilities_normalized = (probabilities_np - probabilities_np.min()) / (probabilities_np.max() - probabilities_np.min())

# Mask values below the threshold
probabilities_masked = np.where(probabilities_normalized >= THRESHOLD, probabilities_normalized, np.nan)

# Create the heatmap with the masked array
plt.figure(figsize=(10, 8))
ax = sns.heatmap(probabilities_masked, annot=False, cmap='viridis', mask=np.isnan(probabilities_masked))
ax.set_title(f'Heatmap of return probabilities (threshold={THRESHOLD})')
ax.set_xlabel(f'Return to Go (100 timesteps / {100 * cfg.dataset.bin_size_ms} ms)')
#%%
prediction = outputs[Output.behavior_pred]
target = outputs[Output.behavior]
is_student = outputs[Output.behavior_query_mask]
# Compute R2
r2 = r2_score(target, prediction)
# print(dataset[0][DataKey.covariate_time][-100:])
is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
plt.plot(is_student_rolling)
valid = is_student_rolling > (model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name]))
print(model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name]))
plt.hlines(model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name]), 0, 1000, )
plt.plot(valid * 1000)

print(f"Computing R2 on {valid.sum()} of {valid.shape} points")
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
ax.set_title(f'{query} {data_label} {modifier} R2: {r2_student:.2f}')
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
xlim = [0, 1500]
xlim = [0, 750]
# xlim = [0, 3000]
# xlim = [0, 5000]
# xlim = [3000, 4000]
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
                alpha=.8,
                linestyle='-',
                linewidth=2,
            )
            first_line = False  # Update the flag as the first line is plotted

def plot_target_pred_overlay(
        target,
        prediction,
        is_student,
        valid_pred,
        label,
        model_label="Pred",
        ax=None,
        palette=palette,
        plot_xlabel=False,
        xlim=None,
):
    ax = prep_plt(ax, big=True)
    palette[0] = 'k'
    r2_subset = r2_score(target[valid_pred], prediction[valid_pred])
    is_student = valid_pred
    if xlim:
        target = target[xlim[0]:xlim[1]]
        prediction = prediction[xlim[0]:xlim[1]]
        is_student = is_student[xlim[0]:xlim[1]]
    # Plot true and predicted values
    ax.plot(target, label=f'True', linestyle='-', alpha=0.5, color=palette[0])
    # ax.plot(prediction, label=f'pred', linestyle='--', alpha=0.75)

    # ax.scatter(
    #     is_student.nonzero(),
    #     prediction[is_student],
    #     label=f'Pred',
    #     alpha=0.5,
    #     color=palette[1],
    #     s=5,
    # )
    model_label = f'{model_label} ({r2_subset:.2f})'
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

labels = outputs[DataKey.covariate_labels.name]
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
        valid[dim::num_dims],
        label=labels[i],
        ax=axs[i],
        plot_xlabel=i == subset_dims[-1], xlim=xlim
    )

plt.tight_layout()


data_label_camera = {
    'odoherty': "O'Doherty",
    'miller': 'IsoEMG',
}
# fig.suptitle(
#     f'{data_label_camera.get(data_label, data_label)} 0-Shot $R^2$ ($\\uparrow$)',
#     fontsize=20,
#     # offset
#     x=0.35,
#     y=0.99,
# )
# fig.suptitle(f'{query}: {data_label_camera.get(data_label, data_label)} Velocity $R^2$ ($\\uparrow$): {r2_student:.2f}')
