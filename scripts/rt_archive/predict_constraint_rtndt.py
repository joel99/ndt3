#%%
# Testing online parity, using open predict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import lightning.pytorch as pl

from sklearn.metrics import r2_score
from context_general_bci.utils import suppress_default_registry
suppress_default_registry()
from context_general_bci.contexts.context_registry import CLOSED_LOOP_DIR
from context_general_bci.model import transfer_model, BrainBertInterface
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import RootConfig, ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.contexts.context_info import BCIContextInfo
from context_general_bci.contexts import context_registry


from context_general_bci.utils import wandb_query_latest
from context_general_bci.analyze_utils import (
    stack_batch, rolling_time_since_student, get_dataloader,
)
from context_general_bci.plotting import prep_plt, data_label_to_target
from context_general_bci.inference import load_wandb_run
from context_general_bci.streaming_utils import (
    precrop_batch,
    postcrop_batch,
    prepend_prompt,
)

query = 'small_40m-0q2by8md'
query = 'small_40m_dense-ggg6z4ii'
query = 'small_40m_dense_q256_nocond-0h0ms77k'
query = 'small_40m_dense_q256_nocond-pgkwe5mj'
context_registry.clear()
context_registry.register([
    *BCIContextInfo.build_from_nested_dir(f'./data/{CLOSED_LOOP_DIR}', task_map={}, alias_prefix='closed_loop_'), # each dataset deposits into its own session folder
])
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
# src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_kinematic_r2')

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
]


# data_label ='indy'
data_label = ''
# data_label = 'p4_grasp'
if data_label:
    target = data_label_to_target(data_label)
else:
    target = [
        # NDT runs
        # OL
        # 'pitt_broad_pitt_co_P4Lab_25_1$',
        # 'pitt_broad_pitt_co_P4Lab_25_2$',
        # 'pitt_broad_pitt_co_P4Lab_25_3$',

        # 'pitt_broad_pitt_co_P4Lab_29_1$',
        # 'pitt_broad_pitt_co_P4Lab_29_2$',
        # 'pitt_broad_pitt_co_P4Lab_29_3$',

        # OLE FBC
        # 'pitt_broad_pitt_co_P4Lab_25_5$',
        # 'pitt_broad_pitt_co_P4Lab_25_6$',
        # 'pitt_broad_pitt_co_P4Lab_29_5$',
        # 'pitt_broad_pitt_co_P4Lab_29_6$',

        # PTest
        # 'closed_loop_pitt_co_PTest_190_1',
        # 'closed_loop_pitt_co_PTest_190_3',
        # 'closed_loop_pitt_co_PTest_190_5',
        'closed_loop_pitt_co_PTest_190_6',
        'closed_loop_pitt_co_PTest_190_7',
    ]
    # data_label = [i for i in DIMS.keys() if dataset.cfg.datasets[0].startswith(i)][0]
    # data_label = 'grasp'
    print(f'Assuming: {data_label}')

# Note: This won't preserve train val split, try to make sure eval datasets were held out
print(cfg.dataset.eval_ratio)
cfg.dataset.datasets = target
cfg.dataset.exclude_datasets = []
cfg.dataset.eval_datasets = []
dataset = SpikingDataset(cfg.dataset)

reference_target = [
    # 'closed_loop_pitt_co_PTest_190_1',
    # 'closed_loop_pitt_co_PTest_190_3',
    # 'closed_loop_pitt_co_PTest_190_4',
    'closed_loop_pitt_co_PTest_190_5',
]
if reference_target:
    reference_cfg = deepcopy(cfg)
    reference_cfg.dataset.datasets = reference_target
    reference_dataset = SpikingDataset(reference_cfg.dataset)
    reference_dataset.build_context_index()
    print(f'Ref: {len(reference_dataset)}')
    prompt = reference_dataset[-1]
else:
    prompt = None

pl.seed_everything(0)
print("Eval length: ", len(dataset))
data_attrs = dataset.get_data_attrs()
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to('cuda')

CONSTRAINT_COND = "normal" # mute, ablate, ablate_prompt
CONSTRAINT_COND = "mute"
# CONSTRAINT_COND = "ablate"
RETURN_COND = "normal" # mute, mute_prompt
RETURN_COND = "amplify"
# RETURN_COND = "mute" # mute, mute_prompt
# RETURN_COND = "mute_prompt" # mute, mute_prompt
NUM_DIMS = len(prompt[DataKey.covariate_labels])
if CONSTRAINT_COND == "ablate_prompt":
    # Actually, at test time, we have normal constraint/reference. Don't alter
    prompt[DataKey.constraint] = prompt[DataKey.constraint][:NUM_DIMS] * 0
    prompt[DataKey.constraint_time] = prompt[DataKey.constraint_time][:NUM_DIMS]
    prompt[DataKey.constraint_space] = prompt[DataKey.constraint_space][:NUM_DIMS]
if RETURN_COND == "mute_prompt":
    prompt[DataKey.task_return] = torch.ones_like(prompt[DataKey.task_return])
    # prompt[DataKey.task_reward] = torch.ones_like(prompt[DataKey.task_reward])


#%%
def eval_model(
    model: BrainBertInterface,
    dataset,
    cue_length_s=3,
    tail_length_s=3,
    precrop_prompt=3, # For simplicity, all precrop for now. We can evaluate as we change precrop length
    postcrop_working=12,
):
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0)
    model.cfg.eval.teacher_timesteps = int(cue_length_s * 1000 / cfg.dataset.bin_size_ms)
    eval_bins = round(tail_length_s * 1000 // cfg.dataset.bin_size_ms)
    prompt_bins = int(precrop_prompt * 1000 // cfg.dataset.bin_size_ms)
    working_bins = int(postcrop_working * 1000 // cfg.dataset.bin_size_ms)
    # total_bins = round(cfg.dataset.pitt_co.chop_size_ms // cfg.dataset.bin_size_ms)
    total_bins = prompt_bins + working_bins

    model.cfg.eval.student_gap = total_bins - eval_bins - model.cfg.eval.teacher_timesteps
    kin_mask_timesteps = torch.zeros(total_bins, device='cuda', dtype=torch.bool)
    kin_mask_timesteps[:model.cfg.eval.teacher_timesteps] = 1
    print(model.cfg.eval)
    if prompt is not None:
        crop_prompt = precrop_batch(prompt, prompt_bins)

    outputs = []
    for batch in dataloader:
        batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # print(batch[DataKey.constraint.name].shape)
        if prompt is not None:
            # breakpoint()
            # print(prompt.keys())
            batch = postcrop_batch(batch, int((cfg.dataset.pitt_co.chop_size_ms - postcrop_working * 1000) // cfg.dataset.bin_size_ms))
            if CONSTRAINT_COND in ["ablate", "ablate_prompt"]:
                batch[DataKey.constraint.name] = batch[DataKey.constraint.name][:, :NUM_DIMS] * 0
                batch[DataKey.constraint_time.name] = batch[DataKey.constraint_time.name][:, :NUM_DIMS] * 0
                batch[DataKey.constraint_space.name] = batch[DataKey.constraint_space.name][:, :NUM_DIMS]
            elif CONSTRAINT_COND == "mute":
                batch[DataKey.constraint.name] = batch[DataKey.constraint.name] * 0
            if RETURN_COND in ["mute", "mute_prompt"]:
                batch[DataKey.task_return.name] = torch.ones_like(batch[DataKey.task_return.name])
                # batch[DataKey.task_reward.name] = torch.ones_like(batch[DataKey.task_reward.name])
            elif RETURN_COND == "amplify":
                # batch[DataKey.task_return.name] = torch.ones_like(batch[DataKey.task_return.name]) * 2
                batch[DataKey.task_reward.name] = torch.ones_like(batch[DataKey.task_reward.name])
                batch[DataKey.task_reward.name][:,::10] = 2
                # batch[DataKey.task_reward.name] = torch.ones_like(batch[DataKey.task_reward.name]) * 2
            # print(batch[DataKey.covariate_labels.name])
            batch = prepend_prompt(batch, crop_prompt)
        # print(batch[DataKey.covariate_labels.name])
        # TODO crop batch

        output = model.predict_simple_batch(
            batch,
            kin_mask_timesteps=kin_mask_timesteps,
            last_step_only=False,
        )
        outputs.append(output)
    outputs = stack_batch(outputs)
    print(outputs[DataKey.covariate_labels.name])
    # print(outputs[Output.behavior_pred].shape)
    # print(outputs[Output.behavior].shape)
    # print(outputs[DataKey.covariate_labels.name])
    prediction = outputs[Output.behavior_pred].cpu()
    # print(prediction.sum())
    target = outputs[Output.behavior].cpu()
    is_student = outputs[Output.behavior_query_mask].cpu().bool()

    # Compute R2
    r2 = r2_score(target, prediction)
    is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
    valid = is_student_rolling > (model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name]))
    # print(gap * len(outputs[DataKey.covariate_labels.name]))
    # plt.plot(is_student_rolling)
    # plt.hlines(gap * len(outputs[DataKey.covariate_labels.name]), 0, 1000, )
    # plt.plot(valid * 1000)

    print(f"Computing R2 on {valid.sum()} of {valid.shape} points")
    mse = torch.mean((target[valid] - prediction[valid])**2, dim=0)
    r2_student = r2_score(target[valid], prediction[valid])

    print(f'MSE: {mse:.4f}')
    # print(f'R2: {r2:.4f}')
    print(f'R2 Student: {r2_student:.4f}')

    return outputs, target, prediction, is_student, valid, r2_student


# for cue_length_s in [3, 6, 9]:
    # eval_model(model, dataset, cue_length_s=cue_length_s)
outputs, target, prediction, is_student, valid, r2_student = eval_model(model, dataset, cue_length_s=3)
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
ax.set_title(f'{query} {data_label} R2: {r2_student:.2f}')
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
# xlim = [0, 750]
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
fig.suptitle(f'{query} Constraint: {CONSTRAINT_COND} Return: {RETURN_COND}')


# %%