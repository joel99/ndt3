# %%
# Testing online parity, using open predict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import lightning.pytorch as pl

from sklearn.metrics import r2_score

from context_general_bci.model import transfer_model, BrainBertInterface
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import (
    RootConfig,
    ModelTask,
    Metric,
    Output,
    DataKey,
    MetaKey,
)
from context_general_bci.contexts import context_registry

from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    stack_batch,
    load_wandb_run,
    prep_plt,
    rolling_time_since_student,
    get_dataloader,
    data_label_to_target,
)
from context_general_bci.streaming_utils import (
    precrop_batch,
    postcrop_batch,
    prepend_prompt,
)

query = "small_40m-0q2by8md"
query = "small_40m_dense-ggg6z4ii"

query = "small_40m_dense_q256_ablate-0grt5zqd"
query = "small_40m_dense_q256_return-1pj8hmj4"
query = "small_40m_dense_q256_ablate_cond_rew-vh12zgxm"
query = "small_40m_dense_q256_return-1ag9txp7"
query = "small_40m_dense_q256_return-sy19ja5h"
query = "small_40m_dense_q256_return-m972f0fr"
query = "small_40m_dense_q256_return-uk9j6tos"

# Replication, 1,2,4,7 -> 8
query = 'small_40m_dense_q256_return-rya8mped'
query = 'small_40m_dense_q256_return-0dxr92bj'

# 2048 1 2 5 13 -> 17 18
query = 'small_40m_dense_q256_return-sx3b7msv'
query = 'small_40m_dense_q256_return-vcbj43ct' # bsz 8

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

tag = 'val_loss'
tag = "val_kinematic_r2"
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
# Parse epoch, format is `val-epoch=<>-other_metrics.ckpt`
ckpt_epoch = int(str(ckpt).split("-")[1].split("=")[1])

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.return_logits,
    Output.return_probs,
]


target = [
    # 'closed_loop_pitt_co_CRS02bLab_2049_1',
    # 'closed_loop_pitt_co_CRS02bLab_2049_2',
    # 'closed_loop_pitt_co_CRS02bLab_2049_4',
    # "closed_loop_pitt_co_CRS02bLab_2049_7",
    # "closed_loop_pitt_co_CRS02bLab_2049_8",
    "closed_loop_pitt_co_CRS02bLab_2045_17",
    "closed_loop_pitt_co_CRS02bLab_2045_18",
]

cfg.dataset.datasets = target
cfg.dataset.exclude_datasets = []
cfg.dataset.eval_datasets = []
dataset = SpikingDataset(cfg.dataset)

reference_target = [
    # 'closed_loop_pitt_co_CRSTest_190_1',
    # 'closed_loop_pitt_co_CRSTest_190_3',
    # 'closed_loop_pitt_co_CRSTest_197_1',
    # 'closed_loop_pitt_co_CRSTest_190_4',
    # 'closed_loop_pitt_co_CRSTest_190_5',
    # 'closed_loop_pitt_co_CRSTest_198_1',
    # 'closed_loop_pitt_co_CRSTest_198_2',
    "closed_loop_pitt_co_CRS02bLab_2045_13",
    # "closed_loop_pitt_co_CRS02bLab_2049_1",
    # "closed_loop_pitt_co_CRS02bLab_2049_2",
    # "closed_loop_pitt_co_CRS02bLab_2049_7",
    # "closed_loop_pitt_co_CRS02bLab_2049_4",
    # 'closed_loop_pitt_co_CRS02bLab_2049_8',
]
if reference_target:
    reference_cfg = deepcopy(cfg)
    reference_cfg.dataset.datasets = reference_target
    reference_dataset = SpikingDataset(reference_cfg.dataset)
    reference_dataset.build_context_index()
    print(len(reference_dataset))
    prompt = reference_dataset[-1]
else:
    prompt = None

pl.seed_everything(0)
print("Eval length: ", len(dataset))
data_attrs = dataset.get_data_attrs()
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")


# %%
def eval_model(
    model: BrainBertInterface,
    dataset,
    cue_length_s=3,
    tail_length_s=3,
    precrop_prompt=3,  # For simplicity, all precrop for now. We can evaluate as we change precrop length
    postcrop_working=12,
):
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0)
    model.cfg.eval.teacher_timesteps = int(
        cue_length_s * 1000 / cfg.dataset.bin_size_ms
    )
    eval_bins = round(tail_length_s * 1000 // cfg.dataset.bin_size_ms)
    prompt_bins = int(precrop_prompt * 1000 // cfg.dataset.bin_size_ms)
    working_bins = int(postcrop_working * 1000 // cfg.dataset.bin_size_ms)
    # total_bins = round(cfg.dataset.pitt_co.chop_size_ms // cfg.dataset.bin_size_ms)
    total_bins = prompt_bins + working_bins

    model.cfg.eval.student_gap = (
        total_bins - eval_bins - model.cfg.eval.teacher_timesteps
    )
    kin_mask_timesteps = torch.ones(total_bins, device="cuda", dtype=torch.bool)
    kin_mask_timesteps[: model.cfg.eval.teacher_timesteps] = 0
    print(model.cfg.eval)
    if prompt is not None:
        crop_prompt = precrop_batch(prompt, prompt_bins)

    outputs = []
    for batch in dataloader:
        batch = {
            k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }
        if prompt is not None:
            # breakpoint()
            # print(prompt.keys())
            # Pseudo model
            # print(f'Before: {batch[DataKey.constraint.name].shape}') # Confirm we actually have new constraint annotations
            # pseudo_prompt = deepcopy(batch)

            batch = postcrop_batch(
                batch,
                int(
                    (cfg.dataset.pitt_co.chop_size_ms - postcrop_working * 1000)
                    // cfg.dataset.bin_size_ms
                ),
            )

            # print(f'After: {batch[DataKey.constraint.name].shape}')
            # crop_prompt = precrop_batch(pseudo_prompt, prompt_bins) # Debug
            # crop_prompt = {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in crop_prompt.items()}
            batch = prepend_prompt(batch, crop_prompt)

        output = model.predict_simple_batch(
            batch,
            kin_mask_timesteps=kin_mask_timesteps,
            last_step_only=False,
        )
        outputs.append(output)
    outputs = stack_batch(outputs)
    print(outputs[DataKey.covariate_labels.name])
    prediction = outputs[Output.behavior_pred].cpu()
    # print(prediction.sum())
    target = outputs[Output.behavior].cpu()
    is_student = outputs[Output.behavior_query_mask].cpu().bool()

    # Compute R2
    # r2 = r2_score(target, prediction)
    is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
    valid = is_student_rolling > (
        model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name])
    )
    # print(gap * len(outputs[DataKey.covariate_labels.name]))
    # plt.plot(is_student_rolling)
    # plt.hlines(gap * len(outputs[DataKey.covariate_labels.name]), 0, 1000, )
    # plt.plot(valid * 1000)

    print(f"Computing R2 on {valid.sum()} of {valid.shape} points")
    mse = torch.mean((target[valid] - prediction[valid]) ** 2, dim=0)
    r2_student = r2_score(target[valid], prediction[valid])

    print(f"Checkpoint: {ckpt_epoch} (tag: {tag})")
    print(f"MSE: {mse:.3f}")
    print(f"R2 Student: {r2_student:.3f}")

    # Get reported metrics
    history = wandb_run.history()
    # drop nan
    history = history.dropna(subset=["epoch"])
    history.loc[:, "epoch"] = history["epoch"].astype(int)
    ckpt_rows = history[history["epoch"] == ckpt_epoch]
    # Cast epoch to int or 0 if nan, use df loc to set in place
    # Get last one
    reported_r2 = ckpt_rows[f"val_{Metric.kinematic_r2.name}"].values[-1]
    reported_loss = ckpt_rows[f"val_loss"].values[-1]
    print(f"Reported R2: {reported_r2:.3f}")
    print(f"Reported Loss: {reported_loss:.3f}")
    return outputs, target, prediction, is_student, valid, r2_student


(outputs, target, prediction, is_student, valid, r2_student) = eval_model(
    # model, dataset, cue_length_s=6, tail_length_s=6
    model, dataset, cue_length_s=3, tail_length_s=9
)

# %%
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)
palette = sns.color_palette(n_colors=2)
colors = [palette[0] if is_student[i] else palette[1] for i in range(len(is_student))]
alpha = [0.1 if is_student[i] else 0.8 for i in range(len(is_student))]
ax.scatter(target, prediction, s=3, alpha=alpha, color=colors)
# target_student = target[is_student]
# prediction_student = prediction[is_student]
# target_student = target_student[prediction_student.abs() < 0.8]
# prediction_student = prediction_student[prediction_student.abs() < 0.8]
# robust_r2_student = r2_score(target_student, prediction_student)
ax.set_xlabel("True")
ax.set_ylabel("Pred")
ax.set_title(f"{query} R2: {r2_student:.2f}")

# %%

# %%
palette = sns.color_palette(n_colors=2)
camera_label = {
    "x": "Vel X",
    "y": "Vel Y",
    "z": "Vel Z",
    "EMG_FCU": "FCU",
    "EMG_ECRl": "ECRl",
    "EMG_FDP": "FDP",
    "EMG_FCR": "FCR",
    "EMG_ECRb": "ECRb",
    "EMG_EDCr": "EDCr",
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
                alpha=0.8,
                linestyle="-",
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
    palette[0] = "k"
    r2_subset = r2_score(target[valid_pred], prediction[valid_pred])
    is_student = valid_pred
    if xlim:
        target = target[xlim[0] : xlim[1]]
        prediction = prediction[xlim[0] : xlim[1]]
        is_student = is_student[xlim[0] : xlim[1]]
    # Plot true and predicted values
    ax.plot(target, label=f"True", linestyle="-", alpha=0.5, color=palette[0])
    model_label = f"{model_label} ({r2_subset:.2f})"
    plot_prediction_spans(ax, is_student, prediction, palette[1], model_label)
    if xlim is not None:
        ax.set_xlim(0, xlim[1] - xlim[0])
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks * cfg.dataset.bin_size_ms / 1000)
    if plot_xlabel:
        ax.set_xlabel("Time (s)")

    ax.set_yticks([-1, 0, 1])
    # Set minor y-ticks
    ax.set_yticks(np.arange(-1, 1.1, 0.25), minor=True)
    # Enable minor grid lines
    ax.grid(which="minor", linestyle=":", linewidth="0.5", color="gray", alpha=0.3)
    ax.set_ylabel(f"{camera_label.get(label, label)} (au)")

    legend = ax.legend(
        loc="upper center",  # Positions the legend at the top center
        bbox_to_anchor=(0.8, 1.1),  # Adjusts the box anchor to the top center
        ncol=len(
            palette
        ),  # Sets the number of columns equal to the length of the palette to display horizontally
        frameon=False,
        fontsize=20,
    )
    # Make text in legend colored accordingly
    for color, text in zip(palette, legend.get_texts()):
        text.set_color(color)


labels = outputs[DataKey.covariate_labels.name]
num_dims = len(labels)
if subset_cov:
    subset_dims = [i for i in range(num_dims) if labels[i] in subset_cov]
    labels = [labels[i] for i in subset_dims]
else:
    subset_dims = range(num_dims)
fig, axs = plt.subplots(
    len(subset_dims), 1, figsize=(16, 2.5 * len(subset_dims)), sharex=True, sharey=True
)

for i, dim in enumerate(subset_dims):
    plot_target_pred_overlay(
        target[dim::num_dims],
        prediction[dim::num_dims],
        is_student[dim::num_dims],
        valid[dim::num_dims],
        label=labels[i],
        ax=axs[i],
        plot_xlabel=i == subset_dims[-1],
        xlim=xlim,
    )

plt.tight_layout()


# fig.suptitle(f'{query}: {data_label_camera.get(data_label, data_label)} Velocity $R^2$ ($\\uparrow$): {r2_student:.2f}')

# %%
