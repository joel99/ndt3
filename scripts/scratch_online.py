# %%
# Testing online parity, using open predict
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import lightning.pytorch as pl
from einops import rearrange

from sklearn.metrics import r2_score

from context_general_bci.model import transfer_model, BrainBertInterface
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import (
    Metric,
    Output,
    DataKey,
)
from context_general_bci.contexts import context_registry

from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    stack_batch,
    load_wandb_run,
    prep_plt,
    rolling_time_since_student,
    get_dataloader,
)
from context_general_bci.streaming_utils import (
    precrop_batch,
    postcrop_batch,
    prepend_prompt,
)

query = 'small_40m_class-tpdlnrii'
query = 'small_40m_class-crzzyj1d'
query = 'small_40m_class-2wmyxnhl'

query = 'small_40m_class-fgf2xd2p' # CRSTest 206_3, 206_4
query = 'small_40m_class-98zvc4s4' # CRS02b 2065_1, 2066_1

query = 'small_40m_4k_prefix_block_loss-nefapbwj' # CRSTest 208 2, 3, 4
query = 'small_40m_4k_prefix_block_loss-zkv3uqb3' # CRSTest 208 33, 34, 35

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

tag = 'val_loss'
tag = "val_kinematic_r2"

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
#%%
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
# Parse epoch, format is `val-epoch=<>-other_metrics.ckpt`
ckpt_epoch = int(str(ckpt).split("-")[1].split("=")[1])

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_logits,
    Output.return_logits,
    Output.return_probs,
]

target = [
    # 'CRS08Lab_59_2$',
    # 'CRS08Lab_59_3$',
    # 'CRS08Lab_59_6$',

    # 'CRSTest_206_3$',
    # 'CRSTest_206_4$',
    # 'CRSTest_207_10'

    # 'CRS02bLab_2065_1$',
    # 'CRS02bLab_2066_1$',

    # 'CRSTest_208_2$',
    # 'CRSTest_208_3$',
    # 'CRSTest_208_4$',

    'CRSTest_208_33$',
    'CRSTest_208_34$',
    'CRSTest_208_35$',
]

cfg.dataset.datasets = target
cfg.dataset.exclude_datasets = []
cfg.dataset.eval_datasets = []
dataset = SpikingDataset(cfg.dataset)

reference_target = [
    # 'CRS08Lab_59_2$',
    # 'CRS08Lab_59_3$',
    # 'CRS08Lab_59_6$',

    # 'CRSTest_206_3$',
    # 'CRSTest_206_4$',

    # 'CRS02bLab_2065_1$',
    # 'CRS02bLab_2066_1$',
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
# Use val for parity with report
train, val = dataset.create_tv_datasets()
data_attrs = dataset.get_data_attrs()
dataset = val
print("Eval length: ", len(dataset))
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

# %%
CUE_S = 0
# CUE_S = 12
TAIL_S = 15
PROMPT_S = 3
PROMPT_S = 0
WORKING_S = 12
WORKING_S = 15

TEMPERATURE = 0.
# TEMPERATURE = 0.5
# TEMPERATURE = 1.0
# TEMPERATURE = 2.0

CONSTRAINT_COUNTERFACTUAL = False
CONSTRAINT_COUNTERFACTUAL = True
# Active assist counterfactual specification
CONSTRAINT_CORRECTION = 0.0
CONSTRAINT_CORRECTION = 1.0
CONSTRAINT_CORRECTION = 0.0
CONSTRAINT_CORRECTION = 0.5
CONSTRAINT_QUERY = 0.5
RETURN_COUNTERFACTUAL = False
# RETURN_COUNTERFACTUAL = True

do_plot = True
do_plot = False

tag = f'Constraint: {CONSTRAINT_CORRECTION}'

def eval_model(
    model: BrainBertInterface,
    dataset,
    cue_length_s=CUE_S,
    tail_length_s=TAIL_S,
    precrop_prompt=PROMPT_S,  # For simplicity, all precrop for now. We can evaluate as we change precrop length
    postcrop_working=WORKING_S,
    constraint_correction=CONSTRAINT_CORRECTION,
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
        if CONSTRAINT_COUNTERFACTUAL:
            assist_constraint = batch[DataKey.constraint.name]
            print(assist_constraint)
            cf_constraint = torch.tensor([
                constraint_correction, constraint_correction, 0, # How much is brain NOT participating, how much active assist is on
            ], dtype=assist_constraint.dtype, device=assist_constraint.device)
            assist_constraint[(assist_constraint != 0).sum(-1) == 2] = cf_constraint
            batch[DataKey.constraint.name] = assist_constraint
        if RETURN_COUNTERFACTUAL:
            assist_return = batch[DataKey.task_return.name]
            assist_return = torch.ones_like(assist_return)
            batch[DataKey.task_return.name] = assist_return

            assist_reward = batch[DataKey.task_reward.name]
            batch[DataKey.task_reward.name] = torch.ones_like(assist_reward)

            print(batch[DataKey.task_return.name].sum())
        if prompt is not None:
            batch = postcrop_batch(
                batch,
                int(
                    (cfg.dataset.pitt_co.chop_size_ms - postcrop_working * 1000)
                    // cfg.dataset.bin_size_ms
                ),
            )
            if len(crop_prompt[DataKey.spikes]) > 0:
                batch = prepend_prompt(batch, crop_prompt)

        output = model.predict_simple_batch(
            batch,
            kin_mask_timesteps=kin_mask_timesteps,
            last_step_only=False,
            temperature=TEMPERATURE
        )
        outputs.append(output)
    outputs = stack_batch(outputs)

    labels = outputs[DataKey.covariate_labels.name][0]
    prediction = outputs[Output.behavior_pred].cpu()
    # print(prediction.sum())
    target = outputs[Output.behavior].cpu()
    is_student = outputs[Output.behavior_query_mask].cpu().bool()
    # We unmask `cue_length` -
    # ! Don't know why behavior queyr mask is not even.
    # print(kin_mask_timesteps)
    print(target.shape, outputs[Output.behavior_query_mask].shape)

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
    # print(target.shape, prediction.shape, valid.shape)
    # print(is_student_rolling.shape)
    loss = outputs[Output.behavior_loss].mean()
    breakpoint()
    mse = torch.mean((target[valid] - prediction[valid]) ** 2, dim=0)
    r2_student = r2_score(target[valid], prediction[valid])
    print(mse)
    print(f"Checkpoint: {ckpt_epoch} (tag: {tag})")
    print(f'Loss: {loss:.3f}')
    print(f"MSE: {mse:.3f}")
    print(f"R2 Student: {r2_student:.3f}")

    def plot_logits(ax, logits, title, bin_size_ms, vmin=-20, vmax=20, truth=None):
        ax = prep_plt(ax, big=True)
        sns.heatmap(logits.cpu().T, ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        if truth is not None:
            ax.plot(truth.cpu().T, color="k", linewidth=2, linestyle="--")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Bhvr (class)")
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks(np.linspace(0, logits.shape[0], 3))
        ax.set_xticklabels(np.linspace(0, logits.shape[0] * bin_size_ms, 3).astype(int))

        # label colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Logit')

    def plot_split_logits(full_logits, labels, cfg, truth=None):
        f, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, sharey=True)

        # Split logits
        stride = len(labels)
        for i, label in enumerate(labels):
            logits = full_logits[i::stride]
            if truth is not None:
                truth_i = truth[i::stride]
            else:
                truth_i = None
            plot_logits(axes[i], logits, label, cfg.dataset.bin_size_ms, truth=truth_i)
        f.suptitle(f"{query} Logits MSE {mse:.3f} Loss {loss:.3f} {tag}")
        plt.tight_layout()

    truth = outputs[Output.behavior].float()
    truth = model.task_pipelines['kinematic_infill'].quantize(truth)
    if do_plot:
        plot_split_logits(outputs[Output.behavior_logits].float(), labels, cfg, truth)

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
    reported_kin_loss = ckpt_rows[f"val_kinematic_infill_loss"].values[-1]
    print(f"Reported R2: {reported_r2:.3f}")
    print(f"Reported Loss: {reported_loss:.3f}")
    print(f"Reported Kin Loss: {reported_kin_loss:.3f}")
    return outputs, target, prediction, is_student, valid, r2_student, mse, loss


(outputs, target, prediction, is_student, valid, r2_student, mse, loss) = eval_model(
    model, dataset,
)

scores = []
for constraint_correction in np.arange(0, 1.1, 0.1):
    (outputs, target, prediction, is_student, valid, r2_student, mse, loss) = eval_model(
        model, dataset, constraint_correction=constraint_correction
    )
    scores.append({
        'constraint_correction': constraint_correction,
        'r2': r2_student,
        'mse': mse.item(),
        'loss': loss.item(),
    })

#%%
import pandas as pd
# Plot all three metrics, side by side
f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
scores = pd.DataFrame(scores)
print(scores['mse'])
for i, metric in enumerate(['r2', 'mse', 'loss']):
    sns.lineplot(
        x='constraint_correction',
        y=metric,
        data=scores,
        ax=axes[i],
    )
    axes[i].set_title(metric)
tag = "Train: 50% | Eval: OL "
f.suptitle(f"{query} {tag}")
# f.suptitle(f"{query} Eval: {dataset.cfg.datasets[0]}")
f.tight_layout()

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


labels = outputs[DataKey.covariate_labels.name][0]
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


fig.suptitle(f'{query} R2: {r2_student:.2f} Temp: {TEMPERATURE}')
# fig.suptitle(f'{query}: {data_label_camera.get(data_label, data_label)} Velocity $R^2$ ($\\uparrow$): {r2_student:.2f}')



