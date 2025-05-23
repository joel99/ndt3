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

from context_general_bci.utils import wandb_query_latest, get_best_ckpt_from_wandb_id
from context_general_bci.analyze_utils import (
    stack_batch,
    rolling_time_since_student,
    get_dataloader,
)
from context_general_bci.plotting import prep_plt
from context_general_bci.inference import load_wandb_run
from context_general_bci.streaming_utils import (
    precrop_batch,
    postcrop_batch,
    prepend_prompt,
)

query = 'small_40m_4k_return_only-82dlavhy'
query = 'small_40m_4k_return-djatdlf0' # 0.8 tv
query = 'base_40m-lq0adxya'
query = 'base_40m-hl9kskvf'

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# tag = 'val_loss'
tag = "val_kinematic_r2"

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
ckpt = get_best_ckpt_from_wandb_id(cfg.wandb_project, wandb_run.id, tag=tag)
# Parse epoch, format is `val-epoch=<>-other_metrics.ckpt`
ckpt_epoch = int(str(ckpt).split("-")[1].split("=")[1])
#%%

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
    Output.behavior_logits,
    Output.return_logits,
]

target = [
    'PTest_215_1$',
    'PTest_215_2$',
    # 'PTest_215_6$',

    # 'P2Lab_2067_2$',
    # 'P2Lab_2067_3$',

    # 'P2Lab_2067_15$',
    # 'P2Lab_2067_16$',

    # 'P3Home_138_10$',
    # 'P3Home_138_11$',
]

cfg.dataset.datasets = target
cfg.dataset.exclude_datasets = []
cfg.dataset.eval_datasets = []
dataset = SpikingDataset(cfg.dataset)

prompt = None
pl.seed_everything(0)
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
TAIL_S = 15
PROMPT_S = 0
WORKING_S = 15

COMPUTE_BUFFER_S = 5.1 # Timestep in seconds to begin computing loss for parity with stream buffer

STREAM_BUFFER_S = 0.
STREAM_BUFFER_S = 5.
# STREAM_BUFFER_S = 1.
# STREAM_BUFFER_S = COMPUTE_BUFFER_S

TEMPERATURE = 0.

CONSTRAINT_COUNTERFACTUAL = False
# CONSTRAINT_COUNTERFACTUAL = True
RETURN_COUNTERFACTUAL = False
# RETURN_COUNTERFACTUAL = True
# Change start of trial
MUTE_BURNIN = ""
# MUTE_BURNIN = "constraint_aa" # Simulate start of trial
# MUTE_BURNIN = "constraint_cl" # Simulate free control
# MUTE_BURNIN = "constraint_eco" # Mute up to the most recent switch that allowed FBC

SUBSET_LABELS = []
# SUBSET_LABELS = ['x', 'f']

def eval_model(
    model: BrainBertInterface,
    dataset,
    cue_length_s=CUE_S,
    tail_length_s=TAIL_S,
    precrop_prompt=PROMPT_S,  # For simplicity, all precrop for now. We can evaluate as we change precrop length
    postcrop_working=WORKING_S,
    stream_buffer_s=STREAM_BUFFER_S, # context length
    compute_buffer_s=COMPUTE_BUFFER_S, # start of eval time (s)
    subset_labels=SUBSET_LABELS,
):
    assert compute_buffer_s > stream_buffer_s, "Compute buffer must be larger than stream buffer"
    dataloader = get_dataloader(dataset, batch_size=1, num_workers=0)
    model.cfg.eval.teacher_timesteps = int(
        cue_length_s * 1000 / cfg.dataset.bin_size_ms
    )
    eval_bins = round(tail_length_s * 1000 // cfg.dataset.bin_size_ms)
    prompt_bins = int(precrop_prompt * 1000 // cfg.dataset.bin_size_ms)
    working_bins = int(postcrop_working * 1000 // cfg.dataset.bin_size_ms)
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
            active_assist = rearrange(torch.tensor([1, 1, 0]).to(assist_constraint.device), 'c -> 1 1 c')
            assist_constraint[(assist_constraint == active_assist).all(-1)] = 0
            batch[DataKey.constraint.name] = assist_constraint
        if RETURN_COUNTERFACTUAL:
            assist_return = batch[DataKey.task_return.name]
            assist_return = torch.ones_like(assist_return)
            batch[DataKey.task_return.name] = assist_return

            assist_reward = batch[DataKey.task_reward.name]
            batch[DataKey.task_reward.name] = torch.ones_like(assist_reward)

            # print(batch[DataKey.task_return.name].sum())
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

        labels = batch[DataKey.covariate_labels.name][0]

        if stream_buffer_s:
            timesteps = batch[DataKey.time.name].max() + 1 # number of distinct timesteps
            buffer_steps = int(stream_buffer_s * 1000 // cfg.dataset.bin_size_ms)
            stream_output = []
            for end_time_exclusive in range(buffer_steps, timesteps + 1): # +1 because range is exlusive
                stream_batch = deepcopy(batch)
                stream_batch = precrop_batch(stream_batch, end_time_exclusive) # Keep to end_time
                # So this is 0 - 49 one first step (end_time=50), we want 700-749 on last step (end_time=750)
                crop_suffix = end_time_exclusive - buffer_steps
                stream_batch = postcrop_batch(stream_batch, crop_suffix) # Take last STREAM_BUFFER_S
                # Delete extraneous batch items to ensure parity with `batchify_inference`
                parity_batch = {k: v for k, v in stream_batch.items() if k in [
                    DataKey.spikes.name,
                    DataKey.time.name,
                    DataKey.position.name,
                    DataKey.bhvr_vel.name,
                    DataKey.covariate_time.name,
                    DataKey.covariate_space.name,
                    DataKey.task_reward.name,
                    DataKey.task_return.name,
                    DataKey.task_return_time.name,
                    DataKey.constraint.name,
                    DataKey.constraint_space.name,
                    DataKey.constraint_time.name,
                ]}
                if MUTE_BURNIN:
                    # Simulate start of trial - no FBC, then some control or AA.
                    if MUTE_BURNIN in ["constraint_aa", "constraint_cl"]:
                        from einops import repeat
                        n_batch = parity_batch[DataKey.constraint.name].shape[0]
                        parity_batch[DataKey.constraint_space.name] = repeat(parity_batch[DataKey.constraint_space.name].unique(), 'n -> b (2 n)', b=n_batch)
                        num_kin = parity_batch[DataKey.constraint_space.name].shape[-1] // 2
                        parity_batch[DataKey.constraint_time.name] = torch.zeros_like(parity_batch[DataKey.constraint_space.name])
                        parity_batch[DataKey.constraint_time.name][..., -num_kin:] = parity_batch[DataKey.covariate_time.name].max() // 2 # i.e. we just switched
                        mock_constraint = repeat(torch.zeros_like(parity_batch[DataKey.constraint_space.name]), 'b n -> b n c', c=3).clone().to(dtype=torch.bfloat16)
                        mock_constraint[..., 0] = 1 # No FBC
                        if MUTE_BURNIN == "constraint_aa":
                            mock_constraint[..., -num_kin:, 1] = 1
                        elif MUTE_BURNIN == "constraint_cl":
                            mock_constraint[..., -num_kin:, 0] = 0 # FBC
                        parity_batch[DataKey.constraint.name] = mock_constraint
                    if MUTE_BURNIN in ["constraint_eco"]:
                        num_kin = len(parity_batch[DataKey.constraint_space.name].unique())
                        no_control_switch = (parity_batch[DataKey.constraint.name] == torch.tensor([1, 0, 0], device=parity_batch[DataKey.constraint.name].device)).all(-1).nonzero(as_tuple=True)
                        if len(no_control_switch) == 0:
                            continue # Don't change the constraint... we always had control in this block
                        last_switch = no_control_switch[1][-1] - (num_kin - 1)
                        # Check if the first constraint was no control as well - in which case we wipe just past this point

                        if 0 in no_control_switch[1]: # If the first switch was no control, we use it and record only the most recent switch to SOME kind of control
                            parity_batch[DataKey.constraint.name] = torch.cat([parity_batch[DataKey.constraint.name][:, :3], parity_batch[DataKey.constraint.name][:, last_switch+3:]], 1)
                            parity_batch[DataKey.constraint_space.name] = torch.cat([parity_batch[DataKey.constraint_space.name][:, :3], parity_batch[DataKey.constraint_space.name][:, last_switch+3:]], 1)
                            parity_batch[DataKey.constraint_time.name] = torch.cat([parity_batch[DataKey.constraint_time.name][:, :3], parity_batch[DataKey.constraint_time.name][:, last_switch+3:]], 1)
                        else: # If the first switch was not no control, include it, but wipe everything until the most recent switch
                            parity_batch[DataKey.constraint.name] = torch.cat([parity_batch[DataKey.constraint.name][:, :3], parity_batch[DataKey.constraint.name][:, last_switch:]], 1)
                            parity_batch[DataKey.constraint_space.name] = torch.cat([parity_batch[DataKey.constraint_space.name][:, :3], parity_batch[DataKey.constraint_space.name][:, last_switch:]], 1)
                            parity_batch[DataKey.constraint_time.name] = torch.cat([parity_batch[DataKey.constraint_time.name][:, :3], parity_batch[DataKey.constraint_time.name][:, last_switch:]], 1)
                    # parity_batch[DataKey.task_reward.name] = torch.zeros_like(parity_batch[DataKey.task_reward.name])
                    # parity_batch[DataKey.task_return.name] = torch.zeros_like(parity_batch[DataKey.task_return.name])
                output = model.predict_simple_batch( # Match streaming API _exactly_, see `rtndt.accelerators` call in CLIMBER
                    parity_batch,
                    kin_mask_timesteps=kin_mask_timesteps,
                    last_step_only=True,
                    temperature=TEMPERATURE
                )
                del output[Output.return_logits]
                stream_output.append(output)
            stream_total = stack_batch(stream_output) # concat behavior preds
            if compute_buffer_s:
                compute_steps = int(compute_buffer_s * 1000 // cfg.dataset.bin_size_ms)
                stream_total[Output.behavior] = batch[DataKey.bhvr_vel.name][0,(compute_steps-1) * len(labels):,0]
                stream_total[Output.behavior_logits] = stream_total[Output.behavior_logits][(compute_steps-buffer_steps) * len(labels):]
                stream_total[Output.behavior_pred] = stream_total[Output.behavior_pred][(compute_steps-buffer_steps) * len(labels):]
            else:
                stream_total[Output.behavior] = batch[DataKey.bhvr_vel.name][0,(buffer_steps-1) * len(labels):,0]
            outputs.append(stream_total)
        else:
            output = model.predict_simple_batch(
                batch,
                kin_mask_timesteps=kin_mask_timesteps,
                last_step_only=False,
            )
            if compute_buffer_s:
                compute_steps = int(compute_buffer_s * 1000 // cfg.dataset.bin_size_ms)
                for k in [Output.behavior_pred, Output.behavior_logits, Output.behavior_query_mask, Output.behavior]:
                    output[k] = output[k][(compute_steps - 1) * len(labels):]
            outputs.append(output)

    outputs = stack_batch(outputs)

    print(f"Checkpoint: {ckpt_epoch} (tag: {tag})")
    prediction = outputs[Output.behavior_pred].cpu()
    target = outputs[Output.behavior].cpu()
    if stream_buffer_s:
        valid = torch.ones(prediction.shape[0], dtype=torch.bool)
        is_student = valid
        loss = 0.
    else:
        is_student = outputs[Output.behavior_query_mask].cpu().bool()
        print(target.shape, outputs[Output.behavior_query_mask].shape)
        is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
        valid = is_student_rolling > (
            model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name])
        )
        loss = outputs[Output.behavior_loss].mean()

    print(f"Computing R2 on {valid.sum()} of {valid.shape} points")
    mse = torch.mean((target[valid] - prediction[valid]) ** 2, dim=0)
    r2_student = r2_score(target[valid], prediction[valid])
    print(f"MSE: {mse:.3f}")
    print(f"R2 Student: {r2_student:.3f}")
    print(f'Loss: {loss:.3f}')

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
        ax.invert_yaxis()
        # label colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.set_ylabel('Logit')

    def plot_split_logits_flat(full_logits, labels, cfg, truth=None, subset_labels=None):
        if subset_labels:
            indices = [labels.index(l) for l in subset_labels]
        else:
            indices = range(len(labels))
        f, axes = plt.subplots(len(indices), 1, figsize=(15, 10), sharex=True, sharey=True)

        # Split logits
        stride = len(labels)
        for i, label in enumerate(labels):
            if i not in indices:
                continue
            logits = full_logits[i::stride]
            if truth is not None:
                truth_i = truth[i::stride]
            else:
                truth_i = None
            # if i == 2: # Grasp dim is dead
                # truth_i = 255 + (255 - truth_i) * 10
            plot_logits(axes[indices.index(i)], logits, label, cfg.dataset.bin_size_ms, truth=truth_i)
        f.suptitle(f"{query} Logits MSE {mse:.3f} Loss {loss:.3f} {tag}")
        plt.tight_layout()

    truth = outputs[Output.behavior].float()
    truth = model.task_pipelines['kinematic_infill'].quantize(truth)
    # Quantize the truth
    breakpoint()
    # if not stream_buffer_s and not compute_buffer_s:
    plot_split_logits_flat(outputs[Output.behavior_logits].float(), labels, cfg, truth, subset_labels=subset_labels)

    # Get reported metrics
    history = wandb_run.history()
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
    return outputs, target, prediction, is_student, valid, r2_student, mse, labels


(outputs, target, prediction, is_student, valid, r2_student, mse, labels) = eval_model(
    model,
    dataset,
    stream_buffer_s=STREAM_BUFFER_S,
    compute_buffer_s=COMPUTE_BUFFER_S,
    subset_labels=SUBSET_LABELS
)

# %%
f = plt.figure(figsize=(10, 10))
ax = prep_plt(f.gca(), big=True)
palette = sns.color_palette(n_colors=2)
colors = [palette[0] if is_student[i] else palette[1] for i in range(len(is_student))]
alpha = [0.1 if is_student[i] else 0.8 for i in range(len(is_student))]
ax.scatter(target, prediction, s=3, alpha=alpha, color=colors)
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
# xlim = [0, 1500]
# xlim = [0, 750]
# xlim = [0, 3000]
# xlim = [0, 5000]
# xlim = [3000, 4000]
xlim = None
subset_cov = []
# subset_cov = ['x', 'f']
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


# labels = outputs[DataKey.covariate_labels.name][0]
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
