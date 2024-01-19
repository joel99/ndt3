# %%
# Testing online parity, using open predict
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import lightning.pytorch as pl
import pandas as pd
from sklearn.metrics import r2_score

from context_general_bci.model import transfer_model, BrainBertInterface
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import (
    Metric,
    Output,
    DataKey,
    MetaKey
)

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
# query = 'small_40m_4k_prefix_block_loss-zkv3uqb3' # CRSTest 208 33, 34, 35 BUGGED
query = 'small_40m_4k_prefix_block_loss-pz6j1cow'
query = 'small_40m_4k_prefix_block_loss-1qla3ato' # OL 208 2, 3, 4
query = 'small_40m_4k_prefix_block_loss-w7wghmc6'

# Trying to get click online
query = 'small_40m_4k_prefix_block_loss-tpm2gllb'
query = 'small_40m_4k_prefix_block_loss-k2lpe653' # https://wandb.ai/joelye9/ndt3/runs/k2lpe653?workspace=user-joelye9
query = 'small_40m_4k_prefix_block_loss-wgvb6m90'
query = 'small_40m_4k_return_only-82dlavhy'
query = 'small_40m_4k_return-lyvk6zuu'

# query = 'small_40m_4k_return-gih4kyon' # NO FINE-TUNING! PRETAINED
query = 'small_40m_4k_return-jf2pdzsl' # ol
# query = 'small_40m_4k_return-pwecifa0' # mixed ol + 50%
query = 'small_40m_4k_return-5g11tdvw' # 50%
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

# tag = 'val_loss'
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
    Output.return_target
]

target = [
    # 'CRSTest_208_2$',
    # 'CRSTest_208_3$',
    # 'CRSTest_208_4$',

    # 'CRSTest_208_33$',
    # 'CRSTest_208_34$',
    # 'CRSTest_208_35$',

    # 'CRSTest_209_13$',
    # 'CRSTest_209_14$',
    # 'CRSTest_209_15$',
    # 'CRSTest_209_16$',
    # 'CRSTest_209_19$',

    'CRS02bLab_2067_2$',
    'CRS02bLab_2067_3$',
    'CRS02bLab_2067_8$',
    'CRS02bLab_2067_9$',

    # 'CRS02bLab_2067_8$',
    # 'CRS02bLab_2067_9$',

    # 'CRS02bLab_2067_15$',
    # 'CRS02bLab_2067_16$',
]

cfg.dataset.datasets = target
cfg.dataset.exclude_datasets = []
cfg.dataset.eval_datasets = []
cfg.dataset.max_tokens = 8192
dataset = SpikingDataset(cfg.dataset)

prompt = None

pl.seed_everything(0)
# Use val for parity with report
train, val = dataset.create_tv_datasets()
data_attrs = dataset.get_data_attrs()
dataset = val
print(dataset.meta_df[MetaKey.session].unique())
subset_datasets = [
#     'ExperimentalTask.pitt_co-CRS02b-2067-closed_loop_pitt_co_CRS02bLab_2067_2',
#     'ExperimentalTask.pitt_co-CRS02b-2067-closed_loop_pitt_co_CRS02bLab_2067_3',
    'ExperimentalTask.pitt_co-CRS02b-2067-closed_loop_pitt_co_CRS02bLab_2067_8',
    'ExperimentalTask.pitt_co-CRS02b-2067-closed_loop_pitt_co_CRS02bLab_2067_9',
]
dataset.subset_by_key(subset_datasets, key=MetaKey.session)

print("Eval length: ", len(dataset))
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to("cuda")

# print(dataset[0][DataKey.covariate_labels.name])
# print(dataset[0][DataKey.task_return])
# print(dataset[0][DataKey.task_return_time])
# plt.plot(dataset[0][DataKey.task_return_time], dataset[0][DataKey.task_return])
# plt.plot(dataset[1][DataKey.task_return_time], dataset[1][DataKey.task_return])
# plt.plot(dataset[2][DataKey.task_return_time], dataset[2][DataKey.task_return])
# plt.plot(dataset[3][DataKey.task_return_time], dataset[3][DataKey.task_return])
# plt.plot(dataset[4][DataKey.task_return_time], dataset[4][DataKey.task_return])
# plt.plot(dataset[5][DataKey.task_return_time], dataset[5][DataKey.task_return])
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
# CONSTRAINT_COUNTERFACTUAL = True
# Active assist counterfactual specification
CONSTRAINT_CORRECTION = 0.0
CONSTRAINT_CORRECTION = 1.0
RETURN_COUNTERFACTUAL = False
# RETURN_COUNTERFACTUAL = True
REWARD_SCALE = 0
REWARD_SCALE = 1. # Vary density of rewards added
REWARD_SCALE = 0.1 # Vary density of rewards added
REWARD_SCRAMBLE = False
# REWARD_SCRAMBLE = True # Vary timing

do_plot = True
# do_plot = False

tag = f'Reward: {REWARD_SCALE} Scramble: {REWARD_SCRAMBLE}'
# tag = f'Constraint: {CONSTRAINT_CORRECTION}'
def plot_logits(
    ax,
    logits,
    title,
    bin_size_ms,
    vmin=0,
    vmax=0,
    truth=None,
    mute_yticks=True,
):
    ax = prep_plt(ax, big=True)
    if not vmin and not vmax:
        sns.heatmap(logits.cpu().T, ax=ax, cmap="RdBu_r")
    else:
        sns.heatmap(logits.cpu().T, ax=ax, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    if truth is not None:
        ax.plot(truth.cpu().T, color="k", linewidth=2, linestyle="--")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Class")
    ax.set_title(title)
    if mute_yticks:
        ax.set_yticks([])
    ax.set_xticks(np.linspace(0, logits.shape[0], 3))
    ax.set_xticklabels(np.linspace(0, logits.shape[0] * bin_size_ms, 3).astype(int))

    # label colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel('Logit')

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
            cf_constraint = torch.tensor([
                constraint_correction, constraint_correction, 0, # How much is brain NOT participating, how much active assist is on
            ], dtype=assist_constraint.dtype, device=assist_constraint.device)
            assist_constraint[(assist_constraint != 0).sum(-1) == 2] = cf_constraint
            batch[DataKey.constraint.name] = assist_constraint
        if RETURN_COUNTERFACTUAL:
            assist_return = batch[DataKey.task_return.name]
            # assist_return = torch.ones_like(assist_return)
            # batch[DataKey.task_return.name] = assist_return

            assist_reward = batch[DataKey.task_reward.name]
            if REWARD_SCALE:
                injected = torch.bernoulli(torch.ones_like(assist_reward) * REWARD_SCALE).int()
                batch[DataKey.task_reward.name] = (assist_reward + injected).clamp(1, 2)

            if REWARD_SCRAMBLE:
                assist_reward = batch[DataKey.task_reward.name]
                assist_reward = assist_reward[:, torch.randperm(assist_reward.shape[1])]
                batch[DataKey.task_reward.name] = assist_reward

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
    # print(target.shape, outputs[Output.behavior_query_mask].shape)

    # Compute R2
    is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
    valid = is_student_rolling > (
        model.cfg.eval.student_gap * len(outputs[DataKey.covariate_labels.name])
    )

    print(f"Computing R2 on {valid.sum()} of {valid.shape} points")

    # print(target.shape, prediction.shape, valid.shape)
    # print(is_student_rolling.shape)
    loss = outputs[Output.behavior_loss].mean()
    mse = torch.mean((target[valid] - prediction[valid]) ** 2, dim=0)
    r2_student = r2_score(target[valid], prediction[valid])
    print(mse)
    print(f"Checkpoint: {ckpt_epoch} (tag: {tag})")
    print(f'Loss: {loss:.3f}')
    print(f"MSE: {mse:.3f}")
    print(f"R2 Student: {r2_student:.3f}")

    def plot_split_logits(full_logits, labels, cfg, truth=None):
        f, axes = plt.subplots(len(labels), 1, figsize=(15, 10), sharex=True, sharey=True)

        # Split logits
        stride = len(labels)
        for i, label in enumerate(labels):
            logits = full_logits[i::stride]
            if truth is not None:
                truth_i = truth[i::stride]
            else:
                truth_i = None
            plot_logits(
                axes[i],
                logits,
                label,
                cfg.dataset.bin_size_ms,
                truth=truth_i,
                vmin=-20,
                vmax=20,
            )
        f.suptitle(f"{query} Logits MSE {mse:.3f} Loss {loss:.3f} {tag}")
        plt.tight_layout()

    truth = outputs[Output.behavior].float()
    # print(truth.shape)
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
    try:
        reported_r2 = ckpt_rows[f"val_{Metric.kinematic_r2.name}"].values[-1]
        reported_loss = ckpt_rows[f"val_loss"].values[-1]
        reported_kin_loss = ckpt_rows[f"val_kinematic_infill_loss"].values[-1]
        print(f"Reported R2: {reported_r2:.3f}")
        print(f"Reported Loss: {reported_loss:.3f}")
        print(f"Reported Kin Loss: {reported_kin_loss:.3f}")
    except IndexError:
        print("No reported metrics found")
    return outputs, target, prediction, is_student, valid, r2_student, mse, loss


(outputs, target, prediction, is_student, valid, r2_student, mse, loss) = eval_model(
    model, dataset,
)

#%%
ax = plt.subplot(1, 1, 1)
ax.invert_yaxis()
plot_logits(
    ax,
    outputs[Output.return_logits].float(),
    f'{query} Pred Return',
    cfg.dataset.bin_size_ms,
    mute_yticks=False,
    truth=outputs[Output.return_target].float() + 0.5, # offset line into bins
    vmin=-10,
    vmax=20,
)

print(outputs[Output.return_target].float())

x_min = 0
# x_min = 120
# x_min = 140
# x_min = 200
# x_min = 2000

# x_max = 20
x_max = 250
# x_max = 800
# x_max = 2500

ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 20)
ax.set_xticks(np.linspace(x_min, x_max, 3))
ax.set_xticklabels(np.linspace(x_min * cfg.dataset.bin_size_ms, x_max * cfg.dataset.bin_size_ms, 3).astype(int))

# ? Shouldn't timestep 0 be very uncertain?
# This is the "teacher forced" performance - not reflective of when we start with a shitty guess.
# TODO See what happens when we seed with a shitty or unrealistic guess
# ! Plot the truth
#%%
import numpy as np
from torch.nn.functional import softmax
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

logits = outputs[Output.return_logits].float()
kappa = 10
logits_opt = torch.linspace(0.0, 1.0, logits.shape[-1]).unsqueeze(0).to(logits.device)
# Sample from log[P(optimality=1|return)*P(return)].
logits_offset = logits + kappa * logits_opt
probs = softmax(logits, -1).cpu()
probs_offset = softmax(logits_offset, -1).cpu()
# Create the logits array
ax = prep_plt(plt.subplot(1, 1, 1), big=True)  # Assuming prep_plt is a predefined function
steps = 1
start = 0
palette = sns.color_palette("mako_r", steps + 1)


# Create a ScalarMappable with the colormap
norm = plt.Normalize(start * cfg.dataset.bin_size_ms, (start + steps) * cfg.dataset.bin_size_ms)
sm = cm.ScalarMappable(cmap="mako_r", norm=norm)

for i, step in enumerate(np.arange(start, start + steps)):
    ax.plot(np.arange(probs.shape[1]), probs[step], color=palette[i])
    ax.plot(np.arange(probs.shape[1]), probs_offset[step], color=palette[i], linestyle="--")

ax.set_xlim(0, 10)

# Add color bar
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Time (ms)')

ax.set_xlabel("pred return")
ax.set_ylabel("prob")