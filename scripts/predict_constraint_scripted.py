#%%
# Testing online parity, using open predict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
import lightning.pytorch as pl

from sklearn.metrics import r2_score

from context_general_bci.model import transfer_model, BrainBertInterface
from context_general_bci.dataset import SpikingDataset
from context_general_bci.config import RootConfig, ModelTask, Metric, Output, DataKey, MetaKey
from context_general_bci.contexts import context_registry

from context_general_bci.utils import wandb_query_latest
from context_general_bci.analyze_utils import (
    stack_batch, load_wandb_run, prep_plt, rolling_time_since_student, get_dataloader,
    data_label_to_target
)
from context_general_bci.streaming_utils import (
    precrop_batch,
    postcrop_batch,
    prepend_prompt,
)

query = 'small_40m-0q2by8md'

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
# data_label = 'crs08_grasp'
if data_label:
    target = data_label_to_target(data_label)
else:
    target = [
        # NDT runs
        # OL
        # 'pitt_broad_pitt_co_CRS08Lab_25_1$',
        # 'pitt_broad_pitt_co_CRS08Lab_25_2$',
        # 'pitt_broad_pitt_co_CRS08Lab_25_3$',

        # 'pitt_broad_pitt_co_CRS08Lab_29_1$',
        # 'pitt_broad_pitt_co_CRS08Lab_29_2$',
        # 'pitt_broad_pitt_co_CRS08Lab_29_3$',

        # OLE FBC
        # 'pitt_broad_pitt_co_CRS08Lab_25_5$',
        'pitt_broad_pitt_co_CRS08Lab_25_6$',
        # 'pitt_broad_pitt_co_CRS08Lab_29_5$',
        # 'pitt_broad_pitt_co_CRS08Lab_29_6$',

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

reference_target = []
reference_target = [
    # 'pitt_broad_pitt_co_CRS08Lab_25_6$',
    # 'pitt_broad_pitt_co_CRS08Lab_25_5$',
]

# reference_target = [
#     'pitt_broad_pitt_co_CRS08Lab_25_1$',
#     'pitt_broad_pitt_co_CRS08Lab_25_2$',
#     'pitt_broad_pitt_co_CRS08Lab_25_3$',
# ]
if reference_target:
    reference_cfg = deepcopy(cfg)
    reference_cfg.dataset.datasets = reference_target
    reference_dataset = SpikingDataset(reference_cfg.dataset)
    reference_dataset.build_context_index()
    print(f'Ref total: {len(reference_dataset)}')
    prompt = reference_dataset[0]
else:
    prompt = None

pl.seed_everything(0)
print("Eval length: ", len(dataset))
data_attrs = dataset.get_data_attrs()
print(data_attrs)
model = transfer_model(src_model, cfg.model, data_attrs)
model.eval()
model = model.to('cuda')

#%%
def eval_model(
    model: BrainBertInterface,
    dataset,
    cue_length_s=3,
    tail_length_s=3,
    precrop_prompt=12, # For simplicity, all precrop for now. We can evaluate as we change precrop length
    postcrop_working=3,
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
        if prompt is not None:
            # breakpoint()
            pseudo_prompt = deepcopy(batch)
            batch = postcrop_batch(batch, int((cfg.dataset.pitt_co.chop_size_ms - postcrop_working * 1000) // cfg.dataset.bin_size_ms))

            crop_prompt = precrop_batch(pseudo_prompt, prompt_bins) # Debug
            crop_prompt = {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in crop_prompt.items()}
            # batch = prepend_prompt(batch, pseudo_crop_prompt)
            # breakpoint()
            batch = prepend_prompt(batch, crop_prompt)
        # TODO crop batch

        output = model.predict_simple_batch(
            batch,
            kin_mask_timesteps=kin_mask_timesteps,
            last_step_only=False,
        )
        outputs.append(output)

    outputs = stack_batch(outputs)

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
    return {
        'cue_length_s': cue_length_s,
        'mse': mse.item(),
        'r2': r2_student
    }

all_metrics = []
for cue_length_s in [3, 6, 9]:
    metrics = eval_model(model, dataset, cue_length_s=cue_length_s)
    all_metrics.append(metrics)
print(all_metrics)
