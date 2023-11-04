#%%
# Testing wise-ft
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# query = 'indy_miller-sweep-ndt3_ft-565zhe19'
query = 'rouse-sweep-ndt3_ft-ufayi36h'
# https://wandb.ai/joelye9/ndt3/runs/ufayi36h

query = 'crs08-sweep-ndt3_ft-st5j8owm'
# https://wandb.ai/joelye9/ndt3/runs/st5j8owm

wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

cfg.model.task.outputs = [
    Output.behavior,
    Output.behavior_pred,
]

if True:
    backbone_wandb = cfg.init_from_id
    backbone_wandb_run = wandb_query_latest(backbone_wandb, allow_running=True, use_display=True)[0]
    backbone_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')

def ensemble_weights(model_a, model_b, alpha):
    # https://github.com/mlfoundations/wise-ft/blob/master/src/wise_ft.py
    # alpha - how much of model_a. model_b should be backbone
    # ? not on same device?
    model_a = model_a.to('cpu')
    model_b = model_b.to('cpu')
    # model has some extra state that doesn't clone
    theta_0 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model_a.state_dict().items()}
    theta_1 = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in model_b.state_dict().items()}
    new_weights = {}
    for k in theta_0.keys():
        if isinstance(theta_0[k], torch.Tensor):
            new_weights[k] = alpha * theta_0[k] + (1 - alpha) * theta_1[k]
        else:
            new_weights[k] = theta_0[k]
    return new_weights


# data_label ='indy_miller'
data_label = 'rouse'
data_label = 'crs08'
# data_label = ''
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
        # 'pitt_broad_pitt_co_CRS02bLab_1942_3', # FBC
        # 'pitt_broad_pitt_co_CRS02bLab_1942_6', # FBC
        # 'pitt_broad_pitt_co_CRS02bLab_1942_7', # Free play
        # 'pitt_broad_pitt_co_CRS02bLab_1942_8', # Free play

        'odoherty_rtt-Indy-20160407_02',
        'odoherty_rtt-Indy-20160627_01',
        'odoherty_rtt-Indy-20161005_06',
        'odoherty_rtt-Indy-20161026_03',
        'odoherty_rtt-Indy-20170131_02',
        "miller_Jango-Jango_20150730_001",
        "miller_Jango-Jango_20150731_001",
        "miller_Jango-Jango_20150801_001",
        "miller_Jango-Jango_20150805_001",

    ]
    # data_label = [i for i in DIMS.keys() if dataset.cfg.datasets[0].startswith(i)][0]
    # data_label = 'grasp'
    print(f'Assuming: {data_label}')

# Note: This won't preserve train val split, try to make sure eval datasets were held out
print(cfg.dataset.eval_ratio)
if cfg.dataset.eval_ratio > 0 and cfg.dataset.eval_ratio < 1:
    # Not super robust... we probably want to make this more like... expand datasets and compute whether overlapped
    dataset = SpikingDataset(cfg.dataset) # Make as original
    eval_dataset = SpikingDataset(cfg.dataset) # Make as eval
    dataset.subset_split(keep_index=True)
    _, dataset = dataset.create_tv_datasets()
    eval_dataset.subset_split(splits=['eval'], keep_index=True)
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
    eval_dataset.subset_by_key(TARGET_DATASETS, key=MetaKey.session)
else:
    cfg.dataset.datasets = target
    cfg.dataset.exclude_datasets = []
    cfg.dataset.eval_datasets = []
    eval_dataset = SpikingDataset(cfg.dataset)
pl.seed_everything(0)
print("Eval length: ", len(eval_dataset))

data_attrs = eval_dataset.get_data_attrs()
print(data_attrs)
CUE_LENGTH_S = 1
# CUE_LENGTH_S = 3
# CUE_LENGTH_S = 9
CUE_LENGTH_S = 30

EVAL_GAP_S = 45 - CUE_LENGTH_S - 0 # TAIL
# EVAL_GAP_S = 45 - CUE_LENGTH_S - 40 # TAIL
# These don't matter, we hijack train path with trainer.test() over trainer.predict()
cfg.model.eval.teacher_timesteps = int(CUE_LENGTH_S * 1000 / cfg.dataset.bin_size_ms)
cfg.model.eval.student_gap = int(EVAL_GAP_S * 1000 / cfg.dataset.bin_size_ms)
cfg.model.eval.use_student = True
cfg.model.eval.use_student = False

trainer = pl.Trainer(
    accelerator='gpu', devices=1, default_root_dir='./data/tmp',
    precision='bf16-mixed',
)
dataloader = get_dataloader(eval_dataset, batch_size=16, num_workers=16)
eval_dataloader = get_dataloader(eval_dataset, batch_size=16, num_workers=16)

results = []
alpha_range = np.linspace(0, 1, 11)
for alpha in alpha_range:
    model = transfer_model(src_model, cfg.model, data_attrs)
    model.load_state_dict(ensemble_weights(model, backbone_model, alpha))

    val_metrics = trainer.test(model, dataloader)[0]
    eval_metrics = trainer.test(model, eval_dataloader)[0] # Avoid testing at once to avoid lightning prefix issues
    # print(f'Test loss: {eval_metrics["test_kinematic_infill_loss"]}')
    results.append(
        {'alpha': alpha,
         'val_kin_loss': val_metrics['test_kinematic_infill_loss'],
         'eval_kin_loss': eval_metrics['test_kinematic_infill_loss']
        }
    )

#%%
# Plot the curve

import pandas as pd
import seaborn as sns
f = plt.figure(figsize=(7, 6))
ax = prep_plt(f.gca())
results = pd.DataFrame(results)
palette = sns.color_palette('viridis', len(alpha_range))
sns.scatterplot(data=results, x='val_kin_loss', y='eval_kin_loss', hue='alpha', ax=ax, palette=palette, s=100)
ax.set_title(f"{query} WISE-FT")
min_x, max_x = ax.get_xlim()
min_y, max_y = ax.get_ylim()
# plot y = x
min_xy = min(min_x, min_y)
max_xy = max(max_x, max_y)
ax.plot([min_xy, max_xy], [min_xy, max_xy], ls="--", c=".3")
ax.set_xlim(min_xy, max_xy)
# Use a colorbar, not a standard legend
ax.get_legend().remove()
cmap = sns.color_palette('viridis', as_cmap=True)
norm = plt.Normalize(alpha_range.min(), alpha_range.max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ticks=alpha_range, label='Alpha (1=FT, 0=Zero-shot)', ax=ax)


# Draw a line connecting consecutive alpha
# for i in range(len(alpha_range) - 1):
    # plt.plot(results['val_kin_loss'].iloc[i:i+2], results['eval_kin_loss'].iloc[i:i+2], c='k', alpha=0.5)