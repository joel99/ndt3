#%%
# Autoregressive inference procedure, for generalist model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pandas as pd
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
    stack_batch, load_wandb_run, prep_plt, rolling_time_since_student, get_dataloader, DIMS
)

queries = [
    'neural_data_monkey-pitt_100',
    'neural_data_monkey-pitt_200',
    'neural_data_monkey-pitt_400',
    'neural_data_monkey-pitt_800',
    'data_monkey_100',
    'data_monkey_200',
    'data_monkey_400',
    'data_monkey_800',
    'data_min_100',
    'data_min_200',
    'data_min_400',
    'data_min_800',
]

def get_eval(query, tag="val_kinematic_r2", target=['rouse.*'], dataset=None, trainer=None):
    print(f"Querying: {query}")
    wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
    print(f"Run: {wandb_run.id}")

    # src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag='val_loss')
    src_model, cfg, old_data_attrs = load_wandb_run(wandb_run, tag=tag)
    cfg.model.task.outputs = [Output.behavior, Output.behavior_pred]

    if dataset is None:
        # Note: This won't preserve train val split, try to make sure eval datasets were held out
        if cfg.dataset.eval_ratio > 0:
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
    data_label = [i for i in DIMS.keys() if dataset.cfg.datasets[0].startswith(i)][0]
    print(f'Eval dataset: {data_label}')
    pl.seed_everything(0)
    print("Eval length: ", len(dataset))
    data_attrs = dataset.get_data_attrs()
    print(data_attrs)

    model = transfer_model(src_model, cfg.model, data_attrs)

    model.cfg.eval.teacher_timesteps = int(50 * 3.) # 0.5s
    model.cfg.eval.student_gap = int(50 * 1.)

    if trainer is None:
        trainer = pl.Trainer(
            accelerator='gpu', devices=1, default_root_dir='./data/tmp',
            precision='bf16-mixed',
        )
    dataloader = get_dataloader(dataset, batch_size=128, num_workers=16)
    outputs = stack_batch(trainer.predict(model, dataloader))

    prediction = outputs[Output.behavior_pred]
    target = outputs[Output.behavior]
    is_student = outputs[Output.behavior_query_mask]
    # Compute R2
    is_student_rolling, trial_change_points = rolling_time_since_student(is_student)
    valid = is_student_rolling > model.cfg.eval.student_gap
    r2_student = r2_score(target[valid], prediction[valid])
    print(f'R2 Student: {r2_student:.4f}')
    print(model.cfg.eval)
    return {
        'Query': query,
        'R2 Student': r2_student,
        'Tune Dataset': cfg.dataset.scale_limit_per_eval_session,
        'Pretrain Model': '_'.join(query.split('_')[:-1]),  # Adjust this based on the actual format of your query strings
    }, dataset, trainer

results_df = []
dataset = None
trainer = None
for query in queries:
    result, dataset, trainer = get_eval(query, dataset=dataset, trainer=trainer)  # Add other parameters as needed
    results_df.append(result)
results_df = pd.DataFrame(results_df)


#%%
# palette = sns.color_palette(n_colors=2)
from matplotlib.ticker import NullFormatter

f = plt.figure(figsize=(8, 6))
ax = prep_plt(f.gca(), big=True)
sns.lineplot(data=results_df, x='Tune Dataset', y='R2 Student', hue='Pretrain Model', ax=ax)
# Also add scatter
sns.scatterplot(data=results_df, x='Tune Dataset', y='R2 Student', hue='Pretrain Model', ax=ax, s=100)
ax.get_legend().remove()
ax.set_xscale('log')
ax.set_xticks([])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([20, 40, 80, 160])
ax.set_xticklabels([40, 80, 160, 320])
ax.set_ylabel('Velocity $R^2$')
ax.set_xlabel('Tuning Data (Minutes)')

# Add three annotation
# ax.annotate('250 NHP + 450 Human', xy=(20, 0.2), xytext=(20, 0.2), fontsize=24)
# ax.annotate('250 NHP', xy=(40, 0.2), xytext=(40, 0.2), fontsize=24)
# ax.annotate('40 NHP', xy=(80, 0.2), xytext=(80, 0.2), fontsize=24)
# print(results_df)