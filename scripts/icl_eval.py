#%%
# Basic script to probe for ICL capabilities
from typing import Dict
import itertools
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import lightning.pytorch as pl

from einops import rearrange, pack, unpack
from sklearn.metrics import r2_score

from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model
from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt
from context_general_bci.utils import get_wandb_run, wandb_query_latest

# Exactly matched to training
# CONTEXT_S_SUITE = [0, 1, 2, 3]
CONTEXT_S_SUITE = [1, 4, 9, 13]
BINS_PER_S = 50

data_label ='indy'
#%%
wandb_run = wandb_query_latest('no_embed', allow_running=True)[0]
# wandb_run = wandb_query_latest('30s_no_embed')[0]
print(f'ICL Eval for: {wandb_run.id}')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run)
# cfg.model.task.outputs = [Output.behavior, Output.behavior_pred] # Don't actually need this, just need the metric
from scripts.predict_scripted import icl_eval
def compute_icl_eval(data_label, context_s=27):
    return 0. # TODO call out to `predict_scripted`
results = []
for context_s in CONTEXT_S_SUITE:
    results.append({
        'data_id': 'eval',
        'context_s': context_s,
        'icl': compute_icl_eval(data_label, context_s)
    })
results = pd.DataFrame(results)
#%%
results = {
    'Model': [
        '70 hr (Expert)',
        '70 hr (Expert)',
        '70 hr (Expert)',
        '70 hr (Expert)',
        '700 hr',
        '700 hr',
        '700 hr',
        '700 hr',
    ],
    'context_s': [0, 1, 2, 3, 0, 1, 2, 3,],
    'icl': [
        0.6663,
        0.6665,
        0.6667,
        0.6720,
        0.6120,
        0.6222,
        0.6321,
        0.6432
    ] # Produced by running for i in {0,1,2,3};do python scripts/predict_scripted.py -i <variant> -d indy -c $i;done
}
# results = {
#     'Model': [
#         '700 hr',
#         '700 hr',
#         '700 hr',
#         '700 hr',
#     ],
#     'context_s': [1, 4, 9, 13],
#     'icl': [
#         0.6144,
#         0.6103,
#         0.6018,
#         0.6255,
#     ]
# }
results = pd.DataFrame(results)
f = plt.figure(figsize=(6, 6))
ax = prep_plt(f.gca(), big=True)

palette = sns.color_palette(
    palette='pastel',
    n_colors=2
)
sns.lineplot(
    data=results,
    x='context_s',
    y='icl',
    hue='Model',
    ax=ax,
    palette=palette,
)
# hline labeled "Single session, 300s" at 0.56
ax.axhline(0.56, ls='--', color='k', label='Single session NDT2, 5 min')
ax.legend(
    loc=(0.0, 0.0),
    bbox_to_anchor=(0.1, 0.1),
    frameon=False
)
ax.set_ylabel("Eval Velocity $R^2$ ($\\uparrow$)")
ax.set_xlabel('Velocity input (s)')