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
from context_general_bci.task_io import create_token_padding_mask
from context_general_bci.contexts import context_registry
from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt
from context_general_bci.utils import get_wandb_run, wandb_query_latest

# Exactly matched to training
CONTEXT_S_SUITE = [0, 1, 2, 3]
BINS_PER_S = 50

EVAL_DATASET_SUITE = [
    "odoherty_rtt-Indy-20160407_02",
    "odoherty_rtt-Indy-20160627_01",
    "odoherty_rtt-Indy-20161005_06",
    "odoherty_rtt-Indy-20161026_03",
    "odoherty_rtt-Indy-20170131_02"
]

wandb_run = wandb_query_latest('no_embed', allow_running=True)[0]
# wandb_run = wandb_query_latest('30s_no_embed')[0]
print(f'ICL Eval for: {wandb_run.id}')
src_model, cfg, old_data_attrs = load_wandb_run(wandb_run)
# cfg.model.task.outputs = [Output.behavior, Output.behavior_pred] # Don't actually need this, just need the metric

def compute_icl_eval(data_id, context_s=27):
    return 0. # TODO call out to `predict_scripted`
results = []
for context_s in CONTEXT_S_SUITE:
    results.append({
        'data_id': 'eval',
        'context_s': context_s,
        'icl': compute_icl_eval(data_id, context_s)
    })
results = pd.DataFrame(results)
#%%
f = plt.figure(figsize=(8, 8))
ax = prep_plt(f.gca(), big=True)

sns.lineplot(

)