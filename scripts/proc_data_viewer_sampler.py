#%%
r""" What does raw data look like? (Preprocessing devbook) """
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import pandas as pd
import torch

import logging
from matplotlib import pyplot as plt
import seaborn as sns

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset
from context_general_bci.tasks import ExperimentalTask

from context_general_bci.analyze_utils import prep_plt, load_wandb_run
from context_general_bci.utils import wandb_query_latest

sample_query = 'base' # just pull the latest run to ensure we're keeping its preproc config
sample_query = '10s_loco_regression'

# Return run
sample_query = 'sparse'

wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss')
run_cfg = cfg.dataset
# run_cfg.datasets = ['pitt_broad.*']

dataset = SpikingDataset(run_cfg)
dataset.build_context_index()
dataset.subset_split()

print(len(dataset))
#%%
has_brain_control = {}
dimensions = {}
for session in dataset.meta_df[MetaKey.session].unique():
    session_df = dataset.meta_df[dataset.meta_df[MetaKey.session] == session]
    all_constraints = []
    for trial in session_df.index:
        if DataKey.constraint in dataset[trial]:
            constraints = dataset[trial][DataKey.constraint]
            all_constraints.append(constraints)
        if session not in dimensions:
            dimensions[session] = dataset[trial][DataKey.covariate_labels]
            break
    # all_constraints = torch.cat(all_constraints, dim=0)
    # print(f'Session {session}: {all_constraints.shape}')
    # has_brain_control[session] = (all_constraints[:, 0] < 1).any()
from pprint import pprint
pprint(dimensions)
# for session in has_brain_control:
#     if not has_brain_control[session]:
#         print(session)

