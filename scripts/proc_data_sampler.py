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
sample_query = 'h512_l6_return'
wandb_run = wandb_query_latest(sample_query, exact=False, allow_running=True)[0]
# print(wandb_run)
_, cfg, _ = load_wandb_run(wandb_run, tag='val_loss', load_model=False)
run_cfg = cfg.dataset
# run_cfg.datasets = ['pitt_broad.*']

# run_cfg.data_keys = [*run_cfg.data_keys, 'cov_min', 'cov_max', 'cov_mean'] # Load these directly, for some reason they're cast
dataset = SpikingDataset(run_cfg)
dataset.build_context_index()
dataset.subset_split()

#%%
from tqdm import tqdm
set_stats = {} # No set level granularity, will have to build from labels later
session_stats = {}
trial_stats = [] # No tracking...
dimensions = {}

def process_session(meta_session):
    session_stats = {}
    trial_stats = []
    session_df = dataset.meta_df[dataset.meta_df[MetaKey.session] == meta_session]
    # breakpoint() # get the session
    all_constraints = []
    session_length = 0
    dim = None
    for trial in session_df.index: # Oh, we don't have set level granularity...
        if DataKey.constraint in dataset[trial]:
            constraints = dataset[trial][DataKey.constraint]
            all_constraints.append(constraints)
        if dim is None:
            dim = dataset[trial][DataKey.covariate_labels]
        mode = torch.mode(dataset[trial][DataKey.bhvr_vel].flatten())
        mode_value = mode.values[0] if len(mode.values.shape) >= 1 else mode.values
        mode_count = (dataset[trial][DataKey.bhvr_vel].flatten() == mode_value).sum()
        total_count = dataset[trial][DataKey.bhvr_vel].flatten().shape[0]
        trial_stats.append({
            'length': dataset[trial][DataKey.time].max(),
            'mode': mode_value.item(),
            'mode_count': mode_count.item(),
            'total_count': total_count,
            'max_return': dataset[trial][DataKey.task_return].max().item(),
        })
        session_length += trial_stats[-1]['length']
    payload = torch.load(session_df.iloc[-1].path)
    for k in ['cov_min', 'cov_max', 'cov_mean']:
        if k not in payload or payload[k] == None:
            payload[k] = torch.zeros(dataset.cfg.behavior_dim)
        elif payload[k].shape[0] == 8: # No force
            payload[k] = torch.cat([payload[k], torch.zeros(1)])
    subject, session, set = meta_session.split('_')[-3:]
    session_stats[meta_session] = {
        "subject": subject,
        "session": int(session),
        "set": int(set),
        "length": session_df.shape[0],
        "dimensions": dim,
        "cov_min": payload["cov_min"],
        "cov_max": payload["cov_max"],
        "cov_mean": payload["cov_mean"],
        "session_length": session_length.item(),
        "has_brain_control": (torch.cat(all_constraints)[:, 0] < 1).any().item(),
    }
    # if len(session_stats) > 10:
        # break # trial
    return session_stats, trial_stats
global_session = {}
global_trial = []
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
unique_sessions = dataset.meta_df[MetaKey.session].unique()
total_sessions = len(unique_sessions)

# with Pool(processes=32) as pool:
    # results = list(tqdm(pool.imap(process_session, unique_sessions), total=len(unique_sessions)))
# with ProcessPoolExecutor(max_workers=8) as executor:
    # results = list(tqdm(executor.map(process_session, unique_sessions), total=total_sessions))
# no multiprocessing
results = []
for meta_session in tqdm(unique_sessions):
    results.append(process_session(meta_session))
    # if len(results) > 2:
        # break
combined_trial_stats = []
combined_session_stats = {}
for session_stat, trial_stat in results:
    combined_trial_stats.extend(trial_stat)
    combined_session_stats.update(session_stat)
# for meta_session in tqdm(dataset.meta_df[MetaKey.session].unique()):
    # session_stats[meta_session], trial_stats = process_session(meta_session)
torch.save({
    'session': session_stats,
    'trial': trial_stats,
}, 'scripts/proc_data_sampler.pt')
from pprint import pprint
# pprint(dimensions)
# for session in has_brain_control:
#     if not has_brain_control[session]:
#         print(session)

