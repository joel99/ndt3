#%%
r"""
    TODO
    Identify .mat files for relevant trials
    Open .mat
    Find the phases (observation) that are relevant for now
    Find the spikes
    Find the observed kinematic traces

    Batch for other sessions
"""
import pandas as pd
import numpy as np
# import xarray as xr
from pathlib import Path
import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch

from context_general_bci.tasks.pitt_co import load_trial
from context_general_bci.analyze_utils import prep_plt
import seaborn as sns

# TODO
# 1. Plot trajectory
# 2. Get success rate, time taken
# ! Note data was incorrectly labeled as a lab session but that's not impt for now

SET_TO_VARIANT = {
    2: 'NDT Human',
    4: 'NDT Subject',
    5: 'OLE',
    6: 'NDT Subject',
    7: 'NDT Human',
    8: 'OLE',
    9: 'OLE',
    10: 'NDT Subject',
    11: 'NDT Human',
}


def extract_reaches(payload):
    reach_key = list(payload['state_strs']).index('Reach') + 1 # 1-indexed
    reach_times = payload['task_states'] == reach_key
    # https://chat.openai.com/share/78e7173b-3586-4b64-8dc9-656eca751526

    # Get indices where reach_times switches from False to True or True to False
    switch_indices = np.where(np.diff(reach_times))[0] + 1  # add 1 to shift indices to the end of each block

    switch_indices = np.concatenate(([0], switch_indices, [len(reach_times)]))

    # Split reach_times and payload['position'] at switch_indices
    reach_times_splits = np.split(reach_times, switch_indices)
    position_splits = np.split(payload['position'], switch_indices)
    target_splits = np.split(payload['target'], switch_indices)
    # Now, we zip together the corresponding reach_times and positions arrays,
    # discarding those where all reach_times are False (no 'Reach' in the trial)
    trial_data = [(pos, times, targets) for pos, times, targets in zip(
        position_splits, reach_times_splits, target_splits
    ) if np.any(times)]
    return trial_data

def get_times(payload):
    # https://github.com/pitt-rnel/motor_learning_BCI/blob/main/utility_fx/calculateAcquisitionTime.m
    reaches = extract_reaches(payload)
    return [np.sum(times) * payload['bin_size_ms'] / 1000 for pos, times, _ in reaches]

def get_path_efficiency(payload):
    # Success weighted by Path Length, in BCI! Who would have thought.
    # https://arxiv.org/pdf/1807.06757.pdf
    # 1/N \Sigma S_i \frac{optimal_i} / \frac{p_i, optimal_i}
    # https://github.com/pitt-rnel/motor_learning_BCI/blob/main/utility_fx/calculate_path_efficiency.m
    reaches = extract_reaches(payload)
    # TODO assuming success atm (need to pull from data)
    spl_individual = []
    for i in reaches:
        pos, times, targets = i
        optimal_length = np.linalg.norm(targets[0] - pos[0])
        path_length = np.sum([np.linalg.norm(pos[i+1] - pos[i]) for i in range(len(pos)-1)])
        spl_individual.append(optimal_length / max(path_length, optimal_length))
    return spl_individual
# def compute_total_reach_time(payload):
#     reaches = extract_reaches(payload)
#     total_time = 0
#     for pos, times in reaches:
#         total_time += np.sum(times)



data_dir = Path('./data/pitt_misc/mat')
session = 13
session_runs = list(data_dir.glob(f'*{session}*fbc.mat'))
all_trials = []
for r in session_runs:
    # r of the format "data/pitt_misc/mat/crs08Lab_session_13_set_11_type_fbc.mat"
    r_set = int(r.name.split('_')[4])

    payload = load_trial(r, key='thin_data')
    times = get_times(payload)
    spls = get_path_efficiency(payload)
    for t, spl in zip(times, spls):
        all_trials.append({
            'r_set': r_set,
            'variant': SET_TO_VARIANT[r_set],
            'time': t,
            'spl': spl,
        })
all_trials = pd.DataFrame(all_trials)

ax = prep_plt()
sns.boxplot(all_trials, y='spl', x='variant', ax=ax)
ax.set_ylabel('Path Efficiency')
ax.set_xlabel('Decoder Variant')
# probably better conveyed as a table

# Pandas to latex table

table = all_trials.groupby(['variant']).agg(['mean', 'std'])
table = table[['spl', 'time']]
table = table.round(2)
table = table.to_latex()
print(table)