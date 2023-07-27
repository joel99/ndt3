#%%
r"""
JY Note to self: This data was imported with data_transfer/transfer_motor.py
You still need to `prep_for_analysis` the QL data.
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
    ('CRS08Home.data.00013', 2): 'NDT2 Human',
    ('CRS08Home.data.00013', 4): 'NDT2 Subject',
    ('CRS08Home.data.00013', 5): 'OLE',
    ('CRS08Home.data.00013', 6): 'NDT2 Subject',
    ('CRS08Home.data.00013', 7): 'NDT2 Human',
    ('CRS08Home.data.00013', 8): 'OLE',
    ('CRS08Home.data.00013', 9): 'OLE',
    ('CRS08Home.data.00013', 10): 'NDT2 Subject',
    ('CRS08Home.data.00013', 11): 'NDT2 Human',

    ('CRS08Home.data.00016', 4): 'OLE (Sup)', # Orochi
    ('CRS08Home.data.00016', 6): 'NDT2 Subject (Unsup)',
    ('CRS08Home.data.00016', 8): 'NDT2 Human (Unsup)',

    ('CRS08Lab.data.00023', 4): 'Mix 0-Shot (2 day)',
    ('CRS08Lab.data.00023', 5): 'OLE',
    ('CRS08Lab.data.00023', 6): 'Human',
    ('CRS08Lab.data.00023', 7): 'ReFIT Tune',
    ('CRS08Lab.data.00023', 8): 'Subject',
    ('CRS08Lab.data.00023', 9): 'Mix',
    ('CRS08Lab.data.00023', 10): 'Subject 0-Shot (2 day)',
}


def extract_reaches(payload):
    reach_key = list(payload['state_strs']).index('Reach') + 1 # 1-indexed
    reach_times = payload['task_states'] == reach_key
    # https://chat.openai.com/share/78e7173b-3586-4b64-8dc9-656eca751526

    # Get indices where reach_times switches from False to True or True to False
    switch_indices = np.where(np.diff(reach_times))[0] + 1  # add 1 to shift indices to the end of each block
    successes = np.zeros(len(switch_indices) // 2, dtype=bool)
    switch_indices = np.concatenate(([0], switch_indices, [len(reach_times)]))

    # Split reach_times and payload['position'] at switch_indices
    reach_times_splits = np.split(reach_times, switch_indices)
    position_splits = np.split(payload['position'], switch_indices)
    target_splits = np.split(payload['target'], switch_indices)

    cumulative_pass = payload['passed']
    assert len(cumulative_pass) == len(successes)
    # convert cumulative into individual successes
    successes[1:] = np.diff(cumulative_pass) > 0
    successes[0] = cumulative_pass[0] > 0

    # Now, we zip together the corresponding reach_times and positions arrays,
    # discarding those where all reach_times are False (no 'Reach' in the trial)
    trial_data = [{
        'pos': pos,
        'times': times,
        'targets': targets,
    } for pos, times, targets in zip(
        position_splits, reach_times_splits, target_splits
    ) if np.any(times)]
    for i, trial in enumerate(trial_data):
        trial['success'] = successes[i]

    return trial_data

def get_times(payload):
    # https://github.com/pitt-rnel/motor_learning_BCI/blob/main/utility_fx/calculateAcquisitionTime.m
    reaches = extract_reaches(payload)
    return [np.sum(i['times']) * payload['bin_size_ms'] / 1000 for i in reaches]

def get_path_efficiency(payload):
    # Success weighted by Path Length, in BCI! Who would have thought.
    # https://arxiv.org/pdf/1807.06757.pdf
    # 1/N \Sigma S_i \frac{optimal_i} / \frac{max(p_i, optimal_i)}
    # https://github.com/pitt-rnel/motor_learning_BCI/blob/main/utility_fx/calculate_path_efficiency.m
    reaches = extract_reaches(payload)
    # TODO assuming success atm (need to pull from data)
    spl_individual = []
    for i in reaches:
        pos, times, targets, success = i['pos'], i['times'], i['targets'], i['success']
        optimal_length = np.linalg.norm(targets[0] - pos[0])
        path_length = np.sum([np.linalg.norm(pos[i+1] - pos[i]) for i in range(len(pos)-1)])
        spl_individual.append(optimal_length * success / max(path_length, optimal_length))
    return spl_individual
# def compute_total_reach_time(payload):
#     reaches = extract_reaches(payload)
#     total_time = 0
#     for pos, times in reaches:
#         total_time += np.sum(times)

handle = 'CRS08Home.data.00013'
handle = 'CRS08Home.data.00016'
handle = 'CRS08Lab.data.00023'

data_dir = Path('./data/pitt_misc/mat')
session = int(handle.split('.')[2])
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
            'variant': SET_TO_VARIANT[(handle, r_set)],
            'time': t,
            'spl': spl,
        })
all_trials = pd.DataFrame(all_trials)

ax = prep_plt()
mode = 'spl'
# mode = 'time'
order = [
    'Subject 0-Shot (2 day)',
    'Mix 0-Shot (2 day)',
    'OLE',
    'Subject',
    'Human',
    'Mix',
    # 'ReFIT Tune',
]
sns.boxplot(all_trials, y=mode, x='variant', ax=ax, order=order)

if mode == 'spl':
    ax.set_ylabel('Success weighted by Path Length')
else:
    ax.set_ylabel('Average Reach Time (s)')
ax.set_xlabel('Decoder Variant')

# Rotate x label
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
# probably better conveyed as a table
ax.set_title(handle)
# Pandas to latex table

table = all_trials.groupby(['variant']).agg(['mean', 'std'])
table = table[['spl', 'time']]
table = table.round(2)
table = table.to_latex()
# print(table)