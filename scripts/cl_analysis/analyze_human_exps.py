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
    ('CRS08Home.data.00013', 2): 'Human',
    ('CRS08Home.data.00013', 4): 'Subject',
    ('CRS08Home.data.00013', 5): 'OLE',
    ('CRS08Home.data.00013', 6): 'Subject',
    ('CRS08Home.data.00013', 7): 'Human',
    ('CRS08Home.data.00013', 8): 'OLE',
    ('CRS08Home.data.00013', 9): 'OLE',
    ('CRS08Home.data.00013', 10): 'Subject',
    ('CRS08Home.data.00013', 11): 'Human',

    ('CRS08Home.data.00016', 4): 'Orochi', # Orochi, not OLE..
    # ('CRS08Home.data.00016', 4): 'OLE', # Orochi, not OLE..
    ('CRS08Home.data.00016', 6): 'Subject Unsup',
    ('CRS08Home.data.00016', 8): 'Human Unsup',

    ('CRS08Lab.data.00023', 4): 'Mix 0-Shot',
    ('CRS08Lab.data.00023', 5): 'OLE',
    ('CRS08Lab.data.00023', 6): 'Human',
    ('CRS08Lab.data.00023', 7): 'ReFIT',
    ('CRS08Lab.data.00023', 8): 'Subject',
    ('CRS08Lab.data.00023', 9): 'Mix',
    ('CRS08Lab.data.00023', 10): 'Subject 0-Shot',

    # ! Protocol defined at this point. 2 wide, 1 close calibration.
    ('CRS08Lab.data.00025', 4): 'OLE 0-Shot', # THIS ONE REQUIRED TUNING
    ('CRS08Lab.data.00025', 5): 'OLE',
    ('CRS08Lab.data.00025', 6): 'Mix 0-Shot',
    ('CRS08Lab.data.00025', 7): 'Subject 0-Shot',
    ('CRS08Lab.data.00025', 8): 'Subject',
    ('CRS08Lab.data.00025', 9): 'Mix Interrupt',
    ('CRS08Lab.data.00025', 10): 'Mix',
    ('CRS08Lab.data.00025', 11): 'Human',
    ('CRS08Lab.data.00025', 12): 'Mix Unsup', # Subject got tired.
    ('CRS08Lab.data.00025', 13): 'Subject Unsup',

    ('CRS08Home.data.00019', 4): 'OLE',
    ('CRS08Home.data.00019', 5): 'OLE 0-Shot', # No tuning, 5 day old
    ('CRS08Home.data.00019', 6): 'Mix Unsup',
    ('CRS08Home.data.00019', 7): 'Subject Unsup',
    ('CRS08Home.data.00019', 8): 'Mix 0-Shot',
    ('CRS08Home.data.00019', 9): 'Subject 0-Shot',
    ('CRS08Home.data.00019', 10): 'Mix',
    ('CRS08Home.data.00019', 11): 'Subject',
}


NEED_TUNING = [
    'CRS08Lab.data.00025', # severe tuning means score should be zero-ed in my analysis
]

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
        trial['id'] = i

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
# handle = 'CRS08Lab.data.00023'
handle = 'CRS08Lab.data.00025'
# handle = 'CRS08Home.data.00019'

def get_handle_df(handle: str, remove_burn_in=True):
    # remove_burn_in: don't count first trial, where participant often is not ready
    data_dir = Path('./data/pitt_misc/mat')
    session = int(handle.split('.')[2])
    session_runs = list(data_dir.glob(f'*session_{session}*fbc.mat'))
    all_trials = []
    for r in session_runs:
        # r of the format "data/pitt_misc/mat/crs08Lab_session_13_set_11_type_fbc.mat"
        r_set = int(r.name.split('_')[4])

        payload = load_trial(r, key='thin_data')
        times = get_times(payload)
        spls = get_path_efficiency(payload)
        trial_count = 0
        for t, spl in zip(times, spls):
            label = SET_TO_VARIANT[(handle, r_set)]
            variant, *status = label.split()
            if status:
                status = ' '.join(status)
            else:
                status = 'Sup'
            all_trials.append({
                'r_set': r_set,
                'variant': variant,
                'status': status,
                'time': t,
                'spl': spl,
                'id': trial_count,
            })
            trial_count += 1
    all_trials = pd.DataFrame(all_trials)
    all_trials['session'] = session
    if remove_burn_in:
        all_trials = all_trials[all_trials['id'] != 0]
    return all_trials


ax = prep_plt(big=True)
# mode = 'spl'
mode = 'time'

if handle == 'CRS08Lab.data.00023':
    order = [
        'Subject 0-Shot',
        'Mix 0-Shot',
        'OLE',
        'Subject',
        'Human',
        'Mix',
        # 'ReFIT Tune',
    ]
elif handle == 'CRS08Lab.data.00025':
    order = [
        # 'OLE 0-Shot',
        'Subject 0-Shot',
        'Mix 0-Shot',
        'Subject Unsup',
        'Mix Unsup',
        'OLE',
        'Subject',
        'Human',
        'Mix',
    ]
elif handle == 'CRS08Home.data.00019':
    order = [
        'OLE 0-Shot',
        'Subject 0-Shot',
        'Mix 0-Shot',
        'Subject Unsup',
        'Mix Unsup',
        'OLE',
        'Subject',
        'Mix',
    ]

df = get_handle_df(handle)
df['Group'] = df['variant'].apply(lambda x: '0-Shot' if '0-Shot' in x else ('Unsupervised' if 'Unsup' in x else 'Other'))
df = df[df['variant'].isin(order)]

x_order = ['0-Shot', 'Unsup', 'Sup']
hue_order = ['OLE', 'Subject', 'Mix']
# hue_order = ['OLE', 'Subject', 'Human', 'Mix']
# Boxplot using Seaborn to show the distribution of data
sns.boxplot(data=df, y=mode, x='status', hue='variant', order=x_order, hue_order=hue_order, ax=ax)
# Use Seaborn strip plot to add data points on top of the boxplot
sns.stripplot(data=df, y=mode, x='status', hue='variant', order=x_order, hue_order=hue_order, dodge=True, jitter=True, s=5, alpha=0.5, ax=ax)

# Add some annotations
ole_index = hue_order.index('OLE')
estimate_group_offset = 0.6
def estimate_offset(index):
    return ((len(hue_order) / 2.) - index) / len(hue_order) * estimate_group_offset
offset = estimate_offset(ole_index)
print(offset)
ole_x, ole_y = '0-Shot', df[df['variant'] == 'OLE'][mode].iloc[0]
ax.text(x_order.index(ole_x) - offset, ole_y, 'X', color='red', fontsize=18, ha='center', va='center')


if mode == 'spl':
    ax.set_ylabel('Success weighted by Path Length')
else:
    ax.set_ylabel('Reach Time (s)')
    ax.set_ylim(0, 10) # timeout
ax.set_xlabel('Decoder Variant')
ax.set_xlabel('')
# Rotate x label
# for tick in ax.get_xticklabels():
    # tick.set_rotation(45)
# probably better conveyed as a table
ax.set_title(handle)
ax.set_title('N=1 (3 min calibration)')
# Pandas to latex table

# Move legend off to right side, middle of plot
# Only include second half of legend, first half is redundant
handles, labels = ax.get_legend_handles_labels()
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(handles[len(handles)//2:], labels[len(labels)//2:], loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, fontsize=18)

table = df.groupby(['variant']).agg(['mean', 'std'])
table = table[['spl', 'time']]
table = table.round(2)
table = table.to_latex()
# print(table)
#%%
handles = [
    # 'CRS08Home.data.00013',
    # 'CRS08Home.data.00016',
    'CRS08Lab.data.00023',
    'CRS08Lab.data.00025',
    'CRS08Home.data.00019',
]

# df = get_handle_df(handle)
df = pd.concat([get_handle_df(h) for h in handles])

aggr_df = df.groupby(['session', 'status', 'variant'], as_index=False).agg(['mean', 'std'])
ax = prep_plt(size='medium')
mode = 'spl'
mode = 'time'
print(aggr_df[mode])
aggr_df = aggr_df[mode].reset_index()
aggr_df = aggr_df[aggr_df['variant'].isin(hue_order)]
aggr_df = aggr_df[aggr_df['status'].isin(x_order)]

# sns.boxplot(data=aggr_df, y='mean', x='status', hue='variant', ax=ax, order=x_order, hue_order=hue_order)
sns.stripplot(
    data=aggr_df,
    y='mean',
    x='status',
    hue='variant',
    order=x_order,
    hue_order=hue_order,
    dodge=True,
    jitter=True,
    s=8,
    alpha=0.8,
    ax=ax
)

# stripplot X's over the need_tuning data
# X's manually annotated, JY gives up trying to automate this
# need_tuning_df = aggr_df[aggr_df['variant'].isin(NEED_TUNING) & aggr_df['status'].isin(['0-Shot'])]
# print(need_tuning_df)
# sns.stripplot(
#     data=need_tuning_df,
#     y='mean',
#     x='status',
#     hue='variant',
#     order=x_order,
#     hue_order=hue_order,
#     dodge=True,
#     jitter=True,
#     s=5,
#     alpha=0.8,
#     ax=ax,
#     color='red',
#     marker='X',
# )


if mode == 'spl':
    ax.set_ylabel('Success weighted by Path Length')
else:
    ax.set_ylabel('Reach Time (s, $\leftarrow$)')
ax.set_xlabel('Decoder Variant')

# relabel x ticks
status_remap = {
    '0-Shot': '0-Shot',
    'Unsup': 'Unsupervised',
    'Sup': 'Supervised',
}
ax.set_xticklabels([status_remap[x] for x in x_order])

# ax.set_title('NDT2 Cursor Control')
# just put as text
# ax.text(0.35, 0.92, 'NDT2 Cursor Control', transform=ax.transAxes, ha='center', va='center', fontsize=18)

ax.set_ylim(1, 5)

# Redo legend
handles, labels = ax.get_legend_handles_labels()
labels[labels.index('Subject')] = 'Session'
labels[labels.index('Mix')] = 'Broad'
ax.legend(
    handles,
    labels,
    title='Model',
    fontsize=16,
    title_fontsize=16,
    frameon=True,
    # loc='upper right', # Not quite high enough
    loc=(0.65, 0.65),
)
ax.set_xlabel('')

# sessions = aggr_df['session'].unique()

# for session in sessions:
#     session_data = aggr_df[aggr_df['session'] == session]
#     x_coords = []
#     y_coords = []
#     for index, row in session_data.iterrows():
#         # Find the correct x position for each status in the session
#         x_pos = x_order.index(row['status'])
#         y_pos = row['mean']

#         x_coords.append(x_pos)
#         y_coords.append(y_pos)

#     # Plot a line connecting the points for this session
#     ax.plot(x_coords, y_coords, color='grey', linestyle='-', linewidth=1, alpha=0.5)
