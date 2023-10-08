r"""
    Miller/Limb lab data under XDS format e.g.
    https://datadryad.org/stash/dataset/doi:10.5061/dryad.cvdncjt7n (Jango, force isometric, 20 sessions, 95 days)
    Data proc libs:
    - https://github.com/limblab/xds
    - https://github.com/limblab/adversarial_BCI/blob/main/xds_tutorial.ipynb
    JY updated the xds repo into a package, clone here: https://github.com/joel99/xds/

    Features EMG data and abundance of isometric tasks.
    No fine-grained analysis - just cropped data for pretraining.
"""
#%%
from typing import List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch

from einops import reduce
from scipy.signal import decimate

from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

from pathlib import Path
import xds_python as xds
from context_general_bci.analyze_utils import prep_plt
import matplotlib.pyplot as plt
import numpy as np

data_path = Path('data/miller/Chewie_CO_2016') # Cursor
data_path = Path('data/miller/Greyson_Key_2019') # EMG + Forces
# data_path = Path('data/miller/Jango_ISO_2015') # EMG + Forces
# data_path = Path('data/miller/Mihili_CO_2014') # Just cursor
# data_path = Path('data/miller/Mihili_RT_2013_2014') # Just cursor
# data_path = Path('data/miller/Spike_ISO_2012') # Cursor, force, emg

data_path = data_path.glob('*.mat').__next__()
print()
my_xds = xds.lab_data(str(data_path.parent), data_path.name) # Load the data using the lab_data class in xds.py
print(my_xds.bin_width)
print('Are there EMGs? %d'%(my_xds.has_EMG))
print('Are there cursor trajectories? %d'%(my_xds.has_cursor))
print('Are there forces? %d'%(my_xds.has_force))

print('\nThe units names are %s'%(my_xds.unit_names))
if my_xds.has_EMG:
    print('\nThe muscle names are %s'%(my_xds.EMG_names))

my_xds.update_bin_data(0.020) # rebin to 20ms

cont_time_frame = my_xds.time_frame
cont_spike_counts = my_xds.spike_counts

# Print total active time etc
all_trials = [*my_xds.get_trial_info('R'), *my_xds.get_trial_info('F')] # 'A' not included
end_times = [trial['trial_end_time'] for trial in all_trials]
start_times = [trial['trial_gocue_time'] for trial in all_trials]
# ? Does the end time indicate the sort of... bin count?
print("Start: ", start_times)
print("End: ", end_times)
print(len(start_times), len(end_times))
if isinstance(start_times[0], np.ndarray):
    start_times = [start[0] for start in start_times]
total_time = sum([end - start for start, end in zip(start_times, end_times)])
print(total_time)

print(f"Total trial time: {total_time:.2f}")
print(f"Estimated recording time: {(my_xds.time_frame[-1] - my_xds.time_frame[0])}")
print(f"Timeframe start: {my_xds.time_frame[0]}")
print(f"Timeframe end: {my_xds.time_frame[-1]}")
print(f"First trial start: {start_times[0]:.2f}")
print(f"Last trial end: {end_times[-1]:.2f}")
print(f"Active trial %: {(total_time / (my_xds.time_frame[-1] - my_xds.time_frame[0])[0] * 100):.2f}%")

print('Shapes')
print('Time frame : ', cont_time_frame.shape)
print('Spike counts : ', cont_spike_counts.shape)

if my_xds.has_EMG:
    cont_EMG = my_xds.EMG
    print('EMG : ', cont_EMG.shape)
if my_xds.has_cursor:
    print('Cursor : ', my_xds.curs_v.shape)
if my_xds.has_force:
    print('Force : ', my_xds.force.shape)
# TODO low pri - there's off by 1 alignment in covariates
#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming spike_bin_counts is your time x channel array
plt.figure(figsize=(10, 10))
sns.heatmap(cont_spike_counts.T, cmap="viridis", cbar=True)
plt.ylabel('Channels')
plt.xlabel('Time')
plt.title('Spike Bin Counts Heatmap')
plt.show()

# Visualize spikes
print(cont_spike_counts.mean())
print(cont_spike_counts.min())
print(cont_spike_counts.max())
#%%
# Visualize some trajectories.

fig, axs = plt.subplots(11, 1, figsize=(10, 22), sharex=True)  # 11 subplots, 1 column

# Prepare each subplot
for ax in axs:
    prep_plt(ax)  # Assuming prep_plt() modifies the passed axis object

def annotate_title(ax, data, label):
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    ax.set_title(f"{label} (Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f})")

# Plot each data series in its own subplot and annotate title
if my_xds.has_cursor:
    annotate_title(axs[0], my_xds.curs_v[:, 0], 'vx')
    axs[0].plot(my_xds.curs_v[:, 0])

    annotate_title(axs[1], my_xds.curs_v[:, 1], 'vy')
    axs[1].plot(my_xds.curs_v[:, 1])
if my_xds.has_force:
    annotate_title(axs[2], my_xds.force[:, 0], 'fx')
    axs[2].plot(my_xds.force[:, 0])

    annotate_title(axs[3], my_xds.force[:, 1], 'fy')
    axs[3].plot(my_xds.force[:, 1])
if my_xds.has_EMG:
    # Plot EMG (assuming 7 dimensions)
    for i in range(7):
        annotate_title(axs[4 + i], my_xds.EMG[:, i], f'e{i}')
        axs[4 + i].plot(my_xds.EMG[:, i])

plt.tight_layout()
plt.show()
