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
try:
    import pyaldata
except:
    logging.info("Pyaldata not installed, please install from https://github.com/NeuralAnalysis/PyalData. Import will fail")

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

data_path = Path('data/miller/adversarial')
file_name = 'Jango_20150730_001.mat'
file_name = 'Jango_20150731_001.mat'
my_xds = xds.lab_data(str(data_path), file_name) # Load the data using the lab_data class in xds.py
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
cont_EMG = my_xds.EMG

# Print total active time etc
all_trials = [*my_xds.get_trial_info('R'), *my_xds.get_trial_info('F')] # 'A' not included
end_times = [trial['trial_end_time'] for trial in all_trials]
start_times = [trial['trial_gocue_time'] for trial in all_trials]
# ? Does the end time indicate the sort of... bin count?
total_time = sum([end - start for start, end in zip(start_times, end_times)])
print('Total trial time: %f'%(total_time))
print('Estimated recording time: %f'%(my_xds.time_frame[-1] - my_xds.time_frame[0]))
print('Shapes')
print(my_xds.force.shape)
print(my_xds.curs_v.shape)
print(cont_time_frame.shape)
print(cont_spike_counts.shape)
print(cont_EMG.shape)
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
annotate_title(axs[0], my_xds.curs_v[:, 0], 'vx')
axs[0].plot(my_xds.curs_v[:, 0])

annotate_title(axs[1], my_xds.curs_v[:, 1], 'vy')
axs[1].plot(my_xds.curs_v[:, 1])

annotate_title(axs[2], my_xds.force[:, 0], 'fx')
axs[2].plot(my_xds.force[:, 0])

annotate_title(axs[3], my_xds.force[:, 1], 'fy')
axs[3].plot(my_xds.force[:, 1])

# Plot EMG (assuming 7 dimensions)
for i in range(7):
    annotate_title(axs[4 + i], my_xds.EMG[:, i], f'e{i}')
    axs[4 + i].plot(my_xds.EMG[:, i])

plt.tight_layout()
plt.show()
