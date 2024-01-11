#%%
from pathlib import Path
import numpy as np

from context_general_bci.utils import loadmat

# Private Rouse data
# to ./data/rouse_precision and placed ./data_extracted at root.
data_dir = Path(
    'data/rouse/'
)

for sample_file in data_dir.glob('*.mat'):
    data = loadmat(sample_file)
    print(f'{sample_file.stem} -- Num channels : {len(data["AllSpikeTimes"])}')
sample_file = data_dir.glob('*.mat').__next__()
data = loadmat(sample_file)

print(data.keys())
#%%
# print(data['AllSpikeTimes'].shape)
print(data['SpikeTimes'].shape) # Nested list - unit x trial x event
print(data['SpikeTimes'][0].shape) # Trials
print(data['SpikeTimes'][0][0].shape) # Events
print(data['SpikeTimes'][0][1].shape) # Events
print(data['CursorPos'].shape)
print(data['SpikeSettings'].keys())
bin_size_ms = 1000 / data['SpikeSettings']['samp_rate'] # 100Hz?
bin_edges_ms = np.arange(0, data['CursorPos'].shape[1] * bin_size_ms + 1, bin_size_ms)
# For monkey A: I need regions - EFGH (M1), ABCD (Premotor) (see README)
# For monkey B: I need regions: EFGH (M1), ABCD (Premotor)
relevant_units = []
for channel, array in enumerate(data['SpikeSettings']['array_by_channel']):
    if array in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        relevant_units.append(channel)
relevant_units = np.array(relevant_units)
# Construct a binned array

for trial, trial_pos in enumerate(data['CursorPos']):
    spike_arr = np.zeros((data['CursorPos'].shape[1], len(data['SpikeTimes'])), dtype=np.uint8)
    for idx, trial_unit in enumerate(data['SpikeTimes']):
        spike_cnt, _ = np.histogram(trial_unit[trial], bins=bin_edges_ms / 1000) # spike time is in s
        spike_arr[:, idx] = spike_cnt
    # Clip out outlier values
    spike_arr[spike_arr > 20] = 0
    # Clip data to when there's valid positions
    valid_pos = np.logical_not(np.isnan(trial_pos[:, 0]))
    print(valid_pos)
    spike_arr = spike_arr[valid_pos, :]
    trial_pos = trial_pos[valid_pos, :]
    break

#%%
print(data['SpikeTimes'][0][0])
print(data['SpikeTimes'][0][1])
# Plot the data
#%%
import scipy.signal as signal
import matplotlib.pyplot as plt
f = plt.figure(figsize=(20, 10))
ax = f.gca()

# ? Why do I suddenly have so many NaNs?
pos = data['CursorPos']
vel = np.gradient(pos, axis=1)
def resample(data, covariate_rate=100):
    base_rate = int(1000 / 20)
    return (
        signal.resample_poly(data, base_rate, covariate_rate, padtype='line', axis=1)
    )

print(pos[0, :, 0])
ax.plot(np.arange(pos.shape[1]), pos[0,:,0] / 10) # For visual scaling
ax.plot(np.arange(pos.shape[1]), vel[0,:,0])
fake_vel = vel.copy()
fake_vel[np.isnan(fake_vel)] = 0
ax.plot(np.arange(0, pos.shape[1], 2), resample(fake_vel)[0, :, 0])
#%%
# Plot the both
import matplotlib.pyplot as plt
import seaborn as sns

# raster plot
# sns.histplot(spike_arr.flatten(), bins=100)
# f = plt.figure(figsize=(20, 10))
# ax = f.gca()
# sns.heatmap(spike_arr, alpha=0.1)
# print(spike_arr[:, 0])
# print(spike_arr[:, 2])
# print(spike_arr[:, 1])
def rasterplot(spike_arr, bin_size_s=0.01, ax=None):
    if ax is None:
        ax = plt.gca()
    for idx, unit in enumerate(spike_arr.T):
        ax.scatter(np.where(unit)[0] * bin_size_s, np.ones(np.sum(unit != 0)) * idx, s=1, c='k', marker='|')
    # ax = sns.heatmap(spike_arr.T, cmap='gray_r', ax=ax) # Not sufficient - further autobinning occurs
    ax.set_xticks(np.arange(0, spike_arr.shape[0], 5000))
    ax.set_xticklabels(np.arange(0, spike_arr.shape[0], 5000) * bin_size_s)
    ax.set_yticks(np.arange(0, spike_arr.shape[1], 40))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron #')
rasterplot(spike_arr, bin_size_s=bin_size_ms / 1000)