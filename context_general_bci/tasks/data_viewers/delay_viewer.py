#%%
from typing import List
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from einops import rearrange, reduce, repeat
from scipy.signal import resample_poly
import logging
logger = logging.getLogger(__name__)

try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS
from context_general_bci.config.presets import ScaleHistoryDatasetConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import chop_vector, compress_vector

import matplotlib.pyplot as plt
import scipy.interpolate as spi
cfg = ScaleHistoryDatasetConfig()
sampling_rate = 1000
datapath = 'data/delay_reach/000121/sub-Reggie/sub-Reggie_ses-20170116T102856_behavior+ecephys.nwb'
task_cfg = cfg.delay_reach
with NWBHDF5IO(datapath, 'r') as io:
    nwbfile = io.read()
    # Note, not all nwb are created equal (though they're similar)
    trial_info = nwbfile.trials
    starts = (trial_info.start_time.data[:] * sampling_rate).astype(int)
    ends = (trial_info.stop_time.data[:] * sampling_rate).astype(int)
    # For this, looking forward to general processing, we'll not obey trial information
    hand_pos_global = nwbfile.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].data # T x 2

    # hand_vel_global = np.gradient(hand_pos_global, axis=0) # T x 2
    timestamps_global = np.round(nwbfile.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].timestamps[:] * sampling_rate).astype(int) # T
    true_samples = np.concatenate([np.array([0]), np.diff(hand_pos_global[:, 0]).nonzero()[0] + 1]) # start of a new value assumed to be sample point
    hand_pos = []
    timestamps = []
    for i in range(0, true_samples.shape[0], 10000): # For some reason we're hanging, if we slice all at once...
        hand_pos.append(hand_pos_global[true_samples[i:i+10000]])
        timestamps.append(timestamps_global[true_samples[i:i+10000]])
    hand_pos_global = np.concatenate(hand_pos, 0)
    timestamps_global = np.concatenate(timestamps, 0)

    target_time = np.arange(0, timestamps_global[-1], cfg.bin_size_ms)
    interp_func = spi.interp1d(timestamps_global, hand_pos_global, kind='linear', bounds_error=False, fill_value=np.nan, axis=0)
    interpolated_position = interp_func(target_time)

    # Detect NaN spans in original data
    nan_spans = np.diff(np.isnan(hand_pos_global).astype(int))

    # Mark corresponding spans in interpolated data as NaN
    for t_start, t_end in zip(timestamps_global[np.where(nan_spans == 1)[0]], timestamps_global[np.where(nan_spans == -1)[0]]):
        idx_start = np.searchsorted(target_time, t_start)
        idx_end = np.searchsorted(target_time, t_end)
        interpolated_position[idx_start:idx_end] = np.nan

    # Step 3: Differentiation to get velocity
    target_vel = np.gradient(interpolated_position, target_time, axis=0, edge_order=1) # *varargs for T x 3
    target_vel = torch.from_numpy(target_vel).float()
    global_args = {}
    if cfg.tokenize_covariates:
        global_args[DataKey.covariate_labels] = ['x', 'y', 'z']
    if task_cfg.minmax:
        # warn about nans
        global_vel = target_vel
        if global_vel.isnan().any():
            logging.warning(f'{torch.isnan(global_vel).any(axis=1).sum()} nan steps found in velocity, masking out for global calculation')
            global_vel = global_vel[~torch.isnan(global_vel).any(axis=1)]
        global_vel = torch.as_tensor(global_vel, dtype=torch.float)
        if global_vel.shape[0] > int(1e6): # Too long for quantile, just crop with warning
            logging.warning(f'Covariate length too long ({global_vel.shape[0]}) for quantile, cropping to 1M')
            global_vel = global_vel[:int(1e6)]
        global_args['cov_mean'] = global_vel.mean(0)
        global_args['cov_min'] = torch.quantile(global_vel, 0.001, dim=0)
        global_args['cov_max'] = torch.quantile(global_vel, 0.999, dim=0)
        target_vel = (target_vel - global_args['cov_mean']) / (global_args['cov_max'] - global_args['cov_min'])
        target_vel = torch.clamp(target_vel, -1, 1)

    spikes = nwbfile.units.to_dataframe()
    # We assume one continuous observation, which should be the case
    interval_ct = spikes.obs_intervals.apply(lambda x: x.shape[0]).unique()
    if len(interval_ct) != 1:
        print(f"Found {len(interval_ct)} unique interval counts (expecting 1); they were {interval_ct}")
    min_obs, max_obs = spikes.obs_intervals.apply(lambda x: x[0,0]).min(), spikes.obs_intervals.apply(lambda x: x[-1,-1]).max()
    min_obs, max_obs = int(min_obs * sampling_rate), int(max_obs * sampling_rate)
    span = max_obs - min_obs + 1
    spike_dense = torch.zeros(span, (len(spikes.spike_times)), dtype=torch.uint8)
    for i, times in enumerate(spikes.spike_times):
        spike_dense[(times * sampling_rate).astype(int) - min_obs, i] = 1
    # Find inner time bounds between spikes and kinematics
    start_time = max(min_obs, timestamps_global[0])
    end_time = min(max_obs, timestamps_global[-1])
    # Crop both
    spike_dense = spike_dense[start_time - min_obs:end_time - min_obs]
    # vel_dense = torch.zeros(span, hand_vel_global.shape[-1], dtype=torch.float) # Surprisingly, we don't have fully continuous hand signals. We'll zero pad for those off periods. # TODO we may want to introduce something besides zero periods.
    # # Put hand vel signals
    # vel_dense.fill_(np.nan) # Just track - we'll reject segments that contain NaNs
    # vel_dense.scatter_(0, repeat(torch.tensor(timestamps_global - start_time), 't -> t d', d=vel_dense.shape[-1]), torch.tensor(hand_vel_global, dtype=torch.float))
    # vel_dense = vel_dense[start_time - min_obs:end_time - min_obs]
# OK, great, now just chop
# spike_dense = compress_vector(spike_dense, task_cfg.chop_size_ms, cfg.bin_size_ms)
# downsample
# vel_dense = torch.as_tensor(resample_poly(vel_dense, int(1000 /  cfg.bin_size_ms), 1000, padtype='line', axis=0))
# Issue...  nan bleeds all over by nan element count heuristic
# bhvr = chop_vector(vel_dense, task_cfg.chop_size_ms, cfg.bin_size_ms) # Effectively a downsample
# We go trialized after all, for simplicity.
# def preproc_vel(trial_vel, global_args):
    # trial_vel = trial_vel[trial_vel.shape[0] % cfg.bin_size_ms:]
    # trial_vel = resample_poly(trial_vel, (1000 / cfg.bin_size_ms), 1000, padtype='line', axis=0)
    # trial_vel = torch.from_numpy(trial_vel).float() # Min max already occurred at global level
    # return trial_vel
#%%
for t in range(50, len(trial_info)):
    t_start, t_end = starts[t], ends[t] + 500
    trial_spikes = spike_dense[t_start - min_obs:t_end - min_obs]
    vel_mask = (target_time >= t_start) & (target_time <= t_end)
    trial_vel = target_vel[vel_mask][-(trial_spikes.shape[0] // cfg.bin_size_ms):]
    # trial_spikes = create_spike_payload(trial_spikes, context_arrays, cfg=cfg) # providing cfg will trigger compression
    # Though it looks like trials are getting cut off, they link straight into the next piece.
print(trial_spikes.shape, trial_vel.shape)
import matplotlib.pyplot as plt
plt.plot(trial_vel[..., 0], label='x')
plt.plot(trial_vel[..., 1], label='y')
plt.plot(trial_vel[..., 2], label='z')