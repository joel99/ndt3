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
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import chop_vector, compress_vector

import matplotlib.pyplot as plt
import scipy.interpolate as spi

sampling_rate = 1000
datapath = 'data/delay_reach/000121/sub-Reggie/sub-Reggie_ses-20170116T102856_behavior+ecephys.nwb'
with NWBHDF5IO(datapath, 'r') as io:
    nwbfile = io.read()
    hand_pos_global = nwbfile.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].data # T x 3
    timestamps_global = np.round(nwbfile.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].timestamps[:] * sampling_rate).astype(int) # T
    # On observation it looks like the hand data is not continuous, so we need to interpolate
    true_samples = np.diff(hand_pos_global[:, 0]).nonzero()[0]
    hand_pos_global = hand_pos_global[true_samples]
    timestamps_global = timestamps_global[true_samples]
    # hand_pos_resampled = np.interp(np.arange(timestamps_global[0], timestamps_global[-1]), timestamps_global, hand_pos_global)
    # resample to appropriate resolution, and compute velocity

    # Step 1: Interpolation to 50Hz
    # target_time = np.arange(0, timestamps_global[-1], 1/50.0)
    # interp_func = spi.interp1d(time, position, kind='linear', bounds_error=False, fill_value=np.nan)
    # interpolated_position = interp_func(target_time)

    # # Detect NaN spans in original data
    # nan_spans = np.diff(np.isnan(position).astype(int))

    # # Mark corresponding spans in interpolated data as NaN
    # for t_start, t_end in zip(time[np.where(nan_spans == 1)[0]], time[np.where(nan_spans == -1)[0]]):
    #     idx_start = np.searchsorted(target_time, t_start)
    #     idx_end = np.searchsorted(target_time, t_end)
    #     interpolated_position[idx_start:idx_end] = np.nan

    # # Step 3: Differentiation to get velocity
    # velocity = np.gradient(interpolated_position, target_time, edge_order=1)

    # hand_pos_global = resample_poly(hand_pos_global, sampling_rate, len(true_samples) // 1000, padtype='line', axis=0)
    # hand_vel_global = np.gradient(hand_pos_global, axis=0) # T x 2
    # print(hand_vel_global.shape)
    print(timestamps_global.shape)
    # print(hand_vel_global[:100])
    plt.plot(timestamps_global[:100], hand_pos_global[:100, 0])
    print(np.diff(hand_pos_global[:100, 0]).nonzero())
    # print(nwbfile.processing['behavior'].data_interfaces["Position"].spatial_series['Hand'].data[:][:10, 0])