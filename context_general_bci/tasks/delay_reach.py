#%%
from typing import List
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from einops import rearrange, reduce, repeat
from scipy.signal import resample_poly
import scipy.interpolate as spi
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

@ExperimentalTaskRegistry.register
class DelayReachLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.delay_reach
    r"""
    - https://dandiarchive.org/dandiset/000121/0.210815.0703 Even-chen et al.
    - Delayed reaching, with PMd + M1; should contain preparatory dynamics.
    # ! JY realizes now that the data scraped from gdrive in `churchland_misc` is exactly this data.
    # ! We prefer to use standardized releases, so we should migrate at some point.
    """

    @classmethod
    def load(
        cls,
        datapath: Path, # path to NWB file
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
        sampling_rate=1000
    ):
        task_cfg = getattr(cfg, task.name)
        meta_payload = {}
        meta_payload['path'] = []
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
        for t in range(len(trial_info)):
            t_start, t_end = starts[t], ends[t]
            trial_spikes = spike_dense[t_start - min_obs:t_end - min_obs]
            if trial_spikes.sum() == 0:
                # Either empty or something anomalous, skip
                continue
            vel_mask = (target_time >= t_start) & (target_time <= t_end)
            trial_vel = target_vel[vel_mask][-(trial_spikes.shape[0] // cfg.bin_size_ms):]
            trial_spikes = create_spike_payload(trial_spikes, context_arrays, cfg=cfg) # providing cfg will trigger compression

            # Crop
            for k, v in trial_spikes.items():
                trial_spikes[k] = v[-task_cfg.chop_size_ms // cfg.bin_size_ms:]
            trial_vel = trial_vel[-task_cfg.chop_size_ms // cfg.bin_size_ms:]
            # Crop start if necessary
            # trial_vel = vel_dense[(t_start * sampling_rate).astype(int) - min_obs:(t_end * sampling_rate).astype(int) - min_obs]

            # Compress, downsample
            # These are short, no need to compress/chop
            # trial_spikes = compress_vector(trial_spikes, task_cfg.chop_size_ms, cfg.bin_size_ms)
        # for t in range(spike_dense.size(0)):
            # trial_spikes = spike_dense[t]
            # trial_vel = bhvr[t]
            # Check NaNs and crop if > 5%
            # nan_pct = (torch.isnan(trial_vel).sum() / trial_vel.numel()).item()
            # if nan_pct > 0.05:
            #     # Skip
            #     continue
            # else:
            #     # Report
            #     if nan_pct > 0:
            #         print(f'Warning: {nan_pct} of velocity data is nan, interpolating')
            #         # Convert PyTorch tensor to NumPy array
            #         trial_vel_np = trial_vel.numpy()

            #         # Loop through each column (dimension)
            #         for i in range(trial_vel_np.shape[1]):
            #             column_data = trial_vel_np[:, i]

            #             # Find indices of NaNs and non-NaNs
            #             nan_idx = np.isnan(column_data)
            #             not_nan_idx = np.logical_not(nan_idx)

            #             # Interpolate
            #             x = np.arange(len(column_data))
            #             column_data[nan_idx] = np.interp(x[nan_idx], x[not_nan_idx], column_data[not_nan_idx])

            #             # Update the column
            #             trial_vel_np[:, i] = column_data

            #         # Convert back to PyTorch tensor
            #         trial_vel = torch.tensor(trial_vel_np)
            single_payload = {
                DataKey.spikes: trial_spikes,
                DataKey.bhvr_vel: trial_vel,
                **global_args
            }
            assert list(trial_spikes.values())[0].shape[0] == trial_vel.shape[0], "Spike and velocity lengths do not match"
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
#%%
# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     sampling_rate = 1000
#     datapath = 'data/delay_reach/000121/sub-Reggie/sub-Reggie_ses-20170116T102856_behavior+ecephys.nwb'
#     with NWBHDF5IO(datapath, 'r') as io:
#         nwbfile = io.read()
#         hand_pos_global = nwbfile.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].data # T x 3
#         timestamps_global = np.round(nwbfile.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].timestamps[:] * sampling_rate).astype(int) # T
#         # On observation it looks like the hand data is not continuous, so we need to interpolate
#         true_samples = np.diff(hand_pos_global[:, 0]).nonzero()[0]
#         hand_pos_global = hand_pos_global[true_samples]
#         timestamps_global = timestamps_global[true_samples]
#         # hand_pos_resampled = np.interp(np.arange(timestamps_global[0], timestamps_global[-1]), timestamps_global, hand_pos_global)
#         # resample to appropriate resolution, and compute velocity

#         # Step 1: Interpolation to 50Hz
#         # target_time = np.arange(0, timestamps_global[-1], 1/50.0)
#         # interp_func = spi.interp1d(time, position, kind='linear', bounds_error=False, fill_value=np.nan)
#         # interpolated_position = interp_func(target_time)

#         # # Detect NaN spans in original data
#         # nan_spans = np.diff(np.isnan(position).astype(int))

#         # # Mark corresponding spans in interpolated data as NaN
#         # for t_start, t_end in zip(time[np.where(nan_spans == 1)[0]], time[np.where(nan_spans == -1)[0]]):
#         #     idx_start = np.searchsorted(target_time, t_start)
#         #     idx_end = np.searchsorted(target_time, t_end)
#         #     interpolated_position[idx_start:idx_end] = np.nan

#         # # Step 3: Differentiation to get velocity
#         # velocity = np.gradient(interpolated_position, target_time, edge_order=1)

#         # hand_pos_global = resample_poly(hand_pos_global, sampling_rate, len(true_samples) // 1000, padtype='line', axis=0)
#         # hand_vel_global = np.gradient(hand_pos_global, axis=0) # T x 2
#         # print(hand_vel_global.shape)
#         print(timestamps_global.shape)
#         # print(hand_vel_global[:100])
#         plt.plot(timestamps_global[:100], hand_pos_global[:100, 0])
#         print(np.diff(hand_pos_global[:100, 0]).nonzero())
#         # print(nwbfile.processing['behavior'].data_interfaces["Position"].spatial_series['Hand'].data[:][:10, 0])