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

@ExperimentalTaskRegistry.register
class DelayReachLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.delay_reach
    r"""
    - https://dandiarchive.org/dandiset/000121/0.210815.0703 Even-chen et al.
    - Delayed reaching, with PMd + M1; should contain preparatory dynamics.
    # ! JY realizes now that the data scraped from gdrive in `churchland_misc` is exactly this data.
    # ! We prefer to use standardized releases, so we should migrate at some point.
    TODO implement
    TODO subset with NWB loader
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
            # For this, looking forward to general processing, we'll not obey trial information
            hand_pos_global = nwbfile.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].data # T x 2
            hand_vel_global = np.gradient(hand_pos_global, axis=0) # T x 2
            timestamps_global = np.round(nwbfile.processing['behavior'].data_interfaces['Position'].spatial_series['Hand'].timestamps[:] * sampling_rate).astype(int) # T
            global_args = {}
            if cfg.tokenize_covariates:
                global_args[DataKey.covariate_labels] = ['x', 'y', 'z']
            if task_cfg.minmax:
                global_vel = np.concatenate(hand_vel_global, 0)
                # warn about nans
                if np.isnan(global_vel).any():
                    logging.warning(f'{global_vel.isnan().sum()} nan values found in velocity, masking out for global calculation')
                    global_vel = global_vel[~np.isnan(global_vel).any(axis=1)]
                global_vel = torch.as_tensor(global_vel, dtype=torch.float)
                if global_vel.shape[0] > int(1e6): # Too long for quantile, just crop with warning
                    logging.warning(f'Covariate length too long ({global_vel.shape[0]}) for quantile, cropping to 1M')
                    global_vel = global_vel[:int(1e6)]
                global_args['cov_mean'] = global_vel.mean(0)
                global_args['cov_min'] = torch.quantile(global_vel, 0.001, dim=0)
                global_args['cov_max'] = torch.quantile(global_vel, 0.999, dim=0)

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
            vel_dense = torch.zeros(span, hand_vel_global.shape[-1], dtype=torch.float) # Surprisingly, we don't have fully continuous hand signals. We'll zero pad for those off periods. # TODO we may want to introduce something besides zero periods.
            # Put hand vel signals
            vel_dense.fill_(np.nan) # Just track - we'll reject segments that contain NaNs
            vel_dense.scatter_(0, repeat(torch.tensor(timestamps_global - start_time), 't -> t d', d=vel_dense.shape[-1]), torch.tensor(hand_vel_global, dtype=torch.float))
            vel_dense = vel_dense[start_time - min_obs:end_time - min_obs]
        # OK, great, now just chop
        breakpoint()
        spike_dense = compress_vector(spike_dense, task_cfg.chop_size_ms, cfg.bin_size_ms)
        # downsample
        vel_dense = torch.as_tensor(resample_poly(vel_dense, int(1000 /  cfg.bin_size_ms), 1000, padtype='line', axis=0))
        # Doesn't seem like nan bleeds all over by nan element count heuristic
        bhvr = chop_vector(vel_dense, task_cfg.chop_size_ms, cfg.bin_size_ms) # Effectively a downsample

        for t in range(spike_dense.size(0)):
            breakpoint() # TODO - looks like we get a lot of NaNs after all, not just a few big breaks
            # Probably need to add in constraints now, to indicate the non-meaningful periods.
            trial_spikes = spike_dense[t]
            trial_vel = bhvr[t]
            # Check NaNs and crop if > 5%
            nan_pct = (torch.isnan(trial_vel).sum() / trial_vel.numel()).item()
            if nan_pct > 0.05:
                # Skip
                continue
            else:
                # Report
                if nan_pct > 0:
                    print(f'Warning: {nan_pct} of velocity data is nan, interpolating')
                    # Convert PyTorch tensor to NumPy array
                    trial_vel_np = trial_vel.numpy()

                    # Loop through each column (dimension)
                    for i in range(trial_vel_np.shape[1]):
                        column_data = trial_vel_np[:, i]

                        # Find indices of NaNs and non-NaNs
                        nan_idx = np.isnan(column_data)
                        not_nan_idx = np.logical_not(nan_idx)

                        # Interpolate
                        x = np.arange(len(column_data))
                        column_data[nan_idx] = np.interp(x[nan_idx], x[not_nan_idx], column_data[not_nan_idx])

                        # Update the column
                        trial_vel_np[:, i] = column_data

                    # Convert back to PyTorch tensor
                    trial_vel = torch.tensor(trial_vel_np)
            single_payload = {
                DataKey.spikes: create_spike_payload(trial_spikes, context_arrays),
                DataKey.bhvr_vel: trial_vel.clone(),
                **global_args
            }
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
