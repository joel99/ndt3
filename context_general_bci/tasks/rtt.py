#%%

from typing import List
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from scipy.interpolate import interp1d
import scipy.signal as signal
from einops import rearrange, reduce

import logging
logger = logging.getLogger(__name__)
try:
    import h5py
except:
    logger.info("h5py not installed, please install with `conda install -c anaconda h5py`")

from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import chop_vector, compress_vector

@ExperimentalTaskRegistry.register
class ODohertyRTTLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.odoherty_rtt
    r"""
    O'Doherty et al RTT data.
    # https://zenodo.org/record/3854034
    # The data was pulled from Zenodo directly via
    # zenodo_get 3854034
    """
    @staticmethod
    def load_raw(datapath: Path, cfg: DatasetConfig, context_arrays: List[str]):
        # Hacky patch to determine the right arrays to use
        if cfg.odoherty_rtt.include_sorted:
            context_arrays = [arr for arr in context_arrays if 'all' in arr]
        else:
            context_arrays = [arr for arr in context_arrays if 'all' not in arr]

        assert cfg.odoherty_rtt.chop_size_ms % cfg.bin_size_ms == 0, "Chop size must be a multiple of bin size"
        with h5py.File(datapath, 'r') as h5file:
            orig_timestamps = np.squeeze(h5file['t'][:])
            time_span = int((orig_timestamps[-1] - orig_timestamps[0]) * cfg.odoherty_rtt.sampling_rate)
            if cfg.odoherty_rtt.load_covariates:
                def resample(data): # Updated 9/10/23: Previous resample produces an undesirable strong artifact at timestep 0. This hopefully removes that and thus also removes outliers.
                    covariate_rate = cfg.odoherty_rtt.covariate_sampling_rate # in Hz
                    return torch.tensor(
                        signal.resample_poly(data, int(1000 / covariate_rate), cfg.bin_size_ms, padtype='line')
                    )
                bhvr_vars = {}
                finger_pos = h5file['finger_pos'][()].T / 100 # into meters
                # bhvr_vars['position'] = torch.tensor(finger_pos).float() # ! Debug
                finger_pos = resample(finger_pos[..., 1:3])
                # ignore orientation if present
                # order is z, -x, -y. We just want x and y.
                finger_vel = np.gradient(finger_pos, axis=0)
                # unit for finger is m/bin, adjust to m/s
                finger_vel = finger_vel * (1000 / cfg.bin_size_ms)
                bhvr_vars[DataKey.bhvr_vel] = torch.tensor(finger_vel).float()

            int_arrays = [h5file[ref][()][:,0] for ref in h5file['chan_names'][0]]
            make_chan_name = lambda array: ''.join([chr(num) for num in array])
            chan_names = [make_chan_name(array) for array in int_arrays]
            chan_arrays = [cn.split()[0] for cn in chan_names]
            assert (
                len(chan_arrays) == 96 and all([c == 'M1' for c in chan_arrays]) or \
                len(chan_arrays) == 192 and all([c == 'M1' for c in chan_arrays[:96]]) and all([c == 'S1' for c in chan_arrays[96:]])
            ), "Only M1 and S1 arrays in specific format are supported"
            if len(chan_names) == 96 and any('S1' in arr for arr in context_arrays):
                logger.error(f'{datapath} only shows M1 but registered {context_arrays}, update RTTContextInfo')
                return None
                raise NotImplementedError
            spike_refs = h5file['spikes'][()].T
            if cfg.odoherty_rtt.include_sorted:
                spike_refs = spike_refs[:96] # only M1. Already quite a lot of units to process without, maintains consistency with other datasets. (not exploring multiarea atm)
                assert all('S1' not in arr for arr in context_arrays), "S1 not supported with sorted units"
            channels, units = spike_refs.shape # units >= 1 are sorted, we just want MUA on unit 0
            mua_unit = 0

            unit_budget = units # We change to include all units instead of just hash for unsorted, much more reasonable
            # unit_budget = units if cfg.odoherty_rtt.include_sorted else 1
            spike_arr = torch.zeros((time_span, channels, unit_budget), dtype=torch.uint8)

            min_spike_time = []
            for c in range(channels):
                if h5file[spike_refs[c, mua_unit]].dtype != float:
                    continue
                unit_range = range(units)
                # unit_range = range(units) if cfg.odoherty_rtt.include_sorted else [mua_unit]
                for unit in unit_range:
                    spike_times = h5file[spike_refs[c, unit]][()]
                    if spike_times.shape[0] == 2: # empty unit
                        continue
                    spike_times = spike_times[0] # most entries are shaped 1 x N (some malformatted are 2x2)
                    if len(spike_times) < 2: # don't bother on low FR (and drop malformatted)
                        continue
                    spike_times = spike_times - orig_timestamps[0]
                    ms_spike_times, ms_spike_cnt = np.unique((spike_times * cfg.odoherty_rtt.sampling_rate).round(6).astype(int), return_counts=True)
                    spike_mask = ms_spike_times < spike_arr.shape[0]
                    ms_spike_times = ms_spike_times[spike_mask]
                    ms_spike_cnt = torch.tensor(ms_spike_cnt[spike_mask], dtype=torch.uint8)
                    spike_arr[ms_spike_times, c, unit] = ms_spike_cnt
                    min_spike_time.append(ms_spike_times[0])
            if not cfg.odoherty_rtt.include_sorted:
                spike_arr = spike_arr.sum(2, keepdim=True)

        min_spike_time = max(min(min_spike_time), 0) # some spikes come before marked trial start
        spike_arr: torch.Tensor = spike_arr[min_spike_time:, :]

        spike_arr = spike_arr.flatten(1, 2)
        if cfg.odoherty_rtt.include_sorted:
            # Filter out extremely low FR units, we can't afford to load everything.
            threshold_count = cfg.odoherty_rtt.firing_hz_floor * (spike_arr.shape[0] / cfg.odoherty_rtt.sampling_rate)
            spike_arr = spike_arr[:, (spike_arr.sum(0) > threshold_count).numpy()]
        return spike_arr, bhvr_vars, context_arrays

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
    ):
        spike_arr, bhvr_vars, context_arrays = cls.load_raw(datapath, cfg, context_arrays)

        full_spikes = compress_vector(spike_arr, cfg.odoherty_rtt.chop_size_ms, cfg.bin_size_ms)
        global_args = {}
        if cfg.odoherty_rtt.load_covariates:
            for bhvr in [DataKey.bhvr_vel]:
                bhvr_vars[bhvr] = chop_vector(bhvr_vars[bhvr], cfg.odoherty_rtt.chop_size_ms, cfg.bin_size_ms)
        if cfg.odoherty_rtt.minmax: # Note we apply after chop, which also includes binning
            global_args['cov_mean'] = torch.tensor([0.0, 0.0]) # Our prior
            # global_args['cov_min'] = torch.quantile(bhvr_vars[bhvr].flatten(end_dim=-2), 0.0001, dim=0) # essentially guard for extreme outliers, but that's it.
            global_args['cov_min'] = torch.quantile(bhvr_vars[bhvr].flatten(end_dim=-2), 0.001, dim=0) # essentially guard for extreme outliers, but that's it.
            global_args['cov_max'] = torch.quantile(bhvr_vars[bhvr].flatten(end_dim=-2), 0.999, dim=0) # Just be consistent with pitt preproc.
            # global_args['cov_max'] = torch.quantile(bhvr_vars[bhvr].flatten(end_dim=-2), 0.9999, dim=0)
            rescale = global_args['cov_max'] - global_args['cov_min']
            rescale[torch.isclose(rescale, torch.tensor(0.))] = 1
            # if (bhvr_vars[bhvr] / rescale).max() > 0.9 or (bhvr_vars[bhvr] / rescale).min() < -0.9: # Looks like there's a long tail, sick.
                # print(bhvr_vars[bhvr].max(), bhvr_vars[bhvr].min())
                # breakpoint()
            bhvr_vars[bhvr] = (bhvr_vars[bhvr] - global_args['cov_mean']) / rescale
            bhvr_vars[bhvr] = torch.clamp(bhvr_vars[bhvr], -1, 1)

        if cfg.tokenize_covariates:
            global_args[DataKey.covariate_labels] = REACH_DEFAULT_KIN_LABELS
        meta_payload = {}
        meta_payload['path'] = []
        for t in range(full_spikes.size(0)):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                DataKey.bhvr_vel: bhvr_vars[DataKey.bhvr_vel][t].clone(),
                **global_args
            }
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
