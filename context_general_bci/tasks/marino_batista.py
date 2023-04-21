#%%
r"""
Closed-source data shared by Batista lab.
"""
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

from context_general_bci.utils import loadmat

import logging

logger = logging.getLogger(__name__)

@ExperimentalTaskRegistry.register
class MarinoBatistaLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.marino_batista_mp_reaching

    BASE_RES = 1000 # hz (i.e. 1ms)

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        **kwargs,
    ):
        mat_dict = loadmat(datapath)['Data']

        state_data = mat_dict['stateData']
        spikes = mat_dict['spikes'] # L [T (ms) x C (neurons)]
        num_trials = len(state_data)
        use_vel = 'reaching' in str(datapath)
        if use_vel:
            marker_data = mat_dict['marker']

        meta_payload = {}
        meta_payload['path'] = []

        arrays_to_use = context_arrays
        for trial_id in range(num_trials):
            trial_time = mat_dict['time'][trial_id]
            trial_spikes = spikes[trial_id]
            if use_vel:
                trial_vel = marker_data[trial_id]['velocity'][:,:2] # x and y, mm/ms -> m/s for consistency with other monkey tasks
                nan_mask = np.isnan(trial_vel[:,0])
                marker_time = marker_data[trial_id]['time']
                marker_time = marker_time[~nan_mask]
                trial_vel = trial_vel[~nan_mask]
                # assumes continuitiy, i.e. nan mask only cropping ends
                intersect_time = np.intersect1d(trial_time, marker_time)
                # subset both spikes and vel to the same time
                trial_spikes = trial_spikes[np.isin(trial_time, intersect_time)]
                trial_vel = trial_vel[np.isin(marker_time, intersect_time)]

                # downsample
                if trial_vel.shape[0] % int(cfg.bin_size_ms) != 0:
                    # crop beginning
                    trial_vel = trial_vel[int(cfg.bin_size_ms) - (trial_vel.shape[0] % int(cfg.bin_size_ms)):]
                trial_vel = decimate(trial_vel, int(cfg.bin_size_ms / 1), axis=0, zero_phase=True)

            single_payload = {
                DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg, spike_bin_size_ms=1),
            }
            if use_vel:
                single_payload[DataKey.bhvr_vel] = torch.tensor(trial_vel.copy())
            single_path = cache_root / f'{trial_id}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)

ExperimentalTaskRegistry.register_manual(ExperimentalTask.marino_batista_mp_bci, MarinoBatistaLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.marino_batista_mp_iso_force, MarinoBatistaLoader)