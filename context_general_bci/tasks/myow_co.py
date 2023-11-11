#%%
# ! Not updated for NDT3 processing (with tokenized covariates etc) due to low volume.
r"""
A handful of clearly sorted ~700 trials from MYOW
https://github.com/nerdslab/myow-neuro
- Adapted from myow/data/monkey_neural_dataset.py

Data was pulled by git clone and copy of raw data folder.
"""
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from context_general_bci.utils import loadmat

from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import PackToChop, get_minmax_norm

import logging

logger = logging.getLogger(__name__)
DYER_DEFAULT_KIN_LABELS = [*REACH_DEFAULT_KIN_LABELS, 'fx', 'fy']
@ExperimentalTaskRegistry.register
class DyerCOLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.dyer_co

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
        mat_dict = loadmat(datapath)

        # load matrix
        trialtable = mat_dict['trial_table']
        neurons = mat_dict['out_struct']['units']
        # pos = np.array(mat_dict['out_struct']['pos'])
        vel = np.array(mat_dict['out_struct']['vel'])
        # acc = np.array(mat_dict['out_struct']['acc'])
        force = np.array(mat_dict['out_struct']['force'])
        covariates = np.concatenate([vel[:, 1:], force[:, 1:]], axis=1).astype(np.float32)
        time = vel[:, 0]

        num_neurons = len(neurons)
        num_trials = trialtable.shape[0]

        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}
        if cfg.tokenize_covariates:
            global_args[DataKey.covariate_labels] = DYER_DEFAULT_KIN_LABELS
        if cfg.dyer_co.minmax:
            _, payload_norm = get_minmax_norm(covariates, center_mean=cfg.dyer_co.center) # we manually normalize later
            global_args.update(payload_norm)
        print(f"Global args: {global_args}")

        arrays_to_use = context_arrays
        # data_list = {'firing_rates': [], 'position': [], 'velocity': [], 'acceleration': [],
                    #  'force': [], 'labels': [], 'sequence': []}

        if cfg.pack_dense:
            packer = PackToChop(cfg.dyer_co.chop_size_ms // cfg.bin_size_ms, cache_root)
        for trial_id in range(num_trials):
            min_T = trialtable[trial_id, 9]
            max_T = trialtable[trial_id, 12]

            binning_period = cfg.bin_size_ms / 1000
            grid = np.arange(min_T, max_T + binning_period, binning_period)
            grids = grid[:-1]
            gride = grid[1:]
            num_bins = len(grids)

            neurons_binned = np.zeros((num_bins, num_neurons))
            # pos_binned = np.zeros((num_bins, 2))
            covariates_binned = np.zeros((num_bins, 4))
            # acc_binned = np.zeros((num_bins, 2))
            # targets_binned = np.zeros((num_bins,))
            # id_binned = np.arange(num_bins)

            # JY: binning edited to be a bit more efficient
            for k in range(num_bins):
                bin_mask = (time >= grids[k]) & (time <= gride[k])
                # if len(pos) > 0:
                    # pos_binned[k, :] = np.mean(pos[bin_mask, 1:], axis=0)
                covariates_binned[k, :] = covariates[bin_mask].mean(0)
                # if len(acc):
                #     acc_binned[k, :] = np.mean(acc[bin_mask, 1:], axis=0)
                # targets_binned[k] = trialtable[trial_id, 1]
            covariates_binned = torch.as_tensor(covariates_binned)
            if cfg.dyer_co.minmax:
                if global_args['cov_mean'] is not None:
                    covariates_binned = (covariates_binned - global_args['cov_mean']) / (global_args['cov_max'] - global_args['cov_min'])
                else:
                    covariates_binned = covariates_binned / global_args['cov_max']
                covariates_binned = torch.clamp(covariates_binned, -1, 1)
            for i in range(num_neurons):
                spike_times = neurons[i]['ts']
                neurons_binned[:, i] = np.histogram(spike_times, grid)[0]
                # for k in range(num_bins):
                #     bin_mask = (spike_times >= grids[k]) & (spike_times <= gride[k])
                #     neurons_binned[k, i] = np.sum(bin_mask) # / binning_period

            # Kill the mask, we don't want it anymore.
            # filter velocity
            # mask = np.linalg.norm(covariates_binned, 2, axis=1) > cfg.dyer_co.velocity_threshold
            # data_list['firing_rates'].append(neurons_binned[mask])
            # data_list['position'].append(pos_binned[mask])
            # data_list['velocity'].append(vel_binned[mask])
            # data_list['acceleration'].append(acc_binned[mask])
            # data_list['force'].append(force_binned[mask])
            # data_list['labels'].append(targets_binned[mask])
            # data_list['sequence'].append(id_binned[mask])
            single_payload = {
                DataKey.spikes: create_spike_payload(neurons_binned, arrays_to_use),
                DataKey.bhvr_vel: covariates_binned,
                **global_args,
                # DataKey.bhvr_acc: torch.tensor(acc_binned[mask]),
                # DataKey.bhvr_force: torch.tensor(force_binned[mask]),
            }
            if cfg.pack_dense:
                packer.pack(single_payload)
            else:
                single_path = cache_root / f'{trial_id}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        if cfg.pack_dense:
            packer.flush()
            meta_payload['path'] = packer.get_paths()
        return pd.DataFrame(meta_payload)
