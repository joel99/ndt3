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
from typing import List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
from einops import reduce, rearrange
import scipy.signal as signal

from context_general_bci.utils import loadmat
from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

def flatten_single(channel_spikes, offsets): # offset in seconds
    # print(channel_spikes)
    filtered = [spike + offset for spike, offset in zip(channel_spikes, offsets) if spike is not None]
    filtered = [spike if len(spike.shape) > 0 else np.array([spike]) for spike in filtered]
    return np.concatenate(filtered)

@ExperimentalTaskRegistry.register
class RouseLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.rouse

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
        breakpoint()
        payload = loadmat(datapath)
        trial_starts = payload['TrialInfo']['trial_start_time'] # Covariate sampled at 100Hz
        trial_spikes = payload['SpikeTimes']
        trial_spikes = [s[0] for s in trial_spikes]
        spikes = [flatten_single(channel, trial_starts) for channel in trial_spikes]
        vel = payload['JoystickPos_disp']
        def resample(data, covariate_rate=100): # Updated 9/10/23: Previous resample produces an undesirable strong artifact at timestep 0. This hopefully removes that and thus also removes outliers.
            base_rate = int(1000 / cfg.bin_size_ms)
            # print(base_rate, covariate_rate, base_rate / covariate_rate)
            return torch.tensor(
                signal.resample_poly(data, base_rate, covariate_rate, padtype='line')
            )
        # TODO vel has NaNs, TODO vel unlikely to exactly match trial spikes timeframe
        # TODO trial spikes is sparse, not dense
        breakpoint()
        vel = resample(vel)

        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}

        if cfg.tokenize_covariates:
            canonical_labels = REACH_DEFAULT_KIN_LABELS
            global_args[DataKey.covariate_labels] = canonical_labels

        def chop_vector(vec: torch.Tensor):
            # vec - already at target resolution, just needs chopping
            chops = round(cfg.miller.chop_size_ms / cfg.bin_size_ms)
            return rearrange(
                vec.unfold(0, chops, chops),
                'trial hidden time -> trial time hidden'
             ) # Trial x C x chop_size (time)

        if cfg.rouse.minmax:
            # Aggregate velocities and get min/max
            global_args['cov_mean'] = vel.mean(0)
            global_args['cov_min'] = torch.quantile(vel, 0.001, dim=0) # sufficient in a quick notebook check.
            global_args['cov_max'] = torch.quantile(vel, 0.999, dim=0)
            rescale = global_args['cov_max'] - global_args['cov_min']
            rescale[torch.isclose(rescale, torch.tensor(0.))] = 1
            vel = (vel - global_args['cov_mean']) / rescale
            vel = torch.clamp(vel, -1, 1)

        # Directly chop trialized data as though continuous - borrowing from LM convention
        vel = chop_vector(vel) # T x H
        full_spikes = chop_vector(torch.as_tensor(spikes, dtype=torch.float))
        assert full_spikes.size(0) == vel.size(0), "Chop size mismatch"
        for t in range(full_spikes.size(0)):
            single_payload = {
                DataKey.spikes: create_spike_payload(full_spikes[t], context_arrays),
                DataKey.bhvr_vel: vel[t].clone(), # T x H
                **global_args,
            }
            single_path = cache_root / f'{t}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)