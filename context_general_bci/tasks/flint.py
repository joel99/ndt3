r"""
From CRCNS DREAM - under Flint 2012. README below.
https://portal.nersc.gov/project/crcns/download/dream/data_sets/Flint_2012

Ben Walker, February 2013
ben-walker@northwestern.edu
__________________________________________
Experiment list
------------------------------------------
All are monkey C, center out
_e1: Same data as Stevenson_2011_e1, subject 1
_e2: Same day, subjects are different recording sessions
_e3: Same day, subjects are different recording sessions

_e4: Same day, subjects are different recording sessions
_e5: Same day, subjects are different recording sessions

__________________________________________
Data Comments:
------------------------------------------
The Neuron structure in the trial field has timestamps of nuerons that fire.
The EMG is for Biceps, Triceps, Anterior and Posterior Deltoids
LFPs are also present.

Subject 1 did not record EMGs.

File 2 was recorded on July 19, 2010.  File 3 was July 20.  File 4 was Aug 31.
File 5 was Sept 1.  I'm not sure when File 1 was recorded.
__________________________________________
Notes:
------------------------------------------
The first subject of data here is also the first subject for the
Stevenson_2011_e1 data set.  Each experiment is one day's worth of recording.
Each 'Subject' is a different recording session on the same day.

All the data is for one monkey, monkey C in the publication.

"Good" trials are ones that had a target off event.  They still might not
have a completed reach from target to target, however.

The target position was not recorded during the experiment.  It has been
estimated and added in.

Data format:
- Spike times, covariates at 100Hz
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
from context_general_bci.config import DataKey, DatasetConfig, ExperimentalConfig, REACH_DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import chop_vector, compress_vector, PackToChop, spike_times_to_dense, get_minmax_norm

def flatten_single(channel_spikes, offsets): # offset in seconds
    # print(channel_spikes)
    filtered = [spike + offset for spike, offset in zip(channel_spikes, offsets) if spike is not None]
    filtered = [spike if len(spike.shape) > 0 else np.array([spike]) for spike in filtered]
    return np.concatenate(filtered)

@ExperimentalTaskRegistry.register
class FlintLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.flint

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
        **kwargs,
    ):
        r"""
            We are currently assuming that the start of covariate data is the same as time=0 for trial SpikeTimes
            TODO do we want a <break> token between trials? For now no.
        """
        exp_task_cfg: ExperimentalConfig = getattr(cfg, task.value)
        payload = loadmat(datapath)
        payload = payload['Subject']['Trial']
        breakpoint()
        all_vels = []
        all_spikes = []
        for trial_data in payload:
            trial_times = trial_data['Time'] # in seconds
            trial_vel = np.array(trial_data['HandVel'])[:,:2] # Last dim is empty
            trial_vel = torch.tensor(
                signal.resample_poly(trial_vel, 10, cfg.bin_size_ms, padtype='line', axis=1), # Default 100Hz
                dtype=torch.float32
            ) # Time x Dim
            all_vels.append(trial_vel)
            spike_times = [(np.array(t['Spikes']) - trial_times[0]) * 1000 for t in trial_data['Neuron']] # List of channel spike times, in ms from trial start

            all_spikes.append(spike_times_to_dense(spike_times, cfg.bin_size_ms, time_end=int(trial_times[-1] * 1000) + 10)) # Timebins x Channel x 1, at bin res
        global_vel = torch.cat(all_vels) # Time x Dim, at bins
        global_spikes = torch.cat(all_spikes) # Timebins x Channel x 1, at bin res

        if cfg.pack_dense:
            packer = PackToChop(exp_task_cfg.chop_size_ms // cfg.bin_size_ms, cache_root)

        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}
        if cfg.tokenize_covariates:
            global_args[DataKey.covariate_labels] = REACH_DEFAULT_KIN_LABELS

        if exp_task_cfg.minmax:
            global_vel, norm_dict = get_minmax_norm(global_vel)
            global_args.update(norm_dict)
        # Hm... hard to use packer individually since we normalize in external function

        # Directly chop trialized data as though continuous - borrowing from LM convention
        global_vel = chop_vector(global_vel, exp_task_cfg.chop_size_ms, cfg.bin_size_ms) # T x H
        global_spikes = chop_vector(global_spikes[..., 0], exp_task_cfg.chop_size_ms, cfg.bin_size_ms).unsqueeze(-1)
        assert global_spikes.size(0) == global_vel.size(0), "Chop size mismatch"
        for t in range(global_spikes.size(0)):
            single_payload = {
                DataKey.spikes: create_spike_payload(global_spikes[t], context_arrays),
                DataKey.bhvr_vel: global_vel[t].clone(), # T x H
                **global_args,
            }
            if cfg.pack_dense:
                packer.pack(single_payload)
            else:
                single_path = cache_root / f'{t}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        if cfg.pack_dense:
            packer.flush()
            meta_payload['path'] = packer.get_paths()
        return pd.DataFrame(meta_payload)