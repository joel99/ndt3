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
from context_general_bci.tasks.preproc_utils import chop_vector, compress_vector, get_minmax_norm

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
        r"""
            We are currently assuming that the start of covariate data is the same as time=0 for trial SpikeTimes
            TODO do we want a <break> token between trials? For now no.
        """
        version_ksu = 'ksu' in dataset_alias # New batch, where position is labeled differently and there are more arrays, see `rouse_viewer2`

        payload = loadmat(datapath)
        channel_trial_spikes = payload['SpikeTimes']
        if not version_ksu:
            channel_trial_spikes = [s[0] for s in channel_trial_spikes]
            pos = payload['JoystickPos_disp'] # Trial x Time x Dim
            # spikes = [flatten_single(channel, trial_starts) for channel in trial_spikes]
        else:
            pos = payload['CursorPos']
        vel = np.gradient(pos, axis=1)
        vel = torch.tensor(
            signal.resample_poly(vel, 10, cfg.bin_size_ms, padtype='line', axis=1), # Default 100Hz
            dtype=torch.float32
        ) # Trial x Time x Dim
        # Note this introduces NaNs at the start of trial, which we just accept for simplicity
        all_vels = []
        all_spikes = []
        # Compact trialized representations
        # We'll take the periods of time when velocity is valid, and build a spike matrix for exactly these times.
        # Then we'll concatenate everything and chop it up into chunks.
        for trial, trial_vel in enumerate(vel): # Time x Dim
            # Find the longest continuous span of non-nan
            trial_spikes = [s[trial] for s in channel_trial_spikes] # Outer is channel, inner is trial
            # For now we assume that the trial spike time is given wrt exact same time as velocity[0]
            valid_times_at_bin_rate = ~torch.isnan(trial_vel).any(1).to(dtype=bool) # Time
            diffs = torch.diff(torch.cat((torch.tensor([False]), valid_times_at_bin_rate, torch.tensor([False]))).to(dtype=int))
            # Find start and end indices
            starts = torch.where(diffs == 1)[0]
            ends = torch.where(diffs == -1)[0]

            # Calculate span lengths and find the longest
            lengths = ends - starts
            longest_span_idx = torch.argmax(lengths)

            # Endpoints of the longest span
            start_longest_span = starts[longest_span_idx]
            end_longest_span = ends[longest_span_idx]
            all_vels.append(trial_vel[start_longest_span:end_longest_span])

            cue_time_start = start_longest_span * cfg.bin_size_ms
            cue_time_end = end_longest_span * cfg.bin_size_ms
            # Create at ms resolution
            trial_spikes_dense = torch.zeros(len(trial_spikes), cue_time_end - cue_time_start, dtype=torch.uint8)
            for channel, channel_spikes in enumerate(trial_spikes):
                if isinstance(channel_spikes, float) or isinstance(channel_spikes, int):
                    channel_spikes = np.array([channel_spikes])
                if channel_spikes is None or len(channel_spikes) == 0:
                    continue
                channel_spikes_ms = torch.as_tensor(channel_spikes * 1000, dtype=int)
                channel_spikes_ms = (channel_spikes_ms[(channel_spikes_ms >= cue_time_start) & (channel_spikes_ms < cue_time_end)] - cue_time_start)
                trial_spikes_dense[channel] = torch.bincount(channel_spikes_ms, minlength=trial_spikes_dense.shape[1])
            all_spikes.append(trial_spikes_dense)
        dense_spikes = torch.cat([compress_vector(s.T, 0, cfg.bin_size_ms) for s in all_spikes]) # Time x Channel x 1, at bin res
        vel = torch.cat(all_vels) # Time x Dim, at bins

        meta_payload = {}
        meta_payload['path'] = []
        global_args = {}

        if cfg.tokenize_covariates:
            canonical_labels = REACH_DEFAULT_KIN_LABELS
            global_args[DataKey.covariate_labels] = canonical_labels

        if cfg.rouse.minmax:
            vel, payload_norm = get_minmax_norm(vel, cfg.rouse.center)
            global_args.update(payload_norm)

        # Directly chop trialized data as though continuous - borrowing from LM convention
        vel = chop_vector(vel, cfg.rouse.chop_size_ms, cfg.bin_size_ms) # T x H
        full_spikes = chop_vector(dense_spikes[..., 0], cfg.rouse.chop_size_ms, cfg.bin_size_ms).unsqueeze(-1)
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