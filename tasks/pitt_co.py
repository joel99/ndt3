#%%
from typing import List
from pathlib import Path
import math
import numpy as np
import torch
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve
from einops import rearrange, reduce

import logging
logger = logging.getLogger(__name__)
try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from ..config import DataKey, DatasetConfig
from ..subjects import SubjectInfo, create_spike_payload
from ..tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry



r"""
    Dev note to self: Pretty unclear how the .mat payloads we're transferring seem to be _smaller_ than n_element bytes. The output spike trials, ~250 channels x ~100 timesteps are reasonably, 25K. But the data is only ~10x this for ~100x the trials.
"""

def extract_ql_data(ql_data):
    # ql_data: .mat['iData']['QL']['Data']
    # Currently just equipped to extract spike snippets
    # If you want more, look at `icms_modeling/scripts/preprocess_mat`
    # print(ql_data.keys())
    # print(ql_data['TASK_STATE_CONFIG'].keys())
    # print(ql_data['TASK_STATE_CONFIG']['state_num'])
    # print(ql_data['TASK_STATE_CONFIG']['state_name'])
    # print(ql_data['TRIAL_METADATA'])
    def extract_spike_snippets(spike_snippets):
        THRESHOLD_SAMPLE = 12./30000
        return {
            "spikes_source_index": spike_snippets['source_index'], # JY: I think this is NSP box?
            "spikes_channel": spike_snippets['channel'],
            "spikes_source_timestamp": spike_snippets['source_timestamp'] + THRESHOLD_SAMPLE,
            # "spikes_snippets": spike_snippets['snippet'], # for waveform
        }

    return {
        **extract_spike_snippets(ql_data['SPIKE_SNIPPET']['ss'])
    }

def events_to_raster(
    events,
    channels_per_array=128,
):
    """
        Tensorize sparse format.
    """
    events['spikes_channel'] = events['spikes_channel'] + events['spikes_source_index'] * channels_per_array
    bins = np.arange(
        events['spikes_source_timestamp'].min(),
        events['spikes_source_timestamp'].max(),
        0.001
    )
    timebins = np.digitize(events['spikes_source_timestamp'], bins, right=False) - 1
    spikes = torch.zeros((len(bins), 256), dtype=torch.uint8)
    spikes[timebins, events['spikes_channel']] = 1
    return spikes


def load_trial(fn, use_ql=True, key='data'):
    # if `use_ql`, use the prebinned at 20ms and also pull out the kinematics
    # else take raw spikes
    # data = payload['data'] # 'data' is pre-binned at 20ms, we'd rather have more raw
    payload = loadmat(fn, simplify_cells=True, variable_names=[key] if use_ql else ['iData'])
    out = {
        'bin_size_ms': 20 if use_ql else 1,
        'use_ql': use_ql,
    }
    if use_ql:
        payload = payload[key]
        spikes = payload['SpikeCount']
        if spikes.shape[1] == 256 * 5:
            standard_channels = np.arange(0, 256 * 5,5) # unsorted, I guess
            spikes = spikes[..., standard_channels]
        out['spikes'] = torch.from_numpy(spikes)
        out['trial_num'] = torch.from_numpy(payload['trial_num'])
        if 'Kinematics' in payload:
            # cursor x, y
            out['position'] = torch.from_numpy(payload['Kinematics']['ActualPos'][:,1:3]) # 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
            out['position'] = out['position'].roll(1, dims=1) # swap x and y
        elif 'pos' in payload:
            out['position'] = torch.from_numpy(payload['pos'][:,1:3]) # 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
            out['position'] = out['position'].roll(1, dims=1) # swap x and y
    else:
        data = payload['iData']
        trial_data = extract_ql_data(data['QL']['Data'])
        out['src_file'] = data['QL']['FileName']
        out['spikes'] = events_to_raster(trial_data)
    return out

@ExperimentalTaskRegistry.register
class PittCOLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.pitt_co
    r"""
    Churchland/Kaufman reaching data, from gdrive. Assorted extra sessions that don't overlap with DANDI release.

    List of IDs
    # - register, make task etc

    """

    @classmethod
    def load(
        cls,
        datapath: Path, # path to matlab file
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        task: ExperimentalTask,
    ):
        assert cfg.bin_size_ms == 20, 'code not prepped for different resolutions'

        meta_payload = {}
        meta_payload['path'] = []
        arrays_to_use = context_arrays
        def chop_vector(vec: torch.Tensor):
            # vec - already at target resolution, just needs chopping
            chop_size = round(cfg.pitt_co.chop_size_ms / cfg.bin_size_ms)
            return rearrange(
                vec.unfold(0, chop_size, chop_size),
                'trial hidden time -> trial time hidden'
             ) # Trial x C x chop_size (time)
        def save_trial_spikes(spikes, i, other_data={}):
            single_payload = {
                DataKey.spikes: create_spike_payload(
                    spikes.clone(), arrays_to_use
                ),
                **other_data
            }
            single_path = cache_root / f'{dataset_alias}_{i}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        def get_velocity(position):
            # position = gaussian_filter1d(position, 2.5, axis=0) # This seems reasonable, but useless since we can't compare to Pitt codebase without below
            position = pd.Series(position.flatten()).interpolate().to_numpy().reshape(-1, 2) # remove intermediate nans
            # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
            position = convolve(position, np.ones((int(500 / cfg.bin_size_ms), 2))/ (500 / cfg.bin_size_ms), mode='same')

            vel = torch.tensor(np.gradient(position, axis=0)).float()
            vel[vel.isnan()] = 0 # extra call to deal with edge values
            return vel
        if not datapath.is_dir() and datapath.suffix == '.mat': # payload style, preproc-ed/binned elsewhere
            payload = load_trial(datapath, key='thin_data')

            # Sanitize
            spikes = payload['spikes']
            elements = spikes.nelement()
            unique, counts = np.unique(spikes, return_counts=True)
            for u, c in zip(unique, counts):
                if u >= 15 or c / elements < 1e-5: # anomalous, suppress. (Some bins randomly report impossibly high counts like 90 (in 20ms))
                    spikes[spikes == u] = 0

            if task == ExperimentalTask.unstructured:  # dont' bother with trial structure
                spikes = chop_vector(spikes)
                for i, trial_spikes in enumerate(spikes):
                    save_trial_spikes(trial_spikes, i)
            else:
                # Iterate by trial, assumes continuity
                for i in payload['trial_num'].unique():
                    session_spikes = payload['spikes'][payload['trial_num'] == i]

                    start_pad = round(500 / cfg.bin_size_ms)
                    end_pad = round(1000 / cfg.bin_size_ms)
                    if session_spikes.size(0) <= start_pad + end_pad: # something's odd about this trial
                        continue
                    # trim edges -- typically a trial starts with half a second of inter-trial and ends with a second of failure/inter-trial pad
                    session_spikes = session_spikes[start_pad:-end_pad]
                    if 'position' in payload and task in [ExperimentalTask.observation]: # We only "trust" in the labels provided by obs (for now). Note we previously tried ortho to no avail, and loading is buggier.
                    # if 'position' in payload and task in [ExperimentalTask.observation, ExperimentalTask.ortho]: # We only "trust" in the labels provided by obs (for now)
                        session_vel = get_velocity(payload['position'][payload['trial_num'] == i])[start_pad:-end_pad]
                    else:
                        session_vel = None
                    if session_spikes.size(0) < round(cfg.pitt_co.chop_size_ms / cfg.bin_size_ms):
                        save_trial_spikes(session_spikes, i, {DataKey.bhvr_vel: session_vel} if session_vel is not None else {})
                    else:
                        end_of_trial = session_spikes.size(0) % round(cfg.pitt_co.chop_size_ms / cfg.bin_size_ms)
                        if end_of_trial > 10: # some arbitrary small threshold...
                            session_spikes_end = session_spikes[-end_of_trial:]
                            if session_vel is not None:
                                session_vel_end = session_vel[-end_of_trial:]
                            save_trial_spikes(session_spikes_end, f'{i}_end', {DataKey.bhvr_vel: session_vel_end} if session_vel is not None else {})

                        session_spikes = chop_vector(session_spikes)
                        if session_vel is not None:
                            session_vel = chop_vector(session_vel)
                        for j, subtrial_spikes in enumerate(session_spikes):
                            save_trial_spikes(subtrial_spikes, f'{i}_trial{j}', {DataKey.bhvr_vel: session_vel[j]} if session_vel is not None else {})

        else: # folder style, preproc-ed on mind
            for i, fname in enumerate(datapath.glob("*.mat")):
                if fname.stem.startswith('QL.Task'):
                    payload = load_trial(fname)
                    single_payload = {
                        DataKey.spikes: create_spike_payload(
                            payload['spikes'], arrays_to_use, cfg, payload['bin_size_ms']
                        ),
                    }
                    if 'position' in payload:
                        single_payload[DataKey.bhvr_vel] = get_velocity(payload['position'])
                    single_path = cache_root / f'{i}.pth'
                    meta_payload['path'].append(single_path)
                    torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)


# Register aliases
ExperimentalTaskRegistry.register_manual(ExperimentalTask.observation, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.ortho, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.fbc, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.unstructured, PittCOLoader)