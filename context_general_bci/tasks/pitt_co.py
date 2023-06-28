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
# from scipy.signal import convolve
import torch.nn.functional as F
from einops import rearrange, reduce

import logging
logger = logging.getLogger(__name__)
try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from context_general_bci.config import DataKey, DatasetConfig, PittConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry


CLAMP_MAX = 15

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


def load_trial(fn, use_ql=True, key='data', copy_keys=True):
    # if `use_ql`, use the prebinned at 20ms and also pull out the kinematics
    # else take raw spikes
    # data = payload['data'] # 'data' is pre-binned at 20ms, we'd rather have more raw
    payload = loadmat(str(fn), simplify_cells=True, variable_names=[key] if use_ql else ['iData'])
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
        elif 'pos' in payload:
            out['position'] = torch.from_numpy(payload['pos'][:,1:3]) # 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
        out['position'] = out['position'].roll(1, dims=1) # Pitt position logs in robot coords, i.e. y, dim 1 is up/down in cursor space, z, dim 2 is left/right in cursor space. Roll so we have x, y
        if 'target' in payload:
            out['target'] = torch.from_numpy(payload['target'][1:3].T) # dimensions flipped here, originally C x T
            out['target'] = out['target'].roll(1, dims=1) # Pitt position logs in robot coords, i.e. y, dim 1 is up/down in cursor space, z, dim 2 is left/right in cursor space. Roll so we have x, y
    else:
        data = payload['iData']
        trial_data = extract_ql_data(data['QL']['Data'])
        out['src_file'] = data['QL']['FileName']
        out['spikes'] = events_to_raster(trial_data)
    if copy_keys:
        for k in payload:
            if k not in out and k not in ['SpikeCount', 'trial_num', 'Kinematics', 'pos', 'target', 'QL', 'iData', 'data']:
                out[k] = payload[k]
    return out

@ExperimentalTaskRegistry.register
class PittCOLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.pitt_co
    r"""
    Churchland/Kaufman reaching data, from gdrive. Assorted extra sessions that don't overlap with DANDI release.

    List of IDs
    # - register, make task etc

    """

    @staticmethod
    def get_velocity(position, kernel=np.ones((int(500 / 20), 2))/ (500 / 20)):
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # position = gaussian_filter1d(position, 2.5, axis=0) # This seems reasonable, but useless since we can't compare to Pitt codebase without below
        int_position = pd.Series(position.flatten()).interpolate()
        position = torch.tensor(int_position).view(-1, position.shape[-1])
        position = F.conv1d(position.T.unsqueeze(1), torch.tensor(kernel).float().T.unsqueeze(1), padding='same')[:,0].T
        vel = torch.as_tensor(np.gradient(position.numpy(), axis=0)).float() # note gradient preserves shape

        # position = pd.Series(position.flatten()).interpolate().to_numpy().reshape(-1, 2) # remove intermediate nans
        # position = convolve(position, kernel, mode='same')
        # vel = torch.tensor(np.gradient(position, axis=0)).float()
        # position = convolve(position, kernel, mode='same') # Nope. this applies along both dimensions. Facepalm.

        vel[vel.isnan()] = 0 # extra call to deal with edge values
        return vel

    @staticmethod
    def ReFIT(positions: torch.Tensor, goals: torch.Tensor, reaction_lag_ms=200, bin_ms=20) -> torch.Tensor:
        # positions, goals: Time x Hidden.
        # defaults for lag experimented in `pitt_scratch`
        empirical = PittCOLoader.get_velocity(positions)
        oracle = goals.roll(reaction_lag_ms // bin_ms) - positions
        magnitudes = torch.linalg.norm(empirical, dim=1)  # Compute magnitudes of original velocities
        angles = torch.atan2(oracle[:, 1], oracle[:, 0])  # Compute angles of velocities

        # Clip velocities with magnitudes below threshold to 0
        # mask = (magnitudes < thresh)
        # magnitudes[mask] = 0.0

        new_velocities = torch.stack((magnitudes * torch.cos(angles), magnitudes * torch.sin(angles)), dim=1)
        new_velocities[:reaction_lag_ms // bin_ms] = torch.nan  # Replace clipped velocities with original ones, for rolled time periods
        # new_velocities[reaction_lag_ms // bin_ms:] = empirical[reaction_lag_ms // bin_ms:]  # Replace clipped velocities with original ones, for rolled time periods
        return new_velocities

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

        if not datapath.is_dir() and datapath.suffix == '.mat': # payload style, preproc-ed/binned elsewhere
            payload = load_trial(datapath, key='thin_data')

            # Sanitize
            spikes = payload['spikes']
            # elements = spikes.nelement()
            unique, counts = np.unique(spikes, return_counts=True)
            for u, c in zip(unique, counts):
                if u >= CLAMP_MAX:
                    spikes[spikes == u] = CLAMP_MAX # clip
                # if u >= 15 or c / elements < 1e-5: # anomalous, suppress to max. (Some bins randomly report impossibly high counts like 90 (in 20ms))
                    # spikes[spikes == u] = 0

            if task == ExperimentalTask.unstructured:  # dont' bother with trial structure
                spikes = chop_vector(spikes)
                for i, trial_spikes in enumerate(spikes):
                    save_trial_spikes(trial_spikes, i)
            else:
                # Iterate by trial, assumes continuity so we grab velocity outside
                # start_pad = round(500 / cfg.bin_size_ms)
                # end_pad = round(1000 / cfg.bin_size_ms)
                # should_clip = False
                exp_task_cfg: PittConfig = getattr(cfg, task.value)

                if (
                    'position' in payload and \
                    task in [ExperimentalTask.observation, ExperimentalTask.ortho, ExperimentalTask.fbc] # and \
                ): # We only "trust" in the labels provided by obs (for now)
                    if len(payload['position']) == len(payload['trial_num']):
                        if exp_task_cfg.closed_loop_intention_estimation == "refit":
                            # breakpoint()
                            session_vel = PittCOLoader.ReFIT(payload['position'], payload['target'], bin_ms=cfg.bin_size_ms)
                        else:
                            session_vel = PittCOLoader.get_velocity(payload['position'])
                        # if session_vel[-end_pad:].abs().max() < 0.001: # likely to be a small bump to reset for next trial.
                        #     should_clip = True
                    else:
                        session_vel = None
                else:
                    session_vel = None
                if exp_task_cfg.respect_trial_boundaries:
                    for i in payload['trial_num'].unique():
                        trial_spikes = payload['spikes'][payload['trial_num'] == i]
                        # trim edges -- typically a trial starts with half a second of inter-trial and ends with a second of failure/inter-trial pad
                        # we assume intent labels are not reliable in this timeframe
                        # if trial_spikes.size(0) <= start_pad + end_pad: # something's odd about this trial
                        #     continue
                        if session_vel is not None:
                            trial_vel = session_vel[payload['trial_num'] == i]
                        # if should_clip:
                        #     trial_spikes = trial_spikes[start_pad:-end_pad]
                        #     if session_vel is not None:
                        #         trial_vel = trial_vel[start_pad:-end_pad]
                        if trial_spikes.size(0) < 10:
                            continue
                        if trial_spikes.size(0) < round(exp_task_cfg.chop_size_ms / cfg.bin_size_ms):
                            save_trial_spikes(trial_spikes, i, {DataKey.bhvr_vel: trial_vel} if session_vel is not None else {})
                        else:
                            chopped_spikes = chop_vector(trial_spikes)
                            if session_vel is not None:
                                chopped_vel = chop_vector(trial_vel)
                            for j, subtrial_spikes in enumerate(chopped_spikes):
                                save_trial_spikes(subtrial_spikes, f'{i}_trial{j}', {DataKey.bhvr_vel: chopped_vel[j]} if session_vel is not None else {})

                            end_of_trial = trial_spikes.size(0) % round(exp_task_cfg.chop_size_ms / cfg.bin_size_ms)
                            if end_of_trial > 10:
                                trial_spikes_end = trial_spikes[-end_of_trial:]
                                if session_vel is not None:
                                    trial_vel_end = trial_vel[-end_of_trial:]
                                save_trial_spikes(trial_spikes_end, f'{i}_end', {DataKey.bhvr_vel: trial_vel_end} if session_vel is not None else {})
                else:
                    # chop both
                    spikes = chop_vector(spikes)
                    if session_vel is not None:
                        session_vel = chop_vector(session_vel)
                    for i, trial_spikes in enumerate(spikes):
                        save_trial_spikes(trial_spikes, i, {DataKey.bhvr_vel: session_vel[i]} if session_vel is not None else {})
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
                        single_payload[DataKey.bhvr_vel] = PittCOLoader.get_velocity(payload['position'])
                    single_path = cache_root / f'{i}.pth'
                    meta_payload['path'].append(single_path)
                    torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)


# Register aliases
ExperimentalTaskRegistry.register_manual(ExperimentalTask.observation, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.ortho, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.fbc, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.unstructured, PittCOLoader)
ExperimentalTaskRegistry.register_manual(ExperimentalTask.pitt_co, PittCOLoader)