#%%
from typing import List
from pathlib import Path
import math
import numpy as np
import torch
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.interpolate import interp1d
from scipy.io import loadmat

from config import DataKey, DatasetConfig
from subjects import SubjectInfo, SubjectArrayRegistry
from tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from einops import rearrange, reduce

import logging

logger = logging.getLogger(__name__)

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


def load_trial(fn, use_ql=True):
    # if `use_ql`, use the prebinned at 20ms and also pull out the kinematics
    # else take raw spikes
    # data = payload['data'] # 'data' is pre-binned at 20ms, we'd rather have more raw
    payload = loadmat(fn, simplify_cells=True, variable_names=['data'] if use_ql else ['iData'])
    out = {
        'bin_size_ms': 20 if use_ql else 1,
        'use_ql': use_ql,
    }
    if use_ql:
        standard_channels = np.arange(0, 256 * 5,5) # unsorted, I guess
        spikes = payload['data']['SpikeCount'][..., standard_channels]
        out['spikes'] = torch.from_numpy(spikes)
        # cursor x, y
        out['position'] = torch.from_numpy(payload['data']['Kinematics']['ActualPos'][:,2:4])
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
    ):
        meta_payload = {}
        meta_payload['path'] = []
        arrays_to_use = context_arrays
        for i, fname in enumerate(datapath.glob("*.mat")):
            if fname.stem.startswith('QL.Task'):
                payload = load_trial(fname)
                trial_spikes = payload['spikes']
                # import pdb;pdb.set_trace()
                assert cfg.bin_size_ms % payload['bin_size_ms'] == 0
                bin_factor = cfg.bin_size_ms // payload['bin_size_ms']
                # crop
                trial_spikes = trial_spikes[len(trial_spikes) % bin_factor:]
                trial_spikes = rearrange(
                    trial_spikes,
                    '(t bin) c -> t bin c', bin=bin_factor
                )
                trial_spikes = reduce(trial_spikes, 't bin c -> t c 1', 'sum')
                trial_spikes = trial_spikes.to(dtype=torch.uint8)
                spike_payload = {}
                for a in arrays_to_use:
                    array = SubjectArrayRegistry.query_by_array(a)
                    if array.is_exact:
                        array = SubjectArrayRegistry.query_by_array_geometric(a)
                        spike_payload[a] = trial_spikes[:, array.as_indices()].clone()
                    else:
                        assert len(arrays_to_use) == 1, "Can't use multiple arrays with non-exact arrays"
                        spike_payload[a] = trial_spikes.clone()
                single_payload = {
                    DataKey.spikes: spike_payload,
                }
                if 'position' in payload:
                    position = payload['position']
                    vel = torch.tensor(np.gradient(position, axis=0)).float()
                    vel[vel.isnan()] = 0
                    assert cfg.bin_size_ms == 20, 'check out rtt code for resampling'
                    single_payload[DataKey.bhvr_vel] = vel
                single_path = cache_root / f'{i}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)