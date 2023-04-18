#%%
from typing import List
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from einops import rearrange, reduce

import logging
logger = logging.getLogger(__name__)

try:
    from pynwb import NWBHDF5IO
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry



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
    ):
        meta_payload = {}
        meta_payload['path'] = []
        arrays_to_use = context_arrays
        for i, fname in enumerate(datapath.glob("*.mat")):
            if fname.stem.startswith('QL.Task'):
                payload = load_trial(fname)
                trial_spikes = payload['spikes']
                single_payload = {
                    DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg=cfg, spike_bin_size_ms=payload['bin_size_ms']),
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
