#%%
r"""
Gallego CO release (10/29/22)
https://doi.org/10.5061/dryad.xd2547dkt

```
In these experiments, monkeys controlled a cursor on a screen using a two-link, planar manipulandum. Monkeys performed a simple center-out task to one of the eight possible targets, after a variable delayed period. During this reaching task, we tracked the endpoint position of the hand using sensors on the manipulandum. In addition to the behavioral data, we collected neural data from one or two of these areas using Blackrock Utah multielectrode arrays, yielding ~100 to ~200 channels of extracellular recordings per monkey. Recordings from these channels were thresholded online to detect spikes, which were sorted offline into putative single units.
```

Data was pulled by manual wget + unzip. (no CLI)

This loader requires the PyalData package
https://github.com/NeuralAnalysis/PyalData
(and likely mat73)
`pip install mat73`
"""
from typing import List
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import torch
try:
    import pyaldata
except:
    logging.info("Pyaldata not installed, please install from https://github.com/NeuralAnalysis/PyalData. Import will fail")

from einops import reduce
from scipy.signal import decimate

from context_general_bci.config import DataKey, DatasetConfig
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry


@ExperimentalTaskRegistry.register
class GallegoCOLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.gallego_co

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
        df: pd.DataFrame = pyaldata.mat2dataframe(datapath, shift_idx_fields=True)
        assert cfg.bin_size_ms % (df.bin_size[0] * 1000) == 0, "bin_size_ms must divide bin_size in the data"
        chop_bins = int(cfg.bin_size_ms / (df.bin_size[0] * 1000))

        def compress_spikes(spikes):
            # make sure we divide evenly
            if spikes.shape[0] % chop_bins != 0:
                spikes = spikes[(spikes.shape[0] % chop_bins):]
            return reduce(spikes, '(t b) h -> t h 1', 'sum', b=chop_bins)

        def compress_vel(vel):
            vel = vel / 100 # cm/s -> m/s
            # Technically we just really need an actual data-based normalizer...
            if vel.shape[0] % chop_bins != 0:
                vel = vel[(vel.shape[0] % chop_bins):]
            vel = torch.tensor(decimate(vel, chop_bins, axis=0).copy(), dtype=torch.float)
            return torch.nan_to_num(vel) # some nan's found, data seems mostly good though? In general probably don't use this velocity

        meta_payload = {}
        meta_payload['path'] = []

        arrays_to_use = context_arrays
        for trial_id in range(len(df)):
            spike_payload = {}
            for array in arrays_to_use:
                if f'{SubjectInfo.unwrap_array(array)}_spikes' in df.columns:
                    spike_payload[array] = torch.tensor(df[f'{SubjectInfo.unwrap_array(array)}_spikes'][trial_id], dtype=torch.uint8)
                    spike_payload[array] = compress_spikes(spike_payload[array])
            single_payload = {
                DataKey.spikes: spike_payload,
                DataKey.bhvr_vel: compress_vel(df.vel[trial_id]), # T x H
            }
            single_path = cache_root / f'{trial_id}.pth'
            meta_payload['path'].append(single_path)
            torch.save(single_payload, single_path)
        return pd.DataFrame(meta_payload)
