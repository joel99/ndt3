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
import h5py

logger = logging.getLogger(__name__)

# Note these comrpise a bunch of different tasks, perhaps worth denoting/splitting them
gdown_ids = {
    # Jenkins, milestone 1 9/2015 -> 1/2016 (vs all in 2009 for DANDI)
    'jenkins': {
        'https://drive.google.com/file/d/1o3X-L7uFH0vVPollVaD64AmDPwc0kXkq/view?usp=share_link',
        'https://drive.google.com/file/d/1MmnXvAMSBvt_eZ8X-CgmOWibHqnk_1vr/view?usp=share_link',
        'https://drive.google.com/file/d/10cXhCDnfDlcwVJc_elZwjQLLsb_d7xYI/view?usp=share_link',
        'https://drive.google.com/file/d/1msGk3H6yPwS4GCcJwZybJFFX6JWcbvYp/view?usp=share_link',
        'https://drive.google.com/file/d/16QJZXe0xQKIVqWV5XhOPDIzPBX1V2CV4/view?usp=share_link',
        'https://drive.google.com/file/d/1pe3gnurM4xY5R9qGQ8ohMi1h2Lv-eJWf/view?usp=share_link',
        'https://drive.google.com/file/d/16QJZXe0xQKIVqWV5XhOPDIzPBX1V2CV4/view?usp=share_link',
        'https://drive.google.com/file/d/1Uxht3GUFdJ9Y0AcyTYfCp7uhwCvX0Ujs/view?usp=share_link',
        'https://drive.google.com/file/d/1hxD7xKu96YEMD8iTuHVF6mSAv-h5xlxG/view?usp=share_link',
    },
    # Reggie, milestone 1 ~ 2017
    'reggie': {
        '151nE5p4OTSwiR7UyW2s9RBMGklYLvYO1',
        '1TFVbWjdTgQ4XgfiRN3ilwfya4LAk9jgB',
        '1m8YxKehWhZlkFn9p9XKk8bWnIhfLy1ja',
        '1-qq1JiOOChq80xasEhtkwUCP3j_b2_v1',
        '1413W9XGLJ2gma1CCXpg1DRDGpl4-uxkG',
        '19euCNYTHipP7IJTGBPtu-4efuShW6qSk',
        '1eePWeHohrhbtBwQg8fJJWjPJDCtWLV3S',
    },
    # Nitschke (9/22-28/2010)
    'nitschke': {
        'https://drive.google.com/file/d/1IHPADrDpwdWEZKVjC1B39NIf_FdSv49k/view?usp=share_link',
        'https://drive.google.com/file/d/1D8KYfy5IwMmEZaKOEv-7U6-4s-7cKINK/view?usp=share_link',
        'https://drive.google.com/file/d/1tp_ezJqvgW5w_e8uNBvrdbGkgf2CP_Sj/view?usp=share_link',
        'https://drive.google.com/file/d/1Im75cmAPuS2dzHJUGw9v5fuy9lx49r6c/view?usp=share_link',
        # skip 9/22 provided in DANDI release
    }
}

if __name__ == '__main__':
    # Pull the various files using `gdown` (pip install gdown)
    # https://github.com/catalystneuro/shenoy-lab-to-nwb
    # -- https://drive.google.com/drive/folders/1mP3MCT_hk2v6sFdHnmP_6M0KEFg1r2g_
    import gdown

    for sid in gdown_ids:
        for gid in gdown_ids[sid]:
            if gid.startswith('http'):
                gid = gid.split('/')[-2]
            if not Path(f'./data/churchland_misc/{sid}-{gid}.mat').exists():
                gdown.download(id=gid, output=f'./data/churchland_misc/{sid}-{gid}.mat', quiet=False)

@ExperimentalTaskRegistry.register
class ChurchlandMiscLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.churchland_misc
    r"""
    Churchland/Kaufman reaching data, from gdrive. Assorted extra sessions that don't overlap with DANDI release.

    List of IDs
    # - register, make task etc

    """

    @classmethod
    def load(
        cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        dataset_alias: str,
        sampling_rate: int = 1000 # Hz
    ):
        meta_payload = {}
        meta_payload['path'] = []
        arrays_to_use = context_arrays
        def save_raster(trial_spikes: torch.Tensor, trial_id: int):
                # trim to valid length and then reshape
                trial_spikes = rearrange(
                    trial_spikes,
                    '(t bin) c -> t bin c', bin=cfg.bin_size_ms
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
                single_path = cache_root / f'{trial_id}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        # Ok, some are hdf5, some are mat (all masquerade with .mat endings)
        try:
            with h5py.File(datapath, 'r') as f:
                data = f['R']
                raise Exception(f'HDF5 for {dataset_alias} not supported yet because we dunno how to extract spikes')
                # https://github.com/catalystneuro/shenoy-lab-to-nwb/issues/8
        except:
            data = loadmat(datapath, simplify_cells=True)
            # data = loadmat(datapath, simplify_cells=True)
            data = pd.DataFrame(data['R'])
        if 'spikeRaster' in data:
            # These are scipy sparse matrices
            array_0 = data['spikeRaster']
            array_1 = data['spikeRaster2']
            time_start = data['timeCueOn']
            time_start = time_start.fillna(0).astype(int)
            for idx, trial in data.iterrows():
                start = time_start[idx]
                spike_raster = np.concatenate([array_0[idx].toarray(), array_1[idx].toarray()], axis=0).T # (time, c)
                spike_raster = torch.from_numpy(spike_raster)[start:]
                save_raster(spike_raster, idx)
        else: # Nitschke format
            data = data[data.hasSpikes == 1]
            # Mark provided a filtering script, but we won't filter as thoroughly as they do for analysis, just needing data validity
            START_KEY = 'commandFlyAppears' # presumably the cue
            END_KEY = 'trialEndsTime'
            for idx, trial in data.iterrows():
                start, end = trial[START_KEY], trial[END_KEY]
                trial_spikes = torch.zeros(end - start, 192, dtype=torch.uint8)
                spike_times = trial['unit']
                assert len(spike_times) == 192, "Expected 192 units"
                for c in range(len(spike_times)):
                    unit_times = spike_times[c]['spikeTimes'] # in ms, apparently
                    if isinstance(unit_times, float):
                        unit_times = np.array(unit_times)
                    unit_times = unit_times[(unit_times > start) & (unit_times < end - 1)] - start # end - 1 for buffer
                    ms_spike_times, ms_spike_cnt = np.unique(np.floor(unit_times), return_counts=True)
                    trial_spikes[ms_spike_times, c] = torch.tensor(ms_spike_cnt, dtype=torch.uint8)
                save_raster(trial_spikes, trial['trialID'])
        return pd.DataFrame(meta_payload)
