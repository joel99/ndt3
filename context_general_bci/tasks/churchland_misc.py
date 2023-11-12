#%%
from typing import List
from pathlib import Path
import math
import numpy as np
import torch
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import resample_poly
from scipy.io import loadmat
from scipy.sparse import csc_matrix
from einops import rearrange, reduce

import logging
logger = logging.getLogger(__name__)
try:
    from pynwb import NWBHDF5IO
    import h5py
except:
    logger.info("pynwb not installed, please install with `conda install -c conda-forge pynwb`")

from context_general_bci.config import DataKey, DatasetConfig, REACH_DEFAULT_3D_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, SubjectArrayRegistry, create_spike_payload, SubjectName
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry
from context_general_bci.tasks.preproc_utils import PackToChop, get_minmax_norm


# Note these comprise a bunch of different tasks, perhaps worth denoting/splitting them
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
    # ! Actually, the Jenkins/Reggie data here corresponds to Even-Chen's study on structure of delay in PMd. (Nitschke data unaccounted for)
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
        task: ExperimentalTask,
        sampling_rate: int = 1000 # Hz
    ):
        if subject.name == SubjectName.reggie:
            raise DeprecationWarning("Reggie data should use DANDI release (DelayedReach) due to covariate concerns in this path")
        meta_payload = {}
        meta_payload['path'] = []
        arrays_to_use = context_arrays
        if cfg.pack_dense:
            packer = PackToChop(cfg.churchland_misc.chop_size_ms // cfg.bin_size_ms, cache_root)
        def save_raster(trial_spikes: torch.Tensor, trial_id: int, other_args: dict = {}):
            single_payload = {
                DataKey.spikes: create_spike_payload(trial_spikes, arrays_to_use, cfg=cfg),
                **other_args
            }
            if cfg.pack_dense:
                single_payload = packer.pack(single_payload)
            else:
                single_path = cache_root / f'{trial_id}.pth'
                meta_payload['path'].append(single_path)
                torch.save(single_payload, single_path)
        # Ok, some are hdf5, some are mat (all masquerade with .mat endings)
        def get_global_args(hand_vels: List[np.ndarray]): # each T x 3
            global_args = {}
            if cfg.tokenize_covariates:
                global_args[DataKey.covariate_labels] = REACH_DEFAULT_3D_KIN_LABELS
            if cfg.churchland_misc.minmax:
                # Aggregate velocities and get min/max. No... vel needs to be per trial
                global_vel = np.concatenate(hand_vels, 0)
                # warn about nans
                if np.isnan(global_vel).any():
                    logging.warning(f'{global_vel.isnan().sum()} nan values found in velocity, masking out for global calculation')
                    global_vel = global_vel[~np.isnan(global_vel).any(axis=1)]
                global_vel, payload_norm = get_minmax_norm(global_vel, center_mean=cfg.churchland_misc.center)
                global_args.update(payload_norm)
            return global_args

        def preproc_vel(trial_vel, global_args):
            # trial_vel: (time, 3)
            # Mirror spike downsample logic - if uneven, crop beginning
            trial_vel = trial_vel[trial_vel.shape[0] % cfg.bin_size_ms:, ]
            trial_vel = resample_poly(trial_vel, 1,  cfg.bin_size_ms, padtype='line', axis=0)
            trial_vel = torch.from_numpy(trial_vel).float()
            if cfg.churchland_misc.minmax:
                if global_args['cov_mean'] is not None:
                    trial_vel = (trial_vel - global_args['cov_mean']) / (global_args['cov_max'] - global_args['cov_min'])
                else:
                    trial_vel = trial_vel / global_args['cov_max']
                trial_vel = torch.clamp(trial_vel, -1, 1)
            return trial_vel
        try:
            with h5py.File(datapath, 'r') as f:
                data = f['R']
                num_trials = data['spikeRaster'].shape[0]
                assert data['spikeRaster2'].shape[0] == num_trials, 'mismatched array recordings'
                # Run through all trials to collect global normalization stats (annoyingly...)
                hand_vel = []
                for i in range(num_trials):
                    hand_vel.append(np.gradient(data[data['handPos'][i, 0]], axis=0)) # comes out T x 3 raw
                global_args = get_global_args(hand_vel)
                for i in range(num_trials):
                    def make_arr(ref):
                        return csc_matrix((
                            data[ref]['data'][:], data[ref]['ir'][:], data[ref]['jc'][:]
                        )).toarray()
                    array_0 = make_arr(data['spikeRaster'][i, 0]).T # (time, c)
                    array_1 = make_arr(data['spikeRaster2'][i, 0]).T
                    # pad each array to size 96 if necessary (apparently some are smaller, but reason isn't recorded)
                    if array_0.shape[1] < 96:
                        array_0 = np.pad(array_0, ((0, 0), (0, 96 - array_0.shape[1])), mode='constant', constant_values=0)
                    if array_1.shape[1] < 96:
                        array_1 = np.pad(array_1, ((0, 0), (0, 96 - array_1.shape[1])), mode='constant', constant_values=0)
                    # print(data[data['timeCueOn'][i, 0]].shape, data[data['timeCueOn'][i, 0]][()])
                    time_start = data[data['timeCueOn'][i, 0]]
                    time_start = 0 if time_start.shape[0] != 1 or np.isnan(time_start[0, 0]) else int(time_start[0, 0])
                    spike_raster = np.concatenate([array_0, array_1], axis=1)
                    spike_raster = torch.from_numpy(spike_raster)[time_start:]
                    if spike_raster.size(1) > 192:
                        print(spike_raster.size(), 'something wrong with raw data')
                        import pdb;pdb.set_trace()
                    trial_vel = preproc_vel(hand_vel[i][time_start:], global_args)
                    other_args = {
                        DataKey.bhvr_vel: trial_vel,
                        **global_args
                    }
                    save_raster(spike_raster, i, other_args)
                if cfg.pack_dense:
                    packer.flush()
                    meta_payload['path'] = packer.get_paths()
                return pd.DataFrame(meta_payload)
        except Exception as e:
            # print(e)
            # import pdb;pdb.set_trace()
            data = loadmat(datapath, simplify_cells=True)
            data = pd.DataFrame(data['R'])
        # breakpoint()
        if 'spikeRaster' in data:
            # These are scipy sparse matrices
            array_0 = data['spikeRaster']
            array_1 = data['spikeRaster2']
            time_start = data['timeCueOn']
            time_start = time_start.fillna(0).astype(int)


            hand_vel = data.apply(lambda x: np.gradient(x['handPos'], axis=1).T, axis=1)
            global_args = get_global_args(hand_vel.values)

            for i, trial in data.iterrows():
                start = time_start[i]
                spike_raster = np.concatenate([array_0[i].toarray(), array_1[i].toarray()], axis=0).T # (time, c)
                spike_raster = torch.from_numpy(spike_raster)[start:]
                trial_vel = preproc_vel(hand_vel[i][start:], global_args)
                other_args = {
                    DataKey.bhvr_vel: trial_vel,
                    **global_args
                }
                save_raster(spike_raster, i, other_args)
        else: # Nitschke format, sparse
            data = data[data.hasSpikes == 1]
            # Mark provided a filtering script, but we won't filter as thoroughly as they do for analysis, just needing data validity
            START_KEY = 'commandFlyAppears' # presumably the cue
            END_KEY = 'trialEndsTime' # in units of ms, I think - and the bhvr has exactly this many timesteps
            # breakpoint()
            # do one iteration to find the global args
            hand_vel = []
            for idx, trial in data.iterrows():
                hand_vel.append(np.stack([
                    np.gradient(trial['HAND'][dim], axis=0) for dim in ['X', 'Y', 'Z']
                ], 1))
            global_args = get_global_args(hand_vel)
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
                # * apparently iter order is not preserved; so we reextract the hand_vel instead of pulling from previous array
                trial_vel = preproc_vel(np.stack([
                    np.gradient(trial['HAND'][dim], axis=0) for dim in ['X', 'Y', 'Z']
                ], 1)[start:end], global_args)
                save_raster(trial_spikes, trial['trialID'], {
                    DataKey.bhvr_vel: trial_vel,
                    **global_args
                })
        if cfg.pack_dense:
            packer.flush()
            meta_payload['path'] = packer.get_paths()
        return pd.DataFrame(meta_payload)
