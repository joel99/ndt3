#%%
from typing import List
from pathlib import Path
import math
import numpy as np
import torch
import torch.distributions as dists
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter1d
# from scipy.signal import convolve
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

import logging
logger = logging.getLogger(__name__)

from context_general_bci.config import DataKey, DatasetConfig, PittConfig, DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectInfo, create_spike_payload
from context_general_bci.tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry


CLAMP_MAX = 15

r"""
    Dev note to self: Pretty unclear how the .mat payloads we're transferring seem to be _smaller_ than n_element bytes. The output spike trials, ~250 channels x ~100 timesteps are reasonably, 25K. But the data is only ~10x this for ~100x the trials.
"""
def compute_return_to_go(rewards: torch.Tensor, horizon=100):
    # rewards: T
    if horizon:
        padded_reward = F.pad(rewards, (0, horizon - 1), value=0)
        return padded_reward.unfold(0, horizon, 1).sum(-1) # T
    reversed_rewards = torch.flip(rewards, [0])
    returns_to_go_reversed = torch.cumsum(reversed_rewards, dim=0)
    return torch.flip(returns_to_go_reversed, [0])

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
    spikes = torch.zeros((len(bins), 256), dtype=int)
    spikes[timebins, events['spikes_channel']] = 1
    return spikes


def load_trial(fn, use_ql=True, key='data', copy_keys=True, limit_dims=8):
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
            # breakpoint()
            out['position'] = torch.from_numpy(payload['Kinematics']['ActualPos'][:,:limit_dims]) # index 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
        elif 'pos' in payload:
            out['position'] = torch.from_numpy(payload['pos'][:,:limit_dims]) # 1 is y, 2 is X. Col 6 is click, src: Jeff Weiss
        if 'position' in out:
            assert len(out['position']) == len(out['trial_num']), "Position and trial num should be same length"

        if 'target' in payload:
            out['target'] = torch.from_numpy(payload['target'][:limit_dims])
        if 'force' in payload:
            out['force'] = torch.from_numpy(payload['force'])
            if out['force'].ndim == 1:
                out['force'] = out['force'].unsqueeze(1)
            assert out['force'].size(-1) == 1, "Force feedback should be 1D"
        if 'brain_control' in payload:
            out['brain_control'] = torch.from_numpy(payload['brain_control']).half() # half as these are very simple fractions
            out['active_assist'] = torch.from_numpy(payload['active_assist']).half()
            out['passive_assist'] = torch.from_numpy(payload['passive_assist']).half()
            assert out['brain_control'].size(-1) == 3, "Brain control should be 3D (3 domains)"
        if 'passed' in payload:
            try:
                if isinstance(payload['passed'], int):
                    out['passed'] = torch.tensor([payload['passed']], dtype=int) # It's 1 trial
                else:
                    out['passed'] = torch.from_numpy(payload['passed'].astype(int))
            except:
                breakpoint()
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
    r"""
        Note: This is called "PittCO as in pitt center out due to dev legacy, but it's really just a general loader for the pitt data.
    """
    name = ExperimentalTask.pitt_co

    @staticmethod
    def smooth(position, kernel=np.ones((int(200 / 20), 1))/ (200 / 20)):
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # This is necessary since 1. our data reports are effector position, not effector command; this is a better target since serious effector failure should reflect in intent
        # and 2. effector positions can be jagged, but intent is (presumably) not, even though intent hopefully reflects command, and 3. we're trying to report intent.
        int_position = pd.Series(position.flatten()).interpolate()
        position = torch.tensor(int_position).view(-1, position.shape[-1])
        position = F.conv1d(position.T.unsqueeze(1), torch.tensor(kernel).float().T.unsqueeze(1), padding='same')[:,0].T
        position[position.isnan()] = 0 # extra call to deal with edge values
        return position

    @staticmethod
    def get_velocity(position, kernel=np.ones((int(200 / 20), 1))/ (200 / 20)):
    # def get_velocity(position, kernel=np.ones((int(500 / 20), 1))/ (500 / 20)):
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # This is necessary since 1. our data reports are effector position, not effector command; this is a better target since serious effector failure should reflect in intent
        # and 2. effector positions can be jagged, but intent is (presumably) not, even though intent hopefully reflects command, and 3. we're trying to report intent.
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
    def ReFIT(positions: torch.Tensor, goals: torch.Tensor, reaction_lag_ms=100, bin_ms=20, oracle_blend=0.25) -> torch.Tensor:
        # positions, goals: Time x Hidden.
        # weight: don't do a full refit correction, weight with original
        # defaults for lag experimented in `pitt_scratch`
        lag_bins = reaction_lag_ms // bin_ms
        empirical = PittCOLoader.get_velocity(positions)
        oracle = goals.roll(lag_bins, dims=0) - positions
        magnitudes = torch.linalg.norm(empirical, dim=1)  # Compute magnitudes of original velocities
        # Oracle magnitude update - no good, visually

        # angles = torch.atan2(empirical[:, 1], empirical[:, 0])  # Compute angles of velocities
        source_angles = torch.atan2(empirical[:, 1], empirical[:, 0])  # Compute angles of original velocities
        oracle_angles = torch.atan2(oracle[:, 1], oracle[:, 0])  # Compute angles of velocities

        # Develop a von mises update that blends the source and oracle angles
        source_concentration = 10.0
        oracle_concentration = source_concentration * oracle_blend

        # Create Von Mises distributions for source and oracle
        source_von_mises = dists.VonMises(source_angles, source_concentration)
        updated_angles = torch.empty_like(source_angles)

        # Mask for the nan values in oracle
        nan_mask = torch.isnan(oracle_angles)

        # Update angles where oracle is not nan
        if (~nan_mask).any():
            # Create Von Mises distributions for oracle where it's not nan
            oracle_von_mises = dists.VonMises(oracle_angles[~nan_mask], oracle_concentration)

            # Compute updated estimate as the circular mean of the two distributions.
            # We weight the distributions by their concentration parameters.
            updated_angles[~nan_mask] = (source_von_mises.concentration[~nan_mask] * source_von_mises.loc[~nan_mask] + \
                                        oracle_von_mises.concentration * oracle_von_mises.loc) / (source_von_mises.concentration[~nan_mask] + oracle_von_mises.concentration)

        # Use source angles where oracle is nan
        updated_angles[nan_mask] = source_angles[nan_mask]
        angles = updated_angles
        angles = torch.atan2(torch.sin(angles), torch.cos(angles))

        new_velocities = torch.stack((magnitudes * torch.cos(angles), magnitudes * torch.sin(angles)), dim=1)
        new_velocities[:reaction_lag_ms // bin_ms] = torch.nan  # We don't know what the goal is before the reaction lag, so we clip it
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
        def chop_vector(vec: torch.Tensor | None): # T x C
            if vec is None:
                return None
            # vec - already at target resolution, just needs chopping
            if vec.size(0) < cfg.pitt_co.chop_size_ms / cfg.bin_size_ms:
                return vec.unsqueeze(0)
            chop_size = round(cfg.pitt_co.chop_size_ms / cfg.bin_size_ms)
            return rearrange(
                vec.unfold(0, chop_size, chop_size),
                'trial hidden time -> trial time hidden'
             ) # Trial x chop_size x hidden

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
            payload = load_trial(datapath, key='thin_data', limit_dims=cfg.pitt_co.limit_kin_dims)
            # Sanitize / renormalize
            spikes = payload['spikes']
            # elements = spikes.nelement()
            unique, counts = np.unique(spikes, return_counts=True)
            for u, c in zip(unique, counts):
                if u >= CLAMP_MAX:
                    spikes[spikes == u] = CLAMP_MAX # clip
                # if u >= 15 or c / elements < 1e-5: # anomalous, suppress to max. (Some bins randomly report impossibly high counts like 90 (in 20ms))
                    # spikes[spikes == u] = 0

            # Iterate by trial, assumes continuity so we grab velocity outside
            exp_task_cfg: PittConfig = getattr(cfg, task.value)

            # * Kinematics (labeled 'vel' as we take derivative of reported position)
            if (
                'position' in payload # and \
                # task in [ExperimentalTask.observation, ExperimentalTask.ortho, ExperimentalTask.fbc, ExperimentalTask.unstructured] # and \ # Unstructured kinematics may be fake, mock data.
            ): # We only "trust" in the labels provided by obs (for now)
                if exp_task_cfg.closed_loop_intention_estimation == "refit" and task in [ExperimentalTask.ortho, ExperimentalTask.fbc]:
                    # breakpoint()
                    covariates = PittCOLoader.ReFIT(payload['position'], payload['target'], bin_ms=cfg.bin_size_ms)
                else:
                    covariates = PittCOLoader.get_velocity(payload['position'])
            else:
                covariates = None

            # * Force
            if 'force' in payload: # Force I believe is often strictly positive in our setting (grasp closure force)

                # This needs a repull - it's got weird, incorrect values.
                # breakpoint()
                # I do believe force velocity is still a helpful, used concept? For more symmetry with other dimensions
                # Just minimize smoothing
                if (payload['force'][~payload['force'].isnan()] != 0).sum() > 10: # Some small number of non-zero, not interesting enough.
                    # Heuristic to identify interesting variability.
                    # breakpoint()
                    pass
                else:
                    print('dud force')
                covariate_force = PittCOLoader.smooth(payload['force'])
                covariates = torch.cat([covariates, covariate_force], 1) if covariates is not None else covariate_force

                # These are mostly Gary's - skip the initial 1s, which has the hand adjust but the participant isn't really paying attn
                spikes = spikes[int(1000 / cfg.bin_size_ms):]
                covariates = covariates[int(1000 / cfg.bin_size_ms):]

            # Apply a policy before normalization - if there's minor variance; these values are supposed to be relatively interpretable
            # So tiny variance is just machine/env noise. Zero that out so we don't include those dims. Src: Gary Blumenthal
            if covariates is not None:
                covariates = covariates - covariates.mean(0)
                covariates[:, (covariates.abs() < 1e-2).all(0)] = 0

            if exp_task_cfg.minmax and covariates is not None: # T x C
                payload['cov_mean'] = covariates.mean(0)
                payload['cov_min'] = covariates.min(0).values
                payload['cov_max'] = covariates.max(0).values
                rescale = payload['cov_max'] - payload['cov_min']
                rescale[torch.isclose(rescale, torch.tensor(0.))] = 1 # avoid div by 0 for inactive dims
                covariates = (covariates - payload['cov_mean']) / rescale # Think this rescales to a bit less than 1
                # TODO we should really sanitize for severely abberant values in a more robust way... or checking for outlier effects
            else:
                payload['cov_mean'] = None
                payload['cov_min'] = None
                payload['cov_max'] = None


            # * Constraints
            brain_control = payload.get('brain_control', None)
            active_assist = payload.get('active_assist', None)
            passive_assist = payload.get('passive_assist', None)

            # * Reward and return!
            passed = payload.get('passed', None)
            trial_num: torch.Tensor = payload['trial_num']
            if passed is not None and trial_num.max() > 1: # Heuristic - single trial means this is probably not a task-based dataset
                trial_change_step = (trial_num.roll(-1, dims=0) != trial_num).nonzero()[:,0] # end of episode timestep

                per_trial_pass = torch.cat([passed[:1], torch.diff(passed)]).to(dtype=int)
                per_trial_pass = torch.clamp(per_trial_pass, max=1) # Literally, clamp that. What does 2x reward even mean? (It shows up sometimes...)
                reward_dense = torch.zeros_like(trial_num, dtype=int) # only 0 or 1 reward
                reward_dense.scatter_(0, trial_change_step, per_trial_pass)
                return_dense = compute_return_to_go(reward_dense, horizon=int((cfg.return_horizon_s * 1000) // cfg.bin_size_ms))
                reward_dense = reward_dense.unsqueeze(-1) # T -> Tx1
                return_dense = return_dense.unsqueeze(-1) # T -> Tx1
            else:
                reward_dense = None
                return_dense = None

            spikes = chop_vector(spikes)
            if brain_control is None or covariates is None:
                chopped_constraints = None
            else:
                # Chop first bc chop is only implemented for 3d
                chopped_constraints = torch.stack([
                    chop_vector(1 - brain_control), # return complement, such that native control is the "0" condition, no constraint
                    chop_vector(active_assist),
                    chop_vector(passive_assist),
                ], 2)
                chopped_constraints = repeat(chopped_constraints, 'trial t dim domain -> trial t dim (domain 3)')[..., :covariates.size(-1)] # Put behavioral control dimension last
                # ! If we ever extend beyond 9 dims, the other force dimensions all belong to the grasp domain: src - Jeff Weiss

            if reward_dense is not None:
                reward_dense = chop_vector(reward_dense)
                return_dense = chop_vector(return_dense)

            # Expecting up to 9D vector (T x 9), 8D from kinematics, 1D from force
            if cfg.tokenize_covariates:
                covariate_dims = []
                covariate_reduced = []
                constraints_reduced = []
                labels = DEFAULT_KIN_LABELS
                if covariates is not None:
                    for i, cov in enumerate(covariates.T):
                        if cov.any(): # i.e. nonempty
                            covariate_dims.append(labels[i])
                            covariate_reduced.append(cov)
                            if chopped_constraints is not None:
                                constraints_reduced.append(chopped_constraints[..., i]) # Subselect behavioral dim
                covariates = torch.stack(covariate_reduced, -1) if covariate_reduced else None
                chopped_constraints = torch.stack(constraints_reduced, -1) if constraints_reduced else None # T x 3 (constriant dim) x B

            other_args = {
                DataKey.bhvr_vel: chop_vector(covariates),
                DataKey.constraint: chopped_constraints,
                DataKey.task_reward: reward_dense,
                DataKey.task_return: return_dense,
            }

            global_args = {}
            if exp_task_cfg.minmax:
                global_args['cov_mean'] = payload['cov_mean']
                global_args['cov_min'] = payload['cov_min']
                global_args['cov_max'] = payload['cov_max']
            if cfg.tokenize_covariates:
                global_args[DataKey.covariate_labels] = covariate_dims


            for i, trial_spikes in enumerate(spikes):
                other_args_trial = {k: v[i] for k, v in other_args.items() if v is not None}
                other_args_trial.update(global_args)
                save_trial_spikes(trial_spikes, i, other_args_trial)

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