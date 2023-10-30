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
NORMATIVE_MAX_FORCE = 25 # Our prior on the full Pitt dataset. Some FBC data reports exponentially large force
NORMATIVE_MIN_FORCE = 0 # according to mujoco system; some decoders report negative force, which is nonsensical
# which is not useful to rescale by.
# https://www.notion.so/joelye/Broad-statistic-check-facb9b6b68a0408090921e4f84f70a6e

NORMATIVE_EFFECTOR_BLACKLIST = {
    'cursor': [3, 4, 5, 8], # Rotation and gz are never controlled in cursor tasks.
}

r"""
    Dev note to self: Pretty unclear how the .mat payloads we're transferring seem to be _smaller_ than n_element bytes. The output spike trials, ~250 channels x ~100 timesteps are reasonably, 25K. But the data is only ~10x this for ~100x the trials.
"""
def compute_return_to_go(rewards: torch.Tensor, horizon=100):
    # rewards: T
    if horizon:
        padded_reward = F.pad(rewards, (0, horizon - 1), value=0)
        return padded_reward.unfold(0, horizon, 1)[..., 1:].sum(-1) # T. Don't include current timestep
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
        effector = payload['effector']
        if len(effector) == 0:
            out['effector'] = ''
        else:
            out['effector'] = effector.lower().strip()
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
        if 'override' in payload:
            out['override_assist'] = torch.from_numpy(payload['override']).half()
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


def interpolate_nan(arr: np.ndarray | torch.Tensor):
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    out = np.zeros_like(arr)
    for i in range(arr.shape[1]):
        x = arr[:, i]
        nans = np.isnan(x)
        non_nans = ~nans
        x_interp = np.interp(np.flatnonzero(nans), np.flatnonzero(non_nans), x[non_nans])
        x[nans] = x_interp
        out[:, i] = x
    return torch.as_tensor(out)

@ExperimentalTaskRegistry.register
class PittCOLoader(ExperimentalTaskLoader):
    r"""
        Note: This is called "PittCO as in pitt center out due to dev legacy, but it's really just a general loader for the pitt data.
    """
    name = ExperimentalTask.pitt_co

    # We have a basic 180ms boxcar smooth to deal with visual noise in rendering. Not really that excessive, still preserves high frequency control characteristics in the data. At lower values, observation targets becomes jagged and unrealistic.
    @staticmethod
    def smooth(position, kernel):
        # kernel: np.ndarray, e.g. =np.ones((int(180 / 20), 1))/ (180 / 20)
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # This is necessary since 1. our data reports are effector position, not effector command; this is a better target since serious effector failure should reflect in intent
        # and 2. effector positions can be jagged, but intent is (presumably) not, even though intent hopefully reflects command, and 3. we're trying to report intent.
        position = interpolate_nan(position)
        # position = position - position[0] # zero out initial position
        # Manually pad with edge values
        # OK to pad because this is beginning and end of _set_ where we expect little derivative (but possibly lack of centering)
        assert kernel.shape[0] % 2 == 1, "Kernel must be odd (for convenience)"
        pad_left, pad_right = int(kernel.shape[0] / 2), int(kernel.shape[0] / 2)
        position = F.pad(position.T, (pad_left, pad_right), 'replicate')
        return F.conv1d(position.unsqueeze(1), torch.tensor(kernel).float().T.unsqueeze(1))[:,0].T

    @staticmethod
    def get_velocity(position, kernel):
        # kernel: np.ndarray, e.g. =np.ones((int(180 / 20), 1))/ (180 / 20)
        # Apply boxcar filter of 500ms - this is simply for Parity with Pitt decoding
        # This is necessary since 1. our data reports are effector position, not effector command; this is a better target since serious effector failure should reflect in intent
        # and 2. effector positions can be jagged, but intent is (presumably) not, even though intent hopefully reflects command, and 3. we're trying to report intent.
        position = PittCOLoader.smooth(position, kernel=kernel)
        return torch.as_tensor(np.gradient(position.numpy(), axis=0)).float() # note gradient preserves shape

    @staticmethod
    def ReFIT(positions: torch.Tensor, goals: torch.Tensor, reaction_lag_ms=100, bin_ms=20, oracle_blend=0.25) -> torch.Tensor:
        # positions, goals: Time x Hidden.
        # weight: don't do a full refit correction, weight with original
        # defaults for lag experimented in `pitt_scratch`
        raise NotImplementedError("Deprecated")
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
        sample_bin_ms=20
    ):
        downsample = cfg.bin_size_ms / sample_bin_ms
        # assert cfg.bin_size_ms == 20, 'code not prepped for different resolutions'
        meta_payload = {}
        meta_payload['path'] = []
        arrays_to_use = context_arrays
        def chop_vector(vec: torch.Tensor | None): # T x C
            # vec - already at target sampling resolution, just needs chopping
            if vec is None:
                return None
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
            # unique, counts = np.unique(spikes, return_counts=True)
            # ! Removing clip, we clip on embed. Avoids dataset, preproc specific clip
            # for u, c in zip(unique, counts):
                # if u >= CLAMP_MAX:
                    # spikes[spikes == u] = CLAMP_MAX # clip

            # Iterate by trial, assumes continuity so we grab velocity outside
            exp_task_cfg: PittConfig = getattr(cfg, task.value)

            # * Kinematics (labeled 'vel' as we take derivative of reported position)
            kernel = np.ones((int(exp_task_cfg.causal_smooth_ms / cfg.bin_size_ms), 1)) / (exp_task_cfg.causal_smooth_ms / cfg.bin_size_ms)
            kernel[-kernel.shape[0] // 2:] = 0 # causal, including current timestep
            if (
                'position' in payload # and \
                # task in [ExperimentalTask.observation, ExperimentalTask.ortho, ExperimentalTask.fbc, ExperimentalTask.unstructured] # and \ # Unstructured kinematics may be fake, mock data.
            ): # We only "trust" in the labels provided by obs (for now)
                if exp_task_cfg.closed_loop_intention_estimation == "refit" and task in [ExperimentalTask.ortho, ExperimentalTask.fbc]:
                    # breakpoint()
                    covariates = PittCOLoader.ReFIT(payload['position'], payload['target'], bin_ms=cfg.bin_size_ms)
                else:
                    covariates = PittCOLoader.get_velocity(payload['position'], kernel=kernel)
            else:
                covariates = None

            # * Force
            if 'force' in payload: # Force I believe is often strictly positive in our setting (grasp closure force)
                if not (payload['force'][~payload['force'].isnan()] != 0).sum() > 10: # Some small number of non-zero, not interesting enough.
                    print('dud force')
                covariate_force = payload['force']
                # clamp
                covariate_force[covariate_force > NORMATIVE_MAX_FORCE] = NORMATIVE_MAX_FORCE
                covariate_force[covariate_force < NORMATIVE_MIN_FORCE] = NORMATIVE_MIN_FORCE
                covariate_force = PittCOLoader.smooth(covariate_force, kernel=kernel) # Gary doesn't compute velocity, just absolute. We follow suit.
                covariates = torch.cat([covariates, covariate_force], 1) if covariates is not None else covariate_force

                # These are mostly Gary's data - skip the initial 1s, which has the hand adjust but the participant isn't really paying attn
                spikes = spikes[int(1000 / cfg.bin_size_ms):]
                covariates = covariates[int(1000 / cfg.bin_size_ms):]
            # breakpoint()
            # Apply a policy before normalization - if there's minor variance; these values are supposed to be relatively interpretable
            # So tiny variance is just machine/env noise. Zero that out so we don't include those dims. Src: Gary Blumenthal
            if covariates is not None:
                payload['cov_mean'] = covariates.mean(0)
                payload['cov_min'] = torch.quantile(covariates, 0.001, dim=0)
                payload['cov_max'] = torch.quantile(covariates, 0.999, dim=0)
                covariates = covariates - covariates.mean(0)
                NOISE_THRESHOLDS = torch.full_like(payload['cov_min'], 0.001)
                # Threshold for force is much higher based on spotchecks. Better to allow noise, than to drop true values? IDK.
                if 'force' in payload: # Force is appended if available
                    NOISE_THRESHOLDS[-covariate_force.size(1):] = 0.008
                covariates[:, (payload['cov_max'] - payload['cov_min']) < NOISE_THRESHOLDS] = 0 # Higher values are too sensitive! We see actual values ranges sometimes around 0.015, careful not to push too high.
            else:
                payload['cov_mean'] = None
                payload['cov_min'] = None
                payload['cov_max'] = None

            if exp_task_cfg.minmax and covariates is not None: # T x C
                rescale = payload['cov_max'] - payload['cov_min']
                rescale[torch.isclose(rescale, torch.tensor(0.))] = 1 # avoid div by 0 for inactive dims
                covariates = covariates / rescale # Think this rescales to a bit less than 1
                covariates = torch.clamp(covariates, -1, 1) # Note dynamic range is typically ~-0.5, 0.5 for -1, 1 rescale like we do. This is for extreme outliers.
                # TODO we should really sanitize for severely abberant values in a more robust way... (we currently instead verify post-hoc in `sampler`)
            if 'effector' in payload and covariates is not None:
                for k in NORMATIVE_EFFECTOR_BLACKLIST:
                    if k in payload['effector']:
                        for dim in NORMATIVE_EFFECTOR_BLACKLIST[k]:
                            if dim < covariates.size(-1):
                                covariates[:, dim] = 0
                        break

            # * Constraints
            brain_control: torch.Tensor | None = payload.get('brain_control', None)
            active_assist: torch.Tensor | None = payload.get('active_assist', None)
            passive_assist: torch.Tensor | None = payload.get('passive_assist', None)
            override_assist: torch.Tensor | None = payload.get('override_assist', None) # Override is sub-domain specific active assist, used for partial domain control e.g. in robot tasks
            """
            Quoting JW:
            ActiveAssist expands the active_assist weight from
            6 domains to 30 dimensions, and then takes the max of
            the expanded active_assist_weight (can be float 0-1)
            and override (0 or 1) to get an effective weight
            for each dimension.
            """
            # clamp each constraint to 0 and 1 - otherwise nonsensical
            if brain_control is not None:
                brain_control = brain_control.int().clamp(0, 1).half()
            if active_assist is not None:
                active_assist = active_assist.int().clamp(0, 1).half()
            if passive_assist is not None:
                passive_assist = passive_assist.int().clamp(0, 1).half()
            if override_assist is not None:
                override_assist = override_assist.int().clamp(0, 1).half()

            # * Reward and return!
            passed = payload.get('passed', None)
            trial_num: torch.Tensor = payload['trial_num']
            if passed is not None and trial_num.max() > 1: # Heuristic - single trial means this is probably not a task-based dataset
                trial_change_step = (trial_num.roll(-1, dims=0) != trial_num).nonzero()[:,0] # * end of episode timestep.
                # * Since this marks end of episode, it also marks when reward is provided

                per_trial_pass = torch.cat([passed[:1], torch.diff(passed)]).to(dtype=int)
                # if (per_trial_pass < 0).any():
                    # breakpoint()
                per_trial_pass = torch.clamp(per_trial_pass, min=0, max=1) # Literally, clamp that. What does > 1 reward even mean? (It shows up sometimes...)
                # In some small # of datasets, num_passed randomly drops (significantly, i.e. not decrement of 1). JY assuming this means some task change to reset counter
                # e.g. CRS02bLab_245_12
                # So we clamp at 0; so only that trial gets DQ-ed; rest of counters should resume as normal
                reward_dense = torch.zeros_like(trial_num, dtype=int) # only 0 or 1 reward
                reward_dense.scatter_(0, trial_change_step, per_trial_pass)
                return_dense = compute_return_to_go(reward_dense, horizon=int((cfg.return_horizon_s * 1000) // cfg.bin_size_ms))
                reward_dense = reward_dense.unsqueeze(-1) # T -> Tx1
                return_dense = return_dense.unsqueeze(-1) # T -> Tx1
                # We need to have tuples <Return, State, Action, Reward> - currently, final timestep still has 1 return
            else:
                reward_dense = None
                return_dense = None

            spikes = chop_vector(spikes)
            # breakpoint()
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
                if override_assist is not None:
                    # breakpoint() # assuming override dimension is Trial T (domain 3) after chop
                    chopped_override = chop_vector(override_assist)
                    chopped_constraints[..., 0, :] = torch.maximum(chopped_constraints[..., 0, :], chopped_override[..., :chopped_constraints.shape[-1]]) # if override is on, brain control is off, which means FBC constraint is 1
                    chopped_constraints[..., 1, :] = torch.maximum(chopped_constraints[..., 1, :], chopped_override[..., :chopped_constraints.shape[-1]]) # if override is on, active assist is on, which means active assist constraint is 1

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
            raise NotImplementedError("Deprecated")
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