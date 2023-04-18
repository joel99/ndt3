from typing import Dict, Any, List
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import torch

import logging

from ..config import DataKey, MetaKey, DatasetConfig
from ..subjects import ArrayID, SubjectInfo, SubjectArrayRegistry, SubjectName
from ..tasks import ExperimentalTask, ExperimentalTaskLoader, ExperimentalTaskRegistry

TrialNum = int
MetadataKey = str

def transform_current(current, normalize=False):
    # approx z-score.
    if normalize:
        return current / 100 -0.5
    else:
        return (current + 0.5 * 100)

def infer_stim_parameters(
    trial_record,
    trial_stim_samples,
    command_time,
    command_channel,
    command_current,
    stim_channels=64,
    time_bin_size_s=0.002,
):
    trial_record = torch.tensor(trial_record).T.unsqueeze(-1) # C T to T C 1
    # Target shape: Trial[time bin x channel x feat]
    r""" EXTRACT STIMULATION PARAMETERS
        Our preprocessing will extract putative stimulation times (`trial_stim_samples`),
        which we need to identify the true stimulation parameters for (such as channel, amplitude).
        We do need to compare timing as some (small but non-negligible) fraction of pulses are dropped.
        TODO: the most technically correct way of doing this is to extract from voltage monitor, i.e. hardware logs right before stimulation
        - We will NEED to use voltage monitor once using multi-channel stimulation.
        Right now, we're referring to the commanded parameters, assuming minimal drift and no egregious dropping (i.e. not robust to severe jitter and assuming each command has an artifact.)

        Note we only use parameters, and not timing, as command timing clearly drifts a ~2ms.
        Assuming effected stim conserves command order, we'll zip backwards.
        i.e. assuming last pulse was effected, then take next detected stim's params from the closest of remaining commands
        # In simplest case, this turns into taking last N
    """
    def simple_backzip(command_time, effected_time):
        effected_time = effected_time + (command_time[-1] - effected_time[-1])
        effected_command_indices = [] # ! Not all commanded are effected (dropped pulses).
        commanded_effect_indices = [] # ! Not all detected artifacts are true artifacts (i.e. transient). While the model should know about this, we're currently just dropping them.
        cmd_i = len(command_time) - 1
        effect_i = len(effected_time) - 1
        while effect_i >= 0 and cmd_i >= 0:
            # assuming alignment at current timestep, a few possible scenarios:
            # either next effect is closer to cmd
            if cmd_i == 0 or \
                abs(command_time[cmd_i] - effected_time[effect_i]) <= abs(command_time[cmd_i-1] - effected_time[effect_i]):
                # In this case, the current effected time likely matches the current commanded time, but it may not be the best fitting effected time
                # (Sometimes we have multiple mini-artifax per command)
                # Sweep back for the best one. In particular, "best" is history biased since
                while (effect_i > 0 and \
                    abs(command_time[cmd_i] - effected_time[effect_i - 1]) <= abs(command_time[cmd_i-1] - effected_time[effect_i - 1])): # As long as effect_i is more likely to belong to current cmd than previous one
                    # abs(command_time[cmd_i] - effected_time[effect_i]) >= abs(command_time[cmd_i] - effected_time[effect_i-1])):
                        effect_i -= 1
                effected_command_indices.append(cmd_i)
                commanded_effect_indices.append(effect_i)
                # print(f"Cmd: {cmd_i}/{len(command_time)} \t Effect: {effect_i}/{len(effected_time)} \t {command_time[cmd_i]}:{effected_time[effect_i]:.4f}")
                # Adjust effected time labels for purpose of matchign with command. (Same as moving command, probably more legible that way.)
                # This presumes (and Jeff confirms) stimulator attempts to maintain diffs rather than absolute command times (dropping commands when infeasible)
                effected_time = effected_time + command_time[cmd_i] - effected_time[effect_i] # careful not to mutate true
                effect_i -= 1
                # In order to account for monotonic drift, we need to constantly re-align effected time to zipped command time
            cmd_i -= 1
        return list(reversed(effected_command_indices)), list(reversed(commanded_effect_indices))
    fs = 30000
    effected_idx, commanded_idx = simple_backzip(command_time, trial_stim_samples / fs)
    # aligned_command_time = command_time + trial_stim_samples[-1] / fs - command_time[-1]

    channels = torch.tensor(command_channel[effected_idx] - 1).long()
    currents = transform_current(torch.tensor(command_current[effected_idx]), normalize=True)
    # While true channel and current info come from command (as far as we know), best timing info should be taken from effected stim itself
    # ! We are discarding the other artifact fragment markers by only reporting true commands here. Consider denoting other artifact times.
    trial_stim_samples = torch.tensor(trial_stim_samples[commanded_idx].astype(int))
    trial_stim_state = torch.zeros(
        trial_record.size(0),
        stim_channels,
        # 1 # count
        3 # count, current, timing
    ).float()
    bin_samples = round(time_bin_size_s * fs)
    timebins = torch.div(trial_stim_samples, bin_samples, rounding_mode='floor').long() # just rounded, add timing cf truncation in preprocessing
    try:
        trial_stim_state[timebins, channels, 0] = 1.
        trial_stim_state[timebins, channels, 1] = currents
        trial_stim_state[timebins, channels, 2] = (trial_stim_samples % bin_samples) / bin_samples - 0.5
    except:
        # Somehow, there are some strange trials where stim is just not recorded properly e.g. only 5 pulses in 1s...?
        logging.error("Invalid stim attempted, skipping trial.")
        import pdb;pdb.set_trace() # figure out what to do with this
        # TODO delete invalid trials from key_df or mark them somehow
    return trial_stim_state

@ExperimentalTaskRegistry.register
class ICMSLoader(ExperimentalTaskLoader):
    name = ExperimentalTask.passive_icms

    @classmethod
    def load(cls,
        datapath: Path,
        cfg: DatasetConfig,
        cache_root: Path,
        subject: SubjectInfo,
        context_arrays: List[str],
        *args
    ):
        r"""
            Loader for data from `icms_modeling` project.
            Takes in a payload containing all trials, fragments according to config.

            1. Preprocess as needed (e.g. trimming max length)
            2. Trialize storage
        """
        _CONDITION_KEY = 'raw_condition'
        import pdb;pdb.set_trace()
        data = torch.load(datapath)
        payload = {
            DataKey.spikes : data['spikes'],
            DataKey.stim : data['stim_time'],
            MetaKey.trial: data['trial_num'],
            'src_file': data['src_file'],
            _CONDITION_KEY: data['condition'],
            'command_time': [c[0] for c in data['command']], # tuple (time, channel (1-64), amp)
            'command_channels': [c[1] for c in data['command']], # tuple (time, channel (1-64), amp)
            'command_current': [c[2] for c in data['command']], # tuple (time, channel (1-64), amp)
        }
        for k in cfg.data_keys:
            assert k in payload, f"Missing {k} in payload"
        if 'stim_dir' in data and not data.get('ignore_exp_info', False):
            exp_info = Path(data['stim_dir']) / 'exp_info.pkl'
            if exp_info.exists():
                with open(exp_info, 'rb') as f:
                    exp_info: Dict[TrialNum, Dict[MetadataKey, Any]] = pickle.load(f)
                for key in exp_info[1]: # 1 is arbitrary. Keys included are listed in `generate_stim` in `icms_modeling` -- e.g. `channels`, `train`, `condition`, etc. `condtiion` is the same as `raw_condition` above.
                    payload[key] = [exp_info[t][key] for t in payload['trial_num']]
        del data
        # TODO store as uint8
        # `meta_info.arrays` is more static, and should dictate the order arrays are cached in (though I expect to overwrite)
        # TODO review whether we should save the configured subset or configure outside of cache
        # arrays_to_use = [a for a in context_info.arrays if a in cfg.passive_icms.arrays] # use config
        arrays_to_use = context_arrays # ignore config

        def extract_arrays(full_spikes: torch.Tensor, arrays_to_use: List[str]) -> Dict[ArrayID, torch.Tensor]:
            r"""
                The bulk of `icms_modeling` experiments ignore motor data since participant behavior was unconstrained + early recordings had noisy motor banks;
                Edit this function to save it down.

                full_spikes: T C H # TODO validate shape
            """
            import pdb;pdb.set_trace()
            if subject.name in [SubjectName.CRS02b, SubjectName.CRS07]:
                info = {}
                for a in arrays_to_use:
                    info[a] = full_spikes[:, SubjectArrayRegistry.query_by_array_geometric(a).as_indices()]
                return info
            raise NotImplementedError

        payload['path'] = []
        for t in payload[MetaKey.trial]:
            single_payload = {}
            for k in cfg.data_keys:
                if k == DataKey.stim:
                    trial_stim_state = infer_stim_parameters(
                        payload[DataKey.spikes][t],
                        payload[DataKey.stim][t],
                        payload['command_time'][t],
                        payload['command_channels'][t],
                        payload['command_current'][t],
                        stim_channels=64, # TODO make configurable
                        time_bin_size_s=cfg.bin_size_ms / 1000.,
                    )
                    single_payload[k] = trial_stim_state
                    # TODO implement in fragmented (but probably not, that's too much work...)
                elif k == DataKey.spikes:
                    single_payload[k] = extract_arrays(payload[k][t], arrays_to_use)
                else: # Honestly have no idea what other keys even are
                    single_payload[k] = payload[k][t]
                single_payload[k] = single_payload[k][:, :cfg.max_trial_length].clone() # TODO alignment?
            import pdb;pdb.set_trace() # TODO check shape
            single_path = cache_root / f"{t}.pth"
            payload['path'].append(single_path)
            torch.save(single_payload, single_path)

        for k in cfg.data_keys:
            del payload[k]
        return pd.DataFrame(payload)