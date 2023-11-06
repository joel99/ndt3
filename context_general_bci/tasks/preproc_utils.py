from typing import List, Tuple, Dict, TypeVar
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, reduce

from context_general_bci.config import DataKey

T = TypeVar('T', torch.Tensor, None)

def compute_return_to_go(rewards: torch.Tensor, horizon=100):
    # Mainly for PittCO
    # rewards: T
    if horizon:
        padded_reward = F.pad(rewards, (0, horizon - 1), value=0)
        return padded_reward.unfold(0, horizon, 1)[..., 1:].sum(-1) # T. Don't include current timestep
    reversed_rewards = torch.flip(rewards, [0])
    returns_to_go_reversed = torch.cumsum(reversed_rewards, dim=0)
    return torch.flip(returns_to_go_reversed, [0])

def crop_subject_handles(subject: str):
    if subject.endswith('Home'):
        subject = subject[:-4]
    elif subject.endswith('Lab'):
        subject = subject[:-3]
    return subject

def get_minmax_norm(covariates: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""
        Get min/max normalization for covariates
        covariates: ... H  trailing dim is covariate dim
        noise_suppression: H - clip away values under this magnitude
    """
    original_shape = covariates.shape
    covariates = covariates.flatten(start_dim=0, end_dim=-2)
    norm = {}
    norm['cov_mean'] = covariates.mean(dim=0)
    norm['cov_min'] = torch.quantile(covariates, 0.001, dim=0)
    norm['cov_max'] = torch.quantile(covariates, 0.999, dim=0)
    rescale = norm['cov_max'] - norm['cov_min']
    rescale[torch.isclose(rescale, torch.tensor(0.))] = 1
    covariates = (covariates - norm['cov_mean']) / rescale
    covariates = torch.clamp(covariates, -1, 1)
    return covariates.reshape(original_shape), norm

def chop_vector(vec: T, chop_size_ms: int, bin_size_ms: int) -> T:
    # vec - T H
    # vec - already at target resolution, just needs chopping. e.g. useful for covariates that have been externally downsampled
    if vec is None:
        return None
    chops = round(chop_size_ms / bin_size_ms)
    if vec.size(0) <= chops:
        return rearrange(vec, 'time hidden -> 1 time hidden')
    else:
        return rearrange(
            vec.unfold(0, chops, chops),
            'trial hidden time -> trial time hidden'
            ) # Trial x C x chop_size (time)

def compress_vector(vec: torch.Tensor, chop_size_ms: int, bin_size_ms: int, compression='sum', sample_bin_ms=1, keep_dim=True):
    # vec: at sampling resolution of 1ms, T C. Useful for things that don't have complicated downsampling e.g. spikes.
    # chop_size_ms: chop size in ms
    # bin_size_ms: bin size in ms - target bin size, after comnpression
    # sample_bin_ms: native res of vec

    if chop_size_ms:
        if vec.size(0) < chop_size_ms // sample_bin_ms:
            # No extra chop needed, just directly compress
            full_vec = vec.unsqueeze(0)
            # If not divisible by subsequent bin, crop
            if full_vec.shape[1] % (bin_size_ms // sample_bin_ms) != 0:
                full_vec = full_vec[:, :-(full_vec.shape[1] % (bin_size_ms // sample_bin_ms)), :]
            full_vec = rearrange(full_vec, 'b time c -> b c time')
        else:
            full_vec = vec.unfold(0, chop_size_ms // sample_bin_ms, chop_size_ms // sample_bin_ms) # Trial x C x chop_size (time)
        full_vec = rearrange(full_vec, 'b c (time bin) -> b time c bin', bin=bin_size_ms // sample_bin_ms)
        if compression != 'last':
            out_str = 'b time c 1' if keep_dim else 'b time c'
            return reduce(full_vec, f'b time c bin -> {out_str}', compression)
        if keep_dim:
            return full_vec[..., -1:]
        return full_vec[..., -1]
    else:
        if vec.shape[0] % (bin_size_ms // sample_bin_ms) != 0:
            vec = vec[:-(vec.shape[0] % (bin_size_ms // sample_bin_ms))]
        vec = rearrange(vec, '(time bin) c -> time c bin', bin=bin_size_ms // sample_bin_ms)
        if compression != 'last':
            out_str = 'time c 1' if keep_dim else 'time c'
            return reduce(vec, f'time c bin -> {out_str}', compression)
        if keep_dim:
            return vec[..., -1:]
        return vec[..., -1]



def spike_times_to_dense(spike_times_ms: List[np.ndarray], bin_size_ms: int, time_start=0, time_end=0) -> torch.Tensor:
    # spike_times_ms: List[Channel] of spike times, in ms from trial start
    # return: Time x Channel x 1, at bin resolution
    # Create at ms resolution
    if time_end != 0:
        spike_times_ms = [s[s < time_end] if s is not None else s for s in spike_times_ms]
    else:
        time_end = max([s[-1] if s is not None else s for s in spike_times_ms])
    dense_bin_count = math.ceil(time_end - time_start)
    if time_start != 0:
        spike_times_ms = [s[s >= time_start] - time_start if s is not None else s for s in spike_times_ms]

    trial_spikes_dense = torch.zeros(len(spike_times_ms), dense_bin_count, dtype=torch.uint8)
    for channel, channel_spikes_ms in enumerate(spike_times_ms):
        if channel_spikes_ms is None:
            continue
        trial_spikes_dense[channel] = torch.bincount(torch.as_tensor(np.round(channel_spikes_ms), dtype=torch.int), minlength=trial_spikes_dense.shape[1])
    trial_spikes_dense = trial_spikes_dense.T # Time x Channel
    return compress_vector(trial_spikes_dense, 0, bin_size_ms)

class PackToChop:
    r"""
        Accumulates data and saves to disk when data reaches chop length.
        General utility.
    """
    def __init__(self, chop_size, save_dir: Path):
        self.chop_size = chop_size
        self.queue = []
        self.running_length = 0
        self.paths = []
        self.save_dir = save_dir
        self.idx = 0
        self.prefix = ""
        # Remove all files directory
        for p in self.save_dir.glob("*.pth"):
            p.unlink()

    def get_paths(self):
        return list(self.save_dir.glob("*.pth"))

    def pack(self, payload):
        self.queue.append(payload)
        self.running_length += payload[DataKey.spikes][list(payload[DataKey.spikes].keys())[0]].shape[0]
        while self.running_length >= self.chop_size:
            self.flush()

    def flush(self):
        if len(self.queue) == 0 or self.running_length == 0:
            return
        # assert self.running_length >= self.chop_size, "Queue length should be at least chop size"
        payload = {}
        crop_last = max(self.running_length - self.chop_size, 0) # This is the _excess_ - i.e. crop as tail. Max: Keep logic well behaved for manual flush calls.
        if crop_last:
            # split the last one
            last = self.queue[-1]
            include, exclude = {}, {}
            for k in last.keys():
                if k == DataKey.spikes:
                    include[k] = {k2: v[:-crop_last] for k2, v in last[DataKey.spikes].items()}
                    exclude[k] = {k2: v[-crop_last:] for k2, v in last[DataKey.spikes].items()}
                elif k == DataKey.bhvr_vel:
                    include[k] = last[k][:-crop_last]
                    exclude[k] = last[k][-crop_last:]
                else:
                    include[k] = last[k]
                    exclude[k] = last[k]
            self.queue[-1] = include

        for key in self.queue[0].keys():
            if key == DataKey.spikes: # Spikes need special treatment
                payload[key] = {}
                for k in self.queue[0][key].keys():
                    payload[key][k] = torch.cat([p[key][k] for p in self.queue])
            elif key == DataKey.bhvr_vel: # Also timeseries
                payload[key] = torch.cat([p[key] for p in self.queue])
            else:
                payload[key] = self.queue[0][key]
        # print(payload[DataKey.bhvr_vel].shape, payload[DataKey.spikes]['Jenkins-M1'].shape)
        torch.save(payload, self.save_dir / f'{self.prefix}{self.idx}.pth')
        self.idx += 1
        if crop_last:
            self.queue = [exclude]
        else:
            self.queue = []
        self.running_length = crop_last