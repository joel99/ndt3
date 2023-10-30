from typing import TypeVar
from pathlib import Path
import torch
from einops import rearrange, reduce

from context_general_bci.config import DataKey

T = TypeVar('T', torch.Tensor, None)

def chop_vector(vec: T, chop_size_ms: int, bin_size_ms: int) -> T:
    # vec - T H
    # vec - already at target resolution, just needs chopping. e.g. useful for covariates that have been externally downsampled
    if vec is None:
        return None
    chops = round(chop_size_ms / bin_size_ms)
    if chops == 0:
        return vec.unsqueeze(0)
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
        full_vec = vec.unfold(0, chop_size_ms // sample_bin_ms, chop_size_ms // sample_bin_ms) # Trial x C x chop_size (time)
        out_str = 'b time c 1' if keep_dim else 'b time c'
        return reduce(
            rearrange(full_vec, 'b c (time bin) -> b time c bin', bin=bin_size_ms // sample_bin_ms),
            f'b time c bin -> {out_str}', compression
        )
    else:
        out_str = 'time c 1' if keep_dim else 'time c'
        return reduce(
            rearrange(vec, '(time bin) c -> time c bin', bin=bin_size_ms // sample_bin_ms),
            f'time c bin -> {out_str}', compression
        )

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

    def get_paths(self):
        return [
            self.save_dir / f'{i}.pth' for i in range(self.idx)
        ]

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
        torch.save(payload, self.save_dir / f'{self.idx}.pth')
        self.idx += 1
        if crop_last:
            self.queue = [exclude]
        else:
            self.queue = []
        self.running_length = crop_last