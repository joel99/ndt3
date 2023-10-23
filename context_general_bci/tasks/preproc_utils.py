from pathlib import Path
import torch
from einops import rearrange, reduce

from context_general_bci.config import DataKey, DatasetConfig


def chop_vector(vec: torch.Tensor, chop_size_ms: int, bin_size_ms: int):
    # vec - T H
    # vec - already at target resolution, just needs chopping
    chops = round(chop_size_ms / bin_size_ms)
    return rearrange(
        vec.unfold(0, chops, chops),
        'trial hidden time -> trial time hidden'
        ) # Trial x C x chop_size (time)

def compress_vector(vec: torch.Tensor, chop_size_ms: int, bin_size_ms: int, compression='sum'):
    # vec: at sampling resolution of 1ms, T C
    if chop_size_ms:
        full_vec = vec.unfold(0, chop_size_ms, chop_size_ms) # Trial x C x chop_size (time)
        return reduce(
            rearrange(full_vec, 'b c (time bin) -> b time c bin', bin=bin_size_ms),
            'b time c bin -> b time c 1', compression
        )
    else:
        return reduce(
            rearrange(vec, '(time bin) c -> time c bin', bin=bin_size_ms),
            'time c bin -> time c 1', compression
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
        if len(self.queue) == 0:
            return
        # assert self.running_length >= self.chop_size, "Queue length should be at least chop size"
        payload = self.queue[0]
        for key in payload.keys():
            if key == DataKey.spikes: # Spikes need special treatment
                payload[key] = {k: torch.cat([p[key][k] for p in self.queue]) for k in payload[key].keys()}
            elif key == DataKey.bhvr_vel: # Also timeseries
                payload[key] = torch.cat([p[key] for p in self.queue])
            else:
                pass # Just keep global args
        torch.save(payload, self.save_dir / f'{self.idx}.pth')
        self.idx += 1

        self.running_length = max(self.running_length - self.chop_size, 0) # Keep logic well behaved for manual flush calls
        if self.running_length:
            # Cut into this
            cropped = {}
            last = self.queue[-1]
            for k in payload.keys():
                if k == DataKey.spikes:
                    cropped[k] = {k2: v[self.running_length:] for k2, v in last[DataKey.spikes].items()}
                elif k == DataKey.bhvr_vel:
                    cropped[k] = last[k][self.running_length:]
                else:
                    cropped[k] = last[k]
            self.queue = [cropped]
        else:
            self.queue = []