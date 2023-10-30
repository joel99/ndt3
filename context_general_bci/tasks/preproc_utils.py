from pathlib import Path
import torch
from einops import rearrange, reduce

from context_general_bci.config import DataKey

def chop_vector(vec: torch.Tensor | None, chop_size_ms: int, bin_size_ms: int):
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