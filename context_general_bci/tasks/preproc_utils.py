import torch
from einops import rearrange, reduce

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