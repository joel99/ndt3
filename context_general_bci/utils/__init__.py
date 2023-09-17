import os
from .loader import loadmat
from .halton import generate_search
from .grid_search import grid_search
import torch

# Has dependencies on typedefs in Config but hopefully that's not a huge issue.
from .ckpts_and_wandb_helpers import *

def suppress_default_registry():
    os.environ['NDT_SUPPRESS_DEFAULT_REGISTRY'] = '1'

def enum_backport(old_inst, new_enum_cls):
    # We run many enum checks but also migrated class modules at some point -- python doesn't recognize them as equal
    # so we add a cast
    return new_enum_cls[old_inst.name]

def sort_A_by_B(A: torch.Tensor, B: torch.Tensor, indices: torch.Tensor | None = None):
    # Generally expecting Batch T * dimensions
    # Sort B along the Time dimension (dim=1) and get the sorting indices
    _, indices = torch.sort(B, dim=1)
    # Sort A using the sorting indices obtained from B
    if indices.ndim != A.ndim:
        indices = indices.unsqueeze(-1).expand(-1, -1, A.shape[-1])
    A_sorted = torch.gather(A, 1, indices)
    return A_sorted, indices
