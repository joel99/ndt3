import os
from .loader import loadmat
from .halton import generate_search
from .grid_search import grid_search

# Has dependencies on typedefs in Config but hopefully that's not a huge issue.
from .ckpts_and_wandb_helpers import *

def suppress_default_registry():
    os.environ['NDT_SUPPRESS_DEFAULT_REGISTRY'] = '1'

def enum_backport(old_inst, new_enum_cls):
    # We run many enum checks but also migrated class modules at some point -- python doesn't recognize them as equal
    # so we add a cast
    return new_enum_cls[old_inst.name]