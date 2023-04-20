import os
from .loader import loadmat
from .halton import generate_search
from .grid_search import grid_search

# Has dependencies on typedefs in Config but hopefully that's not a huge issue.
from .ckpts_and_wandb_helpers import *

def suppress_default_registry():
    os.environ['NDT_SUPPRESS_DEFAULT_REGISTRY'] = '1'