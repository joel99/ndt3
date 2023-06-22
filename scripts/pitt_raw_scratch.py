#%%
r"""
    TODO
    Identify .mat files for relevant trials
    Open .mat
    Find the phases (observation) that are relevant for now
    Find the spikes
    Find the observed kinematic traces

    Batch for other sessions
"""
import pandas as pd
import numpy as np
# import xarray as xr
from pathlib import Path
import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import torch

from context_general_bci.utils import loadmat

data_dir = Path("./data/sample_crs02b_data/mat/CRS02b_test")
mat_file = list(data_dir.glob(f'QL*.mat'))[2]
print(mat_file)
payload = loadmat(mat_file)

#%%
print(payload['data'].keys())
print(payload['data']['TaskStateMasks'].keys())
print(len(payload['data']['TaskStateMasks']['target']))
print(payload['data']['TaskStateMasks']['target'][1]) # 30 x Time
print(np.array(payload['data']['SpikeCount']).shape) # Time x Channels
# How can I tell when the user has control? Just filter for nonzero velocity.

# Found it. This is the spike target.


# print(payload['data']['TaskStateMasks'].keys())
# print(payload['data']['Kinematics'].keys())
#%%
print(payload['iData']['QL']['Data']['TASK_STATE_CONFIG']['target'])
print(len(payload['iData']['QL']['Data']['TASK_STATE_CONFIG']['target']))
print(len(payload['iData']['QL']['Data']['TASK_STATE_CONFIG']['target'][0]))
print(len(payload['iData']['QL']['Data']['TASK_STATE_CONFIG']['target'][1]))
# Target seems to be a 30x7. Status: I don't know what the second dimension is for.