#%%

# Try using pyaldata repo
import pandas as pd
import numpy as np
# import xarray as xr
from pathlib import Path
import os
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from pyaldata import *

data_dir = Path("./data/gallego_co/")
# for fname in data_dir.glob("*.mat"):
    # df = mat2dataframe(fname, shift_idx_fields=True)
    # print(fname)
    # # print(df.columns)
    # for c in df.columns:
    #     if c.endswith('_spikes'):
    #         print(c, df[c][0].shape)
    # import pdb;pdb.set_trace()
# fname = os.path.join(data_dir, "Han_CO_20171207.mat")
# fname = os.path.join(data_dir, "Lando_CO_20171207.mat")
fname = os.path.join(data_dir, "Chewie_CO_20150313.mat")
df = mat2dataframe(fname, shift_idx_fields=True)

# df.head()
#%%
print("original: ", df.M1_spikes[0].shape)
print(df.columns)
print(df.head().bin_size)
print(df.vel[0].shape)
#%%
# print(df.M1_spikes[0].shape)
# print(len(df.M1_spikes))
# print(len(df))