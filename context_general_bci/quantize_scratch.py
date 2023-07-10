#%%
# Generate some random pseudo-zscores

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

QUANTIZE_CLASSES = 32
quantile_buckets = torch.linspace(-1.6, 1.6, QUANTIZE_CLASSES)
test_inputs = torch.randn((100, 10))
def symlog(x):
    return np.sign(x) * np.log(1 + np.abs(x))
def unsymlog(x):
    return np.sign(x) * (np.exp(np.abs(x)) - 1)
# print(quantile_buckets)

# Quantized
test_quantize = torch.bucketize(symlog(test_inputs), quantile_buckets)
print(test_quantize.unique())

# test_quantize = torch.bucketize(test_inputs, log_quantile_buckets)
# print(extended_quantile_buckets)
# instead of representing as buckets, represent as bucket mean
# unquantize = extended_quantile_buckets[test_quantize]
print(quantile_buckets)
print(symlog(test_inputs).max())
print(symlog(test_inputs).min())
extended_quantile_buckets = torch.cat([torch.tensor([-4]), quantile_buckets, torch.tensor([4])])
unquantize = (extended_quantile_buckets[test_quantize] + extended_quantile_buckets[test_quantize + 1]) / 2
unquantize = unsymlog(unquantize)

ax = sns.scatterplot(x=test_inputs.flatten(), y=unquantize.flatten())
ax.set(xlabel='Input', ylabel='Unquantized')
# add grid
ax.grid(True)
# sns.distplot(test_quantize.flatten())