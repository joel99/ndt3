- Data scale goes from 1K → 100K.
- Original model size was ~0.5M, but high dropouts mean effective 0.3M.
    - We scale this 100X to ~30M. Kaplan suggests we may want to vary, but Chinchilla says no, hold it equal.
- Compute should be held constant wrt data available, i.e. since we're not single-epoch, fix # of epochs.

Other things to study:
- basic arch questions (pre-norm, post-norm, mask ratio (MAE He 21), chunk size (likely unique to our domain))