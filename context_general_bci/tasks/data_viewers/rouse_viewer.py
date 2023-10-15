#%%
from pathlib import Path
# Unzipped https://figshare.com/articles/dataset/Processed_data_files_used_in_Schwartze_et_al_2023/23631951
# to ./data/rouse_precision and placed ./data_extracted at root.
data_dir = Path(
    'data/rouse_precision/monk_p/COT_SpikesCombined'
)

sample_file = data_dir.glob('*.mat').__next__()
print(sample_file)
#%%
print("hi")