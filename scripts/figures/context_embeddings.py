#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model, logger

from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt
from context_general_bci.utils import  get_wandb_run, wandb_query_latest

pl.seed_everything(0)


query = 'scale1_1s-z51g0amc'
query = 'human_rtt_task_init-xgcgl55d'
query = 'human_rtt_pitt_init-2dbglumn'
wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
print(wandb_run.id)

src_model, cfg, data_attrs = load_wandb_run(wandb_run, tag='val_loss')
print(cfg.dataset.datasets)


#%%
# Scatter of embedding distance vs temporal distance
import numpy as np
from datetime import datetime
# print(src_model.session_embed.weight) # Session x Dim
# print(data_attrs.context.session) # Session list of form 'ExperimentalTask.odoherty_rtt-Indy-YYYYMMDD_etc'

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from datetime import datetime

subject_targets = ['Indy', 'Loco'] # , 'CRS02b', 'CRS07']
# Convert session string to datetime object
def session_to_datetime(subject, session):
    if subject in ['Indy', 'Loco']:
        # Assuming the date is the next element after 'Indy'
        date_str = session.split('-')[2].split('_')[0]
        return datetime.strptime(date_str, '%Y%m%d')
    elif subject in ['CRS02b', 'CRS07']:
        # Assuming the date is the next element after 'CRS'
        date_str = session.split('-')[1].split('_')[0]
        return datetime.strptime(date_str, '%Y%m%d')
        # TODO...
    else:
        raise ValueError(f'Unknown subject {subject}')

def get_distances_for_subject(subject_filter):
    # Calculate temporal distances
    session_filter_mask = np.array([subject_filter in session for session in data_attrs.context.session])
    print(f'Found {session_filter_mask.sum()} sessions for {subject_filter}')
    session_datetimes = [session_to_datetime(subject_filter, session) for session in np.array(data_attrs.context.session)[session_filter_mask]]
    temporal_distances = np.zeros((len(session_datetimes), len(session_datetimes)))

    for i, d1 in enumerate(session_datetimes):
        for j, d2 in enumerate(session_datetimes):
            temporal_distances[i, j] = (d1 - d2).days
    # Extract unique pairs
    temporal_distances = np.abs(temporal_distances[np.triu_indices(temporal_distances.shape[0], k = 1)])

    # Calculate embedding distances
    embeddings = src_model.session_embed.weight.detach().numpy()
    embeddings = embeddings[session_filter_mask]
    embedding_distances = cosine_distances(embeddings)

    # Extract unique pairs
    embedding_distances = embedding_distances[np.triu_indices(embedding_distances.shape[0], k = 1)]
    return temporal_distances, embedding_distances

temp_ds, emb_ds = zip(*[get_distances_for_subject(subject) for subject in subject_targets])
subjects = np.concatenate([[subject] * len(t) for subject, t in zip(subject_targets, temp_ds)])

temp_ds = np.concatenate(temp_ds)
emb_ds = np.concatenate(emb_ds)

# Throw into a df
import pandas as pd
df = pd.DataFrame({'temporal_distance': temp_ds, 'embedding_distance': emb_ds, 'subject': subjects})

# Scatter plot
ax = prep_plt()
sns.scatterplot(data=df, x='temporal_distance', y='embedding_distance', hue='subject', ax=ax, s=8, alpha=0.8)

# ax.set_xscale('log') # Confusing...
ax.set_xlabel('Days apart')
ax.set_ylabel('Embedding L2')
ax.set_title('Scatter plot of Embedding Distance vs Temporal Distance')

#%%
ax = prep_plt()
sns.histplot(cosine_distances(src_model.session_embed.weight.detach().numpy()), bins=50, legend=False, ax=ax)

#%%
ax = prep_plt()
sns.histplot(cosine_distances(torch.randn_like(src_model.session_embed.weight.detach()).numpy()), bins=50, legend=False, ax=ax)