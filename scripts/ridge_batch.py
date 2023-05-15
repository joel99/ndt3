#%%
# Compute a ridge comparison score.
# TODO could be a _lot_ more efficient by just checking set inventory for the actual relevant sessions instead of computing for all of them...

from typing import List
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf
from einops import rearrange
import pytorch_lightning as pl
from sklearn.linear_model import Ridge # note default metric is r2
from sklearn.model_selection import GridSearchCV

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from copy import deepcopy
import pytorch_lightning as pl
from einops import rearrange

from context_general_bci.contexts import context_registry
from context_general_bci.config import DatasetConfig, DataKey, MetaKey
from context_general_bci.config.presets import FlatDataConfig
from context_general_bci.dataset import SpikingDataset
from context_general_bci.subjects import SubjectInfo, create_spike_payload

from context_general_bci.analyze_utils import prep_plt, DataManipulator
from context_general_bci.tasks.pitt_co import load_trial, PittCOLoader

USE_RAW = False
USE_CAT_SPIKES = False
USE_CAT_SPIKES = True
LAG_MS = 0

# We pull in one actual experiment to make sure we have the right data splits. We run a ridge regression in the loop

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model, logger, BrainBertInterface

from context_general_bci.analyze_utils import stack_batch, load_wandb_run
from context_general_bci.analyze_utils import prep_plt, get_dataloader
from context_general_bci.utils import wandb_query_experiment, get_wandb_run, wandb_query_latest

pl.seed_everything(0)

def get_clean_comp(csv_path):
    local_scores = pd.read_csv(csv_path)
    # R2 is currently either a nan, a single number, or a string with two numbers, parse out the average
    def try_cast(str_or_nan):
        try:
            if isinstance(str_or_nan, float) and np.isnan(str_or_nan):
                return str_or_nan
            elif isinstance(str_or_nan, str) and len(str_or_nan.split(',')) == 1:
                return float(str_or_nan)
            else:
                return np.mean([float(y) for y in str_or_nan.split(',')])
        except:
            return np.nan
    local_scores['R2'] = local_scores['R2'].apply(try_cast)
    # drop rows with type != 'obs'
    local_scores = local_scores[local_scores['Type'] == 'Obs']
    comp_df = local_scores[['Session', 'Sets', 'R2']]
    comp_df = comp_df.rename(columns={'Session': 'session', 'Sets': 'set', 'R2': 'kin_r2'})
    comp_df = comp_df.astype({
        'set': 'int64'
    })
    comp_df['limit'] = 0
    comp_df['variant'] = 'kf_base'
    comp_df['series'] = 'kf_base'
    return comp_df
crs02b_df = get_clean_comp('./scripts/figures/CRS02bSetInventory.csv')
crs02b_df['subject'] = 'CRS02bLab'
crs07_df = get_clean_comp('./scripts/figures/CRS07SetInventory.csv')
crs07_df['subject'] = 'CRS07Lab'
comp_df = pd.concat([crs02b_df, crs07_df])
comp_df['data_id'] = comp_df['subject'].replace('Lab', '').replace('Home', '') \
    + '_' + comp_df['session'].astype(str) + '_' + comp_df['set'].astype(str)


EVAL_DATASETS = [
    'observation_CRS02b_19.*',
    # 'observation_CRS02b_1953_9',
    'observation_CRS07_15.*',
    'observation_CRS07_16.*',
]
# expand by querying alias
EVAL_DATASETS = SpikingDataset.list_alias_to_contexts(EVAL_DATASETS)
EVAL_ALIASES = [x.alias for x in EVAL_DATASETS]

EXPERIMENTS_KIN = [
    f'pitt_v3/probe_5_cross',
]

queries = [
    'human_m5',
]

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed']},
})
print(f'Found {len(runs_kin)} runs. Evaluating on {len(EVAL_ALIASES)} datasets.')
USE_THRESH = False
USE_THRESH = True
eval_data = f'pitt_kin_df_ridge_{"thresh" if USE_THRESH else "unthresh"}.pt'

#%%
def smooth_spikes(
    dataset: SpikingDataset, kernel_sd=80
) -> List[torch.Tensor]:
    # Smooth along time axis
    return [DataManipulator.gauss_smooth(
        rearrange(i[DataKey.spikes].float(), 't s c 1 -> 1 t (s c)'),
        bin_size=dataset.cfg.bin_size_ms,
        kernel_sd=kernel_sd,
    ).squeeze(0) for i in dataset]


def sweep_ridge_fit(train, val, test, cfg: ModelConfig, zero_filt_train=True, zero_filt_eval=True, spike_smth_range = [100, 200, 400, 600]):
    decoders = []
    vals = []

    train_behavior = np.concatenate([i[DataKey.bhvr_vel] for i in train], 0)
    val_behavior = np.concatenate([i[DataKey.bhvr_vel] for i in val], 0)
    test_behavior = np.concatenate([i[DataKey.bhvr_vel] for i in test], 0)

    def single_score(smth, train_bhvr, val_bhvr):
        decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-1, 3, 50)})

        smth_train = smooth_spikes(train, kernel_sd=smth)
        smth_val = smooth_spikes(val, kernel_sd=smth)
        train_rates = np.concatenate(smth_train, 0)
        val_rates = np.concatenate(smth_val, 0)

        if zero_filt_train:
            train_rates = train_rates[(np.abs(train_bhvr) > cfg.task.behavior_metric_thresh).any(-1)]
            train_bhvr = train_bhvr[(np.abs(train_bhvr) > cfg.task.behavior_metric_thresh).any(-1)]
        if zero_filt_eval:
            val_rates = val_rates[(np.abs(val_bhvr) > cfg.task.behavior_metric_thresh).any(-1)]
            val_bhvr = val_bhvr[(np.abs(val_bhvr) > cfg.task.behavior_metric_thresh).any(-1)]
        # print(train_rates.shape, train_bhvr.shape, eval_rates.shape, val_bhvr.shape)
        decoder.fit(train_rates, train_bhvr)
        return decoder.score(val_rates, val_bhvr), decoder

    for i in spike_smth_range:
        val_score, decoder = single_score(i, train_behavior, val_behavior)
        vals.append(val_score)
        decoders.append(decoder)
    # get best
    best_idx = np.argmax(vals)
    best_decoder = decoders[best_idx]
    smth_test = smooth_spikes(test, kernel_sd=spike_smth_range[best_idx])
    smth_rates = np.concatenate(smth_test, 0)
    smth_rates = smth_rates[(np.abs(test_behavior) > cfg.task.behavior_metric_thresh).any(-1)]
    test_behavior = test_behavior[(np.abs(test_behavior) > cfg.task.behavior_metric_thresh).any(-1)]
    return best_decoder.score(smth_rates, test_behavior)

def build_df(runs, mode='nll'):
    df = []
    run = runs[0]
    seen_set = {}
    variant, _frag, *rest = run.name.split('-')
    src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
    pl.seed_everything(seed=cfg.seed)

    ref_df = SpikingDataset(cfg.dataset, use_augment=False)
    tv_ref = deepcopy(ref_df)
    eval_ref = deepcopy(ref_df)
    eval_ref.subset_split(splits=['eval'])
    tv_ref.subset_split()
    train_ref, val_ref = tv_ref.create_tv_datasets()
    def create_sub_dataset(src_dataset, ref_dataset):
        inst_df = deepcopy(src_dataset)
        inst_df.cfg.eval_datasets = [dataset]
        inst_df.cfg.datasets = [dataset]
        inst_df.subset_by_key([EVAL_DATASETS[i].id], key=MetaKey.session)
        valid_keys = list(ref_dataset.meta_df[
            (ref_dataset.meta_df[MetaKey.session] == EVAL_DATASETS[i].id)
        ][MetaKey.unique])
        inst_df.subset_by_key(valid_keys, key=MetaKey.unique)
        return inst_df

    for i, dataset in enumerate(EVAL_ALIASES):
        if 'crs07' in variant and 'CRS07' not in dataset:
            continue # special case sub-evaluation
        if 'crs02b' in variant and 'CRS02b' not in dataset:
            continue
            # We use val _and_ eval to try to be generous and match Pitt settings
        train = create_sub_dataset(ref_df, train_ref)
        val = create_sub_dataset(ref_df, val_ref)
        test = create_sub_dataset(ref_df, eval_ref)

        pl.seed_everything(0)
        eval_r2 = sweep_ridge_fit(train, val, test, cfg=cfg.model)
        eval_no_filt_r2 = sweep_ridge_fit(train, val, test, cfg=cfg.model, zero_filt_train=False, zero_filt_eval=True)

        payload = {
            'ridge_filt' : eval_r2,
            'ridge_no_filt': eval_no_filt_r2,
            'dataset': dataset
        }
        print(payload)
        df.append(payload)
    return pd.DataFrame(df)
kin_df = build_df(runs_kin, mode='kin_r2')
#%%
# Drop rows with nan dataset
kin_df = kin_df.dropna(subset=['dataset'])

kin_df.loc[:, 'subject'] = kin_df['dataset'].apply(lambda x: x.split('_')[1])

kin_df.loc[:, 'data_id'] = kin_df['dataset'].apply(lambda x: '_'.join(x.split('_')[1:4] if bool(x) else ""))

df = pd.concat([kin_df, comp_df])

# NOPE, need to do some data casting
def abbreviate(data_id):
    pieces = data_id.split('_')
    if pieces[0].endswith('Lab'):
        pieces[0] = pieces[0].replace('Lab', '')
    elif data_id[0].endswith('Home'):
        pieces[0] = pieces[0].replace('Home', '')
    return '_'.join(pieces)

df.loc[df['variant'] == 'kf_base', 'data_id'] = df[df['variant'] == 'kf_base']['data_id'].apply(abbreviate)
df.loc[df['variant'] == 'kf_base', 'subject'] = df[df['variant'] == 'kf_base']['subject'].apply(abbreviate)

torch.save(df, eval_data) # for some reason notebook isn't loading, so force it with a shell call and load from here...

kf_ids = df[df['variant'] == 'kf_base']['data_id'].unique()
model_ids = df[df['variant'] != 'kf_base']['data_id'].unique()

nontrivial_ids = df[(df['variant'] == 'kf_base') & (df['kin_r2'] > 0)]['data_id'].unique()
intersect_ids = np.intersect1d(kf_ids, model_ids)
intersect_ids = np.intersect1d(intersect_ids, nontrivial_ids)

sub_df = df[df['data_id'].isin(intersect_ids)]

#%%
# print(sub_df[sub_df['variant'] != 'kf_base']['ridge_filt'].mean())
print(sub_df[sub_df['variant'] != 'kf_base']['ridge_no_filt'].mean()) # No filt edges out

#%%
print(sub_df[(sub_df['variant'] != 'kf_base') & (sub_df['subject'] == 'CRS02b')]['ridge_filt'].mean())
print(sub_df[(sub_df['variant'] != 'kf_base') & (sub_df['subject'] == 'CRS07')]['ridge_filt'].mean())