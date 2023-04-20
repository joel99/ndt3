#%%
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.WARNING) # needed to get `logger` to print
# logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
import pytorch_lightning as pl
from einops import rearrange

# Load BrainBertInterface and SpikingDataset to make some predictions
from context_general_bci.config import RootConfig, ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey
from context_general_bci.dataset import SpikingDataset, DataAttrs
from context_general_bci.model import transfer_model, logger

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
comp_df['data_id'] = comp_df['subject'] + '_' + comp_df['session'].astype(str) + '_' + comp_df['set'].astype(str)



EVAL_DATASETS = [
    # 'observation_CRS02bLab_session_19.*',
    'observation_CRS07Lab_session_15.*',
    'observation_CRS07Lab_session_16.*',
]
# expand by querying alias
EVAL_DATASETS = SpikingDataset.list_alias_to_contexts(EVAL_DATASETS)
EVAL_ALIASES = [x.alias for x in EVAL_DATASETS]

EXPERIMENTS_KIN = [
    # f'pitt_v2/probe',
    f'pitt_v2/probe_01_cross',
    # f'pitt_v2/probe_01',
]

queries = [
    # 'human_obs_limit',
    'human_obs',
    'human_obs_m5',
    # 'human_obs_m75',
    # 'human',
    # 'human_m5',
    # 'human_unsup',
]

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed']},
})
print(f'Found {len(runs_kin)} runs. Evaluating on {len(EVAL_ALIASES)} datasets.')

#%%
USE_THRESH = False
USE_THRESH = True
def get_evals(model, dataloader, runs=8, mode='nll'):
    evals = []
    for i in range(runs):
        pl.seed_everything(i)
        heldin_metrics = stack_batch(trainer.test(model, dataloader, verbose=False))
        if mode == 'nll':
            test = heldin_metrics['test_infill_loss'] if 'test_infill_loss' in heldin_metrics else heldin_metrics['test_shuffle_infill_loss']
        else:
            if USE_THRESH:
                test = heldin_metrics['test_kinematic_r2_thresh']
            else:
                test = heldin_metrics['test_kinematic_r2']
        test = test.mean().item()
        evals.append({
            'seed': i,
            mode: test,
        })
    return pd.DataFrame(evals)[mode].mean()

def get_single_payload(cfg: RootConfig, src_model, run, experiment_set, mode='nll', dataset=None):
    if dataset is None:
        dataset = SpikingDataset(cfg.dataset)
        dataset.subset_split(splits=['eval'])
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    set_limit = run.config['dataset']['scale_limit_per_eval_session']
    cfg.model.task.tasks = [ModelTask.kinematic_decoding] # remove stochastic shuffle
    if USE_THRESH:
        cfg.model.task.metrics = [Metric.kinematic_r2, Metric.kinematic_r2_thresh]
        cfg.model.task.behavior_fit_thresh = 0.1
    model = transfer_model(src_model, cfg.model, data_attrs)
    dataloader = get_dataloader(dataset)

    # the dataset name is of the for {type}_{subject}_session_{session}_set_{set}_....mat
    # parse out the variables
    _, subject, _, session, _, set_num, *_ = dataset.cfg.eval_datasets[0].split('_')

    payload = {
        'limit': set_limit,
        'variant': run.name.split('-')[0],
        'series': experiment_set,
        'data_id': f"{subject}_{session}_{set_num}",
        'subject': subject,
        'session': int(session),
        'set': int(set_num),
        'lr': run.config['model']['lr_init'], # swept
    }
    payload[mode] = get_evals(model, dataloader, mode=mode, runs=1 if mode != 'nll' else 8)
    return payload

def build_df(runs, mode='nll'):
    df = []
    seen_set = {}
    for run in runs:
        variant, _frag, *rest = run.name.split('-')
        if variant not in queries:
            continue
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
        experiment_set = run.config['experiment_set']
        pl.seed_everything(seed=cfg.seed)
        # if (
        #     variant,
        #     run.config['dataset']['eval_datasets'][0],
        #     run.config['model']['lr_init'],
        #     experiment_set
        # ) in seen_set:
        #     continue
        # payload = get_single_payload(cfg, src_model, run, experiment_set, mode=mode)
        # df.append(payload)
        # seen_set[(variant, run.config['dataset']['eval_datasets'][0], run.config['model']['lr_init']), experiment_set] = True

        # Don't split into loop, we might be loading train data...
        # In order to get the correct eval split, we need to use the same set of datasets as train (splits are not per dataset)
        # So construct this and split it repeatedly
        ref_df = SpikingDataset(cfg.dataset)
        tv_ref = deepcopy(ref_df)
        eval_ref = deepcopy(ref_df)
        eval_ref.subset_split(splits=['eval'])
        tv_ref.subset_split()
        train_ref, val_ref = tv_ref.create_tv_datasets()

        for i, dataset in enumerate(EVAL_ALIASES):
            # We use val _and_ eval to try to be generous and match Pitt settings
            inst_df = deepcopy(ref_df)
            inst_df.cfg.eval_datasets = [dataset]
            inst_df.cfg.datasets = [dataset]
            inst_df.subset_by_key([EVAL_DATASETS[i].id], key=MetaKey.session)
            valid_keys = list(val_ref.meta_df[
                (val_ref.meta_df[MetaKey.session] == EVAL_DATASETS[i].id)
            ][MetaKey.unique]) + list(eval_ref.meta_df[
                (eval_ref.meta_df[MetaKey.session] == EVAL_DATASETS[i].id)
            ][MetaKey.unique])
            inst_df.subset_by_key(valid_keys, key=MetaKey.unique)
            # inst_df.subset_split(splits=['eval'])

            # val.subset_by_key([EVAL_DATASETS[i].id], key=MetaKey.session)
            experiment_set = run.config['experiment_set']
            if variant.startswith('sup') or variant.startswith('unsup'):
                experiment_set = experiment_set + '_' + variant.split('_')[0]
            if (
                variant,
                dataset,
                run.config['model']['lr_init'],
                experiment_set
            ) in seen_set:
                continue
            payload = get_single_payload(cfg, src_model, run, experiment_set, mode=mode, dataset=inst_df)
            df.append(payload)
            seen_set[(variant, dataset, run.config['model']['lr_init']), experiment_set] = True
    return pd.DataFrame(df)
kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'series', 'data_id'])
kin_df.drop(columns=['lr'])

# # %%
# kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'series'])
# kin_df.drop(columns=['lr'])
# print(kin_df)


df = pd.concat([kin_df, comp_df])
# Are we actually better or worse than Pitt baselines?
# intersect unique data ids, to get the relevant test set. Also, only compare nontrivial KF slots
kf_ids = df[df['variant'] == 'kf_base']['data_id'].unique()
model_ids = df[df['variant'] == 'human_obs']['data_id'].unique()
nontrivial_ids = df[(df['variant'] == 'kf_base') & (df['kin_r2'] > 0)]['data_id'].unique()
intersect_ids = np.intersect1d(kf_ids, model_ids)
intersect_ids = np.intersect1d(intersect_ids, nontrivial_ids)

sub_df = df[df['data_id'].isin(intersect_ids)]

print(sub_df.groupby(['variant']).mean().sort_values('kin_r2', ascending=False))

#%%
# make pretty seaborn default
subject = 'CRS02bLab'
# subject = 'CRS07Lab'
subject_df = sub_df[sub_df['subject'] == subject]
sns.set_theme(style="whitegrid")
# boxplot
order = sorted(subject_df.variant.unique())
ax = sns.boxplot(data=subject_df, x='variant', y='kin_r2', order=order)
ax.set_ylim(0, 1)
ax.set_ylabel('Vel R2')
ax.set_xlabel('Model variant')
ax.set_title(f'{subject} Perf ({EXPERIMENTS_KIN[0]}) ({"thresh" if USE_THRESH else ""})')
# Rotate xlabels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# 10 yticks
ax.set_yticks(np.linspace(0, 1, 11))

#%%
print(kin_df.groupby(['variant']).mean().sort_values('kin_r2', ascending=False))

#%%
g = sns.catplot(data=sub_df, col='data_id', x='variant', y='kin_r2', kind='bar', col_wrap=4)

def deco(data, **kwargs):
    # set min y to 0
    ax = plt.gca()
    ax = prep_plt(ax)
    ax.set_ylim(0, 1)
    # ax.set_xlabel('Target session trials')
    # ax.set_ylabel('Vel R2')

g.map_dataframe(deco)
# To facet grid
# g = sns.FacetGrid(data=sub_df, col='data_id', hue='variant', col_wrap=4)
# g.map_dataframe(sns.barplot, x='variant', y='kin_r2')

#%%
print(kin_df)