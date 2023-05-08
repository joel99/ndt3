#%%
import logging
import sys
# logging.basicConfig(stream=sys.stdout, level=logging.WARNING) # needed to get `logger` to print
# logging.basicConfig(stream=sys.stdout, level=logging.INFO) # needed to get `logger` to print
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from copy import deepcopy
import pytorch_lightning as pl
from einops import rearrange

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
    # f'online_bci',
    f'pitt_v3/probe_01_cross',
]

queries = [
    # 'human_obs_limit',
    'human_obs_m5',
    'human_obs_m1',
    'human_m5',
    'human_m1',
    # 'human_obs_m5_lr1e5', # note this LR is infeasibly slow for RT. Takes ~46 minutes.
    # 'human_obs_m75',
    # 'human_m5',
    # 'human_m5_lr1e5',
    # 'human_rtt_task_init',
    'human_rtt_pitt_init',
    'human_rtt_pitt_init_m1',
    # 'human_rtt_scratch',
    'crs07_m5',
    'crs07_m1',
    # 'crs07_m5_itertest',
    # 'human_unsup',
    # 'human_aug',
    # 'online_test_tune',
]

trainer = pl.Trainer(accelerator='gpu', devices=1, default_root_dir='./data/tmp')
runs_kin = wandb_query_experiment(EXPERIMENTS_KIN, order='created_at', **{
    "state": {"$in": ['finished', 'failed', 'crashed']},
})
print(f'Found {len(runs_kin)} runs. Evaluating on {len(EVAL_ALIASES)} datasets.')
USE_THRESH = False
USE_THRESH = True
eval_data = f'pitt_kin_df_{"thresh" if USE_THRESH else "unthresh"}.pt'

USE_SECOND_HALF_ONLY = False
USE_SECOND_HALF_ONLY = True # quick sanity check to see that results improve with time. Needed to explain why we're worse than KF baseline all the time

DO_SUB_FBC = False
# DO_SUB_FBC = True
if DO_SUB_FBC:
    query = 'human_10l-j7mq2snc'
    wandb_run = wandb_query_latest(query, allow_running=True, use_display=True)[0]
    task_model, task_cfg, task_attrs = load_wandb_run(wandb_run, )
#%%

def get_evals(model: BrainBertInterface, dataloader, runs=8, mode='nll'):
    evals = []
    for i in range(runs):
        pl.seed_everything(i)
        if 'kin_r2' in mode:
            # ? Ehm... not sure if buggy.
            model.cfg.task.outputs = [Output.behavior, Output.behavior_pred]
            heldin_outputs = stack_batch(trainer.predict(model, dataloader))
            if DO_SUB_FBC:
                offset_bins = 2
            else:
                offset_bins = model.task_pipelines[ModelTask.kinematic_decoding.value].bhvr_lag_bins
            if isinstance(heldin_outputs[Output.behavior_pred], list):
                if USE_SECOND_HALF_ONLY:
                    pred = np.concatenate([p[p.shape[0] // 2:] for p in heldin_outputs[Output.behavior_pred]])
                    true = np.concatenate([t[t.shape[0] // 2:] for t in heldin_outputs[Output.behavior]])
                else:
                    pred = np.concatenate([p[offset_bins:] for p in heldin_outputs[Output.behavior_pred]])
                    true = np.concatenate([t[offset_bins:] for t in heldin_outputs[Output.behavior]])
            else:
                start = heldin_outputs[Output.behavior_pred].shape[1] // 2 if USE_SECOND_HALF_ONLY else offset_bins
                pred = heldin_outputs[Output.behavior_pred][:, start:].flatten(end_dim=-2)
                true = heldin_outputs[Output.behavior][:,start:].flatten(end_dim=-2)
            pred = pred[(true != model.data_attrs.pad_token).any(-1)]
            true = true[(true != model.data_attrs.pad_token).any(-1)]
            if USE_THRESH:
                pred = pred[(np.abs(true) > model.cfg.task.behavior_metric_thresh).any(-1)]
                true = true[(np.abs(true) > model.cfg.task.behavior_metric_thresh).any(-1)]
            # compute r2
            return r2_score(true, pred)

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
        dataset = SpikingDataset(cfg.dataset, use_augment=False)
        dataset.subset_split(splits=['eval'])
    print(dataset.cfg.datasets)
    dataset.build_context_index()
    data_attrs = dataset.get_data_attrs()
    set_limit = run.config['dataset']['scale_limit_per_eval_session']
    cfg.model.task.tasks = [ModelTask.kinematic_decoding] # remove stochastic shuffle
    if USE_THRESH:
        cfg.model.task.metrics = [Metric.kinematic_r2, Metric.kinematic_r2_thresh]
        cfg.model.task.behavior_fit_thresh = 0.1
    model = transfer_model(src_model, cfg.model, data_attrs)
    if DO_SUB_FBC:
        from copy import deepcopy
        from context_general_bci.model_decode import transfer_model as decode_transfer
        extra_embed_map = {'task': (task_model.task_embed, task_model.data_attrs)}
        deployed_data_attrs = deepcopy(model.data_attrs)
        if 'task' in extra_embed_map:
            # Unlike in pretraining, we keep the session embed when switching tasks (a fault of our prertaining. We didn't prep common day labels).
            deployed_data_attrs.context.task = [ExperimentalTask.fbc]
        # model = transfer_model(model, model.cfg, deployed_data_attrs, extra_embed_map=extra_embed_map)
        model = decode_transfer(model, model.cfg, deployed_data_attrs, extra_embed_map=extra_embed_map)
    # dataloader = get_dataloader(dataset, num_workers=0, batch_size=100)
    dataloader = get_dataloader(dataset, num_workers=0, batch_size=1 if DO_SUB_FBC else 100)

    # the dataset name is of the for {type}_{subject}_session_{session}_set_{set}_....mat
    # parse out the variables
    _, subject, session, set_num, *_ = dataset.cfg.eval_datasets[0].split('_')

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
        experiment_set = run.config['experiment_set']
        if variant not in queries:
            continue
        if (
            variant,
            # dataset,
            run.config['model']['lr_init'],
            experiment_set
        ) in seen_set:
            continue
        print('evaling on', run.name)
        src_model, cfg, data_attrs = load_wandb_run(run, tag='val_loss')
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
        ref_df = SpikingDataset(cfg.dataset, use_augment=False)
        tv_ref = deepcopy(ref_df)
        eval_ref = deepcopy(ref_df)
        eval_ref.subset_split(splits=['eval'])
        tv_ref.subset_split()
        train_ref, val_ref = tv_ref.create_tv_datasets()

        for i, dataset in enumerate(EVAL_ALIASES):
            if 'crs07' in variant and 'CRS07' not in dataset:
                continue # special case sub-evaluation
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
            payload = get_single_payload(cfg, src_model, run, experiment_set, mode=mode, dataset=inst_df)
            df.append(payload)
            # seen_set[(variant, dataset, run.config['model']['lr_init']), experiment_set] = True
        seen_set[(variant, run.config['model']['lr_init'], experiment_set)] = True
    return pd.DataFrame(df)
kin_df = build_df(runs_kin, mode='kin_r2')
kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'series', 'data_id'])
kin_df.drop(columns=['lr'])

# # %%
# kin_df = kin_df.sort_values('kin_r2', ascending=False).drop_duplicates(['variant', 'series'])
# kin_df.drop(columns=['lr'])
# print(kin_df)

df = pd.concat([kin_df, comp_df])
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
#%%
df = torch.load(eval_data)
# print(df)
# map kf ids to the correct abbreviated variant
# Are we actually better or worse than Pitt baselines?
# intersect unique data ids, to get the relevant test set. Also, only compare nontrivial KF slots
kf_ids = df[df['variant'] == 'kf_base']['data_id'].unique()
model_ids = df[df['variant'] != 'kf_base']['data_id'].unique()
nontrivial_ids = df[(df['variant'] == 'kf_base') & (df['kin_r2'] > 0)]['data_id'].unique()
intersect_ids = np.intersect1d(kf_ids, model_ids)
intersect_ids = np.intersect1d(intersect_ids, nontrivial_ids)

sub_df = df[df['data_id'].isin(intersect_ids)]

print(sub_df.groupby(['variant']).mean().sort_values('kin_r2', ascending=False))

#%%
# make pretty seaborn default
subject = 'CRS02b'
subject = 'CRS07'
subject_df = sub_df[sub_df['subject'] == subject]
sns.set_theme(style="whitegrid")
# boxplot
order = sorted(subject_df.variant.unique())
palette = sns.color_palette("mako_r", len(order))
ax = sns.boxplot(data=subject_df, x='variant', y='kin_r2', order=order)
# sns.swarmplot(data=subject_df, x='variant', y='kin_r2', hue=, order=order, ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel('Vel R2')
ax.set_xlabel('Model variant')
ax.set_title(f'{subject} Perf ({EXPERIMENTS_KIN[0]}) ({"thresh" if USE_THRESH else ""}, {"second half" if USE_SECOND_HALF_ONLY else ""})')
# Rotate xlabels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

# 10 yticks
ax.set_yticks(np.linspace(0, 1, 11))

#%%
print(kin_df.groupby(['variant']).mean().sort_values('kin_r2', ascending=False))

#%%
one_one_df = sub_df[sub_df['variant'].isin(['kf_base', 'human_rtt_pitt_init'])]
# one_one_df = sub_df[sub_df['variant'].isin(['kf_base', 'human_m5'])]
g = sns.catplot(data=one_one_df, col='data_id', x='variant', y='kin_r2', kind='bar', col_wrap=4)

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
# Reshape the dataframe using pivot_table
scatter_df = one_one_df.pivot_table(index='data_id', columns='variant', values='kin_r2').reset_index()
# Create scatter plot
scatter_variants = scatter_df.columns[1:]
if scatter_variants[0] != 'kf_base':
    scatter_variants = scatter_variants[::-1]
sns.scatterplot(data=scatter_df, x=scatter_variants[0], y=scatter_variants[1], hue='data_id', legend=False)
# sns.scatterplot(data=scatter_df, x='kf_base', y='human_m5', hue='data_id', legend=False)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

# Add labels and diagonal reference line
plt.xlabel(f'{scatter_variants[0]} Kin R2')
plt.ylabel(f'{scatter_variants[1]} Kin R2')
plt.title('Performance Comparison of KF Base and Human M5')
# Seems like there might be some data where model has no training data at all, unluckily. But that contributes maybe 0.01 drop at most.

#%%
from scipy import stats

# Perform paired t-test
t, p = stats.ttest_rel(scatter_df['human_m5'], scatter_df['kf_base'])

# Print test results
if p < 0.05:
    print("Human M5 performance is significantly greater than KF Base performance (p = {:.3f})".format(p))
else:
    print("There is no significant difference between Human M5 and KF Base performance (p = {:.3f})".format(p))
#%%
print(df[df['data_id'] == 'CRS07_157_5'])
# %%
