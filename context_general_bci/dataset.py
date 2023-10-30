from typing import List, Any, Dict, Union
import copy
import json
import os
from pathlib import Path
from math import ceil
import itertools
import logging
from concurrent.futures import ThreadPoolExecutor

from omegaconf import OmegaConf
from dataclasses import dataclass, field
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from einops import rearrange, repeat

import lightning.pytorch as pl

from context_general_bci.config import DatasetConfig, MetaKey, DataKey, DEFAULT_KIN_LABELS
from context_general_bci.subjects import SubjectArrayRegistry
from context_general_bci.contexts import context_registry, ContextInfo
from context_general_bci.tasks import ExperimentalTask
from context_general_bci.augmentations import augmentations

r"""
    Stores range of contexts provided by a dataset.
    Data will serve attributes as index of provided contexts.
    The model should probably unique parameters for each context (JY thinks they should be embeddings).
    - `subject` context will _determine_ data shape.
    1. Simple is to get a linear layer per subject.
    2. Odd but maybe workable is to pad to some max length (e.g. 128, current generation Utah array).
    3. Stretch; too much of a separate research endeavor -- use Set networks to learn a subject-based cross-attention operator to downproject.
    These contexts are not independent. In fact, they're almost hierarchical.
    Subject -> Array -> Session, Task -> session.
"""

# Padding tokens
LENGTH_KEY = 'length'
CHANNEL_KEY = 'channel_counts'
# TODO deprecate when remodularizing data loaders... (where essentially we will just track data, position, trial, and pack densely)
COVARIATE_LENGTH_KEY = 'covariate_length' # we need another length tracker for padded sequences of covariates in the flat case
COVARIATE_CHANNEL_KEY = 'covariate_channel_counts' # essentially for heldout channels only (deprecated)

CONSTRAINT_LENGTH_KEY = 'constraint_length' # needed for sparse constraints
RETURN_LENGTH_KEY = 'return_length'

HELDOUT_CHANNEL_KEY = 'heldout_channel_counts'

r"""
    I really can't figure a good normalization scheme in light of the fact that we're supposed to be able to adapt to arbitrary magnitudes for ICL phase.
    For now, we will force a kind of registration with the understanding that data should be brought into a dynamic range of 0.1-10.
    Future covariates should have a normalization scheme that roughly respects this.
"""

logger = logging.getLogger(__name__)
@dataclass
class ContextAttrs:
    r"""
        Each of these can potentially be embedded
    """
    subject: List[str] = field(default_factory=list)
    # subject: List[SubjectName] = field(default_factory=list)
    array: List[str] = field(default_factory=list) # should be prefixed with subject
    session: List[str] = field(default_factory=list) # unique ID
    task: List[str] = field(default_factory=list) # experimental task
    # task: List[ExperimentalTask] = field(default_factory=list) # experimental task
    # Types are indeed the enums but if we swap dacite whines and can't parse from wandb

@dataclass
class DataAttrs:
    bin_size_ms: int
    spike_dim: int
    max_channel_count: int
    context: ContextAttrs
    max_arrays: int = 1 # patch, todo remove default

    # Task specific
    rtt_heldout_channel_count: int = 0 # Only for NLB, kinda hacky
    maze_heldout_channel_count: int = 0

    behavior_dim: int = 2 # This is the _max_ number of features expected, in NDT2 simply also the readout dim. Will compare first N dimensions if fewer are available.

    pad_token: int = 0
    # This pad token applies for both content and space in enc/dec flow
    # The "Time" associated with padding is determined on a per-modality basis (either min time or max time)
    # In decoder-only models, space has sorting priority, and "Space" for padding is also set to max position (hardcoded to 32)
    max_trial_length: int = 0

    serve_tokens: bool = False # if true, serves flat data tokens with additional keys for annotations (e.g. array + timestep) instead of structured data (e.g. with array dimensions)
    serve_tokens_flat: bool = False
    neurons_per_token: int = 8

    sparse_constraints: bool = False
    sparse_rewards: bool = False # also counts for return
    tokenize_covariates: bool = False
    semantic_covariates: bool = False

    @property
    def max_spatial_tokens(self):
        return max(self.behavior_dim, self.max_spatial_tokens_neural)

    @property
    def max_spatial_tokens_neural(self):
        per_array = ceil(self.max_channel_count / self.neurons_per_token)
        if self.serve_tokens:
            return per_array
        else:
            return per_array * self.max_arrays

class SpikingDataset(Dataset):
    r"""
        Generic container for spiking data from intracortical microelectrode recordings.
        Intended to wrap multiple (large) datasets, hence stores time series in disk, not memory.
        In order to be schema agnostic, we'll maintain metadata in a pandas dataframe and larger data (time series) will be stored in a file, per trial.
"        Some training modes may open a file and not use all info in said file, but if this turns into an issue we can revisit and split the files as well.

        Design considerations:
        - Try to be relatively agnostic (needs to deal with QuickLogger + NWB)
        # ? Will it be important to have a preprocessed cache? If trials are exploded, and we have distinct padding requirements, we might need per-trial processing. We might just store exploded values after preprocessing. But then, what if we need to update preprocessing?
        - Then we need to re-process + update exploded values. Simple as that.

        Some notes on metadata:
        - MetaKey.Subject column stores SubjectName (OrderedEnum), so that we can vet subjects exist before starting training. May work with SubjectInfo classes
        - Can we "mixin" time-varying data, or maybe simpler to just be a separate codepath in this class.
    """
    def __init__(self, cfg: DatasetConfig, use_augment: bool = True, override_preprocess_path=False, debug=False):
        super().__init__()
        if not isinstance(cfg, OmegaConf):
            cfg: DatasetConfig = OmegaConf.create(cfg)
        self.cfg = cfg
        assert DataKey.spikes in cfg.data_keys, "Must have spikes"
        if self.cfg.serve_tokenized_flat:
            assert self.cfg.serve_tokenized, 'codepaths assume serve_tokenized is true if serve_tokenized_flat is true'
        if self.cfg.datasets:
            contexts = self.list_alias_to_contexts(self.cfg.datasets)
            if getattr(self.cfg, 'data_blacklist', ''):
                # load txt
                with open(self.cfg.data_blacklist, 'r') as f:
                    blacklist = f.readlines()
                    blacklist = [b.strip() for b in blacklist]
                exclude_contexts = self.list_alias_to_contexts(blacklist)
            else:
                exclude_contexts = []
            if getattr(self.cfg, 'exclude_datasets', []):
                exclude_contexts.extend(self.list_alias_to_contexts(self.cfg.exclude_datasets))
            eval_contexts = self.list_alias_to_contexts(self.cfg.eval_datasets)
            exclude_contexts = [c for c in exclude_contexts if c not in eval_contexts]
            contexts = [c for c in contexts if c not in exclude_contexts]
            if not contexts:
                raise Exception(f"No contexts {self.cfg.datasets} left in dataset.")
            if debug or True:
                results = [self.load_single_session(c, override_preprocess_path=override_preprocess_path) for c in contexts]
            else: # Hm, thread pool also doesn't seem to surface errors properly, as segfaults on busy machines...?
                with ThreadPoolExecutor(max_workers=32) as executor: # Not processpool as it's mildly inconvenient to refactor our this preprocessing to a pickleable step right now
                    results = list(executor.map(lambda c: self.load_single_session(c, override_preprocess_path=override_preprocess_path), contexts))
            self.meta_df = pd.concat(results).reset_index(drop=True)
            # self.meta_df = pd.concat([self.load_single_session(c, override_preprocess_path=override_preprocess_path) for c in contexts]).reset_index(drop=True)

            if 'split' in self.meta_df.columns and len(self.meta_df['split'].unique()) > 1:
                logger.warning("Non-train splits found in meta_df. Subsetting is expected.")
        else:
            self.meta_df = None
        self.context_index = None
        self.subsetted = False
        breakpoint()
        self.max_bins = round(self.cfg.max_length_ms / self.cfg.bin_size_ms)
        self.mark_eval_split_if_exists()
        self.cache = {}
        self.z_score = torch.load(self.cfg.z_score) if self.cfg.z_score else None
        self.augment = use_augment and bool(self.cfg.augmentations)

    @property
    def loaded(self):
        return self.meta_df is not None

    @staticmethod
    def preprocess_path(cfg: DatasetConfig, session_path: Path, override_preprocess_path: bool) -> Path:
        if override_preprocess_path:
            return session_path.parent / session_path.stem / cfg.preprocess_suffix
        return cfg.root_dir / cfg.preprocess_suffix / session_path.relative_to(cfg.root_dir)

    def validate_meta(self, meta_df: pd.DataFrame):
        for k in self.cfg.meta_keys:
            if k == MetaKey.subject:
                unique_subjects = meta_df[MetaKey.subject].unique()
                for s in unique_subjects:
                    assert SubjectArrayRegistry.query_by_subject(s) is not None, f"Subject {s} not found registered."
            elif k == MetaKey.array:
                pass # no validation
            else:
                assert k in meta_df.columns, f"Requested meta key {k} not loaded in meta_df"

    def preproc_version(self, task: ExperimentalTask):
        version = {
            'max_trial_length': self.cfg.max_trial_length, # defunct
            'bin_size_ms': self.cfg.bin_size_ms,
            'tokenize_covariates': self.cfg.tokenize_covariates,
            'return_horizon_s': self.cfg.return_horizon_s,
        }
        # breakpoint()
        task_cfg = getattr(self.cfg, task.value)
        # version.update(task_cfg.reproc_dict())
        # Extremely hacky, IDK how to get cfg class methods working,
        task_dict = OmegaConf.to_container(task_cfg, resolve=True)
        for k, v in task_dict.items():
            version[k] = v
        return version

    def checksum_diff(self, version_path: Path, task: ExperimentalTask):
        # load json in session path
        if not version_path.exists():
            return True
        with open(version_path, 'r') as f:
            cached_preproc_version = json.load(f)
        # breakpoint()
        # ! patch - don't compare arrays
        current = self.preproc_version(task)
        cached_preproc_version.pop('arrays', None)
        current.pop('arrays', None)
        if 'heldout_neurons' in cached_preproc_version:
            cached_preproc_version.pop('heldout_neurons')
        if 'heldout_neurons' in current:
            current.pop('heldout_neurons')
        return current != cached_preproc_version

    @staticmethod
    def list_alias_to_contexts(path_or_alias_list: List[Union[Path, str]]) -> List[ContextInfo]:
        # sorted wrapper for more safety
        return sorted([c for p in path_or_alias_list for c in SpikingDataset.aliases_to_contexts(p)])

    @staticmethod
    def aliases_to_contexts(session_path_or_alias: Union[Path, str]) -> List[ContextInfo]:
        if isinstance(session_path_or_alias, str):
            # Try alias
            context_meta = context_registry.query(alias=session_path_or_alias)
            if context_meta is None:
                session_path = Path(session_path_or_alias)
                context_meta = [context_registry.query_by_datapath(session_path)]
            elif not isinstance(context_meta, list):
                context_meta = [context_meta]
            return sorted(context_meta)
        else:
            return [context_registry.query_by_datapath(session_path_or_alias)]

    def mark_eval_split_if_exists(self):
        if not self.cfg.eval_datasets:
            return
        assert self.loaded, "Must load meta_df before loading eval datasets"
        if 'split' not in self.meta_df:
            self.meta_df['split'] = 'train'
        else:
            self.meta_df['split'] = self.meta_df['split'].fillna('train')
        eval_metas = self.list_alias_to_contexts(self.cfg.eval_datasets)
        eval_ids = [m.id for m in eval_metas]
        eval_pool = self.meta_df[(self.meta_df[MetaKey.session].isin(eval_ids)) & (self.meta_df['split'] == 'train')]
        if sorted(eval_ids) != sorted(eval_pool[MetaKey.session].unique()):
            raise Exception(f"Requested datasets {sorted(eval_ids)} not all found. Found {sorted(eval_pool[MetaKey.session].unique())}")
        if self.cfg.eval_split_continuous:
            eval_subset = eval_pool.iloc[-int(self.cfg.eval_ratio * len(eval_pool)):] # take tail end, and we'll take head for train split
        else:
            if self.cfg.eval_ratio < 1:
                eval_subset = eval_pool.sample(frac=self.cfg.eval_ratio, random_state=self.cfg.eval_seed)
            else:
                eval_subset = eval_pool
        self.meta_df['split'] = self.meta_df['split'].mask(self.meta_df.index.isin(eval_subset.index), 'eval')

    def load_single_session(self, context_meta: ContextInfo, override_preprocess_path: bool=False) -> pd.DataFrame:
        session_path = context_meta.datapath
        # breakpoint()
        if not (hash_dir := self.preprocess_path(self.cfg, session_path, override_preprocess_path)).exists() or \
            self.checksum_diff(hash_dir / 'preprocess_version.json', context_meta.task):
            # TODO consider filtering meta df to be more lightweight (we don't bother right now because some nonessential attrs can be useful for analysis)
            os.makedirs(hash_dir, exist_ok=True)
            # Clear out the dir, we're regenerating
            for f in os.listdir(hash_dir):
                os.remove(hash_dir / f)
            meta = context_meta.load(self.cfg, hash_dir)
            if meta is None:
                logger.info('No metadata loaded, assuming debug mode. Continuing...')
                return None
            meta.to_csv(hash_dir / 'meta.csv')
            with open(hash_dir / 'preprocess_version.json', 'w') as f:
                json.dump(self.preproc_version(context_meta.task), f)
        else:
            meta = pd.read_csv(hash_dir / 'meta.csv')
            del meta[f'Unnamed: 0'] # index column
        for k in self.cfg.meta_keys:
            if k == MetaKey.array:
                data_arrays = getattr(context_meta, k.name)
                # Filter arrays using task configuration
                task_arrays = getattr(self.cfg, context_meta.task.name).arrays
                if task_arrays: # if task subset is defined, use task array naming (which may be aliases)
                    # keep the aliases that are relevant for this dataset - (slight hack)
                    context_array = [a for a in task_arrays if SubjectArrayRegistry.resolve_alias(a)[0] in data_arrays]
                    # context_array = [a for a in context_array if a in resolved_arrays]
                    if len(context_array) == 0:
                        raise Exception(
                            f"Session {session_path} has arrays {data_arrays} which has no elements in task configuration {task_arrays}.\n"
                            f"Remove or reconfigure (did you remember to add subject handle)?"
                        )
                else:
                    context_array = data_arrays
                for i in range(self.cfg.max_arrays):
                    meta[f'array_{i}'] = context_array[i] if i < len(context_array) else ""
                if len(context_array) > self.cfg.max_arrays:
                    logging.error(
                        f"Session {session_path} has more than {self.cfg.max_arrays} arrays."
                        f"Is this the right session? Or is max array setting to low?"
                        f"Or did you remember to truncate used arrays in task configuration?"
                    )
                    raise Exception()
            elif k == MetaKey.session:
                # never conflate sessions (even if they have the same tag)
                meta[k] = context_meta.session_embed_id
            elif k == MetaKey.unique:
                continue # filled below
            elif k == MetaKey.subject:
                meta[k] = context_meta.subject.name
            else:
                meta[k] = getattr(context_meta, k.name)
        meta[MetaKey.unique] = meta[MetaKey.session] + '-' + meta.index.astype(str) # unique per _trial_ INDEX in dataset
        self.validate_meta(meta)

        # invalid_paths = []
        # for k in self.cfg.data_keys:
        #     if k == DataKey.bhvr_vel:
        #         # load all and check bhvr exists, drop trials that don't
        #         for trial_path in meta.path:
        #             payload = torch.load(trial_path)
        #             if DataKey.bhvr_vel not in payload:
        #                 invalid_paths.append(trial_path)
        # if invalid_paths:
        #     logger.warning(f"Removing {len(invalid_paths)} of {len(meta)} trials from {context_meta.datapath} because they lack requested behavior velocity")
        #     meta = meta[~meta.path.isin(invalid_paths)]
        return meta

    @property
    def pad_value(self):
        return self.cfg.pad_value if self.cfg.serve_tokenized else 0

    def apply_augment(self, data: Dict[DataKey, torch.Tensor]):
        sampled_ops = np.random.choice(self.cfg.augmentations, self.cfg.randaug_num) # RandAugment
        for op in sampled_ops:
            data = augmentations[op](data)
        return data

    def __getitem__(self, index):
        r"""
            dict of arrays

            spikes: torch.Tensor, Batch x Time x Array x Channel x H
            * we give array dim (as opposed to flattening into channel to make array embeddings possible
        """
        trial: Path = self.meta_df.iloc[index]
        if len(self) <= self.cfg.auto_in_memory_thresh and trial.path in self.cache:
            return self.cache[trial.path]
        # * Potential optimization point to load onto GPU directly
        meta_items = {}
        for k in self.cfg.meta_keys:
            if k == MetaKey.unique:
                continue # don't serve
            if k == MetaKey.array: # doing string comparisons probably isn't the fastest thing in the world
                def map_array(a):
                    return self.context_index[k.name].index(a)
                meta_items[k] = torch.tensor([
                    map_array(trial[f'array_{i}']) for i in range(self.cfg.max_arrays)
                ])
            else:
                meta_items[k] = torch.tensor(self.context_index[k.name].index(trial[k])) # Casting in collater might be faster?

        r"""
            Currently we store spikes in a split-array format as a dict of tensors T C H.
            We must use the IDs to reconstruct the stack we want.
        """
        data_items = {}

        payload = torch.load(trial.path)
        # May process redundant data if we are using a subset of arrays, but easier than mucking with logic below
        if self.augment:
            payload = self.apply_augment(payload)

        channel_counts = [] # 1 value per array in base + serve_tokenized. 1 value per token for `serve_tokenized_flat`
        # Note the main reason we track channel_counts for `serve_tokenized_flat` is because we already implemented the unsplit version for `serve_tokenized` but would now like something easier.
        # while heldout channels are never provided in multiple shapes
        # the alternative to padding is to define custom readout via DataAttrs
        # we would rather maintain consistent interface and pad
        # heldout_channel_counts = []
        # import pdb;pdb.set_trace()
        for k in self.cfg.data_keys:
            if k == DataKey.spikes:
                array_spikes = []
                if self.cfg.serve_tokenized:
                    times = []
                    positions = []
                    space = 0
                for i in range(self.cfg.max_arrays):
                    alias = trial[f'array_{i}']
                    if alias == '':
                        continue # empty, ignore
                    alias_arrays = SubjectArrayRegistry.resolve_alias(alias) # list of strs
                    array_group = torch.cat([payload[k][a] for a in alias_arrays], dim=-2) # T C' H
                    if self.cfg.max_channels:
                        array_group = array_group[:,:self.cfg.max_channels] # crop
                    if self.cfg.permute_channels:
                        perm = self.channel_perms[trial[MetaKey.session]]
                        perm  = perm[perm < array_group.shape[-2]]
                        array_group = array_group[:,perm]
                    # * Note to get array tokenization to respect array boundaries, use non-alias full array references
                    pad_amount = (self.cfg.neurons_per_token - array_group.size(-2) % self.cfg.neurons_per_token) % self.cfg.neurons_per_token
                    array_group = F.pad(array_group, (0, 0, 0, pad_amount), value=self.cfg.pad_spike_value)
                    tokenized_spikes = array_group.unfold(1, self.cfg.neurons_per_token, self.cfg.neurons_per_token) # time space H channel_in_token
                    array_spikes.append(rearrange(tokenized_spikes, 'time space h c -> time space c h'))
                    time, token_space = tokenized_spikes.size(0), tokenized_spikes.size(1) # track across aliases and arrays
                    times.append(repeat(torch.arange(time), 'time -> time space', space=token_space))
                    positions.append(repeat(torch.arange(space, space+token_space), 'space -> time space', time=time))
                    space += token_space
                    channel_counts.append(torch.full((time, token_space), fill_value=self.cfg.neurons_per_token, dtype=torch.long))
                    if pad_amount:
                        channel_counts[-1][:,-1] = self.cfg.neurons_per_token - pad_amount
                data_items[k] = rearrange(torch.cat(array_spikes, 1), 't s c h -> (t s) c h')
                data_items[DataKey.time] = rearrange(torch.cat(times, 1), 't s -> (t s)')
                data_items[DataKey.position] = rearrange(torch.cat(positions, 1), 't s -> (t s)')
                data_items[CHANNEL_KEY] = rearrange(torch.cat(channel_counts, 1), 't s -> (t s)')
            else:
                if k == DataKey.heldout_spikes and self.cfg.heldout_key_spoof_shape:
                    data_items[k] = torch.full(list(self.cfg.heldout_key_spoof_shape), fill_value=self.pad_value)
                elif k == DataKey.bhvr_vel:
                    if k not in payload:
                        if self.cfg.tokenize_covariates:
                            if self.cfg.pad_positions:
                                data_items[k] = torch.zeros(len(DEFAULT_KIN_LABELS), 1)
                                data_items[DataKey.covariate_time] = torch.tensor([self.cfg.max_trial_length] * len(DEFAULT_KIN_LABELS), dtype=int)
                                data_items[DataKey.covariate_space] = torch.arange(len(DEFAULT_KIN_LABELS))
                                data_items[DataKey.covariate_labels] = DEFAULT_KIN_LABELS
                            else:
                                data_items[k] = torch.zeros((1, 1)) # null
                                data_items[DataKey.covariate_time] = torch.tensor([self.cfg.max_trial_length], dtype=int)
                                if self.cfg.semantic_positions:
                                    cov_space = torch.tensor([DEFAULT_KIN_LABELS.index('null')], dtype=int)
                                    cov_labels = ['null']
                                else:
                                    cov_space = torch.zeros(1, dtype=int)
                                    cov_labels = ['null']
                                data_items[DataKey.covariate_space] = cov_space
                                data_items[DataKey.covariate_labels] = cov_labels
                        else:
                            data_items[k] = torch.zeros((1, self.cfg.behavior_dim))
                            data_items[DataKey.covariate_time] = torch.tensor([self.cfg.max_trial_length], dtype=int)
                            data_items[DataKey.covariate_space] = torch.tensor([0], dtype=int)
                    else:
                        # breakpoint()
                        mean, std = self.cfg.z_score_default_mean, self.cfg.z_score_default_std
                        if self.z_score and trial[MetaKey.session] in self.z_score:
                            per_zscore = self.z_score[trial[MetaKey.session]]
                            mean = per_zscore['mean']
                            std = per_zscore['std']
                        cov = (payload[k] - mean) / std
                        if self.cfg.tokenize_covariates:
                            cov_labels = payload[DataKey.covariate_labels] # if DataKey.covariate_labels in payload else payload['covariate_dims'] # TODO deprecate 'covariate_dims'
                            # if 'f' not in cov_labels:
                                # breakpoint()
                            # breakpoint()
                            base_space = torch.tensor([DEFAULT_KIN_LABELS.index(i) for i in cov_labels], dtype=int) if self.cfg.semantic_positions else torch.arange(cov.size(1))
                            if self.cfg.pad_positions:
                                # Add space, change data itself, add base labels
                                other_space = torch.tensor([i for i in range(len(DEFAULT_KIN_LABELS)) if i not in base_space], dtype=int)
                                base_space = torch.cat([base_space, other_space])
                                cov_labels = [*cov_labels, *[DEFAULT_KIN_LABELS[i] for i in other_space]]
                                cov = F.pad(cov, (0, len(other_space)), value=self.pad_value)
                            data_items[DataKey.covariate_space] = repeat(base_space, 'b -> (t b)', t=cov.size(0))
                            data_items[DataKey.covariate_time] = repeat(torch.arange(cov.size(0)), 't -> (t b)', b=cov.size(1))
                            cov = rearrange(cov, 't b -> (t b) 1')
                            data_items[DataKey.covariate_labels] = cov_labels
                        else:
                            data_items[DataKey.covariate_time] = torch.arange(cov.size(0))
                            data_items[DataKey.covariate_space] = torch.zeros(cov.size(0), dtype=int)
                        data_items[k] = cov
                elif k == DataKey.constraint: # T x Constraint_Dim x Bhvr_dim
                    # Current implementation assumes fixed shape
                    if self.cfg.sparse_constraints:
                        if k not in payload:
                            bhvr_dim = payload[DataKey.bhvr_vel].size(-1) if DataKey.bhvr_vel in payload else 1
                            default_dim = bhvr_dim if self.cfg.tokenize_covariates else self.cfg.behavior_dim
                            data_items[k] = torch.zeros((1, 3, default_dim)) # add an initial token indicating no constraint
                            data_items[DataKey.constraint_time] = torch.tensor([0], dtype=int) # Constraint kicks things off, not vice versa.
                        else:
                            # check for change
                            constraint_dense = payload[k]
                            # Low-pri - should be slightly more efficient to only serve a constraint change per covariate dimension, not for all dimensions at once (there only needs to be one `.any`)
                            change_steps = torch.cat([torch.tensor([0]), (constraint_dense[1:] != constraint_dense[:-1]).any(1).any(1).nonzero().squeeze(1) + 1])
                            # T x 3 x Bhvr_Dim
                            data_items[k] = constraint_dense[change_steps]
                            data_items[DataKey.constraint_time] = change_steps
                        # breakpoint()
                        if self.cfg.tokenize_covariates:
                            data_items[DataKey.constraint_space] = repeat(torch.arange(data_items[k].size(-1)), 'b -> (t b)', t=data_items[k].size(0))
                            data_items[DataKey.constraint_time] = repeat(data_items[DataKey.constraint_time], 't -> (t b)', b=data_items[k].size(-1))
                            data_items[k] = rearrange(data_items[k], 't c b -> (t b) c')
                    else:
                        assert not self.cfg.tokenize_covariates, "Not implemented"
                        if k not in payload: # e.g. monkey data - assume native control
                            data_items[k] = torch.zeros_like(payload[DataKey.bhvr_vel])
                        else:
                            data_items[k] = payload[k]
                        # If not sparse, we don't need to create constraint time, as code reuses covariate time.
                elif k == DataKey.task_return:
                    # Default policy - if querying for reward and payload doesn't have it, just return nothing (which at most becomes padding), so stream is effectively unconditioned
                    if k not in payload: # add padding so things "compile"
                        data_items[DataKey.task_return] = torch.tensor([self.pad_value]).unsqueeze(0)
                        data_items[DataKey.task_reward] = torch.tensor([self.pad_value]).unsqueeze(0)
                        data_items[DataKey.task_return_time] = torch.tensor([0], dtype=int) # Using a max return is no good. We will crop it out in long seqs. This design in general should be refactored out eventually
                        # TODO think harder about this padding special case
                        # data_items[DataKey.task_return_time] = torch.tensor([self.cfg.max_trial_length], dtype=int)
                    else:
                        # Not sure this is legitimate
                        # breakpoint()
                        if self.cfg.sparse_rewards:
                            return_dense = payload[k]
                            change_steps = torch.cat([torch.tensor([0]), (return_dense[1:] != return_dense[:-1]).any(1).nonzero().squeeze(1) + 1])
                            if change_steps.max() > self.cfg.max_trial_length:
                                raise Exception(f"Trial {trial.path} has return horizon {change_steps.max()} which exceeds max_trial_length {self.cfg.max_trial_length}")
                            data_items[k] = return_dense[change_steps]
                            data_items[DataKey.task_return_time] = change_steps
                            data_items[DataKey.task_reward] = payload[DataKey.task_reward][change_steps]
                        else:
                            data_items[k] = payload[k]
                            data_items[DataKey.task_return_time] = torch.arange(payload[k].size(0)) # create, for simplicity, though we might technically mirror `DataKey.time` if we must...
                            data_items[DataKey.task_reward] = payload[DataKey.task_reward]
                        # +1 since 0 is reserved for padding. Note that since this is a dataloader-level offset... um...
                        data_items[DataKey.task_reward] = data_items[DataKey.task_reward] + 1
                        data_items[DataKey.task_return] = data_items[DataKey.task_return] + 1
                else:
                    data_items[k] = payload[k]
        out = {
            **data_items,
            **meta_items,
        }
        if len(self) <= self.cfg.auto_in_memory_thresh and trial.path not in self.cache:
            self.cache[trial.path] = out
        return out

    def __len__(self):
        return len(self.meta_df)

    def tokenized_collater(self, batch):
        r"""
            batch: list of dicts

            # ! Note - cropping calculation ensures that the tokens allocated per batch is roughly limited,
            # ! However, since dimensions are not consistent across the batch, uneven padding will still bring total tokens per trial above the limit
            # ! This can only be resolved if we lower the assembly to here (not sure how we would), or use nested tensors or something like that.
            # TODO take a look.
            # For now, not a huge deal - order of error will be maybe 20-30%.
        """
        stack_batch = defaultdict(list)
        space_lengths = torch.tensor([len(b[DataKey.position].unique()) + len(b[DataKey.covariate_space].unique()) for b in batch])
        # We only check spike + covariate for cropping since return and constraints should be sparse and add negligible tokens
        time_budget = (self.cfg.max_tokens // space_lengths)
        if self.max_bins:
            time_budget = time_budget.min(torch.tensor(self.max_bins))
        crop_start_limit = (torch.tensor([b[DataKey.time].max() for b in batch]) - time_budget).max(torch.tensor(1))
        crop_start = torch.randint(0, 10000, (len(batch),), dtype=torch.long) % crop_start_limit
        covariate_key = None
        for i, b in enumerate(batch):
            for k in b.keys():
                if isinstance(k, DataKey):
                    if k == DataKey.constraint:
                        constraint = b[k]
                        if self.cfg.sparse_constraints: # sparse and time delimited, check time
                            # Assumes constraint time is available
                            constraint_mask = (b[DataKey.constraint_time] < crop_start[i] + time_budget[i]) & (b[DataKey.constraint_time] >= crop_start[i])
                            if not constraint_mask.any():
                                # breakpoint()
                                constraint_mask = (b[DataKey.constraint_time] < crop_start[i] + time_budget[i]) # There should always be one, since there's always a constraint specified at start of trial.
                                # Get the latest timestep specified (which is before first crop timestep)
                                last_valid = b[DataKey.constraint_time][constraint_mask].max()
                                constraint_mask = (b[DataKey.constraint_time] == last_valid) # Identify all tokens at that timestep, should only be a few
                                # breakpoint()
                                b[DataKey.constraint_time] = torch.where(constraint_mask, crop_start[i], b[DataKey.constraint_time]) # Bump up time to start of crop
                            constraint = constraint[constraint_mask]
                            if DataKey.constraint_space in b:
                                constraint_space = b[DataKey.constraint_space][constraint_mask]
                                stack_batch[DataKey.constraint_space].append(constraint_space)
                            constraint_time = b[DataKey.constraint_time][constraint_mask] - crop_start[i]
                            stack_batch[DataKey.constraint_time].append(constraint_time)
                        else:
                            raise NotImplementedError
                            print("Shouldn't we be subsetting dense constraints according to covariate time")
                            breakpoint()
                        stack_batch[k].append(constraint.float()) # Cast explicitly, prediction function complains
                        # stack_batch[k].append(constraint)
                    elif k in [DataKey.constraint_time, DataKey.constraint_space]:
                        continue # treated above
                    elif k == DataKey.task_return:
                        # if b[k].shape[0] > 1:
                        #     breakpoint()
                        task_return = b[k]
                        task_reward = b[DataKey.task_reward]
                        task_return_time = b[DataKey.task_return_time]
                        if self.cfg.sparse_rewards:
                            # assumes return time is present, note we are aware of diff with constraints
                            time_mask = (b[DataKey.task_return_time] < crop_start[i] + time_budget[i]) & (b[DataKey.task_return_time] >= crop_start[i])
                            if not time_mask.any(): # Return is always declared at start of trial, so we should always have at least one timestep below
                                time_mask = (b[DataKey.task_return_time] < crop_start[i] + time_budget[i])
                                last_valid = b[DataKey.task_return_time][time_mask].max()
                                time_mask = (b[DataKey.task_return_time] == last_valid)
                                b[DataKey.task_return_time] = torch.where(time_mask, crop_start[i], b[DataKey.task_return_time])
                            task_return = task_return[time_mask]
                            task_reward = task_reward[time_mask]
                            task_return_time = b[DataKey.task_return_time][time_mask] - crop_start[i] # assumes time starts at 0
                        stack_batch[DataKey.task_return_time].append(task_return_time)
                        stack_batch[k].append(task_return)
                        stack_batch[DataKey.task_reward].append(task_reward)
                    elif k in [DataKey.task_return_time, DataKey.task_reward]:
                        continue # treated above
                    elif k == DataKey.bhvr_vel:
                        covariate_key = k
                        covariate = b[k]
                        covariate_time_mask = (b[DataKey.covariate_time] < crop_start[i] + time_budget[i]) & (b[DataKey.covariate_time] >= crop_start[i])
                        if not covariate_time_mask.any():
                            covariate_time_mask[-1] = True # ensure we have at least one timestep, even if OOB (optimization should more or less ignore)
                        covariate = covariate[covariate_time_mask]
                        covariate_space = b[DataKey.covariate_space][covariate_time_mask]
                        covariate_time = b[DataKey.covariate_time][covariate_time_mask] - crop_start[i]
                        stack_batch[DataKey.covariate_time].append(covariate_time)
                        stack_batch[DataKey.covariate_space].append(covariate_space)
                        stack_batch[k].append(covariate)
                    elif k in [DataKey.covariate_time, DataKey.covariate_space]:
                        continue # treated above
                    elif k == DataKey.covariate_labels:
                        stack_batch[k].append(b[k])
                    elif k in [DataKey.spikes]:
                        spike_time_mask = (b[DataKey.time] < crop_start[i] + time_budget[i]) & (b[DataKey.time] >= crop_start[i])
                        stack_batch[DataKey.time].append(b[DataKey.time][spike_time_mask] - crop_start[i])
                        stack_batch[DataKey.position].append(b[DataKey.position][spike_time_mask])
                        stack_batch[CHANNEL_KEY].append(b[CHANNEL_KEY][spike_time_mask])
                        stack_batch[k].append(b[k][spike_time_mask])
                    elif k in [DataKey.time, DataKey.covariate_space]:
                        continue # treated above
                else:
                    if k == CHANNEL_KEY:
                        continue # Treated above
                    stack_batch[k].append(b[k])
        lengths = torch.tensor([el.size(0) for el in stack_batch[DataKey.spikes]])
        if covariate_key is not None:
            covariate_lengths = torch.tensor([el.size(0) for el in stack_batch[covariate_key]])
            # Covariate channel functionality deprecated
            # covariate_channels = torch.tensor([el.size(1) for el in stack_batch[covariate_key]])
            # Manually pad to max channels
            # covariate_max = covariate_channels.max()
            # pad_els = [0] + [0, 0] * (stack_batch[covariate_key][0].ndim - 2)
            # for i, el in enumerate(stack_batch[covariate_key]):
                # stack_batch[covariate_key][i] = F.pad(el, (*pad_els, covariate_max - el.size(1)), value=self.pad_value)
        if DataKey.constraint_time in stack_batch: # sparse, can't just use covariate length
            constraint_lengths = torch.tensor([el.size(0) for el in stack_batch[DataKey.constraint]])
        if DataKey.task_return_time in stack_batch:
            task_return_lengths = torch.tensor([el.size(0) for el in stack_batch[DataKey.task_return]])
        for k in stack_batch.keys():
            if k == DataKey.covariate_labels:
                # stack_batch[k] = list(itertools.chain.from_iterable(stack_batch[k])) # Just for logging
                continue # Just leave it alone, we need to know which dims are which
            elif isinstance(k, DataKey) or (k == CHANNEL_KEY):
                # This padding injects pad values into time/space. The alternate is to assign time/space at collation time, but this is not as flexible, I'd rather individual trials specify their times.
                if k in [
                    DataKey.time,
                    DataKey.constraint_time,
                    DataKey.task_return_time,
                    DataKey.covariate_time
                ]:
                    pad_value = self.cfg.max_trial_length
                elif k in [
                    DataKey.position,
                    DataKey.constraint_space,
                    DataKey.covariate_space,
                ]:
                    pad_value = self.pad_value # We could hypothetically serve a different space, but right now we leave as default; since space manipulation is a model level concern atm.
                else:
                    pad_value = self.pad_value
                stack_batch[k] = pad_sequence(
                    stack_batch[k],
                    batch_first=True,
                    padding_value=pad_value)
            else:
                stack_batch[k] = torch.stack(stack_batch[k])
        stack_batch[LENGTH_KEY] = lengths
        if DataKey.constraint_time in stack_batch:
            stack_batch[CONSTRAINT_LENGTH_KEY] = constraint_lengths
        if DataKey.task_return_time in stack_batch:
            stack_batch[RETURN_LENGTH_KEY] = task_return_lengths
        if covariate_key is not None:
            stack_batch[COVARIATE_LENGTH_KEY] = covariate_lengths

        # return dict(stack_batch) # cast back to dict as pytorch distributed can act up with defaultdicts
        # Cast out, for compile
        new_batch = {}
        for k in stack_batch:
            if isinstance(k, DataKey) or isinstance(k, MetaKey):
                new_batch[k.name] = stack_batch[k]
            else:
                new_batch[k] = stack_batch[k]
        return new_batch

    def collater_factory(self):
        if not self.cfg.pad_batches:
            raise NotImplementedError("Need to implement trimming")

        if self.cfg.serve_tokenized:
            # Design decisions for cropping sequences
            # Note we don't take randomized slices over full datasets - (like in NLP) -- this is added complexity that will not obviously be useful
            # We don't want to slice over full corpus, but within a dataset may be worth it if we have many short trials.
            # TODO - (I'm really uncertain about modeling multiple sequences at one step, e.g. with/without <sep>. Will consider in the future)
            # We want to crop aligned to whole timesteps so we don't end up with partial data tokens and full covariates
            # We don't want to just pick a time as data with fewer overall channels will result in shorter sequences
            # We want to align based on token budget.
            # So let's compute the token budget, and then compute the timesteps we can afford based on data, and crop based on that.
            return self.tokenized_collater
        else:
            def collater(batch):
                r"""
                    batch: list of dicts
                """
                stack_batch = {}
                for k in batch[0].keys():
                    crop_seq = [b[k] for b in batch]
                    # TODO randomize crop
                    if self.max_bins and isinstance(k, DataKey):
                        # Leading dimension for DataKeys should be time
                        crop_seq = [b[k][-self.max_bins:] for b in batch] # terminal crop - most trials have long start paddings (e.g. Gallego)
                    if k == DataKey.spikes:
                        stack_batch[LENGTH_KEY] = torch.tensor([cs.shape[0] for cs in crop_seq])
                    if k in [DataKey.spikes, DataKey.bhvr_vel]: # T A C H
                        stack_batch[k] = pad_sequence(crop_seq, batch_first=True)
                    else:
                        stack_batch[k] = torch.stack(crop_seq, 0)
                return stack_batch
            return collater

    def build_context_index(self):
        if self.context_index is not None:
            logging.info("Building context index; any previous DataAttrs may be invalidated.")
        assert self.loaded, "Must load data before building context index"
        context = {}
        for k in self.cfg.meta_keys:
            if k == MetaKey.unique:
                continue # Only used as identifier, never served
            elif k == MetaKey.array:
                all_arrays = sorted(
                    pd.concat(self.meta_df[f'array_{i}'] for i in range(self.cfg.max_arrays)).unique()
                ) # This automatically includes the padding "", as index 0 if it's present in df
                context[MetaKey.array.name] = all_arrays
            else:
                assert k in self.meta_df.columns, f"Key {k} not in metadata"
                context[k.name] = sorted(self.meta_df[k].unique()) # convert key from enum so we can build contextattrs
        self.context_index: Dict[str, List] = context
        if getattr(self.cfg, 'permute_channels'):
            self.channel_perms = {
                s: torch.randperm(self.cfg.max_channels) for s in self.meta_df[MetaKey.session].unique()
            }

    def get_data_attrs(self):
        r"""
            Provide information about unique context such as
            - participants in dataset (and array information)
            - sessions used
            - tasks attempted.
            To be consumed by model to determine model IO.
        """
        if self.context_index is None:
            self.build_context_index()
        return DataAttrs(
            bin_size_ms=self.cfg.bin_size_ms,
            max_channel_count=self.cfg.max_channels,
            max_arrays=self.cfg.max_arrays,
            spike_dim=1, # Higher dims not supported right now
            context=ContextAttrs(**self.context_index),
            rtt_heldout_channel_count=self.cfg.nlb_rtt.heldout_neurons,
            maze_heldout_channel_count=self.cfg.nlb_maze.heldout_neurons,
            behavior_dim=self.cfg.behavior_dim,
            pad_token=self.pad_value,
            max_trial_length=self.cfg.max_trial_length,
            serve_tokens=self.cfg.serve_tokenized,
            serve_tokens_flat=self.cfg.serve_tokenized_flat,
            neurons_per_token=self.cfg.neurons_per_token,
            sparse_constraints=self.cfg.sparse_constraints,
            sparse_rewards=self.cfg.sparse_rewards,
            tokenize_covariates=self.cfg.tokenize_covariates,
            semantic_covariates=self.cfg.semantic_positions,
        )

    # ==================== Data splitters ====================
    @property
    def split_keys(self):
        return self.meta_df[self.cfg.split_key].unique().copy()

    def get_key_indices(self, key_values, key: MetaKey=MetaKey.unique):
        return self.meta_df[self.meta_df[key].isin(key_values)].index

    def subset_by_key(self,
        key_values: List[Any], key: Union[MetaKey, str]=MetaKey.unique, allow_second_subset=True, na=None,
        keep_index=False, message_prefix="",
    ):
        r"""
            # ! In place
        """
        if len(key_values) == 0:
            logging.info("No keys provided, ignoring subset.")
            return
        if self.subsetted:
            assert allow_second_subset
            logging.warning("Dataset has already been subsetted.")
        if na is not None:
            self.meta_df[key] = self.meta_df[key].fillna(na)
        subset = self.meta_df[key].isin(key_values)
        logging.info(f"{message_prefix}: Subset dataset by {key} to {subset.sum()} / {len(self.meta_df)}")
        self.meta_df = self.meta_df[self.meta_df[key].isin(key_values)]
        self.meta_df = self.meta_df.reset_index(drop=True)
        if not keep_index:
            self.build_context_index()
        self.subsetted = True
        self.cache = {}

    def tv_split_by_split_key(self, train_ratio=0.8, seed=None):
        keys = self.split_keys
        if seed is None:
            seed = self.cfg.dataset_seed
        pl.seed_everything(seed)
        np.random.shuffle(keys)
        tv_cut = int(train_ratio * len(keys))
        train_keys, val_keys = keys[:tv_cut], keys[tv_cut:]
        return train_keys, val_keys

    def create_tv_datasets(self, **kwargs):
        r"""
            Keys determine how we split up our dataset.
            Default by trial, or more specific conditions
            Assumes balanced dataset
        """
        if self.context_index is None:
            self.build_context_index()
        train_keys, val_keys = self.tv_split_by_split_key(**kwargs)
        train = copy.deepcopy(self)
        train.subset_by_key(train_keys, key=self.cfg.split_key, keep_index=True, message_prefix="Train:")
        val = copy.deepcopy(self)
        val.subset_by_key(val_keys, key=self.cfg.split_key, keep_index=True, message_prefix="Val:")
        assert train.context_index == val.context_index, "Context index mismatch between train and val (some condition is unavailable, not supported)"
        return train, val

    def merge(self, data_other: Any): # should be type Self but surprisingly this is a 3.11 feature (I thought I used it before?)
        self.meta_df = pd.concat([self.meta_df, data_other.meta_df])
        self.meta_df = self.meta_df.reset_index(drop=True)
        self.build_context_index()
        # TODO think about resetting data attrs - this should be called before any data attr call

    def subset_split(self, splits=['train'], keep_index=False):
        if 'split' in self.meta_df.columns:
            self.subset_by_key(key_values=splits, key='split', na='train', keep_index=keep_index, message_prefix=splits)
        else:
            logger.warning("No split column found, assuming all data is train.")

    def subset_scale(self, limit_per_session=0, limit_per_eval_session=0, ratio=1.0, limit=0, keep_index=False):
        # Random scale-down of data
        if limit_per_session > 0 or limit_per_eval_session > 0:
            keys = None
            eval_keys = []
            train_keys = []
            eval_datasets = [ctx.id for ctx in self.list_alias_to_contexts(self.cfg.eval_datasets)]

            eval_session_df = self.meta_df[self.meta_df[MetaKey.session].isin(eval_datasets)]
            if not limit_per_eval_session:
                limit_per_eval_session = limit_per_session # default is to obey regular limit
            if self.cfg.eval_split_continuous:
                eval_keys = eval_session_df.groupby([MetaKey.session]).apply(lambda x: x.iloc[:limit_per_eval_session])[MetaKey.unique]
            else:
                try:
                    eval_keys = eval_session_df.groupby([MetaKey.session]).apply(lambda x: x.sample(limit_per_eval_session))[MetaKey.unique]
                except ValueError:
                    print("Did not have enough data in some eval session to sample. Diagnostic:")
                    print(eval_session_df.groupby([MetaKey.session]).apply(lambda x: x.shape[0]))
                    raise ValueError

            train_session_df = self.meta_df[~self.meta_df[MetaKey.session].isin(eval_datasets)]
            if limit_per_session:
                if self.cfg.eval_split_continuous:
                    train_keys = train_session_df.groupby([MetaKey.session]).apply(lambda x: x.iloc[:limit_per_session])[MetaKey.unique]
                else:
                    train_keys = train_session_df.groupby([MetaKey.session]).apply(lambda x: x.sample(limit_per_session))[MetaKey.unique]
            else: # default is to assume no limit
                train_keys = train_session_df[MetaKey.unique]
            keys = pd.concat([eval_keys, train_keys])
            self.subset_by_key(
                key_values=keys,
                keep_index=keep_index,
                message_prefix=f"Scale {limit_per_session} (eval {limit_per_eval_session}) per session"
            )
        elif limit > 0:
            self.subset_by_key(
                key_values=self.meta_df.sample(limit)[MetaKey.unique],
                keep_index=keep_index,
                message_prefix=f"Scale {limit}"
            )
        elif ratio < 1:
            self.subset_by_key(
                key_values=self.meta_df.sample(frac=ratio)[MetaKey.unique],
                keep_index=keep_index,
                message_prefix=f"Scale {ratio}"
            )


class SpikingDataModule(pl.LightningDataModule):
    r"""
        A PL module mainly for autoscaling batch size, for sweeping.
    """
    def __init__(self, batch_size, num_workers, train: SpikingDataset, val, test=[]) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train = train
        if not isinstance(val, list):
            val = [val]
        if not isinstance(test, list):
            test = [test]
        self.val: List[SpikingDataset] = val
        self.test: List[SpikingDataset] = test
        self.num_workers = num_workers

    def setup(self, stage: str=""):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train, shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train.tokenized_collater,
            # collate_fn=functools.partial(self.train.tokenized_collater, self.train),
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset, shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                collate_fn=dataset.tokenized_collater,
                # collate_fn=functools.partial(dataset.tokenized_collater, dataset),
            ) for dataset in self.val]

    def test_dataloader(self):
        if len(self.test) == 0:
            return None
        return [DataLoader(
            dataset, shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=dataset.tokenized_collater,
            # collate_fn=functools.partial(dataset.tokenized_collater, dataset),
        ) for dataset in self.test]

