r"""
    ! LEGACY patch - old checkpoints will import this definition (see load_from_checkpoint patch)
    Eventually deprecate this file once we're working with new dirs.
"""




from typing import List, Any, Optional, Dict, Union
import copy
import json
import os
from pathlib import Path
from math import ceil
import re
import itertools
from datetime import datetime
from omegaconf import OmegaConf
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from einops import rearrange, repeat

import pytorch_lightning as pl

from context_general_bci.config import DatasetConfig, MetaKey, DataKey
from context_general_bci.subjects import SubjectArrayRegistry
from context_general_bci.contexts import context_registry, ContextInfo
from context_general_bci.tasks import ExperimentalTask

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
COVARIATE_LENGTH_KEY = 'covariate_length' # we need another length tracker for padded sequences of covariates in the flat case
COVARIATE_CHANNEL_KEY = 'covariate_channel_counts' # essentially for heldout channels only
HELDOUT_CHANNEL_KEY = 'heldout_channel_counts'

logger = logging.getLogger(__name__)
@dataclass
class ContextAttrs:
    r"""
        Each of these can potentially be embedded
    """
    subject: List[str] = field(default_factory=list)
    array: List[str] = field(default_factory=list) # should be prefixed with subject
    session: List[str] = field(default_factory=list) # unique ID
    task: List[str] = field(default_factory=list) # experimental task

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

    behavior_dim: int = 2
    pad_token: int = 20 # this needs to be a value that definitely won't appear as a natural spike count for your used bin size.
    serve_tokens: bool = False # if true, serves flat data tokens with additional keys for annotations (e.g. array + timestep) instead of structured data (e.g. with array dimensions)
    serve_tokens_flat: bool = False
    neurons_per_token: int = 8

    @property
    def max_spatial_tokens(self):
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

        TODO implement. Should span several task settings.
        - ICMS
        - Motor cortex data

        Other notes:
        - Can we "mixin" time-varying data, or maybe simpler to just be a separate codepath in this class.
    """
    def __init__(self, cfg: DatasetConfig, use_augment: bool = True):
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
            self.meta_df = pd.concat([self.load_single_session(c) for c in contexts]).reset_index(drop=True)
            # self.meta_df = pd.concat([self.load_single_session(c) for c in contexts]).reset_index(drop=True)
            if 'split' in self.meta_df.columns and len(self.meta_df['split'].unique()) > 1:
                logger.warning("Non-train splits found in meta_df. Subsetting is expected.")
        else:
            self.meta_df = None
        self.context_index = None
        self.subsetted = False
        self.max_bins = round(self.cfg.max_length_ms / self.cfg.bin_size_ms)
        self.mark_eval_split_if_exists()
        self.cache = {}
        self.augment = use_augment and bool(self.cfg.augmentations)

    @property
    def loaded(self):
        return self.meta_df is not None

    @staticmethod
    def preprocess_path(cfg: DatasetConfig, session_path: Path) -> Path:
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
            'bin_size_ms': self.cfg.bin_size_ms
        }
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
        eval_subset = eval_pool.sample(frac=self.cfg.eval_ratio, random_state=self.cfg.eval_seed)
        self.meta_df['split'] = self.meta_df['split'].mask(self.meta_df.index.isin(eval_subset.index), 'eval')

    def load_single_session(self, context_meta: ContextInfo) -> pd.DataFrame:
        session_path = context_meta.datapath
        if not (hash_dir := self.preprocess_path(self.cfg, session_path)).exists() or \
            self.checksum_diff(hash_dir / 'preprocess_version.json', context_meta.task):
            # TODO consider filtering meta df to be more lightweight (we don't bother right now because some nonessential attrs can be useful for analysis)
            os.makedirs(hash_dir, exist_ok=True)
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
                meta[k] = context_meta.id
            elif k == MetaKey.unique:
                continue # filled below
            elif k == MetaKey.subject:
                meta[k] = context_meta.subject.name
            else:
                meta[k] = getattr(context_meta, k.name)
        meta[MetaKey.unique] = meta[MetaKey.session] + '-' + meta.index.astype(str) # unique per _trial_ INDEX in dataset
        self.validate_meta(meta)

        return meta

    @property
    def pad_value(self):
        return self.cfg.pad_value if self.cfg.serve_tokenized else 0

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
                    if alias == '': # empty, should only occur for i >= 1, so we can identify the right shape from previous entries (squeezing out the array dim)
                        if not self.cfg.serve_tokenized:
                            array_group = torch.full_like(array_spikes[0][:,0], fill_value=self.pad_value)
                            array_spikes.append(rearrange(array_group, 't c h -> t () c h'))
                        if not self.cfg.serve_tokenized_flat:
                            channel_counts.append(torch.tensor(0))
                    else:
                        alias_arrays = SubjectArrayRegistry.resolve_alias(alias) # list of strs
                        array_group = torch.cat([payload[k][a] for a in alias_arrays], dim=-2) # T C' H
                        if self.cfg.max_channels:
                            array_group = array_group[:,:self.cfg.max_channels] # crop
                        if getattr(self.cfg, 'permute_channels', False):
                            perm = self.channel_perms[trial[MetaKey.session]]
                            perm  = perm[perm < array_group.shape[-2]]
                            array_group = array_group[:,perm]
                        if not self.cfg.serve_tokenized_flat:
                            channel_counts.append(array_group.shape[-2])
                        # * Note to get array tokenization to respect array boundaries, use non-alias full array references
                        if self.cfg.serve_tokenized:
                            pad_amount = (self.cfg.neurons_per_token - array_group.size(-2) % self.cfg.neurons_per_token) % self.cfg.neurons_per_token
                            array_group = F.pad(array_group, (0, 0, 0, pad_amount), value=getattr(self.cfg, 'pad_spike_value', 0))
                            tokenized_spikes = array_group.unfold(1, self.cfg.neurons_per_token, self.cfg.neurons_per_token) # time space H channel_in_token
                            array_spikes.append(rearrange(tokenized_spikes, 'time space h c -> time space c h'))
                            time, token_space = tokenized_spikes.size(0), tokenized_spikes.size(1) # track across aliases and arrays
                            times.append(repeat(torch.arange(time), 'time -> time space', space=token_space))
                            positions.append(repeat(torch.arange(space, space+token_space), 'space -> time space', time=time))
                            space += token_space
                            if self.cfg.serve_tokenized_flat:
                                channel_counts.append(torch.full((time, token_space), fill_value=self.cfg.neurons_per_token, dtype=torch.long))
                                if pad_amount:
                                    channel_counts[-1][:,-1] = self.cfg.neurons_per_token - pad_amount
                        else:
                            if self.cfg.max_channels:
                                pad_amount = self.cfg.max_channels - array_group.size(-2)
                                array_group = F.pad(array_group, (0, 0, 0, pad_amount), value=self.pad_value)
                            array_spikes.append(rearrange(array_group, 't c h -> t () c h'))
                if self.cfg.serve_tokenized:
                    data_items[k] = torch.cat(array_spikes, 1) # T x S x C x H
                    data_items[DataKey.time] = torch.cat(times, 1)
                    data_items[DataKey.position] = torch.cat(positions, 1)
                    if self.cfg.serve_tokenized_flat:
                        data_items[CHANNEL_KEY] = torch.cat(channel_counts, 1)
                else:
                    data_items[k] = torch.cat(array_spikes, 1) # T A C H
            else:
                if k == DataKey.heldout_spikes and getattr(self.cfg, 'heldout_key_spoof_shape', []):
                    data_items[k] = torch.full(list(self.cfg.heldout_key_spoof_shape), fill_value=self.pad_value)
                else:
                    data_items[k] = payload[k]
        out = {
            **data_items,
            **meta_items,
        }
        if self.cfg.max_channels and not self.cfg.serve_tokenized_flat:
            out[CHANNEL_KEY] = torch.tensor(channel_counts) # of length arrays (subsumes array mask, hopefully)
            # if heldout_channel_counts:
                # out[HELDOUT_CHANNEL_KEY] = torch.tensor(heldout_channel_counts)

        if len(self) <= self.cfg.auto_in_memory_thresh and trial.path not in self.cache:
            self.cache[trial.path] = out
        return out

    def __len__(self):
        return len(self.meta_df)

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
            def collater(batch):
                r"""
                    batch: list of dicts
                """
                stack_batch = defaultdict(list)
                space_lengths = torch.tensor([b[DataKey.spikes].size(1) for b in batch])
                if not self.cfg.serve_tokenized_flat: # account for padding in space calculation
                    space_lengths = torch.full_like(space_lengths, space_lengths.max())
                time_budget = (self.cfg.max_tokens // space_lengths)
                if self.max_bins:
                    time_budget = time_budget.min(torch.tensor(self.max_bins))
                # TODO separate out/remove this cropping logic for flat spacetime
                crop_start_limit = (torch.tensor([b[DataKey.spikes].size(0) for b in batch]) - time_budget).max(torch.tensor(1))
                crop_start = torch.randint(0, 10000, (len(batch),), dtype=torch.long) % crop_start_limit
                # ! currently, DataKey.time is with respect to trial time; what we really need is a time relative to the crop_start
                # ! we have several ways of instancing this, but we're picking the most convenient for now anticipating near future changes
                covariate_key = None
                for i, b in enumerate(batch):
                    for k in b.keys():
                        if isinstance(k, DataKey):
                            item = b[k][crop_start[i]:crop_start[i]+time_budget[i]]
                            if k == DataKey.time:
                                item = item - item[0]
                            if self.cfg.serve_tokenized_flat:
                                if k in [DataKey.spikes, DataKey.time, DataKey.position]:
                                    # These keys have spatial dimensions that we will serve flat to maximize data throughput across heterogeneous trials
                                    item = item.flatten(0, 1) # T S H -> Token H
                                else:
                                    covariate_key = k # will need separate padding, track for later
                            else: # pad in space
                                if k == DataKey.spikes: # B T S C H
                                    item = F.pad(item, (0, 0, 0, 0, 0, space_lengths[i] - item.size(1)), value=self.pad_value)
                                elif k in [DataKey.time, DataKey.position]:
                                    # ! Note... this "pad_value" is set for spikes, and will definitely conflict with meaningful values for
                                    # time and position. This shouldn't be a problem right now, as we're using LENGTH_KEY and CHANNEL_KEY to determine what's padding and not when computing masks.
                                    # ! For the flat case, we should directly ID a pad value (e.g. 0) for time/position and adjust in code.
                                    item = F.pad(item, (0, space_lengths[i] - item.size(1)), value=self.pad_value)
                            stack_batch[k].append(item)
                        else:
                            if self.cfg.serve_tokenized_flat and k == CHANNEL_KEY: # determine cropped channel count
                                b[k] = b[k][crop_start[i]:crop_start[i]+time_budget[i]]
                                stack_batch[k].append(b[k].flatten(0, 1))
                            else:
                                stack_batch[k].append(b[k])
                lengths = torch.tensor([el.size(0) for el in stack_batch[DataKey.spikes]])
                if covariate_key is not None:
                    covariate_lengths = torch.tensor([el.size(0) for el in stack_batch[covariate_key]])
                    covariate_channels = torch.tensor([el.size(1) for el in stack_batch[covariate_key]])
                    # Manually pad to max channels
                    covariate_max = covariate_channels.max()
                    pad_els = [0] + [0, 0] * (stack_batch[covariate_key][0].ndim - 2)
                    for i, el in enumerate(stack_batch[covariate_key]):
                        stack_batch[covariate_key][i] = F.pad(el, (*pad_els, covariate_max - el.size(1)), value=self.pad_value)
                for k in stack_batch.keys():
                    if isinstance(k, DataKey) or (self.cfg.serve_tokenized_flat and k == CHANNEL_KEY):
                        stack_batch[k] = pad_sequence(stack_batch[k], batch_first=True, padding_value=self.pad_value)
                    else:
                        stack_batch[k] = torch.stack(stack_batch[k])
                stack_batch[LENGTH_KEY] = lengths
                if covariate_key is not None:
                    stack_batch[COVARIATE_LENGTH_KEY] = covariate_lengths
                    stack_batch[COVARIATE_CHANNEL_KEY] = covariate_channels
                return dict(stack_batch) # cast back to dict as pytorch distributed can act up with defaultdicts
            return collater
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
            serve_tokens=self.cfg.serve_tokenized,
            serve_tokens_flat=self.cfg.serve_tokenized_flat,
            neurons_per_token=self.cfg.neurons_per_token,
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
            if limit_per_eval_session:
                eval_keys = eval_session_df.groupby([MetaKey.session]).apply(lambda x: x.sample(limit_per_eval_session))[MetaKey.unique]
            else: # default is to obey regular limit
                eval_keys = eval_session_df.groupby([MetaKey.session]).apply(lambda x: x.sample(limit_per_session))[MetaKey.unique]

            train_session_df = self.meta_df[~self.meta_df[MetaKey.session].isin(eval_datasets)]
            if limit_per_session:
                train_keys = train_session_df.groupby([MetaKey.session]).apply(lambda x: x.sample(limit_per_session))[MetaKey.unique]
            else: # default is to assume no limit
                train_keys = train_session_df[MetaKey.unique]

            keys = pd.concat([eval_keys, train_keys])
            self.subset_by_key(
                key_values=keys,
                keep_index=keep_index,
                message_prefix=f"Scale {limit_per_session} per session"
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
        self.val = val
        self.test = test
        self.num_workers = num_workers

    def setup(self, stage: str=""):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train, shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train.collater_factory(),
        )

    def val_dataloader(self):
        return [
            DataLoader(
                dataset, shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                collate_fn=dataset.collater_factory(),
            ) for dataset in self.val]

    def test_dataloader(self):
        if len(self.test) == 0:
            return None
        for dataset in self.test:
            return [DataLoader(
                dataset, shuffle=False,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
                collate_fn=dataset.collater_factory(),
            ) for dataset in self.test]

