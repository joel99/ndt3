from typing import Tuple, Dict, List, Optional, Any, Mapping, Union
import dataclasses
import time
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack, unpack # baby steps...
from omegaconf import OmegaConf, ListConfig, DictConfig
from dacite import from_dict
import logging
from pprint import pformat

from context_general_bci.config import (
    ModelConfig,
    ModelTask,
    Metric,
    Output,
    EmbedStrat,
    DataKey,
    MetaKey,
    Architecture,
)

from context_general_bci.dataset import DataAttrs, LENGTH_KEY, CHANNEL_KEY, COVARIATE_LENGTH_KEY, COVARIATE_CHANNEL_KEY
from context_general_bci.subjects import subject_array_registry, SortedArrayInfo
# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
from context_general_bci.components import (
    SpaceTimeTransformer,
    ReadinMatrix,
    ReadinCrossAttention,
    ContextualMLP,
)
from context_general_bci.task_io import task_modules, SHUFFLE_KEY, create_temporal_padding_mask
from context_general_bci.utils import enum_backport

logger = logging.getLogger(__name__)

class BrainBertInterface(pl.LightningModule):
    r"""
        I know I'll end up regretting this name.
    """
    def __init__(self, cfg: ModelConfig, data_attrs: DataAttrs):
        super().__init__() # store cfg
        self.save_hyperparameters(logger=False)
        if not isinstance(cfg, ModelConfig):
            # attempt forward port
            conf_container = OmegaConf.to_container(cfg)
            for item in [
                'session_embed_strategy',
                'subject_embed_strategy',
                'task_embed_strategy',
                'array_embed_strategy',
                'readin_strategy',
                'readout_strategy',
                'stim_embed_strategy',
                'heldout_neuron_embed_strategy',
                'spike_embed_style',
            ]:
                conf_container[item] = enum_backport(conf_container[item], EmbedStrat)
            conf_container['arch'] = enum_backport(conf_container['arch'], Architecture)
            for item, class_remap in [
                ('decode_strategy', EmbedStrat),
                ('tasks', ModelTask),
                ('metrics', Metric),
                ('outputs', Output),
                ('behavior_target', DataKey),
            ]:
                if isinstance(conf_container['task'][item], list):
                    conf_container['task'][item] = [
                        enum_backport(x, class_remap) for x in conf_container['task'][item]
                    ]
                else:
                    conf_container['task'][item] = enum_backport(conf_container['task'][item], class_remap)
            cfg = OmegaConf.merge(ModelConfig(), from_dict(data_class=ModelConfig, data=conf_container))
        self.cfg = cfg
        self.data_attrs = data_attrs

        r"""
            Make cfg use correct module refs for enums via a backport after migration
        """

        assert self.data_attrs.max_channel_count % self.cfg.neurons_per_token == 0, "Neurons per token must divide max channel count"
        if self.data_attrs.serve_tokens:
            assert self.cfg.array_embed_strategy == EmbedStrat.none, 'array IDs serving not implemented for spatially tokenized data'
            assert self.cfg.transform_space, 'Transform space must be true if serving (spacetime) tokens'
            assert self.data_attrs.neurons_per_token == self.cfg.neurons_per_token, \
                f"Neurons per token served by data ({self.data_attrs.neurons_per_token}) must match model token size {self.cfg.neurons_per_token}"
        if self.data_attrs.serve_tokens_flat:
            assert self.cfg.transformer.flat_encoder, "Flat encoder must be true if serving flat tokens"
        assert self.cfg.arch == Architecture.ndt, "ndt is all you need"
        if self.cfg.transformer.n_layers == 0: # debug for parity
            self.backbone = nn.Identity()
            self.backbone.out_size = self.cfg.hidden_size
        else:
            self.backbone = SpaceTimeTransformer(
                self.cfg.transformer,
                max_spatial_tokens=data_attrs.max_spatial_tokens,
                debug_override_dropout_out=getattr(cfg.transformer, 'debug_override_dropout_io', False),
                context_integration=getattr(cfg.transformer, 'context_integration', 'in_context'),
                embed_space=cfg.transformer.embed_space,
            )
        self.bind_io()
        self.novel_params: List[str] = [] # for fine-tuning
        num_updates = sum(tp.does_update_root for tp in self.task_pipelines.values())
        assert num_updates <= 1, "Only one task pipeline should update the root"

        if self.cfg.layer_norm_input:
            self.layer_norm_input = nn.LayerNorm(data_attrs.max_channel_count)

        self.token_proc_approx = 0
        self.token_seen_approx = 0
        self.detach_backbone_for_task = False

    def diff_cfg(self, cfg: ModelConfig):
        r"""
            Check if new cfg is different from self.cfg (POV of old model)
        """
        self_copy = self.cfg.copy()
        self_copy = OmegaConf.merge(ModelConfig(), self_copy) # backport novel config
        cfg = OmegaConf.merge(ModelConfig(), cfg)

        # Things that are allowed to change on init (actually most things should be allowed to change, but just register them explicitly here as needed)

        for safe_attr in [
            'decoder_layers', # ! assuming we're freshly initializing, this is kind of not safe
            'decoder_context_integration', # ^
            'dropout',
            'weight_decay',
            'causal',
            'task',
            'lr_init',
            'lr_schedule',
            'lr_ramp_steps',
            'lr_ramp_init_factor',
            'lr_decay_steps',
            'lr_min',
            'accelerate_new_params',
            'tune_decay',
            'val_iters',
            'extra_task_embed_ckpt',
            'extra_subject_embed_ckpt',
            'closed_loop_crop_bins'
        ]:
            setattr(self_copy, safe_attr, getattr(cfg, safe_attr))
        recursive_diff_log(self_copy, cfg)
        return self_copy != cfg

    def bind_io(self):
        r"""
            Add context-specific input/output parameters.
            Has support for re-binding IO, but does _not_ check for shapes, which are assumed to be correct.
            This means we rebind
            - embeddings
            - flags
            - task_modules
            Shapes are hidden sizes for flags/embeddings, and are configured via cfg.
            From this "same cfg" assumption - we will assume that
            `context_project` and `readin` are the same.


            Ideally, we will just bind embedding layers here, but there may be some MLPs.
        """
        for attr in ['session', 'subject', 'task', 'array']:
            if getattr(self.cfg, f'{attr}_embed_strategy') is not EmbedStrat.none:
                assert getattr(self.data_attrs.context, attr), f"{attr} embedding strategy requires {attr} in data"
                if len(getattr(self.data_attrs.context, attr)) == 1:
                    logger.warning(f'Using {attr} embedding strategy with only one {attr}. Expected only if tuning.')

        # We write the following repetitive logic explicitly to maintain typing
        project_size = self.cfg.hidden_size

        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            if self.cfg.session_embed_strategy == EmbedStrat.token and self.cfg.session_embed_token_count > 1:
                self.session_embed = nn.Parameter(torch.randn(len(self.data_attrs.context.session), self.cfg.session_embed_token_count, self.cfg.session_embed_size) / math.sqrt(self.cfg.session_embed_size))
                self.session_flag = nn.Parameter(torch.randn(self.cfg.session_embed_token_count, self.cfg.session_embed_size) / math.sqrt(self.cfg.session_embed_size))
            else:
                self.session_embed = nn.Embedding(len(self.data_attrs.context.session), self.cfg.session_embed_size)
                if self.cfg.session_embed_strategy == EmbedStrat.concat:
                    project_size += self.cfg.session_embed_size
                elif self.cfg.session_embed_strategy == EmbedStrat.token:
                    assert self.cfg.session_embed_size == self.cfg.hidden_size
                    if self.cfg.init_flags:
                        self.session_flag = nn.Parameter(torch.randn(self.cfg.session_embed_size) / math.sqrt(self.cfg.session_embed_size))
                    else:
                        self.session_flag = nn.Parameter(torch.zeros(self.cfg.session_embed_size))

        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            if self.cfg.subject_embed_strategy == EmbedStrat.token and getattr(self.cfg, 'subject_embed_token_count', 1) > 1:
                self.subject_embed = nn.Parameter(torch.randn(len(self.data_attrs.context.subject), self.cfg.subject_embed_token_count, self.cfg.subject_embed_size) / math.sqrt(self.cfg.subject_embed_size))
                self.subject_flag = nn.Parameter(torch.randn(self.cfg.subject_embed_token_count, self.cfg.subject_embed_size) / math.sqrt(self.cfg.subject_embed_size))
            else:
                self.subject_embed = nn.Embedding(len(self.data_attrs.context.subject), self.cfg.subject_embed_size)
                if self.cfg.subject_embed_strategy == EmbedStrat.concat:
                    project_size += self.cfg.subject_embed_size
                elif self.cfg.subject_embed_strategy == EmbedStrat.token:
                    assert self.cfg.subject_embed_size == self.cfg.hidden_size
                    if self.cfg.init_flags:
                        self.subject_flag = nn.Parameter(torch.randn(self.cfg.subject_embed_size) / math.sqrt(self.cfg.subject_embed_size))
                    else:
                        self.subject_flag = nn.Parameter(torch.zeros(self.cfg.subject_embed_size))

        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            self.array_embed = nn.Embedding(
                len(self.data_attrs.context.array),
                self.cfg.array_embed_size,
                padding_idx=self.data_attrs.context.array.index('') if '' in self.data_attrs.context.array else None
            )
            self.array_embed.weight.data.fill_(0) # Don't change by default
            if self.cfg.array_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.array_embed_size
            elif self.cfg.array_embed_strategy == EmbedStrat.token:
                assert self.cfg.array_embed_size == self.cfg.hidden_size
                if self.cfg.init_flags:
                    self.array_flag = nn.Parameter(torch.randn(self.data_attrs.max_arrays, self.cfg.array_embed_size) / math.sqrt(self.cfg.array_embed_size))
                else:
                    self.array_flag = nn.Parameter(torch.zeros(self.data_attrs.max_arrays, self.cfg.array_embed_size))

        if self.cfg.task_embed_strategy is not EmbedStrat.none:
            if self.cfg.task_embed_strategy == EmbedStrat.token and getattr(self.cfg, 'task_embed_token_count', 1) > 1:
                self.task_embed = nn.Parameter(torch.randn(len(self.data_attrs.context.task), self.cfg.task_embed_token_count, self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
                self.task_flag = nn.Parameter(torch.randn(self.cfg.task_embed_token_count, self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
            else:
                self.task_embed = nn.Embedding(len(self.data_attrs.context.task), self.cfg.task_embed_size)
                if self.cfg.task_embed_strategy == EmbedStrat.concat:
                    project_size += self.cfg.task_embed_size
                elif self.cfg.task_embed_strategy == EmbedStrat.token:
                    assert self.cfg.task_embed_size == self.cfg.hidden_size
                    if self.cfg.init_flags:
                        self.task_flag = nn.Parameter(torch.randn(self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
                    else:
                        self.task_flag = nn.Parameter(torch.zeros(self.cfg.task_embed_size))

        if project_size is not self.cfg.hidden_size:
            self.context_project = nn.Sequential(
                nn.Linear(project_size, self.cfg.hidden_size),
                nn.ReLU() if self.cfg.activation == 'relu' else nn.GELU(),
            )

        if self.data_attrs.max_channel_count > 0: # there is padding
            channel_count = self.data_attrs.max_channel_count
        else:
            # * Just project all channels.
            # Doesn't (yet) support separate array projections.
            # Doesn't (yet) support task-subject specific readin.
            # ? I am unclear how Talukder managed to have mixed batch training if different data was shaped different sizes.
            # * Because we only ever train on one subject in this strategy, all registered arrays must belong to that subject.
            # * A rework will be needed if we want to do this lookup grouped per subject
            assert self.cfg.readin_strategy == EmbedStrat.project, 'Ragged array readin only implemented for project readin strategy'
            assert len(self.data_attrs.context.subject) <= 1, "Only implemented for single subject (likely need padding for mixed batches)"

            # for a in self.data_attrs.context.array:
            #     assert not isinstance(subject_array_registry.query_by_array(a), SortedArrayInfo), "actual mixed readins per session not yet implemented"
            channel_count = sum(
                subject_array_registry.query_by_array(a).get_channel_count() for a in self.data_attrs.context.array
            ) * self.data_attrs.spike_dim

        if self.cfg.transform_space:
            assert self.cfg.spike_embed_style in [EmbedStrat.project, EmbedStrat.token]
            if self.cfg.spike_embed_dim:
                spike_embed_dim = self.cfg.spike_embed_dim
            else:
                assert self.cfg.hidden_size % self.cfg.neurons_per_token == 0, "hidden size must be divisible by neurons per token"
                spike_embed_dim = round(self.cfg.hidden_size / self.cfg.neurons_per_token)
            if self.cfg.spike_embed_style == EmbedStrat.project:
                if getattr(self.cfg, 'debug_project_space', False):
                    self.readin = nn.Linear(channel_count, channel_count)
                else:
                    self.readin = nn.Linear(1, spike_embed_dim)
            elif self.cfg.spike_embed_style == EmbedStrat.token:
                assert self.cfg.max_neuron_count > self.data_attrs.pad_token, "max neuron count must be greater than pad token"
                self.readin = nn.Embedding(self.cfg.max_neuron_count, spike_embed_dim, padding_idx=self.data_attrs.pad_token if self.data_attrs.pad_token else None) # I'm pretty confident we won't see more than 20 spikes in 20ms but we can always bump up
                # Ignore pad token if set to 0 (we're doing 0 pad, not entirely legitimate but may be regularizing)
        else:
            if self.cfg.readin_strategy == EmbedStrat.project or self.cfg.readin_strategy == EmbedStrat.token:
                # Token is the legacy default
                self.readin = nn.Linear(channel_count, self.cfg.hidden_size)
            elif self.cfg.readin_strategy == EmbedStrat.unique_project:
                self.readin = ReadinMatrix(channel_count, self.cfg.
                hidden_size, self.data_attrs, self.cfg)
            elif self.cfg.readin_strategy == EmbedStrat.contextual_mlp:
                self.readin = ContextualMLP(channel_count, self.cfg.hidden_size, self.cfg)
            elif self.cfg.readin_strategy == EmbedStrat.readin_cross_attn:
                self.readin = ReadinCrossAttention(channel_count, self.cfg.hidden_size, self.data_attrs, self.cfg)
        if self.cfg.readout_strategy == EmbedStrat.unique_project:
            self.readout = ReadinMatrix(
                self.cfg.hidden_size,
                self.cfg.readout_dim,
                # self.cfg.readout_dim if getattr(self.cfg, 'readout_dim', 0) else channel_count,
                self.data_attrs,
                self.cfg
            )
            # like PC readout
        elif self.cfg.readout_strategy == EmbedStrat.contextual_mlp:
            self.readout = ContextualMLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg)
            # for simplicity, project out to hidden size - task IO will take care of the other items

        # TODO add readin for the stim array (similar attr)
        # if DataKey.stim in self.data_attrs.<ICMS_ATTR>:
        #   raise NotImplementedError

        def get_target_size(k: ModelTask):
            if k == ModelTask.heldout_decoding:
                # even more hacky - we know only one of these is nonzero at the same time
                return max(
                    self.data_attrs.rtt_heldout_channel_count,
                    self.data_attrs.maze_heldout_channel_count,
                )
            return channel_count
        self.task_pipelines = nn.ModuleDict({
            k.value: task_modules[k](
                self.cfg.hidden_size if task_modules[k].unique_space and self.cfg.readout_strategy is not EmbedStrat.none \
                    else self.backbone.out_size,
                get_target_size(k),
                self.cfg,
                self.data_attrs
            ) for k in self.cfg.task.tasks
        })

    def _wrap_key(self, prefix, key):
        return f'{prefix}.{key}'

    def _wrap_keys(self, prefix, named_params):
        out = []
        for n, p in named_params:
            out.append(self._wrap_key(prefix, n))
        return out

    def try_transfer(self, module_name: str, transfer_module: Any = None, transfer_data_attrs: Optional[DataAttrs] = None):
        if (module := getattr(self, module_name, None)) is not None:
            if transfer_module is not None:
                if isinstance(module, nn.Parameter):
                    assert module.data.shape == transfer_module.data.shape
                    # Currently will fail for array flag transfer, no idea what the right policy is right now
                    module.data = transfer_module.data
                else:
                    if isinstance(module, ReadinMatrix):
                        assert transfer_data_attrs is not None, "Must provide data attrs for readin matrix transfer"
                        module.load_state_dict(transfer_module.state_dict(), transfer_data_attrs)
                    else:
                        module.load_state_dict(transfer_module.state_dict())
                logger.info(f'Transferred {module_name} weights.')
            else:
                # if isinstance(module, nn.Parameter):
                #     self.novel_params.append(self._wrap_key(module_name, module_name))
                # else:
                #     self.novel_params.extend(self._wrap_keys(module_name, module.named_parameters()))
                logger.info(f'New {module_name} weights.')

    def try_transfer_embed(
        self,
        embed_name: str, # Used for looking up possibly existing attribute
        new_attrs: List[str],
        old_attrs: List[str],
        transfer_embed: Union[nn.Embedding, nn.Parameter],
    ) -> Union[nn.Embedding, nn.Parameter]:
        if transfer_embed is None:
            logger.info(f'Found no weights to transfer for {embed_name}.')
            return
        if new_attrs == old_attrs:
            self.try_transfer(embed_name, transfer_embed)
            return
        if not hasattr(self, embed_name):
            return
        embed = getattr(self, embed_name)
        if not old_attrs:
            logger.info(f'New {embed_name} weights.')
            return
        if not new_attrs:
            logger.warning(f"No {embed_name} provided in new model despite old model dependency. HIGH CHANCE OF ERROR.")
            return
        num_reassigned = 0
        def get_param(embed):
            if isinstance(embed, nn.Parameter):
                return embed
            return getattr(embed, 'weight')
        # Backport pre: package enum to string (enums from old package aren't equal to enums from new package)
        old_attrs = [str(a) for a in old_attrs]
        for n_idx, target in enumerate(new_attrs):
            if str(target) in old_attrs:
                get_param(embed).data[n_idx] = get_param(transfer_embed).data[old_attrs.index(str(target))]
                num_reassigned += 1
        # for n_idx, target in enumerate(new_attrs):
        #     if target in old_attrs:
        #         get_param(embed).data[n_idx] = get_param(transfer_embed).data[old_attrs.index(target)]
        #         num_reassigned += 1
        logger.info(f'Reassigned {num_reassigned} of {len(new_attrs)} {embed_name} weights.')
        if num_reassigned == 0:
            logger.warning(f'No {embed_name} weights reassigned. HIGH CHANCE OF ERROR.')
        if num_reassigned < len(new_attrs):
            logger.warning(f'Incomplete {embed_name} weights reassignment, accelerating learning of all.')

    def transfer_io(self, transfer_model: pl.LightningModule):
        r"""
            The logger messages are told from the perspective of a model that is being transferred to (but in practice, this model has been initialized and contains new weights already)
        """
        logger.info("Rebinding IO...")

        transfer_data_attrs: DataAttrs = transfer_model.data_attrs
        transfer_cfg: ModelConfig = transfer_model.cfg
        if self.cfg.task != transfer_cfg.task:
            logger.info(pformat(f'Task config updating.. (first logged is new config)'))
            recursive_diff_log(self.cfg.task, transfer_cfg.task)
            #  from {transfer_cfg.task} to {self.cfg.task}'))
        # def try_transfer_embed(
        #     embed_name: str, # Used for looking up possibly existing attribute
        #     new_attrs: List[str],
        #     old_attrs: List[str] ,
        # ) -> nn.Embedding:
        #     if new_attrs == old_attrs:
        #         try_transfer(embed_name)
        #         return
        #     if not hasattr(self, embed_name):
        #         return
        #     embed = getattr(self, embed_name)
        #     if not old_attrs:
        #         # if isinstance(embed, nn.Parameter):
        #         #     self.novel_params.append(self._wrap_key(embed_name, embed_name))
        #         # else:
        #         #     self.novel_params.extend(self._wrap_keys(embed_name, embed.named_parameters()))
        #         logger.info(f'New {embed_name} weights.')
        #         return
        #     if not new_attrs:
        #         logger.warning(f"No {embed_name} provided in new model despite old model dependency. HIGH CHANCE OF ERROR.")
        #         return
        #     num_reassigned = 0
        #     def get_param(embed):
        #         if isinstance(embed, nn.Parameter):
        #             return embed
        #         return getattr(embed, 'weight')
        #     for n_idx, target in enumerate(new_attrs):
        #         if target in old_attrs:
        #             get_param(embed).data[n_idx] = get_param(getattr(transfer_model, embed_name)).data[old_attrs.index(target)]
        #             num_reassigned += 1
        #     logger.info(f'Reassigned {num_reassigned} of {len(new_attrs)} {embed_name} weights.')
        #     if num_reassigned == 0:
        #         logger.warning(f'No {embed_name} weights reassigned. HIGH CHANCE OF ERROR.')
        #     if num_reassigned < len(new_attrs):
        #         # There is no non-clunky granular grouping assignment (probably) but we don't need it either
        #         logger.warning(f'Incomplete {embed_name} weights reassignment, accelerating learning of all.')
        #         # if isinstance(embed, nn.Parameter):
        #         #     self.novel_params.append(self._wrap_key(embed_name, embed_name))
        #         # else:
        #         #     self.novel_params.extend(self._wrap_keys(embed_name, embed.named_parameters()))
        self.try_transfer_embed(
            'session_embed', self.data_attrs.context.session, transfer_data_attrs.context.session,
            getattr(transfer_model, 'session_embed', None)
        )
        try:
            self.try_transfer_embed(
                'subject_embed', self.data_attrs.context.subject, transfer_data_attrs.context.subject,
                getattr(transfer_model, 'subject_embed', None)
            )
            self.try_transfer_embed(
                'task_embed', self.data_attrs.context.task, transfer_data_attrs.context.task,
                getattr(transfer_model, 'task_embed', None)
            )
            self.try_transfer_embed(
                'array_embed', self.data_attrs.context.array, transfer_data_attrs.context.array,
                getattr(transfer_model, 'array_embed', None)
            )
        except:
            print("Failed extra embed transfer, likely no impt reason (model e.g. didn't have.)")

        self.try_transfer('session_flag', getattr(transfer_model, 'session_flag', None))
        try:
            self.try_transfer('subject_flag', getattr(transfer_model, 'subject_flag', None))
            self.try_transfer('task_flag', getattr(transfer_model, 'task_flag', None))
            self.try_transfer('array_flag', getattr(transfer_model, 'array_flag', None))
        except:
            print("Failed extra embed transfer, likely no impt reason (model e.g. didn't have.)")

        self.try_transfer('context_project', getattr(transfer_model, 'context_project', None))
        self.try_transfer('readin', getattr(transfer_model, 'readin', None), transfer_data_attrs=transfer_data_attrs)
        self.try_transfer('readout', getattr(transfer_model, 'readout', None), transfer_data_attrs=transfer_data_attrs)

        for k in self.task_pipelines:
            if k in transfer_model.task_pipelines:
                logger.info(f"Transferred task pipeline {k}.")
                self.task_pipelines[k].load_state_dict(transfer_model.task_pipelines[k].state_dict(), strict=False)
            else:
                logger.info(f"New task pipeline {k}.")
                self.novel_params.extend(self._wrap_keys(f'task_pipelines.{k}', self.task_pipelines[k].named_parameters()))

    def freeze_embed(self):
        logger.info("Freezing embed.")
        def freeze_if_exists(attr: str):
            if hasattr(self, attr):
                if isinstance(getattr(self, attr), nn.Parameter):
                    getattr(self, attr).requires_grad = False
                else:
                    for p in getattr(self, attr).parameters():
                        p.requires_grad = False
        freeze_if_exists('session_embed')
        freeze_if_exists('subject_embed')
        freeze_if_exists('task_embed')
        freeze_if_exists('array_embed')
        freeze_if_exists('session_flag')
        freeze_if_exists('subject_flag')
        freeze_if_exists('task_flag')
        freeze_if_exists('array_flag')

    def freeze_backbone(self):
        logger.info("Freezing backbone.")
        for p in self.backbone.parameters():
            p.requires_grad = False
        # self.backbone.eval() # No, we still want dropout

    def freeze_non_embed(self):
        logger.info("Freezing non-embed.")
        for m in [self.backbone, self.task_pipelines, self.readin]:
            for p in m.parameters():
                p.requires_grad = False

    def _prepare_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
            Format spikes and context into tokens for backbone.
            In:
                spikes: B T A C H=1 (features provided on channel dim for principles but functionally useless)
                or B (Token) C H if `serve_tokens_flat`
            Returns:
                state_in: B x T x A x H (A should be flattened in backbone)
                static_context: List(T') [B x H]
                temporal_context: List(?) [B x T x H]
        """
        temporal_context = []
        for task in self.cfg.task.tasks:
            temporal_context.extend(self.task_pipelines[task.value].get_temporal_context(batch))

        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            if self.cfg.session_embed_token_count > 1:
                session: torch.Tensor = self.session_embed[batch[MetaKey.session]] # B x K x H
            else:
                session: torch.Tensor = self.session_embed(batch[MetaKey.session]) # B x H
        else:
            session = None
        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            if getattr(self.cfg, 'subject_embed_token_count', 1) > 1:
                subject: torch.Tensor = self.subject_embed[batch[MetaKey.subject]]
            else:
                subject: torch.Tensor = self.subject_embed(batch[MetaKey.subject]) # B x H
        else:
            subject = None
        if self.cfg.task_embed_strategy is not EmbedStrat.none:
            if self.cfg.task_embed_token_count > 1:
                task: torch.Tensor = self.task_embed[batch[MetaKey.task]]
            else:
                task: torch.Tensor = self.task_embed(batch[MetaKey.task])
        else:
            task = None
        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            array: torch.Tensor = self.array_embed(batch[MetaKey.array])
        else:
            array = None
        if self.cfg.transform_space:
            # collapse space/array, channel/feature --> # b t s h
            state_in = torch.as_tensor(batch[DataKey.spikes], dtype=int)
            if self.data_attrs.serve_tokens_flat:
                reshape = 'b t c h -> b t (c h)'
            elif self.data_attrs.serve_tokens:
                reshape = 'b t s c h -> b t s (c h)'
            else:
                # do _not_ collapse array here, mostly for code compatibility with existing pathways
                reshape = 'b t a (chunk c) h -> b t a chunk (c h)'
            state_in = rearrange(state_in, reshape, c=self.cfg.neurons_per_token)
            if self.cfg.spike_embed_style == EmbedStrat.token:
                state_in = self.readin(state_in)
            elif self.cfg.spike_embed_style == EmbedStrat.project:
                if getattr(self.cfg, 'debug_project_space', False):
                    state_in = self.readin(state_in.float())
                else:
                    state_in = self.readin(state_in.float().unsqueeze(-1))
            else:
                raise NotImplementedError
            if not getattr(self.cfg, 'debug_project_space', False):
                state_in = state_in.flatten(-2, -1) # b t s h
            # state_in = rearrange(state_in,
            #     'b t s chunk h -> b t s (chunk h)' if self.data_attrs.serve_tokens else \
            #     'b t a s_a chunk h -> b t a s_a (chunk h)'
            # ) # yes, we rearrange twice... better for alternative control flows..
        else: # --> b t a h
            state_in = torch.as_tensor(rearrange(
                batch[DataKey.spikes], 'b t a c h -> b t a (c h)'
            ), dtype=torch.float)
            if self.cfg.readin_strategy == EmbedStrat.contextual_mlp or self.cfg.readout_strategy == EmbedStrat.contextual_mlp:
                batch['session'] = session # hacky
            if self.cfg.readin_strategy in [EmbedStrat.contextual_mlp, EmbedStrat.unique_project]:
                state_in = self.readin(state_in, batch) # b t a h
            elif self.cfg.readin_strategy == EmbedStrat.readin_cross_attn: # deprecated
                state_in = self.readin(state_in, session, subject, array)
            else: # standard project
                state_in = self.readin(state_in)
        if self.cfg.encode_decode or self.cfg.task.decode_separate: # TODO decouple - or at least move after flag injection below
            # cache context
            batch['session'] = session
            batch['subject'] = subject
            batch['task'] = task

        static_context = []
        project_context = [] # only for static info
        # Note we may augment padding tokens below but if attn is implemented correctly that should be fine
        def _add_context(context: torch.Tensor, flag: torch.Tensor, strategy: EmbedStrat):
            if strategy is EmbedStrat.none:
                return
            if strategy == EmbedStrat.token:
                context = context + flag
                static_context.extend(context.unbind(1) if context.ndim == 3 else [context])
            elif strategy == EmbedStrat.token_add:
                assert not self.cfg.transform_space, 'not implemented'
                state_in = state_in + rearrange(context, 'b h -> b 1 1 h')
            elif strategy == EmbedStrat.concat:
                assert not self.cfg.transform_space, 'not implemented'
                context = repeat(context, 'b h -> b t 1 h', t=state_in.shape[1])
                project_context.append(session)
        _add_context(session, getattr(self, 'session_flag', None), self.cfg.session_embed_strategy)
        _add_context(subject, getattr(self, 'subject_flag', None), self.cfg.subject_embed_strategy)
        _add_context(task, getattr(self, 'task_flag', None), self.cfg.task_embed_strategy)
        # array embed deprecated
        # if self.cfg.array_embed_strategy is not EmbedStrat.none: # Note we check earlier that this doesn't accidentally get set for space-time, not supported yet (we need to pass/infer array metadata)
        #     assert not self.cfg.transform_space, 'not implemented'
        #     if self.cfg.array_embed_strategy == EmbedStrat.token:
        #         array = array + self.array_flag
        #         static_context.extend(array.unbind(1)) # path not yet tested
        #     elif self.cfg.array_embed_strategy == EmbedStrat.token_add:
        #         state_in = state_in + rearrange(array, 'b a h -> b 1 a h') # redundant op since array uses 0s for padding
        #     elif self.cfg.array_embed_strategy == EmbedStrat.concat:
        #         array = repeat(array, 'b a h -> b t a h', t=state_in.shape[1])
        #         project_context.append(array)
        # # TODO support temporal embed + temporal project
        # # Do not concat static context - list default is easier to deal with
        # # static_context = rearrange(static_context, 't0 b h -> b t0 h') if static_context else None
        # if project_context: # someone wanted it
        #     raise NotImplementedError # not tested
        #     # B T' H, and we want to merge into B T A H (specifically add T' to each token)
        #     augmented_tokens, ps = pack([state_in, *project_context], 'b * a h')
        #     augmented_tokens = self.context_project(augmented_tokens)
        #     state_in = rearrange(augmented_tokens, ps, 'b (t a) h', t=state_in.size(1))
        # if self.cfg.layer_norm_input:
        #     state_in = self.layer_norm_input(state_in)
        return state_in, static_context, temporal_context

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # returns backbone features B T S H


        state_in, trial_context, temporal_context = self._prepare_inputs(batch)
        temporal_padding_mask = create_temporal_padding_mask(state_in, batch)
        if DataKey.extra in batch and not self.data_attrs.serve_tokens_flat: # serve_tokens_flat is enc dec, don't integrate extra (query) tokens in enc
            state_in = torch.cat([state_in, batch[DataKey.extra]], dim=1)
            if temporal_padding_mask is not None: # Implicit - if we have extra that warrants padding, base certainly warrants padding
                extra_padding_mask = create_temporal_padding_mask(batch[DataKey.extra], batch, length_key=COVARIATE_LENGTH_KEY)
                temporal_padding_mask = torch.cat([temporal_padding_mask, extra_padding_mask], dim=1)

        # Note that fine-grained channel mask doesn't matter in forward (sub-token padding is handled in loss calculation externally)
        # But we do want to exclude fully-padded arrays from computation
        if self.data_attrs.serve_tokens_flat:
            space_padding_mask = None
        elif self.data_attrs.serve_tokens:
            assert not DataKey.extra in batch, 'not implemented'
            allocated_space_tokens = torch.ceil(batch[CHANNEL_KEY] / self.cfg.neurons_per_token).sum(1) # B
            space_comparison = torch.arange(state_in.size(2), device=state_in.device)[None, :]
            space_padding_mask = space_comparison >= allocated_space_tokens[:, None] # -> B A
        else:
            assert not DataKey.extra in batch, 'not implemented'
            space_padding_mask = batch[CHANNEL_KEY] == 0  if CHANNEL_KEY in batch else None # b x a of ints < c
        if self.cfg.transformer.n_layers == 0:
            outputs = state_in
        else:
            outputs: torch.Tensor = self.backbone(
                state_in,
                trial_context=trial_context,
                temporal_context=temporal_context,
                temporal_padding_mask=temporal_padding_mask,
                space_padding_mask=space_padding_mask,
                causal=self.cfg.causal,
                times=batch.get(DataKey.time, None),
                positions=batch.get(DataKey.position, None),
            ) # B x T x S x H or B x Token x H (flat)
        # if outputs.isnan().any():
            # import pdb;pdb.set_trace()
        return outputs

    def _step(self, batch: Dict[str, torch.Tensor], eval_mode=False) -> Dict[str, torch.Tensor]:
        r"""
            batch provided contains all configured data_keys and meta_keys
            - The distinction with `forward` is not currently clear, but `_step` is specifically oriented around training.
            Which means it'll fiddle with the payload itself and compute losses

            TODO:
            - Fix: targets are keyed/id-ed per task; there is just a single target variable we're hoping is right
            - ?: Ideally the payloads could be more strongly typed.

            We use modules to control the task-specific readouts, but this isn't multi-task first
            So a shared backbone is assumed. And a single "batch" exists for all paths.
            And moreover, any task-specific _input_ steps (such as masking/shifting) is not well interfaced right now
            (currently overloading `batch` variable, think more clearly either by studying HF repo or considering other use cases)

            Shapes:
                spikes: B T A/S C H=1 (C is electrode channel) (H=1 legacy decision, hypothetically could contain other spike features)
                - if serve_tokens: third dim is space, else it's array
                - if serve tokens flat: Time x A/S is flattened
                stim: B T C H
                channel_counts: B A (counts per array)
        """
        batch_out: Dict[str, torch.Tensor] = {}
        if Output.spikes in self.cfg.task.outputs:
            batch_out[Output.spikes] = batch[DataKey.spikes][..., 0]
        for task in self.cfg.task.tasks:
            self.task_pipelines[task.value].update_batch(batch, eval_mode=eval_mode)
        features = self(batch) # B T S H
        if self.cfg.log_backbone_norm:
            # expected to track sqrt N. If it's not, then we're not normalizing properly
            self.log('backbone_norm', torch.linalg.vector_norm(
                features.flatten(0, -2), dim=-1
            ).mean(), on_epoch=True, batch_size=features.size(0))

        if not self.cfg.transform_space:
            # no unique strategies will be tried for spatial transformer (its whole point is ctx-robustness)
            if self.cfg.readout_strategy == EmbedStrat.mirror_project:
                unique_space_features = self.readin(features, batch, readin=False)
            elif self.cfg.readout_strategy in [EmbedStrat.unique_project, EmbedStrat.contextual_mlp]:
                unique_space_features = self.readout(features, batch)

        # Create outputs for configured task
        running_loss = 0
        task_order = self.cfg.task.tasks
        if self.cfg.task.kl_lambda > 0 and ModelTask.kinematic_decoding in self.cfg.task.tasks:
            task_order = [ModelTask.kinematic_decoding]
            for t in self.cfg.task.tasks:
                if t != ModelTask.kinematic_decoding:
                    task_order.append(t)
        if getattr(self.cfg.task, 'decode_use_shuffle_backbone', False):
            task_order = [ModelTask.shuffle_infill]
            for t in self.cfg.task.tasks:
                if t != ModelTask.shuffle_infill:
                    task_order.append(t)
        for i, task in enumerate(task_order):
            pipeline_features = unique_space_features if self.task_pipelines[task.value].unique_space and self.cfg.readout_strategy is not EmbedStrat.none else features
            if 'infill' not in task.value and self.detach_backbone_for_task:
                pipeline_features = pipeline_features.detach()
            update = self.task_pipelines[task.value](
                batch,
                pipeline_features,
                eval_mode=eval_mode
            )
            for k in update:
                if 'update' in str(k):
                    if k == 'update_features':
                        features = update[k]
                    batch[k] = update[k]
                else:
                    batch_out[k] = update[k]
            if 'loss' in update and self.cfg.task.task_weights[i] > 0:
                batch_out[f'{task.value}_loss'] = update['loss']
                running_loss = running_loss + self.cfg.task.task_weights[i] * update['loss']
        batch_out['loss'] = running_loss
        return batch_out

    @torch.inference_mode()
    def predict(
        self, batch: Dict[str, torch.Tensor], transform_logrates=True, mask=True,
        eval_mode=True,
        # eval_mode=False,
    ) -> Dict[str, torch.Tensor]:
        r"""
            Note: kind of annoying to change keywords here manually (no args can be passed in)
            batch should provide info needed by model. (responsibility of user)
            Output is always batched (for now)
        """
        # breakpoint()
        if self.data_attrs.serve_tokens and not self.data_attrs.serve_tokens_flat:
            raise NotImplementedError
        # there are data keys and meta keys, that might be coming in unbatched
        batch_shapes = {
            DataKey.spikes: '* t token_chan h' if self.data_attrs.serve_tokens else '* t a c h',
            DataKey.heldout_spikes: '* t c h',
            DataKey.stim: '* t c h', # TODO review
            DataKey.bhvr_vel: '* t h',
            MetaKey.session: '*',
            MetaKey.subject: '*',
            MetaKey.task: '*',
            MetaKey.array: '* a',
            LENGTH_KEY: '*',
            COVARIATE_LENGTH_KEY: '*',
            COVARIATE_CHANNEL_KEY: '*',
            CHANNEL_KEY: '* a', # or '* token'
            DataKey.time: '* t',
            DataKey.position: '* t',
        }

        pack_info = {}
        for k in batch:
            batch[k], pack_info[k] = pack([batch[k]], batch_shapes[k])
        batch_out: Dict[str, torch.Tensor] = {}
        # auto-debug
        for k in [MetaKey.session, MetaKey.subject, MetaKey.task]:
            if k in batch:
                batch_out[k] = batch[k]
        if Output.spikes in self.cfg.task.outputs:
            assert self.data_attrs.serve_tokens_flat or not self.data_attrs.serve_tokens, "Not implemented, needs assembling"
            if self.data_attrs.serve_tokens_flat:
                batch_out[Output.spikes] = unflatten(batch[DataKey.spikes], batch[DataKey.time], batch[DataKey.position])
                batch_out['time'] = batch[DataKey.time].clone() # pre mask
                batch_out['position'] = batch[DataKey.position].clone() # pre mask
            else:
                batch_out[Output.spikes] = batch[DataKey.spikes][..., 0]
        if mask:
            for k in self.cfg.task.tasks:
                self.task_pipelines[k.value].update_batch(batch, eval_mode=eval_mode)
        features = self(batch)
        if self.cfg.readout_strategy == EmbedStrat.mirror_project:
            unique_space_features = self.readin(features, batch, readin=False)
        elif self.cfg.readout_strategy in [EmbedStrat.unique_project, EmbedStrat.contextual_mlp]:
            unique_space_features = self.readout(features, batch)
        task_order = self.cfg.task.tasks
        if self.cfg.task.kl_lambda > 0 and ModelTask.kinematic_decoding in self.cfg.task.tasks:
            task_order = [ModelTask.kinematic_decoding]
            for t in self.cfg.task.tasks:
                if t != ModelTask.kinematic_decoding:
                    task_order.append(t)
        if getattr(self.cfg.task, 'decode_use_shuffle_backbone', False):
            task_order = [ModelTask.shuffle_infill]
            for t in self.cfg.task.tasks:
                if t != ModelTask.shuffle_infill:
                    task_order.append(t)
        for task in task_order:
            update = self.task_pipelines[task.value](
                batch,
                unique_space_features if self.task_pipelines[task.value].unique_space and self.cfg.readout_strategy is not EmbedStrat.none  else features,
                compute_metrics=False,
                eval_mode=eval_mode
            )
            for k in update:
                if 'update' in str(k):
                    if k == 'update_features':
                        features = update[k]
                    batch[k] = update[k]
                else:
                    batch_out[k] = update[k]

        if self.data_attrs.serve_tokens_flat and Output.logrates in batch_out:
            batch_out[Output.logrates] = unflatten(batch_out[Output.logrates], batch_out['time'], batch_out['position'])
        if transform_logrates:
            if Output.logrates in batch_out:
                if self.data_attrs.serve_tokens_flat:
                    logger.warning('Assuming square data for rate transform')
                    batch_out[Output.rates] = self.unpad_and_transform_rates(batch_out[Output.logrates])
                else:
                    batch_out[Output.rates] = self.unpad_and_transform_rates(
                        batch_out[Output.logrates], batch[LENGTH_KEY], batch[CHANNEL_KEY] if CHANNEL_KEY in batch else None
                    )
            if Output.heldout_logrates in batch_out:
                if self.data_attrs.serve_tokens_flat:
                    logger.warning('Assuming square data for rate transform')
                    batch_out[Output.heldout_rates] = self.unpad_and_transform_rates(batch_out[Output.heldout_logrates])
                else:
                    batch_out[Output.heldout_rates] = self.unpad_and_transform_rates(
                        batch_out[Output.heldout_logrates], batch[LENGTH_KEY]
                    )
        return batch_out

    def predict_step(
        self, batch, *args, transform_logrates=True, mask=True, **kwargs
        # self, batch, *args, transform_logrates=True, mask=False, **kwargs
    ):
        return self.predict(batch, transform_logrates=transform_logrates, mask=mask)


    # === Model state ===
    def get_extra_state(self) -> Any:
        return {
            'token_proc_approx': self.token_proc_approx,
            'token_seen_approx': self.token_seen_approx,
            'novel_params': self.novel_params, # for continued training on fine-tuned model
        }

    def set_extra_state(self, state: Any):
        self.token_proc_approx = state['token_proc_approx']
        self.token_seen_approx = state['token_seen_approx']
        if 'novel_params' in state:
            self.novel_params = state['novel_params']

    # ==================== Utilities ====================
    def unpad_and_transform_rates(self, logrates: torch.Tensor, lengths: Optional[torch.Tensor] = None, channels: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
            logrates: raw, padded predictions from model, B T A H
            out: B T C
        """
        # unpad logrates using LENGTH_KEY and CHANNEL_KEY
        logrates, ps = pack([logrates], 'b t * h')
        assert channels is None or (channels == channels[0].unsqueeze(0)).all(), "Heterogenuous arrays not supported for evaluation (why would you want that anyway)"
        logrates = logrates.unbind()
        if lengths is not None:
            logrates = [l[:b, ...] for l, b in zip(logrates, lengths)]
        if channels is not None:
            cat_rates: List[torch.Tensor] = []
            for lograte, array_channels in zip(logrates, channels):
                cat_rates.append(torch.cat([lograte[:, i, :array_channels[i]] for i in range(len(array_channels))], -1))
            logrates = cat_rates
        else:
            logrates = [lr.squeeze(-2) for lr in logrates]
        # B T C
        # Now a potentially heterogenuous list of T x C, with varying T and or C
        if all(lograte.size() == logrates[0].size() for lograte in logrates[1:]):
            logrates = torch.stack(logrates)
        # NLB expects units of spikes / bin (search "spikes/bin" in https://github.dev/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb)
        return self.transform_rates(logrates, exp=True, normalize_hz=False)

    def transform_rates(
        self,
        logrates: Union[List[torch.Tensor], torch.Tensor],
        exp=True,
        normalize_hz=False
    ) -> torch.Tensor:
        r"""
            Convenience wrapper for analysis.
            logrates: Raw model output from forward pass. Can be list of batches predictions.
            exp: Should exponentiate?
            normalize_hz: Should normalize to spikes per second (instead of spikes per bin)?
        """
        def _transform(single: torch.Tensor):
            if exp:
                single = single.exp()
            if normalize_hz:
                single = single / self.data_attrs.bin_size_ms
            return single.cpu()
        out = logrates
        if isinstance(out, list):
            out = [_transform(o) for o in out]
        else:
            out = _transform(out)
        return out

    # ==================== Optimization ====================
    def common_log(self, metrics, prefix='', **kwargs):
        for m in metrics:
            if not isinstance(m, Metric) and not isinstance(m, Output) and 'update' not in m: # log misc, mostly task losses
                self.log(f'{prefix}_{m}', metrics[m], **kwargs)
        for m in self.cfg.task.metrics:
            if m == Metric.kinematic_r2 or m == Metric.kinematic_r2_thresh:
                labels = ['x', 'y', 'z']
                for i, r2 in enumerate(metrics[m]):
                    self.log(f'{prefix}_{m.value}_{labels[i]}', r2, **kwargs)
                self.log(f'{prefix}_{m.value}', metrics[m].mean(), **kwargs)
            else:
                self.log(f'{prefix}_{m.value}', metrics[m], **kwargs)

    def training_step(self, batch, batch_idx):
        if [ModelTask.shuffle_infill in self.cfg.task.tasks] and (self.cfg.log_token_proc_throughput or self.cfg.log_token_seen_throughput):
            self.token_proc_approx += batch[DataKey.spikes].size(0) * batch[DataKey.spikes].size(1)
            self.token_seen_approx += (batch[LENGTH_KEY].sum() * (1 - self.cfg.task.mask_ratio)).item()
        metrics = self._step(batch)
        if [ModelTask.shuffle_infill in self.cfg.task.tasks] and (self.cfg.log_token_proc_throughput or self.cfg.log_token_seen_throughput):
            if self.trainer.is_global_zero:
                if self.cfg.log_token_proc_throughput:
                    token_proc_approx = self.all_gather(self.token_proc_approx).sum()
                    self.log('token_proc', token_proc_approx, rank_zero_only=True)
                if self.cfg.log_token_seen_throughput:
                    token_count_approx = self.all_gather(self.token_seen_approx).sum()
                    self.log('token_seen', token_count_approx, rank_zero_only=True)

        self.common_log(metrics, prefix='train')
        return metrics['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        all_metrics = []
        if getattr(self.cfg, 'val_iters', 1) > 1:
            clean = deepcopy(batch) # not intended to be efficient, quick and dirty
        for i in range(getattr(self.cfg, 'val_iters', 1)):
            if i > 0:
                batch = deepcopy(clean)
            all_metrics.append(self._step(batch))
        metrics = {}
        for k in all_metrics[0]:
            if isinstance(all_metrics[0][k], torch.Tensor):
                metrics[k] = torch.stack([m[k] for m in all_metrics]).mean(0)
            else:
                metrics[k] = np.vstack([m[k] for m in all_metrics]).mean(0)

        # if Metric.kinematic_r2 in metrics:
            # print('Val debug: ', metrics[Metric.kinematic_r2])
        self.common_log(metrics, prefix='val' if dataloader_idx == 0 else 'eval', sync_dist=True, add_dataloader_idx=False)
        # return None metrics['loss']
        # if dataloader_idx == 0:
            # return metrics['loss']

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        r"""
            Note test step isn't capable of returning non-metrics. (use `predict` to get outputs)
        """
        metrics = self._step(batch, eval_mode=False)
        # metrics = self._step(batch, eval_mode=True)
        self.common_log(metrics, prefix='test')
        return metrics

    # def get_context_parameters(self):
    # what the heck, this api is called wrong, IDK
    #     # for use in layer-wise LR decay
    #     params = []
    #     for embed in ['session_embed', 'subject_embed', 'task_embed', 'array_embed']:
    #         if hasattr(self, embed):
    #             if isinstance(getattr(self, embed), nn.Parameter):
    #                 params.append(getattr(self, embed))
    #             else:
    #                 params.extend(*getattr(self, embed).parameters())
    #     return params

    def configure_optimizers(self):
        scheduler = None
        if self.cfg.tune_decay > 0.0: # layer-wise LR decay
            # fix readin
            # accelerate context
            # decay decoder, encoder (Kaiming MAE strategy https://arxiv.org/abs/2111.06377)
            # Position embeddings are fixed (for simplicity)
            # for simplicity
            grouped_params = [
                # {"params": self.readin.parameters(), 'lr': 0}, # don't tune
                {
                    "params": [p for n, p in self.named_parameters() if ('session_embed' in n or 'subject_embed' in n or 'task_embed' in n or 'array_embed' in n)],
                    'lr': self.cfg.lr_init * self.cfg.accelerate_new_params
                },
            ]
            decayed_lr = self.cfg.lr_init * self.cfg.accelerate_new_params
            # Decoder
            for k in self.task_pipelines:
                if k not in [ModelTask.infill.value, ModelTask.shuffle_infill.value, ModelTask.kinematic_decoding.value, ModelTask.heldout_decoding.value]:
                    raise NotImplementedError
                # Supported pipelines use "out" and "decoder" terminology for final readout and transformer decoder, respectively
                pipeline = self.task_pipelines[k]
                grouped_params.append({"params": pipeline.out.parameters(), 'lr': decayed_lr})
                if not hasattr(pipeline, 'decoder'):
                    continue
                if hasattr(pipeline.decoder, 'final_norm'):
                    grouped_params.append({"params": pipeline.decoder.final_norm.parameters(), 'lr': decayed_lr})
            for i in reversed(range(self.cfg.decoder_layers)):
                for k in self.task_pipelines:
                    if k not in [ModelTask.infill.value, ModelTask.shuffle_infill.value, ModelTask.kinematic_decoding.value, ModelTask.heldout_decoding.value]:
                        raise NotImplementedError
                    if not hasattr(pipeline, 'decoder'):
                        continue
                    pipeline = self.task_pipelines[k]
                    decayed_lr *= self.cfg.tune_decay
                    # Supported pipelines use "out" and "decoder" terminology for final readout and transformer decoder, respectively
                    grouped_params.append({"params": pipeline.decoder.transformer_encoder.layers[i].parameters(), 'lr': decayed_lr})
            # Encoder
            if hasattr(self.backbone, 'final_norm'):
                grouped_params.append({"params": self.backbone.final_norm.parameters(), 'lr': decayed_lr})
            for i in reversed(range(self.cfg.transformer.n_layers)):
                decayed_lr *= self.cfg.tune_decay
                grouped_params.append({"params": self.backbone.transformer_encoder.layers[i].parameters(), 'lr': decayed_lr})
        elif self.novel_params and self.cfg.accelerate_new_params > 1.0:
            params = list(self.named_parameters()) # As of 2/24/23 all my parameters are named, this better stay the case
            accel_flag = lambda name: name in self.novel_params or ('session_embed' in name or 'subject_embed' in name or 'task_embed' in name or 'array_embed' in name)
            grouped_params = [
                {"params": [p for n, p in params if accel_flag(n)], 'lr': self.cfg.lr_init * self.cfg.accelerate_new_params},
                {"params": [p for n, p in params if not accel_flag(n)], 'lr': self.cfg.lr_init},
            ]
        else:
            # grouped_params = self.parameters()
            # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BaseFinetuning.html#lightning.pytorch.callbacks.BaseFinetuning
            grouped_params = filter(lambda p: p.requires_grad, self.parameters())
        try:
            # from apex.optimizers import FusedAdam
            # optimizer_cls = FusedAdam # In JY's experience, about 5% speedup on 3090 in PT 1.13
            # However, literally spontaneous bug emerged where this doesn't train at all. What...?
            # And this was after successfully training and not touching anything else...?
            # The only plausible candidate is that env deactivating and reactivating lost some apex-critical state?
            # IDK.
            optimizer_cls = optim.AdamW
        except ImportError:
            logger.info("Didn't find Apex optimizer, defaulting to Pytorch AdamW")
            optimizer_cls = optim.AdamW
        optimizer = optimizer_cls(
            grouped_params,
            lr=self.cfg.lr_init,
            weight_decay=self.cfg.weight_decay
        )
        if self.cfg.lr_schedule == 'linear_warmup':
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.cfg.lr_ramp_init_factor,
                total_iters=self.cfg.lr_ramp_steps
            )
        elif self.cfg.lr_schedule == 'cosine_warmup':
            scheduler = optim.lr_scheduler.ChainedScheduler([
                optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=self.cfg.lr_ramp_init_factor,
                    total_iters=self.cfg.lr_ramp_steps
                ),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.lr_decay_steps,
                    eta_min=self.cfg.lr_min
                ),
            ])
        else:
            assert self.cfg.lr_schedule == 'fixed', f"Unknown lr_schedule {self.cfg.lr_schedule}"
        out = {
            'optimizer': optimizer,
            'monitor': 'val_loss'
        }
        if scheduler is not None:
            out['lr_scheduler'] = scheduler
        return out

    # def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    #     super().on_load_checkpoint(checkpoint)
    #     import pdb;pdb.set_trace()
    #     # TODO hook diff_cfg for LR and reset LR schedule if LR changed
    #     return
    # ? No hope, IDK how to do this; just use `init_from_id` if you messed up the schedule

# === Model loading ===
def transfer_cfg(src_cfg: ModelConfig, target_cfg: ModelConfig):
    r"""
        Copy src_cfg into target_cfg
        Motivation: Some cfg we don't want to bother repeatedly specifying; just take from the init-ing ckpt.
        Should be mutually exclusive from `diff_cfg` list.
    """
    src_cfg = OmegaConf.merge(ModelConfig(), src_cfg) # backport novel config
    for attr in [
        "hidden_size",
        "activation",
        # "weight_decay", # new regularization moved to diff_cfg
        # "dropout", # new regularization moved to diff cfg
        "session_embed_size",
        "session_embed_strategy",
        "subject_embed_size",
        "subject_embed_strategy",
        "array_embed_size",
        "array_embed_strategy",
        "task_embed_size",
        "task_embed_strategy",
        "readin_strategy",
        "transformer",
        "readout_strategy",
        "readout_dim",
        "readin_dim",
        "transform_space",
        "encode_decode",
        "spike_embed_style",
    ]:
        setattr(target_cfg, attr, getattr(src_cfg, attr))

# Note - I tried coding this as an override, but PTL `save_hyperparams()` acts up (trying to the save the `self` parameter, apparently) - even when passing explicitly that I just want to save `cfg` and `data_attrs`.
def load_from_checkpoint(
    checkpoint_path: str,
    cfg: Optional[ModelConfig] = None,
    data_attrs: Optional[DataAttrs] = None,
    use_ckpt_model_cfg: bool = False,
):
    r"""
        Specifically, model topology is determined by data_attrs.
        data_attrs thus must be saved and loaded with a model to make sense of it.
        However, if we're initializing from another checkpoint, we want to know its data_attrs, but not save it as the new attrs. To avoid doing this while still hooking into PTL `save_hyperparameters()`, we do a manual state_dict transfer of two model instances (one with old and one with new topology.)

        Args:
        - cfg: override, new cfg
        - data_attrs: override, new data_attrs
        cfg level changes are _expected_ to not affect topology,
        BUT TODO e.g. it's unclear if novel weight decay declaration means optimizer is reinitialized?
    """
    try:
        old_model = BrainBertInterface.load_from_checkpoint(checkpoint_path)
    except Exception as e: # we migrated library directory into a subfolder and old checkpoints may need paths to project dir registered
        logger.warning(e)
        logger.warning("Failed to load checkpoint, assuming old format and retrying after registering project dir...")
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        old_model = BrainBertInterface.load_from_checkpoint(checkpoint_path)

    if cfg is None and data_attrs is None:
        return old_model
    if cfg is not None:
        transfer_cfg(src_cfg=old_model.cfg, target_cfg=cfg)
        # import pdb;pdb.set_trace()
        if old_model.diff_cfg(cfg):
            raise Exception("Unsupported config diff")
    else:
        cfg = old_model.cfg
    if data_attrs is None:
        data_attrs = old_model.data_attrs
    new_cls = BrainBertInterface(cfg=cfg, data_attrs=data_attrs)
    new_cls.backbone.load_state_dict(old_model.backbone.state_dict())
    new_cls.transfer_io(old_model)
    return new_cls

def transfer_model(
    old_model: BrainBertInterface, new_cfg: ModelConfig, new_data_attrs: DataAttrs,
    extra_embed_map: Dict[str, Tuple[Any, DataAttrs]] = {}
):
    r"""
        Transfer model to new cfg and data_attrs.
        Intended to be for inference
    """
    if new_cfg is None and new_data_attrs is None:
        return old_model
    if new_cfg is not None:
        transfer_cfg(src_cfg=old_model.cfg, target_cfg=new_cfg)
        if old_model.diff_cfg(new_cfg):
            raise Exception("Unsupported config diff")
    else:
        new_cfg = old_model.cfg
    if new_data_attrs is None:
        new_data_attrs = old_model.data_attrs
    new_cls = BrainBertInterface(cfg=new_cfg, data_attrs=new_data_attrs)
    new_cls.backbone.load_state_dict(old_model.backbone.state_dict())
    new_cls.transfer_io(old_model)

    for embed_key in extra_embed_map:
        logger.info(f"Transferring extra {embed_key}...")
        extra_embed, extra_attrs = extra_embed_map[embed_key]
        new_cls.try_transfer_embed(f'{embed_key}_embed', getattr(new_cls.data_attrs.context, embed_key), getattr(extra_attrs.context, embed_key), extra_embed)

    return new_cls

# Utilities

def recursive_diff_log(cfg1: Union[DictConfig, ListConfig], cfg2: Union[DictConfig, ListConfig], prefix=""):
    # cfg intended as new, semantically
    if not isinstance(cfg1, DictConfig): # Don't step into ListConfigs
        if cfg1 != cfg2:
            logger.info(f"{prefix} diff: {cfg1} vs {cfg2}")
    else:
        # iterate through attributes
        for attr in cfg1:
            if attr not in cfg2:
                logger.info(f"cfg1 has {attr} but cfg2 does not")
            else:
                recursive_diff_log(getattr(cfg1, attr), getattr(cfg2, attr), prefix=attr)
        for attr in cfg2:
            if attr not in cfg1:
                logger.info(f"cfg2 has {attr} but cfg1 does not")


def unflatten(
    flat_data: torch.Tensor,
    time: torch.Tensor,
    position: torch.Tensor,
    default_value=-100,
):
    r"""
        Unflatten data into (time, position) space
        Args:
            flat_data: (batch, flat ~= time*position, token_chan, ...)
            time: (batch, flat_time (len time*position))
            position: (batch, flat_position (len time * position))
        Returns:
            assembled: (batch, time, channel)
    """
    b, _, token_chan, *rest = flat_data.size()
    time_min, time_max = time.min(), time.max()
    position_min, position_max = position.min(), position.max()
    assembled = torch.full(
        (b, time_max - time_min + 1, position_max - position_min + 1, token_chan, *rest),
        default_value,
        device=flat_data.device,
        dtype=flat_data.dtype,
    )
    assembled[ # no scatter needed, merely need to select the specified indices
        torch.arange(b, device=flat_data.device)[:, None],
        time - time_min,
        position - position_min,
    ] = flat_data
    assembled = assembled.flatten(start_dim=2)
    return assembled