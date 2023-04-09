r"""
We adapt BrainBertInterface for online decoding specifically.
That is, we remove the flexible interface so we can compile the model as well.
"""
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any, Mapping
import dataclasses
import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack, unpack # baby steps...
from omegaconf import OmegaConf, ListConfig, DictConfig
import logging
from pprint import pformat
from model import unflatten, transfer_cfg, recursive_diff_log, BrainBertInterface

from config import (
    ModelConfig,
    ModelTask,
    Metric,
    Output,
    EmbedStrat,
    DataKey,
    MetaKey,
    Architecture,
)

from data import DataAttrs, LENGTH_KEY, CHANNEL_KEY, COVARIATE_LENGTH_KEY, COVARIATE_CHANNEL_KEY
from subjects import subject_array_registry, SortedArrayInfo
# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
from components import (
    SpaceTimeTransformer,
    ReadinMatrix,
    ReadinCrossAttention,
    ContextualMLP,
)
from task_io import task_modules, SHUFFLE_KEY, create_temporal_padding_mask, TaskPipeline

logger = logging.getLogger(__name__)

# Assumes FLAT TOKEN SERVING
batch_shapes = {
    DataKey.spikes: '* t token_chan h',
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

TIMESTEPS = 100

class SkinnyBehaviorRegression(nn.Module):
    r"""
        For online. Do not use for training.
    """

    def __init__(
        self, cfg: ModelConfig, data_attrs: DataAttrs,
    ):
        super().__init__()
        self.decode_cross_attn = True
        self.cfg = cfg.task

        self.time_pad = cfg.transformer.max_trial_length
        self.decoder = SpaceTimeTransformer(
            cfg.transformer,
            max_spatial_tokens=0,
            n_layers=cfg.decoder_layers,
            allow_embed_padding=True,
            context_integration='cross_attn',
            embed_space=False
        )
        self.out = nn.Linear(cfg.hidden_size, data_attrs.behavior_dim)
        self.causal = cfg.causal
        self.spacetime = cfg.transform_space
        self.bhvr_lag_bins = round(self.cfg.behavior_lag / data_attrs.bin_size_ms)
        assert self.bhvr_lag_bins >= 0, "behavior lag must be >= 0, code not thought through otherwise"

        if getattr(self.cfg, 'decode_normalizer', ''):
            # See `data_kin_global_stat`
            zscore_path = Path(self.cfg.decode_normalizer)
            assert zscore_path.exists(), f'normalizer path {zscore_path} does not exist'
            self.register_buffer('bhvr_mean', torch.load(zscore_path)['mean'])
            self.register_buffer('bhvr_std', torch.load(zscore_path)['std'])

            # TODO need to unnormalize with these
        else:
            self.bhvr_mean = None
            self.bhvr_std = None

        self.register_buffer('decode_token', repeat(torch.zeros(cfg.hidden_size), 'h -> 1 t h', t=TIMESTEPS))
        self.register_buffer('decode_time', torch.arange(TIMESTEPS).unsqueeze(0)) # 20ms bins, 2s
        if self.causal and self.cfg.behavior_lag_lookahead:
            self.decode_time = self.decode_time + self.bhvr_lag_bins

    def forward(self, backbone_features: torch.Tensor, src_time: torch.Tensor, trial_context: List[torch.Tensor]) -> torch.Tensor:
        backbone_features: torch.Tensor = self.decoder(
            self.decode_tokens,
            trial_context=trial_context,
            times=self.decode_time,
            positions=None,
            space_padding_mask=None,
            causal=self.causal,
            memory=backbone_features,
            memory_times=src_time,
        )
        bhvr = self.out(backbone_features)
        if self.bhvr_lag_bins:
            bhvr = bhvr[:, :-self.bhvr_lag_bins]
        return bhvr[:,-1]



class BrainBertInterfaceDecoder(pl.LightningModule):
    r"""
        I know I'll end up regretting this name.
    """
    def __init__(self, cfg: ModelConfig, data_attrs: DataAttrs):
        super().__init__() # store cfg
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.data_attrs = data_attrs
        assert self.data_attrs.max_channel_count % self.cfg.neurons_per_token == 0, "Neurons per token must divide max channel count"
        if self.data_attrs.serve_tokens:
            assert self.cfg.array_embed_strategy == EmbedStrat.none, 'array IDs serving not implemented for spatially tokenized data'
            assert self.cfg.transform_space, 'Transform space must be true if serving (spacetime) tokens'
            assert self.data_attrs.neurons_per_token == self.cfg.neurons_per_token, \
                f"Neurons per token served by data ({self.data_attrs.neurons_per_token}) must match model token size {self.cfg.neurons_per_token}"
        if self.data_attrs.serve_tokens_flat:
            assert self.cfg.transformer.flat_encoder, "Flat encoder must be true if serving flat tokens"
        self.backbone = SpaceTimeTransformer(
            self.cfg.transformer,
            max_spatial_tokens=data_attrs.max_spatial_tokens,
            debug_override_dropout_out=getattr(cfg.transformer, 'debug_override_dropout_io', False),
            context_integration=getattr(cfg.transformer, 'context_integration', 'in_context'),
            embed_space=cfg.transformer.embed_space,
        )
        self.bind_io()

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
        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            assert self.data_attrs.context.session, "Session embedding strategy requires session in data"
            if len(self.data_attrs.context.session) == 1:
                logger.warning('Using session embedding strategy with only one session. Expected only if tuning.')
        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            assert self.data_attrs.context.subject, "Subject embedding strategy requires subject in data"
            if len(self.data_attrs.context.subject) == 1:
                logger.warning('Using subject embedding strategy with only one subject. Expected only if tuning.')
        if self.cfg.array_embed_strategy is not EmbedStrat.none:
            assert self.data_attrs.context.array, "Array embedding strategy requires array in data"
            if len(self.data_attrs.context.array) == 1:
                logger.warning('Using array embedding strategy with only one array. Expected only if tuning.')
        if self.cfg.task_embed_strategy is not EmbedStrat.none:
            assert self.data_attrs.context.task, "Task embedding strategy requires task in data"
            if len(self.data_attrs.context.task) == 1:
                logger.warning('Using task embedding strategy with only one task. Expected only if tuning.')

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
            self.subject_embed = nn.Embedding(len(self.data_attrs.context.subject), self.cfg.subject_embed_size)
            if self.cfg.subject_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.subject_embed_size
            elif self.cfg.subject_embed_strategy == EmbedStrat.token:
                assert self.cfg.subject_embed_size == self.cfg.hidden_size
                if self.cfg.init_flags:
                    self.subject_flag = nn.Parameter(torch.randn(self.cfg.subject_embed_size) / math.sqrt(self.cfg.subject_embed_size))
                else:
                    self.subject_flag = nn.Parameter(torch.zeros(self.cfg.subject_embed_size))

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

        self.decoder = SkinnyBehaviorRegression(self.cfg, self.data_attrs)

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
        def try_transfer(module_name: str):
            if (module := getattr(self, module_name, None)) is not None:
                if (transfer_module := getattr(transfer_model, module_name, None)) is not None:
                    if isinstance(module, nn.Parameter):
                        assert module.data.shape == transfer_module.data.shape
                        # Currently will fail for array flag transfer, no idea what the right policy is right now
                        module.data = transfer_module.data
                    else:
                        if isinstance(module, ReadinMatrix):
                            module.load_state_dict(transfer_module.state_dict(), transfer_data_attrs)
                        else:
                            module.load_state_dict(transfer_module.state_dict())
                    logger.info(f'Transferred {module_name} weights.')
                else:
                    logger.info(f'New {module_name} weights.')
        def try_transfer_embed(
            embed_name: str, # Used for looking up possibly existing attribute
            new_attrs: List[str],
            old_attrs: List[str] ,
        ) -> nn.Embedding:
            if new_attrs == old_attrs:
                try_transfer(embed_name)
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
            for n_idx, target in enumerate(new_attrs):
                if target in old_attrs:
                    get_param(embed).data[n_idx] = get_param(getattr(transfer_model, embed_name)).data[old_attrs.index(target)]
                    num_reassigned += 1
            logger.info(f'Reassigned {num_reassigned} of {len(new_attrs)} {embed_name} weights.')
            if num_reassigned == 0:
                logger.warning(f'No {embed_name} weights reassigned. HIGH CHANCE OF ERROR.')
            if num_reassigned < len(new_attrs):
                logger.warning(f'Incomplete {embed_name} weights reassignment, accelerating learning of all.')
        try_transfer_embed('session_embed', self.data_attrs.context.session, transfer_data_attrs.context.session)
        try_transfer_embed('subject_embed', self.data_attrs.context.subject, transfer_data_attrs.context.subject)
        try_transfer_embed('task_embed', self.data_attrs.context.task, transfer_data_attrs.context.task)

        try_transfer('session_flag')
        try_transfer('subject_flag')
        try_transfer('task_flag')

    def forward(
        self,
        spikes: torch.Tensor # T x C x H # ! should be int dtype, double check
    ) -> torch.Tensor: # out is behavior, T x 2
        # do the reshaping yourself
        # ! Assumes this divides evenly
        spikes = spikes.unfold(1, self.cfg.neurons_per_token, self.cfg.neurons_per_token).flatten(-2)
        t, c, h = spikes.size()
        time = repeat(torch.arange(spikes.size(0), device=spikes.device), 't -> (t c)')
        position = repeat(torch.arange(c, device=spikes.device), 'c -> (t c)', t=t)

        session = self.session_embed(torch.zeros(1, dtype=torch.uint8, device=spikes.device)) + self.session_flag
        subject = self.subject_embed(torch.zeros(1, dtype=torch.uint8, device=spikes.device)) + self.subject_flag
        task = self.task_embed(torch.zeros(1, dtype=torch.uint8, device=spikes.device)) + self.task_flag
        trial_context = [session, subject, task]

        state_in = self.readin(spikes).flatten(-2).unsqueeze(0)

        features: torch.Tensor = self.backbone(
            state_in,
            trial_context=trial_context,
            temporal_context=[],
            temporal_padding_mask=None,
            space_padding_mask=None,
            causal=self.cfg.causal,
            times=time,
            positions=position,
        ) # B x Token x H (flat)
        return self.decoder(features, time, trial_context)


def transfer_model(old_model: BrainBertInterface, new_cfg: ModelConfig, new_data_attrs: DataAttrs):
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
    new_cls = BrainBertInterfaceDecoder(cfg=new_cfg, data_attrs=new_data_attrs)
    new_cls.backbone.load_state_dict(old_model.backbone.state_dict())
    new_cls.transfer_io(old_model)
    # TODO more safety in this loading... like the injector is unhappy
    new_cls.decoder.load_state_dict(old_model.task_pipelines[ModelTask.kinematic_decoding.value].state_dict(), strict=False)
    return new_cls