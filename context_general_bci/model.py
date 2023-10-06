from typing import Tuple, Dict, List, Optional, Any, Mapping, Union
from copy import deepcopy
import math
import numpy as np
import torch
from torch import nn, optim
from torch.nn import init
import torch.nn.functional as F
import lightning.pytorch as pl
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
    DEFAULT_KIN_LABELS,
    BatchKey
)

from context_general_bci.dataset import DataAttrs, LENGTH_KEY, CHANNEL_KEY, COVARIATE_LENGTH_KEY, COVARIATE_CHANNEL_KEY
from context_general_bci.subjects import subject_array_registry, SortedArrayInfo
# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
from context_general_bci.components import (
    SpaceTimeTransformer
)
from context_general_bci.task_io import task_modules
from context_general_bci.utils import (
    enum_backport,
    sort_A_by_B,
    unflatten,
    cosine_schedule,
)

logger = logging.getLogger(__name__)

# For autoregressive. If using a common position space to sort, this defines the canonical order.
# Not sure I really believe in separators - the space input should cue the requisite modality.
MODALITY_SPACE_RANGE_START = { # These include both human readable aliases for convenience, but code mainly reference task pipeline names
    'padding': 0, # also trial context receives these space values
    'trial': 0, # also trial context receives these space values

    'constraints': 1, # 1-10. If tokenized, there are as many constraint dims as behavior dims. We allocate max of 10 behavior dims for now.

    'spike': 11, # 11-20. Max of 10 spike dims (32 neurons per -> 320 neurons, IIRC 288 was max for NDT2)
    'spike_context': 11,
    'spike_infill': 11,

    'return': 21, # Only 1 dimension needed for return.
    'return_context': 21,
    'return_infill': 21,

    'covariate': 22, # 22-31. Max of 10 covariate dims. Separator token possibly include.
    'kinematic_classification': 22,
    'kinematic_infill': 22,
    'kinematic_context': 22,
}
MAX_KINEMATIC_DIMS = 10

def cm3leon_init(m):
    if isinstance(m, nn.Linear):
        init.trunc_normal_(m.weight, std=6e-3, a=-3, b=3)
    elif isinstance(m, nn.MultiheadAttention):
        init.trunc_normal_(m.in_proj_weight, std=6e-3, a=-3, b=3)
        # Initialize bias terms if they exist
        if m.in_proj_bias is not None:
            nn.init.constant_(m.in_proj_bias, 0)

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
        assert data_attrs.serve_tokens_flat, 'NDT3 assumes flat serving of tokens'
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
            # Max space can be manipulated in model in next_step path; thus model is responsible for determining max space to encode. If not, use raw max token expected
            max_spatial_tokens = self.cfg.max_spatial_position if self.cfg.next_step_prediction else data_attrs.max_spatial_tokens
            self.backbone = SpaceTimeTransformer(
                self.cfg.transformer,
                max_spatial_tokens=max_spatial_tokens,
                debug_override_dropout_out=cfg.transformer.debug_override_dropout_io,
                context_integration=cfg.transformer.context_integration,
                embed_space=cfg.transformer.embed_space,
                allow_embed_padding=True,
            )
        self.bind_io()

        if self.cfg.cm3leon_init:
            self.backbone.apply(cm3leon_init)
        self.novel_params: List[str] = [] # for fine-tuning
        modifies = []
        for tp in self.task_pipelines.values():
            modifies.extend(tp.modifies)
        assert len(set(modifies)) == len(modifies), f"Task pipelines ({len(modifies)}) oversubscribed must modify different keys, found ({modifies})"

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
            'use_full_encode',
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
            if self.cfg.subject_embed_strategy == EmbedStrat.token and self.cfg.subject_embed_token_count > 1:
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
            if self.cfg.task_embed_strategy == EmbedStrat.token and self.cfg.task_embed_token_count > 1:
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
        if self.cfg.readout_strategy == EmbedStrat.unique_project:
            assert False, "deprecated"
        elif self.cfg.readout_strategy == EmbedStrat.contextual_mlp:
            assert False, "deprecated"

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
                self.backbone.out_size,
                get_target_size(k),
                self.cfg,
                self.data_attrs
            ) for k in self.cfg.task.tasks
        })

        if self.cfg.next_step_prediction: # special tokens
            self.start_of_sentence = nn.Parameter(torch.randn(self.cfg.hidden_size) / math.sqrt(self.cfg.hidden_size))
            # Checks on spatial tokens
            assert self.data_attrs.max_spatial_tokens_neural < MODALITY_SPACE_RANGE_START['return'] -  MODALITY_SPACE_RANGE_START['spike']
            assert self.cfg.max_spatial_position >= max(MODALITY_SPACE_RANGE_START.values()) + MAX_KINEMATIC_DIMS


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
                        module.load_state_dict(transfer_module.state_dict(), strict=False)
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
        for m in [self.backbone, self.task_pipelines]:
            for p in m.parameters():
                p.requires_grad = False

    def _prepare_trial_context(self, batch: Dict[BatchKey, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
            Format spikes and context into tokens for backbone.
            In:
                spikes: B T A C H=1 (features provided on channel dim for principles but functionally useless)
                or B (Token) C H if `serve_tokens_flat`
            Returns:
                static_context: List(T') [B x H]
        """
        assert self.cfg.array_embed_strategy == EmbedStrat.none, "Array embed strategy deprecated"

        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            if self.cfg.session_embed_token_count > 1:
                session: torch.Tensor = self.session_embed[batch[MetaKey.session]] # B x K x H
            else:
                session: torch.Tensor = self.session_embed(batch[MetaKey.session]) # B x H
        else:
            session = None
        if self.cfg.subject_embed_strategy is not EmbedStrat.none:
            if self.cfg.subject_embed_token_count > 1:
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

        if self.cfg.encode_decode or self.cfg.task.decode_separate: # TODO decouple - or at least move after flag injection below
            # cache context
            batch['session'] = session
            batch['subject'] = subject
            batch['task'] = task

        static_context: List[torch.Tensor] = []
        # Note we may augment padding tokens below but if attn is implemented correctly that should be fine
        def _add_context(context: torch.Tensor, flag: torch.Tensor, strategy: EmbedStrat):
            if strategy is EmbedStrat.none:
                return
            # assume token strategy
            context = context + flag
            static_context.append(context)
        _add_context(session, getattr(self, 'session_flag', None), self.cfg.session_embed_strategy)
        _add_context(subject, getattr(self, 'subject_flag', None), self.cfg.subject_embed_strategy)
        _add_context(task, getattr(self, 'task_flag', None), self.cfg.task_embed_strategy)
        if not static_context:
            return [], [], [], []
        metadata_context = pack(static_context, 'b * h')[0]
        return (
            metadata_context,
            torch.zeros(metadata_context.size()[:2], device=metadata_context.device, dtype=int), # time
            torch.zeros(metadata_context.size()[:2], device=metadata_context.device, dtype=int), # space
            torch.zeros(metadata_context.size()[:2], device=metadata_context.device, dtype=bool), # padding
        )

    @property
    def do_kin_maskout(self):
        if self.cfg.kinematic_token_maskout_schedule == "cosine":
            return True
        elif self.cfg.kinematic_token_maskout_schedule == "random":
            return True
        else:
            return self.cfg.kinematic_token_maskout > 0

    @property
    def kin_maskout(self):
        if self.cfg.kinematic_token_maskout_schedule == "cosine":
            maskout = cosine_schedule(
                time=torch.as_tensor(self.current_epoch),
                T=self.cfg.lr_decay_steps,
                start=self.cfg.kinematic_token_maskout_start,
                end=self.cfg.kinematic_token_maskout
            )
        elif self.cfg.kinematic_token_maskout_schedule == "random":
            maskout = (torch.rand(1) * (self.cfg.kinematic_token_maskout_start - self.cfg.kinematic_token_maskout) + self.cfg.kinematic_token_maskout)[0]
        elif self.cfg.kinematic_token_maskout_schedule in ["", "constant"]:
            maskout = self.cfg.kinematic_token_maskout
        else:
            raise ValueError(f"Unknown kinematic token maskout schedule {self.cfg.kinematic_token_maskout_schedule}")
        return maskout

    def assemble_pipeline(self, batch: Dict[BatchKey, torch.Tensor], prefix=False) -> Tuple[
        List[str], List[Any],
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        # modalities is _target_ at timestep, roll forward to determine input modality
        # TODO we should bake metadata context into a regular pipeline, right now left alone due to special transfer logic
        trial_context, trial_times, trial_space, trial_padding = self._prepare_trial_context(batch) # metadata context
        tks, tps = list(self.task_pipelines.keys()), list(self.task_pipelines.values())
        pipeline_context, pipeline_times, pipeline_space, pipeline_padding = zip(*[
            tp.get_context(batch) for tp in tps
        ])
        tks.append('trial')

        pipeline_context = [*pipeline_context, trial_context] # tuples
        pipeline_times = [*pipeline_times, trial_times]
        pipeline_space = [*pipeline_space, trial_space]
        pipeline_padding = [*pipeline_padding, trial_padding]
        filtered = [i for i, p in enumerate(pipeline_context) if p != []]
        # breakpoint()
        tks = [tks[i] for i in filtered]

        pipeline_context = [pipeline_context[i] for i in filtered] # embedded at this point
        pipeline_times = [pipeline_times[i] for i in filtered]
        pipeline_space = [pipeline_space[i] for i in filtered]
        pipeline_padding = [pipeline_padding[i] for i in filtered]


        # Merge context into single seq (in NDT3, data/neuro is not revealed to backbone)
        if self.cfg.next_step_prediction:
            # Update positions for later subsequent canonical order, before we pack and lose track of which modalities are which
            for i, (tk, s) in enumerate(zip(tks, pipeline_space)):
                pipeline_space[i] = s + MODALITY_SPACE_RANGE_START[tk]
            modalities = [torch.full_like(s, filtered[i], dtype=torch.uint8) for i, s in enumerate(pipeline_space)] # track original task pipeline index
            modalities, _ = pack(modalities, 'b *')
        else:
            for i, (tk, s) in enumerate(zip(tks, pipeline_space)):
                pipeline_space[i] = (s + 1) if tk != 'trial' else s
            modalities = None

        pipeline_context, ps = pack(pipeline_context, 'b * h')
        times, _ = pack(pipeline_times, 'b *')
        space, _ = pack(pipeline_space, 'b *')
        pipeline_padding, _ = pack(pipeline_padding, 'b *')

        mask = None

        if self.cfg.next_step_prediction:
            # Pack and Sort. Time is the major sort key, space is minor. We pre-allocate space per modality

            # breakpoint()
            # TODO this op may be redundant - we may be able to address it directly in data loader
            times[pipeline_padding] = self.cfg.transformer.max_trial_length # Assumes dataloader currently doesn't serve pad time especially
            space[pipeline_padding] = self.cfg.max_spatial_position # Assumes dataloader currently doesn't serve pad space especially
            order = times * self.cfg.max_spatial_position + space

            # * ps becomes useless, is that ok? It's fine - we need to create a modality mask so subsequent task pipelines can map out their desired targets
            pipeline_context, indices = sort_A_by_B(pipeline_context, order)
            times, _ = sort_A_by_B(times, order, indices)
            space, _ = sort_A_by_B(space, order, indices)
            pipeline_padding, _ = sort_A_by_B(pipeline_padding, order, indices)
            # breakpoint()
            # assert (pipeline_padding.diff(1).sum(1) <= 1).all(), "Padding should be contiguous and at end of trial..."
            modalities, _ = sort_A_by_B(modalities, order, indices)

            # breakpoint()
            # As _input_, we provide the previous step (teacher-forcing).
            # Output targets are maintained (individual tasks are responsible for tracking this)
            pipeline_context = pipeline_context.roll(1, dims=1)
            if self.training:
                # breakpoint()
                if self.cfg.token_maskout > 0:
                    mask = torch.rand(pipeline_context.size(1), device=pipeline_context.device) < self.cfg.token_maskout
                    pipeline_context[:, mask] = 0
                elif self.do_kin_maskout:
                    is_kinematic_input = (modalities == tks.index('kinematic_infill')).roll(1, dims=1)
                    is_kinematic_input[:, 0] = False
                    mask = torch.rand(pipeline_context.size(1), device=pipeline_context.device) < self.kin_maskout
                    if prefix and self.cfg.task.context_prompt_time_thresh > 0:
                        # Essentially - maskout only begins at timestamps past prompt threshold.
                        mask = mask & (times >= self.cfg.task.context_prompt_time_thresh)
                    elif prefix and self.cfg.task.context_prompt_time_thresh < 0:
                        # Wer still want mask to only apply at timestamps past prompt threshold, but we from end of trial.
                        # print(times.shape)
                        non_pad_times = times.clone()
                        non_pad_times[pipeline_padding] = -1
                        times_from_end = times - non_pad_times.max(-1, keepdim=True).values
                        mask = mask & (times_from_end >= self.cfg.task.context_prompt_time_thresh)
                    mask = is_kinematic_input & mask
                    pipeline_context[mask] = 0
            pipeline_context[:, 0] = self.start_of_sentence

        return (
            tks, ps,
            pipeline_context,
            times,
            space,
            pipeline_padding,
            modalities,
            mask # tokens with no cue input, used for optional loss block
        )

    def forward(self, batch: Dict[BatchKey, torch.Tensor], use_prefix=False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]:
        r"""
            returns backbone features B T H, and timesteps B T
            modalities is flag indicating _target_ modality.
        """
        tks, ps, pipeline_context, times, space, pipeline_padding, modalities, zero_mask = self.assemble_pipeline(
            batch,
            prefix=use_prefix
        )
        outputs: torch.Tensor = self.backbone(
            pipeline_context,
            autoregressive=self.cfg.next_step_prediction,
            padding_mask=None if self.cfg.next_step_prediction else pipeline_padding, # suppress padding if flash attn-able
            causal=self.cfg.causal,
            times=times,
            positions=space,
        ) # B x Token x H (flat)
        if self.cfg.use_full_encode:
            return outputs, times, space, pipeline_padding, modalities, zero_mask
        else:
            outputs = unpack(outputs, ps, 'b * h')
            times = unpack(times, ps, 'b *')
            space = unpack(space, ps, 'b *')
            pipeline_padding = unpack(pipeline_padding, ps, 'b *')
            if 'shuffle_infill' in tks:
                enc_index = tks.index('shuffle_infill') # TODO replace with something that targets the spike context provider...
            else:
                enc_index = tks.index('spike_context')
            return outputs[enc_index], times[enc_index], space[enc_index], pipeline_padding[enc_index], None, None

    def _step(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False) -> Dict[BatchKey, torch.Tensor]:
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
        """
        # breakpoint()
        batch_out: Dict[BatchKey | Output, torch.Tensor] = {}
        if Output.spikes in self.cfg.task.outputs:
            batch_out[Output.spikes] = batch[DataKey.spikes][..., 0]
        for task in self.cfg.task.tasks:
            self.task_pipelines[task.value].update_batch(batch, eval_mode=eval_mode)
        if self.cfg.task.prefix_ratio > 0:
            use_prefix = torch.rand(1) < self.cfg.task.prefix_ratio
            prefix_loss = use_prefix
        else:
            use_prefix = True # feel free to use if available
            prefix_loss = False

        features, times, space, padding, modalities, zero_mask = self(batch, use_prefix=use_prefix) # B T H
        if self.cfg.log_backbone_norm:
            # expected to track sqrt N. If it's not, then we're not normalizing properly
            self.log('backbone_norm', torch.linalg.vector_norm(
                features.flatten(0, -2), dim=-1
            ).mean(), on_epoch=True, batch_size=features.size(0))

        # Create outputs for configured task
        running_loss = 0
        for i, task in enumerate(self.cfg.task.tasks):
            should_detach = 'infill' not in task.value and self.detach_backbone_for_task
            if self.cfg.next_step_prediction:
                sub_features = features[modalities == i] # Only route relevant features, tasks shouldn't be doing anything. # B* H (flattened)
                sub_times = times[modalities == i]
                sub_space = space[modalities == i]
                sub_padding = padding[modalities == i]
                sub_loss_mask = None if zero_mask is None else zero_mask[modalities == i]
                # breakpoint() # Check the shape here. Also, check the modality mask, unclear we provide the right features if some task pipeline provides nothing
            else:
                sub_features = features
                sub_times = times
                sub_space = space
                sub_padding = padding
                sub_loss_mask = zero_mask
            update = self.task_pipelines[task.value](
                batch,
                sub_features.detach() if should_detach else sub_features,
                sub_times,
                sub_space,
                sub_padding,
                loss_mask=sub_loss_mask if prefix_loss else None,
                eval_mode=eval_mode
            )
            batch_out.update(update)
            if 'loss' in update and self.cfg.task.task_weights[i] > 0:
                batch_out[f'{task.value}_loss'] = update['loss']
                running_loss = running_loss + self.cfg.task.task_weights[i] * update['loss']
        batch_out['loss'] = running_loss
        return batch_out

    @torch.inference_mode()
    def predict(
        self, batch: Dict[BatchKey, torch.Tensor], transform_logrates=True, mask=True,
        eval_mode=True,
        # eval_mode=False,
    ) -> Dict[BatchKey | Output, torch.Tensor]:
        r"""
            Note: kind of annoying to change keywords here manually (no args can be passed in)
            batch should provide info needed by model. (responsibility of user)
            Output is always batched (for now)
        """
        assert self.data_attrs.serve_tokens_flat, "Not implemented"
        # there are data keys and meta keys, that might be coming in unbatched
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
            DataKey.covariate_time: '* t',
            DataKey.covariate_space: '* t',
            DataKey.covariate_labels: '*',
            DataKey.constraint: '* t constraint_dim',
            DataKey.constraint_space: '* t',
            DataKey.constraint_time: '* t',
            DataKey.task_return: '* t h',
            DataKey.task_reward: '* t h',
            DataKey.task_return_time: '* t',
            # DataKey.task_return_space: '* t',
            'constraint_length': '*',
            'return_length': '*',
        }
        pack_info = {}
        for k in batch:
            if k == DataKey.covariate_labels:
                continue
            batch[k], pack_info[k] = pack([batch[k]], batch_shapes[k])
        batch_out: Dict[str | DataKey | MetaKey | Output, torch.Tensor] = {}
        # auto-debug
        for k in [MetaKey.session, MetaKey.subject, MetaKey.task]:
            if k in batch:
                batch_out[k] = batch[k]
        if Output.spikes in self.cfg.task.outputs:
            assert self.data_attrs.serve_tokens_flat or not self.data_attrs.serve_tokens, "Not implemented, needs assembling"
            if self.data_attrs.serve_tokens_flat:
                batch_out[Output.spikes] = unflatten(batch[DataKey.spikes], batch[DataKey.time], batch[DataKey.position])
                batch_out[DataKey.time] = batch[DataKey.time].clone() # pre mask
                batch_out[DataKey.position] = batch[DataKey.position].clone() # pre mask
            else:
                batch_out[Output.spikes] = batch[DataKey.spikes][..., 0]

        for k in self.cfg.task.tasks:
            self.task_pipelines[k.value].update_batch(batch, eval_mode=eval_mode)

        if self.cfg.next_step_prediction:
            # Autoregressive inference (no beam search atm - in practice we need one step at a time anw)
            tks, ps, pipeline_context, times, space, pipeline_padding, modalities, zero_mask = self.assemble_pipeline(batch)

            # There are only certain tokens I want model predictions for - the tokens that have kinematic modality targets.
            to_infer_indices = torch.tensor([i for i, tk in enumerate(tks) if tk == 'kinematic_infill'], device=space.device)
            to_infer_mask = torch.isin(modalities, to_infer_indices)
            if self.cfg.eval.limit_timesteps: # Evaluating full length is slow with KV cache, we need to iterate faster
                logger.warning('Assuming even batches for cropped prediction!!!')
                # breakpoint()
                first_step_time = ((times >= self.cfg.eval.limit_timesteps) & (times != self.data_attrs.max_trial_length)).any(0)
                if first_step_time.any():
                    first_step_time = first_step_time.nonzero()[0][0].item()
                    to_infer_mask[:, first_step_time:] = False
            proc_step = 0
            raw_stream = []
            stream_mask = []
            cue_mask = [torch.zeros_like(to_infer_mask[:, 0])] # initially not student cue
            main_seq = torch.zeros_like(times, dtype=batch[DataKey.bhvr_vel].dtype) # B T
            main_seq[modalities == tks.index('kinematic_infill')] = batch[DataKey.bhvr_vel].flatten()
            target_stream = []
            # breakpoint()
            predicted_to = 0 # Exclusive, do we have a prediction up till this step?
            predict_until = 0 # The goalpost hasn't been set yet.
            need_student_slice = (times >= self.cfg.eval.teacher_timesteps).any(0)
            # Want the first slice (batch wise) where anyone needs student force; predict up to that step (exclusvie)
            # breakpoint()
            if not need_student_slice.any():
                predict_until = times.size(1)
            else:
                predict_until = need_student_slice.nonzero()[0][0].item() # Predict_until is exclusive.

            if self.cfg.eval.maskout_last_n:
                # We don't immediately load student, so we need to keep a copy on hand. For convenience, we copy full stream
                student_stream = pipeline_context.clone()
                # Identify the kinematics up to n steps before the first student slice, and zero it out
                # breakpoint()
                is_kinematic_input = (modalities == tks.index('kinematic_infill')).roll(1, dims=1)
                is_kinematic_input[:, 0] = False
                blacklist_kin_times = (times < self.cfg.eval.teacher_timesteps) \
                    & (times >= self.cfg.eval.teacher_timesteps - self.cfg.eval.maskout_last_n) \
                    & is_kinematic_input
                pipeline_context[blacklist_kin_times] = 0
            while proc_step < times.size(1):
                # Jump to the next inferrable step
                if not to_infer_mask[:, proc_step].any():
                    proc_step += 1
                    continue
                if proc_step + 1 > predicted_to:
                    if proc_step + 1 > predict_until: # If we want step 100, and we haven't predicted until 101 exclusive, we need to predict until 101
                        predict_until = proc_step + 1
                    outputs = self.backbone( # No, this isn't enough. If I want a prediction at proc_step, I need to predict until proc_step+1
                        pipeline_context[:, :predict_until], # We want predictions at the current step - provide input up to current step
                        autoregressive=True,
                        padding_mask=None,
                        causal=self.cfg.causal,
                        times=times[:, :predict_until],
                        positions=space[:, :predict_until],
                    )
                    predicted_to = predict_until
                    # breakpoint()
                    # The question hereafter is - is a prediction for proc_step ready?
                # Sample the output from the kinematic pipeline
                decode = self.task_pipelines['kinematic_infill'](
                    batch,
                    outputs[:, proc_step: proc_step + 1],
                    times[:, proc_step: proc_step + 1],
                    space[:, proc_step: proc_step + 1],
                    pipeline_padding[:, proc_step: proc_step + 1],
                    compute_metrics=False,
                    temperature=self.cfg.eval.temperature,
                )

                # We run prediction even if modality is wrong; we slice out correct trials only when forced.
                raw_pred = decode[Output.behavior_pred]
                # breakpoint()
                raw_stream.append(raw_pred)
                target_stream.append(main_seq[:, proc_step:proc_step+1]) # Mark relevant tokens in timestep
                stream_mask.append(to_infer_mask[:, proc_step:proc_step+1]) # Mark relevant tokens in timestep

                # Need to decode and quantize again... (redundant work but IDRC)
                # Greedy decoding - subset to only the relevant pieces
                # No student replacement - just debugging atm!
                re_enc: torch.Tensor = self.task_pipelines['kinematic_infill'].encode_cov(raw_pred)
                if self.cfg.eval.use_student:
                    if self.cfg.eval.student_prob < 1:
                        re_enc = torch.where(
                            torch.rand_like(re_enc) < self.cfg.eval.student_prob,
                            re_enc,
                            0
                        )
                else:
                    re_enc.zero_() # Mirrors Maskout

                if proc_step < times.size(1) - 1:
                    # Will the next step need a student?
                    should_student = times[:, proc_step+1] >= self.cfg.eval.teacher_timesteps
                    cue_mask.append(should_student)
                    # Only student force the tokens that we predicted - hence use `to_infer_mask` of current step
                    if self.cfg.eval.maskout_last_n:
                        # Essentially keep the student stream updated; but only copy up to the last N steps. Meanwhile, true stream should be zero-ed out
                        student_stream[:, proc_step+1][
                            to_infer_mask[:, proc_step] & should_student
                        ] = re_enc[
                            to_infer_mask[:, proc_step] & should_student
                        ]
                        re_enc.zero_()
                        pipeline_context[:, proc_step+1][
                            to_infer_mask[:, proc_step] & should_student
                        ] = re_enc[
                            to_infer_mask[:, proc_step] & should_student
                        ]
                        veil_time = times[:, proc_step:proc_step + 1] - self.cfg.eval.maskout_last_n
                        time_mask = times[:, :proc_step+1] < veil_time
                        pipeline_context[:, :proc_step + 1][time_mask] = student_stream[:, :proc_step + 1][time_mask]
                    else:
                        pipeline_context[:, proc_step+1][
                            to_infer_mask[:, proc_step] & should_student
                        ] = re_enc[
                            to_infer_mask[:, proc_step] & should_student
                        ]
                proc_step += 1
                if True or proc_step % 100 == 0:
                    print(f'Inferred {proc_step} of {times.size(1)} steps.')
            raw_stream = torch.cat(raw_stream, 1) # B T
            stream_mask = torch.cat(stream_mask, 1) # B T
            target_stream = torch.cat(target_stream, 1) # B T
            cue_mask = torch.stack(cue_mask, 1) # B T
            if cue_mask.size(1) > stream_mask.size(1): # crop last step
                cue_mask = cue_mask[:, :stream_mask.size(1)]

            # In order to ID the right raws across batches, track behavior in flat datastream timeline
            # breakpoint()
            batch_out = {
                Output.behavior_pred: raw_stream[stream_mask], # Row major flattening. Should produce coherent outputs, discontinuities at trials.
                Output.behavior: target_stream[stream_mask],
                Output.behavior_query_mask: cue_mask[stream_mask],
            }
        else:
            features, times, space, padding, modalities = self(batch)
            for i, task in enumerate(self.cfg.task.tasks):
                if self.cfg.next_step_prediction:
                    sub_features = features[modalities == i] # Only route relevant features, tasks shouldn't be doing anything. # B* H (flattened)
                    sub_times = times[modalities == i]
                    sub_space = space[modalities == i]
                    sub_padding = padding[modalities == i]
                else:
                    sub_features = features
                    sub_times = times
                    sub_space = space
                    sub_padding = padding
                update = self.task_pipelines[task.value](
                    batch,
                    sub_features,
                    sub_times,
                    sub_space,
                    sub_padding,
                    compute_metrics=False,
                    eval_mode=eval_mode
                )
                batch_out.update(update)
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
    def common_log(
        self,
        metrics,
        prefix='',
        kinematic_labels=DEFAULT_KIN_LABELS,
        **kwargs
    ):
        for m in metrics:
            if 'loss' in m:
                self.log(f'{prefix}_{m}', metrics[m], **kwargs)
        for m in self.cfg.task.metrics:
            if m == Metric.kinematic_r2 or m == Metric.kinematic_r2_thresh:
                if not self.data_attrs.tokenize_covariates: # Heterogeneous, just hangs the DDP procs. Either we maintain the global list and report 0s, or we drop.
                    # For now, let's just drop.
                    for i, r2 in enumerate(metrics[str(m)]):
                        self.log(f'{prefix}_{m.value}_{kinematic_labels[i]}', r2, **kwargs)
                self.log(f'{prefix}_{m.value}', metrics[str(m)].mean(), **kwargs)
            else:
                self.log(f'{prefix}_{m.value}', metrics[str(m)], **kwargs)
        self.log('kin_maskout', self.kin_maskout, **kwargs)

    def training_step(self, batch, batch_idx):
        # if batch_idx > 2:
        #     return None # Override, debug
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

        self.common_log(
            metrics,
            prefix='train',
            kinematic_labels=batch[DataKey.covariate_labels] if DataKey.covariate_labels in batch else DEFAULT_KIN_LABELS,
        )
        return metrics['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        all_metrics = []
        if self.cfg.val_iters > 1:
            assert not self.data_attrs.tokenize_covariates, "We don't know how to combine multiple different R2s at the moment"
            clean = deepcopy(batch) # not intended to be efficient, quick and dirty
        for i in range(self.cfg.val_iters):
            if i > 0:
                batch = deepcopy(clean)
            all_metrics.append(self._step(batch))
        metrics = {}
        for k in all_metrics[0]:
            if isinstance(all_metrics[0][k], torch.Tensor):
                metrics[k] = torch.stack([m[k] for m in all_metrics]).mean(0)
            else:
                metrics[k] = np.vstack([m[k] for m in all_metrics]).mean(0)

        self.common_log(
            metrics,
            prefix='val' if dataloader_idx == 0 else 'eval',
            sync_dist=True,
            add_dataloader_idx=False,
            kinematic_labels=batch[DataKey.covariate_labels] if DataKey.covariate_labels in batch else DEFAULT_KIN_LABELS,
        )
        # return None metrics['loss']
        # if dataloader_idx == 0:
            # return metrics['loss']

    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        r"""
            Note test step isn't capable of returning non-metrics. (use `predict` to get outputs)
        """

        metrics = self._step(batch, eval_mode=False)
            # kinematic_labels=batch[DataKey.covariate_labels] if DataKey.covariate_labels in batch else DEFAULT_KIN_LABELS,
        # )
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
        Does not load optimizer state (TODO, get that)
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