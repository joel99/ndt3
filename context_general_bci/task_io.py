from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import abc
import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce, einsum, pack, unpack # baby steps...
from einops.layers.torch import Rearrange
from sklearn.metrics import r2_score
import logging

logger = logging.getLogger(__name__)

from context_general_bci.config import (
    ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey,
    BatchKey
)

from context_general_bci.dataset import (
    DataAttrs,
    LENGTH_KEY,
    CHANNEL_KEY,
    COVARIATE_LENGTH_KEY,
    COVARIATE_CHANNEL_KEY,
    CONSTRAINT_LENGTH_KEY,
    RETURN_LENGTH_KEY
)
from context_general_bci.contexts import context_registry, ContextInfo
from context_general_bci.components import SpaceTimeTransformer
from context_general_bci.utils import sort_A_by_B

# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
SHUFFLE_KEY = "shuffle"

r"""
Utilities
"""

def logsumexp(x):
    c = x.max()
    return c + (x - c).exp().sum().log()

def apply_shuffle(item: torch.Tensor, shuffle: torch.Tensor):
    # item: B T *
    # shuffle: T
    return item.transpose(1, 0)[shuffle].transpose(1, 0)

def apply_shuffle_2d(item: torch.Tensor, shuffle: torch.Tensor):
    # item: B T *
    # shuffle: T
    # return item.transpose(1, 0)[shuffle].transpose(1, 0)

    batch_size, time_dim = shuffle.shape

    # Create an index tensor to represent the batch dimension
    batch_idx = torch.arange(batch_size)[:, None].repeat(1, time_dim)

    # Use gather to apply different permutations to each batch
    return item[batch_idx, shuffle]

def temporal_pool(batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, temporal_padding_mask: torch.Tensor, pool='mean', override_time=0):
    # Originally developed for behavior regression, extracted for heldoutprediction
    # Assumption is that bhvr is square!
    # This path assumes DataKey.time is not padded!
    time_key = DataKey.time
    if 'update_time' in batch:
        time_key = 'update_time'
    pooled_features = torch.zeros(
        backbone_features.shape[0],
        (override_time if override_time else batch[time_key].max() + 1) + 1, # 1 extra for padding
        backbone_features.shape[-1],
        device=backbone_features.device,
        dtype=backbone_features.dtype
    )
    time_with_pad_marked = torch.where(
        temporal_padding_mask,
        batch[time_key].max() + 1,
        batch[time_key]
    )
    pooled_features = pooled_features.scatter_reduce(
        src=backbone_features,
        dim=1,
        index=repeat(time_with_pad_marked, 'b t -> b t h', h=backbone_features.shape[-1]),
        reduce=pool,
        include_self=False
    )
    # print(torch.allclose(pooled_features[:,0,:] - backbone_features[:,:7,:].mean(1), torch.tensor(0, dtype=torch.float), atol=1e-6))
    backbone_features = pooled_features[:,:-1] # remove padding
    # assume no padding (i.e. all timepoints have data) - this is square assumption
    temporal_padding_mask = torch.ones(backbone_features.size(0), backbone_features.size(1), dtype=bool, device=backbone_features.device)

    # Scatter "padding" to remove square assumption. Output is padding iff all timepoints are padding
    temporal_padding_mask = temporal_padding_mask.float().scatter_reduce(
        src=torch.zeros_like(batch[time_key]).float(),
        dim=1,
        index=batch[time_key],
        reduce='prod',
        include_self=False
    ).bool()
    return backbone_features, temporal_padding_mask

class PoissonCrossEntropyLoss(nn.CrossEntropyLoss):
    r"""
        Poisson-softened multi-class cross entropy loss
        JY suspects multi-context spike counts may be multimodal and only classification can support this?
    """
    def __init__(self, max_count=20, soften=True, **kwargs):
        super().__init__(**kwargs)
        self.soften = soften
        if self.soften:
            poisson_map = torch.zeros(max_count+1, max_count+1)
            for i in range(max_count):
                probs = torch.distributions.poisson.Poisson(i).log_prob(torch.arange(max_count+1)).exp()
                poisson_map[i] = probs / probs.sum()
            self.register_buffer("poisson_map", poisson_map)

    def forward(self, logits, target, *args, **kwargs):
        # logits B C *
        # target B *
        target = target.long()
        if self.soften:
            class_second = [0, -1, *range(1, target.ndim)]
            og_size = target.size()
            soft_target = self.poisson_map[target.flatten()].view(*og_size, -1)
            target = soft_target.permute(class_second)
        return super().forward(
            logits,
            target,
            *args,
            **kwargs,
        )

class TaskPipeline(nn.Module):
    r"""
        Task IO - manages decoder layers, loss functions
        i.e. is responsible for returning loss, decoder outputs, and metrics
    """
    modifies: List[DataKey] = [] # Which DataKeys are altered in use of this Pipeline? (We check to prevent multiple subscriptions)

    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ) -> None:
        super().__init__()
        self.cfg = cfg.task
        self.pad_value = data_attrs.pad_token
        self.serve_tokens = data_attrs.serve_tokens
        self.serve_tokens_flat = data_attrs.serve_tokens_flat

    @abc.abstractproperty
    def handle(self) -> str:
        r"""
            Handle for identifying task
        """
        raise NotImplementedError

    def get_context(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor | List, torch.Tensor | List, torch.Tensor | List, torch.Tensor | List]:
        r"""
            Context for covariates that should be embedded.
            (e.g. behavior, stimuli, ICMS)
            JY co-opting to also just track separate covariates that should possibly be reoncstructed (but main model doesn't know to do this atm, may need to signal.)
            returns:
            - a sequence of embedded tokens (B T H)
            - their associated timesteps. (B T)
            - their associated space steps (B T)
            - padding mask (B T)
            Defaults to empty list for packing op
        """
        return [], [], [], []

    def get_conditioning_context(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor | List, torch.Tensor | List, torch.Tensor | List, torch.Tensor | List]:
        r"""
            For task specific trial _input_. (B T H)
            Same return as above.
        """
        raise NotImplementedError # TODO still not consumed in main model
        return None, None, None, None

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        r"""
            Currently redundant with get_context - need to refactor.
            It could be that this forces a one-time modification.
            Update batch in place for modifying/injecting batch info.
        """
        return batch

    def get_trial_query(self, batch: Dict[str, torch.Tensor]):
        r"""
            For task specific trial _query_. (B H)
        """
        raise NotImplementedError # nothing in main model to use this
        return []

    def extract_trial_context(self, batch, detach=False):
        trial_context = []
        for key in ['session', 'subject', 'task']:
            if key in batch and batch[key] is not None:
                trial_context.append(batch[key] if not detach else batch[key].detach())
        return trial_context

    def forward(
            self,
            batch,
            backbone_features: torch.Tensor,
            backbone_times: torch.Tensor,
            backbone_space: torch.Tensor,
            backbone_padding: torch.Tensor,
            compute_metrics=True,
            eval_mode=False
        ) -> torch.Tensor:
        r"""
            By default only return outputs. (Typically used in inference)
            - compute_metrics: also return metrics.
            - eval_mode: Run IO in eval mode (e.g. no masking)
        """
        raise NotImplementedError

class ContextPipeline(TaskPipeline):
    # Doesn't do anything, just injects tokens
    # Responsible for encoding a piece of the datastream
    def forward(self, *args, **kwargs):
        return {}

    @abc.abstractmethod
    def get_context(self, batch: Dict[str, torch.Tensor]):
        raise NotImplementedError

class ConstraintPipeline(ContextPipeline):
    r"""
        Note this pipeline is mutually exclusive with the dense implementation in CovariateReadout.
    """

    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ) -> None:
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.inject_constraint_tokens = data_attrs.sparse_constraints and self.cfg.encode_constraints # Injects as dimvarying context
        if self.cfg.encode_constraints:
            if self.inject_constraint_tokens and self.cfg.use_constraint_cls:
                # Not obvious we actually need yet _another_ identifying cls if we're also encoding others, but can we afford a zero token if no constraints are active...?
                self.constraint_cls = nn.Parameter(torch.randn(cfg.hidden_size))
            self.constraint_dims = nn.Parameter(torch.randn(3, cfg.hidden_size))
        # self.norm = nn.LayerNorm(cfg.hidden_size) # * Actually no, don't norm the linear projection...

    def encode_constraint(self, constraint: torch.Tensor) -> torch.Tensor:
        # constraint: Out is B T H Bhvr_Dim for sparse/tokenized, or B T H if dense
        if not self.cfg.decode_tokenize_dims:
            # In the dense decode_tokenize_dims path, tokens already arrive rearranged due to crop batch. Flattening occurs outside this func
            constraint_embed = einsum(constraint, self.constraint_dims, 'b t constraint d, constraint h -> b t h d')
            if self.inject_constraint_tokens and self.cfg.use_constraint_cls:
                constraint_embed = constraint_embed + rearrange(self.constraint_cls, 'h -> 1 1 h 1')
            if not self.inject_constraint_tokens: # reduce (pretty crude - we can't tell which dim is constrained like thus)
                constraint_embed = constraint_embed.mean(-1) # B T H
        else:
            constraint_embed = einsum(constraint, self.constraint_dims, 'b t constraint, constraint h -> b t h')
            if self.inject_constraint_tokens and self.cfg.use_constraint_cls:
                constraint_embed = constraint_embed + rearrange(self.constraint_cls, 'h -> 1 1 h')
        return constraint_embed


    def get_context(self, batch: Dict[str, torch.Tensor]):
        assert self.cfg.encode_constraints and self.inject_constraint_tokens, 'constraint pipeline only for encoding tokenized constraints'
        constraint = batch[DataKey.constraint]

        constraint_embed = self.encode_constraint(constraint) # b t h d
        time = batch[DataKey.constraint_time]
        if self.cfg.decode_tokenize_dims:
            assert DataKey.constraint_space in batch, 'constraint space must be provided, inference deprecated on tokenized path'
            space = batch[DataKey.constraint_space]
        else:
            logger.warning('Deprecated constraint path! JY does not remember what preconditions are for this path')
            bhvr_attr_factor = constraint_embed.size(1) // time.size(1) if self.cfg.decode_tokenize_dims else constraint_embed.size(-1)
            space = repeat(torch.arange(bhvr_attr_factor, device=constraint_embed.device), 'd -> b (t d)', b=constraint_embed.size(0), t=time.size(1))
            time = repeat(time, 'b t -> b (t d)', d=bhvr_attr_factor)
        padding = create_token_padding_mask(
            constraint,
            batch,
            length_key=CONSTRAINT_LENGTH_KEY, # 9/15/23: length is compatible on tokenized path. No need for special length treatment
            # multiplicity=bhvr_attr_factor if self.cfg.decode_tokenize_dims else 1,
        ) # Make it before constraint is flattened
        if not self.cfg.decode_tokenize_dims: # if not already flattened
            padding = repeat(padding, 'b t -> b (t d)', d=bhvr_attr_factor)
            constraint_embed = rearrange(constraint_embed, 'b t h d -> b (t d) h')
        return (
            constraint_embed,
            time,
            space,
            padding,
        )


class DataPipeline(TaskPipeline):
    def get_masks(
        self,
        batch: Dict[str, torch.Tensor],
        channel_key=CHANNEL_KEY,
        length_key=LENGTH_KEY,
        ref: torch.Tensor | None = None,
        compute_channel=True,
        shuffle_key=SHUFFLE_KEY,
        encoder_frac=0,
        padding_mask: Optional[torch.Tensor]=None,
    ):
        r"""
            length_key: token-level padding info
            channel_key: intra-token padding info
            encoder_frac: All masks are used for metric computation, which implies it's being run after decoding. Decode tokens are always on the tail end of shuffled seqs, so we pull this length of tail if provided.
        """
        # loss_mask: b t *
        if ref is None:
            ref: torch.Tensor = batch[DataKey.spikes][..., 0]
        loss_mask = torch.ones(ref.size(), dtype=torch.bool, device=ref.device)

        if padding_mask is None:
            padding_mask = create_token_padding_mask(ref, batch, length_key=length_key, shuffle_key=shuffle_key)
            if encoder_frac:
                padding_mask = padding_mask[..., encoder_frac:]
        length_mask = ~(padding_mask & torch.isnan(ref).any(-1))

        loss_mask = loss_mask & length_mask.unsqueeze(-1)

        if channel_key in batch and compute_channel: # only some of b x a x c are valid
            assert ref.ndim >= 3 # Channel dimension assumed as dim 2
            comparison = repeat(torch.arange(ref.size(2), device=ref.device), 'c -> 1 1 c')

            # Note no shuffling occurs here because 1. channel_key shuffle is done when needed earlier
            # 2. no spatial shuffling occurs so we do need to apply_shuffle(torch.arange(c))
            channels = batch[channel_key] # b x a of ints < c (or b x t)
            if channels.ndim == 1:
                channels = channels.unsqueeze(1)
            channel_mask = comparison < rearrange(channels, 'b t -> b t 1') # dim 2 is either arrays (base case) or tokens (flat)
            loss_mask = loss_mask & channel_mask
        else:
            loss_mask = loss_mask[..., 0] # don't specify channel dim if not used, saves HELDOUT case
            channel_mask = None
        return loss_mask, length_mask, channel_mask


class RatePrediction(DataPipeline):
    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        if self.serve_tokens_flat:
            assert Metric.bps not in self.cfg.metrics, "bps metric not supported for flat tokens"
        readout_size = cfg.neurons_per_token if cfg.transform_space else channel_count
        if self.cfg.unique_no_head:
            decoder_layers = []
        elif self.cfg.linear_head:
            decoder_layers = [nn.Linear(backbone_out_size, readout_size)]
        else:
            decoder_layers = [
                nn.Linear(backbone_out_size, backbone_out_size),
                nn.ReLU() if cfg.activation == 'relu' else nn.GELU(),
                nn.Linear(backbone_out_size, readout_size)
            ]

        if not cfg.lograte:
            decoder_layers.append(nn.ReLU())

        if cfg.transform_space and not self.serve_tokens: # if serving as tokens, then target has no array dim
            # after projecting, concatenate along the group dimension to get back into channel space
            decoder_layers.append(Rearrange('b t a s_a c -> b t a (s_a c)'))
        self.out = nn.Sequential(*decoder_layers)

        if getattr(self.cfg, 'spike_loss', 'poisson') == 'poisson':
            self.loss = nn.PoissonNLLLoss(reduction='none', log_input=cfg.lograte)
        elif self.cfg.spike_loss == 'cross_entropy':
            self.loss = PoissonCrossEntropyLoss(reduction='none', soften=self.cfg.cross_ent_soften)

    @torch.no_grad()
    def bps(
        self, rates: torch.Tensor, spikes: torch.Tensor, is_lograte=True, mean=True, raw=False,
        length_mask: Optional[torch.Tensor]=None, channel_mask: Optional[torch.Tensor]=None,
        block=False
    ) -> torch.Tensor:
        r""" # tensors B T A C
            Bits per spike, averaged over channels/trials, summed over time.
            Convert extremely uninterpretable NLL into a slightly more interpretable BPS. (0 == constant prediction for BPS)
            For evaluation.
            length_mask: B T
            channel_mask: B A C

            block: Whether to get null from full batch (more variable, but std defn)
        """
        # convenience logic for allowing direct passing of record with additional features
        if is_lograte:
            logrates = rates
        else:
            logrates = (rates + 1e-8).log()
        if spikes.ndim == 5 and logrates.ndim == 4:
            spikes = spikes[..., 0]
        assert spikes.shape == logrates.shape
        nll_model: torch.Tensor = self.loss(logrates, spikes)
        spikes = spikes.float()
        if length_mask is not None:
            nll_model[~length_mask] = 0.
            spikes[~length_mask] = 0
        if channel_mask is not None:
            nll_model[~channel_mask.unsqueeze(1).expand_as(nll_model)] = 0.
            # spikes[~channel_mask.unsqueeze(1).expand_as(spikes)] = 0 # redundant

        nll_model = reduce(nll_model, 'b t a c -> b a c', 'sum')

        if length_mask is not None:
            mean_rates = reduce(spikes, 'b t a c -> b 1 a c', 'sum') / reduce(length_mask, 'b t -> b 1 1 1', 'sum')
        else:
            mean_rates = reduce(spikes, 'b t a c -> b 1 a c')
        if block:
            mean_rates = reduce(mean_rates, 'b 1 a c -> 1 1 a c', 'mean').expand_as(spikes)
        mean_rates = (mean_rates + 1e-8).log()
        nll_null: torch.Tensor = self.loss(mean_rates, spikes)

        if length_mask is not None:
            nll_null[~length_mask] = 0.
        if channel_mask is not None:
            nll_null[~channel_mask.unsqueeze(1).expand_as(nll_null)] = 0.

        nll_null = nll_null.sum(1) # B A C
        # Note, nanmean used to automatically exclude zero firing trials. Invalid items should be reported as nan.s here
        bps_raw: torch.Tensor = ((nll_null - nll_model) / spikes.sum(1) / np.log(2))
        if raw:
            return bps_raw
        bps = bps_raw[(spikes.sum(1) != 0).expand_as(bps_raw)].detach()
        if bps.isnan().any() or bps.mean().isnan().any():
            return 0 # Stitch is crashing for some reason...
            # import pdb;pdb.set_trace() # unnatural - this should only occur if something's really wrong with data
        if mean:
            return bps.mean()
        return bps

    @staticmethod
    def create_linear_head(
        cfg: ModelConfig, in_size: int, out_size: int, layers=1
    ):
        assert cfg.transform_space, 'Classification heads only supported for transformed space'
        if cfg.task.spike_loss == 'poisson':
            classes = 1
        elif cfg.task.spike_loss == 'cross_entropy':
            classes = cfg.max_neuron_count
        out_layers = [
            nn.Linear(in_size, out_size * classes)
        ]
        if layers > 1:
            out_layers.insert(0, nn.ReLU() if cfg.activation == 'relu' else nn.GELU())
            out_layers.insert(0, nn.Linear(in_size, in_size))
        if cfg.task.spike_loss == 'poisson':
            if not cfg.lograte:
                out_layers.append(nn.ReLU())
        else:
            out_layers.append(
                Rearrange('b t (s c) -> b c t s', c=classes)
            )
        return nn.Sequential(*out_layers)

class SpikeContext(ContextPipeline):

    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.spike_embed_style = cfg.spike_embed_style

        if cfg.spike_embed_dim:
            spike_embed_dim = cfg.spike_embed_dim
        else:
            assert cfg.hidden_size % cfg.neurons_per_token == 0, "hidden size must be divisible by neurons per token"
            spike_embed_dim = round(cfg.hidden_size / cfg.neurons_per_token)
        if self.spike_embed_style == EmbedStrat.project:
            self.readin = nn.Linear(1, spike_embed_dim)
        elif self.spike_embed_style == EmbedStrat.token:
            assert cfg.max_neuron_count > data_attrs.pad_token, "max neuron count must be greater than pad token"
            self.readin = nn.Embedding(cfg.max_neuron_count, spike_embed_dim, padding_idx=data_attrs.pad_token if data_attrs.pad_token else None)
            # I'm pretty confident we won't see more than 20 spikes in 20ms but we can always bump up
            # Ignore pad token if set to 0 (we're doing 0 pad, not entirely legitimate but may be regularizing)

    @property
    def handle(self):
        return 'spike'

    def encode(self, batch):
        state_in = torch.as_tensor(batch[DataKey.spikes], dtype=int)
        state_in = rearrange(state_in, 'b t c h -> b t (c h)')
        if self.spike_embed_style == EmbedStrat.token:
            state_in = self.readin(state_in)
        elif self.spike_embed_style == EmbedStrat.project:
            state_in = self.readin(state_in.float().unsqueeze(-1))
        else:
            raise NotImplementedError
        state_in = state_in.flatten(-2, -1)
        return state_in

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        batch[DataKey.padding] = create_token_padding_mask(
            batch[DataKey.spikes], batch,
            length_key=LENGTH_KEY, # Use the right key, if there's no shuffle # TODO fix dataloader to load LENGTH_KEY as SPIKE_LENGTH_KEY (make spikes less special)
        )
        return batch

    def get_context(self, batch: Dict[str, torch.Tensor]):
        spikes = self.encode(batch)
        time = batch[DataKey.time]
        space = batch[DataKey.position]
        padding = batch[DataKey.padding] # Padding should be made in the `update` step
        # print(f'Spike Space range: [{space.min()}, {space.max()}]')
        return spikes, time, space, padding

class SelfSupervisedInfill(RatePrediction):
    modifies = [DataKey.spikes]
    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        spikes = batch[DataKey.spikes]
        target = spikes[..., 0]
        if eval_mode:
            batch.update({
                # don't actually mask
                'is_masked': torch.zeros(spikes.size()[:-2], dtype=torch.bool, device=spikes.device),
                'spike_target': target
            })
            return batch
        is_masked = torch.bernoulli(
            # ! Spatial-masking seems slightly worse on RTT, revisit with tuning + neuron dropout
            torch.full(spikes.size()[:2], self.cfg.mask_ratio, device=spikes.device)
            # torch.full(spikes.size()[:-2], self.cfg.mask_ratio, device=spikes.device)
        ) # B T S or B Token - don't mask part of a token
        if not self.serve_tokens_flat:
            is_masked = is_masked.unsqueeze(-1) # mock spatial masking
            is_masked = is_masked.expand(*spikes.shape[:2], spikes.shape[2]) # B T S
        mask_type = torch.rand_like(is_masked)
        mask_token = mask_type < self.cfg.mask_token_ratio
        mask_random = (mask_type >= self.cfg.mask_token_ratio) & (mask_type < self.cfg.mask_token_ratio + self.cfg.mask_random_ratio)
        is_masked = is_masked.bool()
        mask_token, mask_random = (
            mask_token.bool() & is_masked,
            mask_random.bool() & is_masked,
        )

        spikes = spikes.clone()
        if self.cfg.mask_random_shuffle:
            assert not self.serve_tokens, 'shape not updated'
            b, t, a, c, _ = spikes.shape
            if LENGTH_KEY in batch:
                times = rearrange(batch[LENGTH_KEY], 'b -> b 1 1') # 1 = a
            else:
                times = torch.full((b, 1, a), t, device=spikes.device)
            # How can we generate a random time if we have different bounds? Use a large number and take modulo, roughly fair
            # (note permute doesn't really work if we have ragged times, we risk shuffling in padding)
            random_draw = torch.randint(0, 100000, (b, t, a), device=times.device) % times

            # Use random_draw to index spikes and extract a tensor of size b t a c 1
            # TODO update this
            time_shuffled_spikes = spikes.gather(1, random_draw.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, c, -1))
            spikes[mask_random] = time_shuffled_spikes[mask_random]
        else:
            if self.serve_tokens and not self.serve_tokens_flat: # ! Spatial-masking seems slightly worse on RTT, revisit with tuning + neuron dropout
                mask_random = mask_random.expand(-1, -1, spikes.size(2))
                mask_token = mask_token.expand(-1, -1, spikes.size(2))
            spikes[mask_random] = torch.randint_like(spikes[mask_random], 0, spikes[spikes != self.pad_value].max().int().item() + 1)
        spikes[mask_token] = 0 # use zero mask per NDT (Ye 21) # TODO revisit for spatial mode; not important in causal mode
        batch.update({
            DataKey.spikes: spikes,
            'is_masked': is_masked,
            'spike_target': target,
        })
        return batch

    def forward(
            self,
            batch,
            backbone_features: torch.Tensor,
            backbone_times: torch.Tensor,
            backbone_space: torch.Tensor,
            backbone_padding: torch.Tensor,
            compute_metrics=True,
            eval_mode=False
        ) -> torch.Tensor:
        assert False, "Deprecated"
        rates: torch.Tensor = self.out(backbone_features)
        batch_out = {}
        if Output.logrates in self.cfg.outputs:
            # rates as B T S C, or B T C
            # assert self.serve_tokens_flat or (not self.serve_tokens), 'non-flat token logic not implemented'
            # TODO torch.gather the relevant rate predictions
            assert not self.serve_tokens, 'shape not updated, not too sure what to do here'
            batch_out[Output.logrates] = rates

        if not compute_metrics:
            return batch_out
        spikes = batch['spike_target']
        loss: torch.Tensor = self.loss(rates, spikes)
        # Infill update mask
        loss_mask, length_mask, channel_mask = self.get_masks(batch)
        if Metric.all_loss in self.cfg.metrics:
            batch_out[Metric.all_loss] = loss[loss_mask].mean().detach()
        loss_mask = loss_mask & batch['is_masked'].unsqueeze(-1) # add channel dim
        # loss_mask = loss_mask & rearrange(batch['is_masked'], 'b t s -> b t s 1')
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        if Metric.bps in self.cfg.metrics:
            batch_out[Metric.bps] = self.bps(
                rates, spikes,
                length_mask=length_mask,
                channel_mask=channel_mask
            )

        return batch_out

class SpikeBase(SpikeContext, RatePrediction):
    modifies = [DataKey.spikes]

    def forward(
            self,
            batch,
            backbone_features: torch.Tensor,
            backbone_times: torch.Tensor,
            backbone_space: torch.Tensor,
            backbone_padding: torch.Tensor,
            compute_metrics=True,
            eval_mode=False
    ) -> torch.Tensor:
        assert compute_metrics, "No direct outputs supported, code inference separately"
        # ! We assume that backbone features arrives in a batch-major, time-minor format, that has already been flattened
        # We need to similarly flatten
        # Time-sorting respects original served DataKey.spikes order (this should be true, but we should check)
        target = batch[DataKey.spikes][..., 0]
        rates = self.out(backbone_features) # B x H
        loss = self.loss(rates, target.flatten(0, 1))
        comparison = repeat(torch.arange(loss.size(-1), device=loss.device), 'c -> t c', t=loss.size(0))
        # cf self.get_loss_mask
        loss_mask = ~backbone_padding.unsqueeze(-1) # B -> B x 1
        channel_mask = (comparison < batch[CHANNEL_KEY].flatten().unsqueeze(-1))
        loss_mask = loss_mask & channel_mask
        loss = loss[loss_mask].mean()
        return { 'loss': loss }


class ShuffleInfill(SpikeBase):
    r"""
        Technical design decision note:
        - JY instinctively decided to split up inputs and just carry around split tensors rather than the splitting metadata.
        - This is somewhat useful in the end (rather than the unshuffling solution) as we can simply collect the masked crop
        - However the code is pretty dirty and this may eventually change

    """

    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        assert not Metric.bps in self.cfg.metrics, 'not supported'
        assert self.serve_tokens and self.serve_tokens_flat, 'other paths not implemented'
        assert cfg.encode_decode, 'non-symmetric evaluation not implemented (since this task crops)'
        # ! Need to figure out how to wire different parameters e.g. num layers here
        self.decoder = SpaceTimeTransformer(
            cfg.transformer,
            max_spatial_tokens=data_attrs.max_spatial_tokens,
            n_layers=cfg.decoder_layers,
            debug_override_dropout_in=getattr(cfg.transformer, 'debug_override_dropout_io', False),
            context_integration=cfg.transformer.context_integration,
            embed_space=cfg.transformer.embed_space,
            allow_embed_padding=True,
        )
        self.max_spatial = data_attrs.max_spatial_tokens
        self.causal = cfg.causal
        # import pdb;pdb.set_trace()
        self.out = RatePrediction.create_linear_head(cfg, cfg.hidden_size, cfg.neurons_per_token)
        self.decode_cross_attn = getattr(cfg, 'spike_context_integration', 'in_context') == 'cross_attn'
        self.injector = TemporalTokenInjector(
            cfg,
            data_attrs,
            reference='spike_target',
        )

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        super().update_batch(batch, eval_mode=eval_mode)
        return self.crop_batch(self.cfg.mask_ratio, batch, eval_mode=eval_mode, shuffle=True)

    def crop_batch(self, mask_ratio: float, batch: Dict[str, torch.Tensor], eval_mode=False, shuffle=True):
        r"""
            Shuffle inputs, keep only what we need for evaluation
        """
        spikes = batch[DataKey.spikes]
        target = spikes[..., 0]
        if eval_mode:
            # manipulate keys so that we predict for all steps regardless of masking status (definitely hacky)
            batch.update({
                f'{self.handle}_target': target,
                f'{self.handle}_encoder_frac': spikes.size(1),
                # f'{DataKey.time}_target': batch[DataKey.time],
                # f'{DataKey.position}_target': batch[DataKey.position],
            })
            return batch
        # spikes: B T H (no array support)
        if shuffle:
            shuffle = torch.randperm(spikes.size(1), device=spikes.device) # T mask
        else:
            shuffle = torch.arange(spikes.size(1), device=spikes.device)
        if self.cfg.context_prompt_time_thresh:
            shuffle_func = apply_shuffle_2d
            nonprompt_time = (batch[DataKey.time] > self.cfg.context_prompt_time_thresh) # B x T mask
            shuffle = shuffle.unsqueeze(0).repeat(spikes.size(0), 1)
            nonprompt_time_shuffled = shuffle_func(nonprompt_time, shuffle).int() # bool not implemented for CUDA
            shuffle = sort_A_by_B(shuffle, nonprompt_time_shuffled) # B x T
        else:
            shuffle_func = apply_shuffle
        # Mask ratio becomes a comment on the remainder of the data
        encoder_frac = round((1 - mask_ratio) * spikes.size(1))
        # shuffle_spikes = spikes.gather(1, shuffle.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spikes.size(2), spikes.size(3)))
        for key in [DataKey.time, DataKey.position, DataKey.padding, CHANNEL_KEY]:
            if key in batch:
                shuffled = shuffle_func(batch[key], shuffle)
                batch.update({
                    key: shuffled[:, :encoder_frac],
                    f'{key}_target': shuffled[:, encoder_frac:],
                })
        # import pdb;pdb.set_trace()
        target = shuffle_func(target, shuffle)[:,encoder_frac:]
        batch.update({
            DataKey.spikes: shuffle_func(spikes, shuffle)[:,:encoder_frac],
            f'{self.handle}_target': target,
            # f'{self.handle}_encoder_frac': encoder_frac, # ! Deprecating
        })
        batch[f'{self.handle}_query'] = self.injector.make_query(target)
        return batch

    def get_loss_mask(self, batch: Dict[str, torch.Tensor], loss: torch.Tensor, padding_mask: torch.Tensor | None = None):
        # get_masks
        loss_mask = torch.ones(loss.size(), device=loss.device, dtype=torch.bool)
        # note LENGTH_KEY and CHANNEL_KEY are for padding tracking
        # while DataKey.time and DataKey.position are for content
        if padding_mask is not None:
            loss_mask = loss_mask & ~padding_mask.unsqueeze(-1)
        else:
            assert False, 'Deprecated encoder_frac dependent path'
            length_mask = ~create_token_padding_mask(None, batch, length_key=LENGTH_KEY, shuffle_key=f'{self.handle}_{SHUFFLE_KEY}')
            if LENGTH_KEY in batch:
                length_mask = length_mask[..., batch[f'{self.handle}_encoder_frac']:]
                loss_mask = loss_mask & length_mask.unsqueeze(-1)
        if CHANNEL_KEY in batch:
            # CHANNEL_KEY padding tracking has already been shuffled
            # And within each token, we just have c channels to track, always in order
            comparison = repeat(torch.arange(loss.size(-1), device=loss.device), 'c -> 1 t c', t=loss.size(1)) # ! assuming flat - otherwise we need the space dimension as well.
            channel_mask = comparison < batch[f'{CHANNEL_KEY}_target'].unsqueeze(-1) # unsqueeze the channel dimension
            loss_mask = loss_mask & channel_mask
        return loss_mask

    def forward(
            self,
            batch,
            backbone_features: torch.Tensor,
            backbone_times: torch.Tensor,
            backbone_space: torch.Tensor,
            backbone_padding: torch.Tensor,
            compute_metrics=True,
            eval_mode=False
        ) -> torch.Tensor:
        batch_out = {}
        target = batch[f'{self.handle}_target'] # B T H
        if not eval_mode:
            decode_tokens = batch[f'{self.handle}_query']
            decode_time = batch[f'{DataKey.time}_target']
            decode_space = batch[f'{DataKey.position}_target']
            decode_padding = batch[f'{DataKey.padding}_target']
        else:
            breakpoint() # JY is not sure of the flow here, TODO
            assert False, "Need to account for unified stream (use_full_encode)"
            decode_tokens = backbone_features
            decode_time = batch[DataKey.time]
            decode_space = batch[DataKey.position]
            decode_padding = None
            # token_padding_mask = create_token_padding_mask(
            #     None, batch,
            #     length_key=f'{LENGTH_KEY}', # Use the default length key that comes with dataloader
            #     shuffle_key=f'{self.handle}_{SHUFFLE_KEY}',
            # ) # Padding mask for full seq
        if self.decode_cross_attn:
            other_kwargs = {
                'memory': backbone_features,
                'memory_times': backbone_times,
                'memory_padding_mask': backbone_padding
            }
        else:
            decode_tokens = torch.cat([backbone_features, decode_tokens], dim=1)
            decode_time = torch.cat([backbone_times, decode_time], 1)
            decode_space = torch.cat([backbone_space, decode_space], 1)
            decode_padding = torch.cat([backbone_padding, decode_padding], 1)
            other_kwargs = {}

        decode_features: torch.Tensor = self.decoder(
            decode_tokens,
            padding_mask=decode_padding,
            times=decode_time,
            positions=decode_space,
            causal=self.causal,
            **other_kwargs,
        )

        if not self.decode_cross_attn:
            decode_features = decode_features[:, -(decode_tokens.size(1)-backbone_features.size(1)):]
        rates = self.out(decode_features)

        if Output.logrates in self.cfg.outputs:
            assert False, 'no chance this is still accurate'
            # out is B T C, we want B T' C, and then to unshuffle
            if eval_mode:
                # we're doing a full query for qualitative eval
                unshuffled = rates
            else:
                all_tokens = torch.cat([
                    torch.full(batch[DataKey.spikes].size()[:-1], float('-inf'), device=rates.device),
                    rates
                ], dim=1)
                unshuffled = apply_shuffle(all_tokens, batch[f'{self.handle}_{SHUFFLE_KEY}'].argsort())
            batch_out[Output.logrates] = unshuffled  # unflattening occurs outside
        if not compute_metrics:
            return batch_out
        loss: torch.Tensor = self.loss(rates, target) # b t' c
        loss_mask = self.get_loss_mask(batch, loss, padding_mask=decode_padding[:,-rates.size(1):]) # shuffle specific
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        return batch_out

class NextStepPrediction(RatePrediction):
    r"""
        One-step-ahead modeling prediction. Teacher-forced (we don't use force self-consistency, to save on computation)
        Revamped for NDT3, matching GATO.
    """
    modifies = []

    def __init__(self, backbone_out_size: int, channel_count: int, cfg: ModelConfig, data_attrs: DataAttrs, **kwargs):
        super().__init__(backbone_out_size, channel_count, cfg, data_attrs, **kwargs)
        self.start_token = nn.Parameter(torch.randn(cfg.hidden_size))
        self.separator_token = nn.Parameter(torch.randn(cfg.hidden_size)) # Delimits action modality, per GATO. # TODO ablate

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        spikes = batch[DataKey.spikes]
        target = spikes[..., 0]
        batch.update({
            DataKey.spikes: torch.cat([
                rearrange(self.start_token, 'h -> () () () h').expand(spikes.size(0), 1, spikes.size(2), -1),
                spikes.roll(1, dims=1)[:, 1:]
            ], 1),
            'spike_target': target,
        })

    def forward(
        self,
        batch,
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor,
        backbone_space: torch.Tensor,
        backbone_padding: torch.Tensor,
        compute_metrics=True,
        eval_mode=False
    ) -> torch.Tensor:
        assert False, "Deprecated"
        rates: torch.Tensor = self.out(backbone_features)
        batch_out = {}
        if Output.logrates in self.cfg.outputs:
            batch_out[Output.logrates] = rates

        if not compute_metrics:
            return batch_out
        loss: torch.Tensor = self.loss(rates, batch['spike_target'])
        loss_mask, length_mask, channel_mask = self.get_masks(batch)
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        if Metric.bps in self.cfg.metrics:
            batch_out[Metric.bps] = self.bps(
                rates, batch['spike_target'],
                length_mask=length_mask,
                channel_mask=channel_mask
            )

        return batch_out

class TemporalTokenInjector(nn.Module):
    r"""
        The in-place "extra" pathway assumes will inject `extra` series for someone else to process.
        It is assumed that the `extra` tokens will be updated elsewhere, and directly retrievable for decoding.
        - There is no code regulating this update, i'm only juggling two tasks at most atm.
        In held-out case, I'm routing update in `ShuffleInfill` update
    """
    def __init__(
        self, cfg: ModelConfig, data_attrs: DataAttrs, reference: DataKey, force_zero_mask=False
    ):
        super().__init__()
        self.reference = reference
        self.cfg = cfg.task
        if force_zero_mask:
            self.register_buffer('cls_token', torch.zeros(cfg.hidden_size))
        else:
            self.cls_token = nn.Parameter(torch.randn(cfg.hidden_size)) # This class token indicates bhvr, specific order of bhvr (in tokenized case) is indicated by space

        if self.cfg.decode_tokenize_dims:
            # this logic is for covariate decode, not heldout neurons
            assert reference != DataKey.spikes, "Decode tokenization should not run for spikes, this is for covariate exps"
        self.pad_value = data_attrs.pad_token
        self.max_space = data_attrs.max_spatial_tokens

    def make_query(self, reference: torch.Tensor):
        r"""
            Much simpler abstraction to just make a few tokens from a flat ref
        """
        b, t, *_ = reference.size() # reference should already be tokenized to desired res
        return repeat(self.cls_token, 'h -> b t h', b=b, t=t)

    def inject(self, batch: Dict[str, torch.Tensor], in_place=False, injected_time: torch.Tensor | None = None, injected_space: torch.Tensor | None = None):
        # create tokens for decoding with (inject them into seq or return them)
        # Assumption is that behavior time == spike time (i.e. if spike is packed, so is behavior), and there's no packing
        b, t, *_ = batch[self.reference].size() # reference should already be tokenized to desired res
        if injected_time is None:
            injected_time = torch.arange(t, device=batch[self.reference].device)
            injected_time = repeat(injected_time, 't -> b t', b=b)

        injected_tokens = repeat(self.cls_token, 'h -> b t h',
            b=b,
            t=t, # Time (not _token_, i.e. in spite of flat serving)
        )
        if injected_space is None:
            if batch[self.reference].ndim > 3:
                injected_space = torch.arange(self.max_space, device=batch[self.reference].device)
                injected_space = repeat(injected_space, 's -> b t s', b=b, t=t)
            else:
                injected_space = torch.zeros(
                    (b, t), device=batch[self.reference].device, dtype=torch.long
                )
        # I want to inject padding tokens for space so nothing actually gets added on that dimension
        if in_place:
            batch[DataKey.extra] = injected_tokens # B T H
            batch[DataKey.extra_time] = injected_time
            batch[DataKey.extra_position] = injected_space
        return injected_tokens, injected_time, injected_space

class ReturnContext(ContextPipeline):
    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.is_sparse = data_attrs.sparse_rewards
        self.max_return = cfg.max_return + 1 if data_attrs.pad_token is not None else cfg.max_return
        self.return_enc = nn.Embedding(
            self.max_return, # It will rarely be
            cfg.hidden_size,
            padding_idx=data_attrs.pad_token,
        )
        self.reward_enc = nn.Embedding(
            3 if data_attrs.pad_token is not None else 2, # 0 or 1, not a parameter for simple API convenience
            cfg.hidden_size,
            padding_idx=data_attrs.pad_token,
        )
        # self.norm = nn.LayerNorm(cfg.hidden_size)

    def get_context(self, batch: Dict[str, torch.Tensor]):
        # if batch[DataKey.task_return].numel() == 0:
        #     breakpoint()
        # if batch[DataKey.task_return].max() >= self.max_return - 1:
        #     print('Return max: ', batch[DataKey.task_return].max(dim=1), batch[MetaKey.session][batch[DataKey.task_return].argmax(dim=1)])
        #     batch[DataKey.task_return] = torch.clamp(batch[DataKey.task_return], max=self.max_return - 2) # Really got to understand what's happening here... guard against off by 1 errors.
        #     breakpoint()
        # * Don't understand why we're OOB-ing based on dataloader, it's one of these two. We need a data check, but scrape is taking a while.
        batch[DataKey.task_return] = batch[DataKey.task_return].clamp(min=0, max=self.max_return-1) # Really got to understand what's happening here... guard against off by 1 errors.
        batch[DataKey.task_return_time] = batch[DataKey.task_return_time].clamp(min=0)
        batch[DataKey.task_reward] = batch[DataKey.task_reward].clamp(min=0, max=2)
        # if batch[DataKey.task_return].min() < 0:
        #     print('Return min: ', batch[DataKey.task_return].min(dim=1), batch[MetaKey.session][batch[DataKey.task_return].argmin(dim=1)])
        #     # clamp # TODO bake down, we shouldn't need a posthoc fix like this. Understand the data
        #     breakpoint()
        # if batch[DataKey.task_reward].max() >= 3:
        #     print('Reward max: ', batch[DataKey.task_reward].max(dim=1), batch[MetaKey.session][batch[DataKey.task_reward].argmax(dim=1)])
        #     batch[DataKey.task_reward] = torch.clamp(batch[DataKey.task_reward], max=2)
        #     breakpoint()
        # if batch[DataKey.task_reward].min() < 0: # assumes pad token atm
        #     print('Reward min: ', batch[DataKey.task_reward].min(dim=1), batch[MetaKey.session][batch[DataKey.task_reward].argmin(dim=1)])
        #     # This should have already happened, but just in case.
        #     batch[DataKey.task_reward] = torch.clamp(batch[DataKey.task_reward], min=0)
        #     breakpoint()

        # print('Reward max: ', batch[DataKey.task_reward].max())
        # print(f'Time max: {batch[DataKey.task_return_time].max()} min: {batch[DataKey.task_return_time].min()}')
        return_embed = self.return_enc(batch[DataKey.task_return])
        reward_embed = self.reward_enc(batch[DataKey.task_reward])
        times = batch[DataKey.task_return_time]
        space = torch.zeros_like(times)
        padding = create_token_padding_mask(
            return_embed, batch, length_key=RETURN_LENGTH_KEY
        ) # Don't need a separate update step unless we need the retrieve padding at later time.
        return (
            return_embed + reward_embed,
            # self.norm(return_embed + reward_embed),
            times,
            space,
            padding
        )

class ReturnInfill(ReturnContext):
    def __init__(self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        # TODO - we need return targets that are appropriately in range in preproc. Don't use this task until we have sanitized return data.
        self.out = nn.Linear(backbone_out_size, self.max_return)
        # TODO - we should merge this pipeline with covariate into a generic classification pipeline.

    def forward(
        self,
        batch,
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor,
        backbone_space: torch.Tensor,
        backbone_padding: torch.Tensor,
        compute_metrics=True,
        eval_mode=False
    ) -> torch.Tensor:
        assert compute_metrics, "No direct outputs supported, code inference separately"
        target = batch[DataKey.task_return].flatten()
        pred = self.out(backbone_features)
        return {
            'loss': F.cross_entropy(pred, target, reduction='none', label_smoothing=self.cfg.decode_label_smooth)[~backbone_padding].mean()
        }


class CovariateReadout(DataPipeline, ConstraintPipeline):
    r"""
        Base class for decoding (regression/classification) of covariates.
        Constraints may be packed in here because the encoding is fused with behavior in the non-sparse case, but we may want to refactor that out.
    """
    modifies = [DataKey.bhvr_vel, DataKey.constraint, DataKey.constraint_time]

    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.served_tokenized_covariates = data_attrs.tokenize_covariates
        self.served_semantic_covariates = data_attrs.semantic_covariates
        if self.inject_constraint_tokens: # if they're injected, we don't need these params in kinematic
            if hasattr(self, 'constraint_cls'):
                del self.constraint_cls
            if hasattr(self, 'constraint_dims'):
                del self.constraint_dims
        assert self.cfg.decode_strategy == EmbedStrat.token, 'Non-token decoding deprecated'
        self.decode_cross_attn = cfg.decoder_context_integration == 'cross_attn'
        self.reference_cov = self.cfg.behavior_target
        self.injector = TemporalTokenInjector(
            cfg,
            data_attrs,
            None, # deprecating reference while trying ot clean up terminology # self.cfg.behavior_target if not self.encodes else f'{self.handle}_target',
            force_zero_mask=self.decode_cross_attn and not self.cfg.decode_tokenize_dims
        )
        self.cov_dims = data_attrs.behavior_dim
        self.covariate_blacklist_dims = torch.tensor(self.cfg.covariate_blacklist_dims)
        if self.cfg.decode_separate: # If we need additional cross-attention to decode. Not needed if mask tokens are procssed earlier.
            self.decoder = SpaceTimeTransformer(
                cfg.transformer,
                max_spatial_tokens=self.cov_dims if self.cfg.decode_tokenize_dims else 0,
                n_layers=cfg.decoder_layers,
                allow_embed_padding=True,
                context_integration=cfg.decoder_context_integration,
                embed_space=self.cfg.decode_tokenize_dims
            )

        self.causal = cfg.causal
        self.spacetime = cfg.transform_space
        assert self.spacetime, "Only spacetime transformer is supported for token decoding"
        self.bhvr_lag_bins = round(self.cfg.behavior_lag / data_attrs.bin_size_ms)
        assert self.bhvr_lag_bins >= 0, "behavior lag must be >= 0, code not thought through otherwise"
        assert not (self.bhvr_lag_bins and self.encodes), "behavior lag not supported with encoded covariates as encoding uses shuffle mask which breaks simple lag shift"

        self.session_blacklist = []
        if self.cfg.blacklist_session_supervision:
            ctxs: List[ContextInfo] = []
            try:
                for sess in self.cfg.blacklist_session_supervision:
                    sess = context_registry.query(alias=sess)
                    if isinstance(sess, list):
                        ctxs.extend(sess)
                    else:
                        ctxs.append(sess)
                for ctx in ctxs:
                    if ctx.id in data_attrs.context.session:
                        self.session_blacklist.append(data_attrs.context.session.index(ctx.id))
            except:
                print("Blacklist not successfully loaded, skipping blacklist logic (not a concern for inference)")

        if self.cfg.decode_normalizer:
            # See `data_kin_global_stat`
            zscore_path = Path(self.cfg.decode_normalizer)
            assert zscore_path.exists(), f'normalizer path {zscore_path} does not exist'
            self.register_buffer('bhvr_mean', torch.load(zscore_path)['mean'])
            self.register_buffer('bhvr_std', torch.load(zscore_path)['std'])
        else:
            self.bhvr_mean = None
            self.bhvr_std = None
        if self.encodes:
            self.initialize_readin(cfg.hidden_size)
        self.initialize_readout(cfg.hidden_size)

    @property
    def encodes(self):
        return self.cfg.covariate_mask_ratio < 1.0

    @abc.abstractmethod
    def initialize_readin(self):
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_readout(self):
        raise NotImplementedError

    @property
    def handle(self):
        return 'covariate'

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode = False):
        batch[f'{self.handle}_{DataKey.padding}'] = create_token_padding_mask(
            batch[self.cfg.behavior_target], batch,
            length_key=f'{self.handle}_{LENGTH_KEY}',
        )
        return self.crop_batch(self.cfg.covariate_mask_ratio, batch, eval_mode=eval_mode) # Remove encode

    def crop_batch(self, mask_ratio: float, batch: Dict[str, torch.Tensor], eval_mode=False, shuffle=True):
        covariates = batch[self.cfg.behavior_target] # B (T Cov_Dims) 1 if tokenized, else  B x T x Cov_Dims,
        # breakpoint()
        if DataKey.covariate_time not in batch:
            cov_time = torch.arange(covariates.size(1), device=covariates.device)
            if self.cfg.decode_tokenize_dims:
                cov_time = repeat(cov_time, 't -> b (t d)', b=covariates.size(0), d=self.cov_dims)
            else:
                cov_time = repeat(cov_time, 't -> b t', b=covariates.size(0))
            batch[DataKey.covariate_time] = cov_time
        if DataKey.covariate_space not in batch:
            if self.cfg.decode_tokenize_dims: # Here in is the implicit padding for space position, to fix.
                cov_space = repeat(
                    torch.arange(covariates.size(2), device=covariates.device),
                    'd -> b (t d)', b=covariates.size(0), t=covariates.size(1)
                )
            else:
                # Technically if data arrives as b t* 1, we can still use above if-case circuit
                cov_space = torch.zeros_like(batch[DataKey.covariate_time])
            batch[DataKey.covariate_space] = cov_space

        if not self.encodes: # Just make targets, exit
            batch[f'{DataKey.covariate_time}_target'] = batch[DataKey.covariate_time]
            batch[f'{DataKey.covariate_space}_target'] = batch[DataKey.covariate_space]
            batch[f'{self.handle}_{DataKey.padding}_target'] = batch[f'{self.handle}_{DataKey.padding}']
        else:
            if self.cfg.decode_tokenize_dims and not self.served_tokenized_covariates:
                covariates = rearrange(covariates, 'b t bhvr_dim -> b (t bhvr_dim) 1')
                if self.cfg.encode_constraints:
                    batch[DataKey.constraint] = rearrange(batch[DataKey.constraint], 'b t constraint bhvr_dim -> b (t bhvr_dim) constraint')
                batch[f'{self.handle}_{LENGTH_KEY}'] = batch[f'{self.handle}_{LENGTH_KEY}'] * self.cov_dims
            if eval_mode: # TODO FIX eval mode implementation # First note we aren't even breaking out so these values are overwritten
                breakpoint()
                batch.update({
                    f'{self.handle}_target': covariates,
                })
            if shuffle:
                shuffle = torch.randperm(covariates.size(1), device=covariates.device)
            else:
                shuffle = torch.arange(covariates.size(1), device=covariates.device)
            # breakpoint()
            if self.cfg.context_prompt_time_thresh:
                shuffle_func = apply_shuffle_2d
                nonprompt_time = (batch[DataKey.covariate_time] >= self.cfg.context_prompt_time_thresh) # B x T mask
                shuffle = repeat(shuffle, 't -> b t', b=covariates.size(0))
                nonprompt_time_shuffled = shuffle_func(nonprompt_time, shuffle).int() # bool not implemented for CUDA
                shuffle = sort_A_by_B(shuffle, nonprompt_time_shuffled) # B x T
            else:
                shuffle_func = apply_shuffle
            encoder_frac = round((1 - mask_ratio) * covariates.size(1))
            # TODO deprecate if we go multi-trial-streams, or next-step
            # If we have non-behavioral data (i.e. scrape is malformatted)
            # It'll just have one padding token.
            # Make sure that's the target, else we'll throw in the decoder for having a null query
            if covariates.size(1) == 1:
                encoder_frac = 0
            def shuffle_key(key):
                if key in batch:
                    shuffled = shuffle_func(batch[key], shuffle)
                    batch.update({
                        key: shuffled[:, :encoder_frac],
                        f'{key}_target': shuffled[:, encoder_frac:],
                    })
            for key in [
                DataKey.covariate_time,
                DataKey.covariate_space,
                f'{self.handle}_{DataKey.padding}',
            ]:
                shuffle_key(key)
            if self.cfg.encode_constraints and not self.inject_constraint_tokens:
                for key in [
                    DataKey.constraint,
                    DataKey.constraint_time
                ]:
                    shuffle_key(key)
            splits = [encoder_frac, covariates.size(1) - encoder_frac]
            enc, target = torch.split(shuffle_func(covariates, shuffle), splits, dim=1)
            # if target.size(1) == 0:
                # breakpoint() # Wat
            batch.update({
                self.cfg.behavior_target: enc,
                f'{self.handle}_target': target,
                f'{self.handle}_encoder_frac': encoder_frac,
            })
        batch[f'{self.handle}_query'] = self.injector.make_query(self.get_target(batch))
        return batch

    def get_context(self, batch: Dict[str, torch.Tensor]):
        if self.cfg.covariate_mask_ratio == 1.0:
            # return super().get_context(batch)
            return [], [], [], []
        enc = self.encode_cov(batch[self.cfg.behavior_target])
        if self.cfg.encode_constraints and not self.inject_constraint_tokens:
            constraint = self.encode_constraint(batch[DataKey.constraint]) # B T H Bhvr_Dim. Straight up not sure how to collapse non-losslessly - we just mean pool for now.
            enc = enc + constraint
        # if batch[DataKey.covariate_space].max() > 7:
            # print(f'Space range: [{batch[DataKey.covariate_space].min()}, {batch[DataKey.covariate_space].max()}]')
            # breakpoint()
        return (
            enc,
            batch[DataKey.covariate_time],
            batch[DataKey.covariate_space],
            batch[f'{self.handle}_{DataKey.padding}']
        )

    @abc.abstractmethod
    def encode_cov(self, covariate: torch.Tensor) -> torch.Tensor: # B T Bhvr_Dims or possibly B (T Bhvr_Dims)
        raise NotImplementedError

    def get_cov_pred(
        self,
        batch: Dict[str, torch.Tensor],
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor,
        backbone_space: torch.Tensor,
        backbone_padding: torch.Tensor,
        eval_mode=False,
        batch_out={}
    ) -> torch.Tensor:
        r"""
            returns: flat seq of predictions, B T' H' (H' is readout dim, regression) or B C T' (classification)
        """
        # breakpoint()
        if self.cfg.decode_separate:
            if self.cfg.decode_time_pool: # B T H -> B T H
                assert False, "Deprecated, currently would pool across modalities... but time is available if you still wanna try"
                backbone_features, backbone_padding = temporal_pool(batch, backbone_features, backbone_padding, pool=self.cfg.decode_time_pool)
                if Output.pooled_features in self.cfg.outputs:
                    batch_out[Output.pooled_features] = backbone_features.detach()

            decode_tokens = batch[f'{self.handle}_query']
            decode_time = batch[f'{DataKey.covariate_time}_target']
            decode_space = batch[f'{DataKey.covariate_space}_target']
            decode_padding = batch[f'{self.handle}_{DataKey.padding}_target']
            if not self.inject_constraint_tokens and self.cfg.encode_constraints:
                decode_tokens = decode_tokens + self.encode_constraint(
                    batch[f'{DataKey.constraint}_target'],
                )

            # Re-extract src time and space. Only time is always needed to dictate attention for causality, but current implementation will re-embed time. JY doesn't want to asymmetrically re-embed only time, so space is retrieved. Really, we need larger refactor to just pass in time/space embeddings externally.
            if self.cfg.decode_time_pool:
                assert False, "Deprecated"
                assert not self.encodes, "not implemented"
                assert not self.cfg.decode_tokenize_dims, 'time pool not implemented with tokenized dims'
                src_time = decode_time
                src_space = torch.zeros_like(decode_space)
                if backbone_features.size(1) < src_time.size(1):
                    # We want to pool, but encoding doesn't necessarily have all timesteps. Pad to match
                    backbone_features = F.pad(backbone_features, (0, 0, 0, src_time.size(1) - backbone_features.size(1)), value=0)
                    backbone_padding = F.pad(backbone_padding, (0, src_time.size(1) - backbone_padding.size(1)), value=True)

            # allow looking N-bins of neural data into the "future"; we back-shift during the actual loss comparison.
            if self.causal and self.cfg.behavior_lag_lookahead:
                decode_time = decode_time + self.bhvr_lag_bins

            if self.decode_cross_attn:
                other_kwargs = {
                    'memory': backbone_features,
                    'memory_times': backbone_times,
                    'memory_padding_mask': backbone_padding,
                }
            else:
                assert not self.cfg.decode_tokenize_dims, 'non-cross attn not implemented with tokenized dims'
                if backbone_padding is not None:
                    decode_padding = torch.cat([backbone_padding, decode_padding], dim=1)
                logger.warning('This is untested code where we flipped order of padding declarations. Previously extra padding was declared after we concatenated backbone, but this did not make sense')
                decode_tokens = torch.cat([backbone_features, decode_tokens], dim=1)
                decode_time = torch.cat([backbone_times, decode_time], dim=1)
                decode_space = torch.cat([backbone_space, decode_space], dim=1)
                other_kwargs = {}
            # print('Src stream: ', decode_tokens.shape, decode_padding.shape, decode_time.shape, decode_space.shape)
            # print('Cross stream:', other_kwargs['memory'].shape, other_kwargs['memory_padding_mask'].shape, other_kwargs['memory_times'].shape)
            # if decode_tokens.size(1) == 0:
                # breakpoint() # Wat is this data...
            # print(decode_time.max(), backbone_space.max(), decode_space.max())
            # if decode_time.max() > 1500: # debug
            # if decode_time.max() > self.cfg.task.max_trial_length:
                # raise ValueError(f'Decode Trial length {decode_time.max()} exceeds max trial length {self.cfg.max_trial_length}')
            backbone_features: torch.Tensor = self.decoder(
                decode_tokens,
                padding_mask=decode_padding,
                times=decode_time,
                positions=decode_space,
                causal=self.causal,
                **other_kwargs
            )
        # crop out injected tokens, -> B T H
        if not self.decode_cross_attn:
            backbone_features = batch[:, -decode_tokens.size(1):]
        return self.out(backbone_features)

    def get_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.cfg.covariate_mask_ratio == 1.0:
            tgt = batch[self.cfg.behavior_target]
        else:
            tgt = batch[f'{self.handle}_target']
        if self.bhvr_mean is not None:
            tgt = tgt - self.bhvr_mean
            tgt = tgt / self.bhvr_std
        return tgt

    def simplify_logits_to_prediction(self, bhvr: torch.Tensor):
        # no op for regression, argmax + dequantize for classification
        return bhvr

    @abc.abstractmethod
    def compute_loss(self, bhvr, bhvr_tgt):
        pass

    def forward(
        self,
        batch,
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor,
        backbone_space: torch.Tensor,
        backbone_padding: torch.Tensor,
        compute_metrics=True,
        eval_mode=False
    ) -> torch.Tensor:
        batch_out = {}
        # breakpoint()
        bhvr = self.get_cov_pred(
            batch,
            backbone_features,
            backbone_times,
            backbone_space,
            backbone_padding,
            eval_mode=eval_mode,
            batch_out=batch_out
        ) # * flat (B T D)
        # bhvr is still shuffled and tokenized..

        # At this point (computation and beyond) it is easiest to just restack tokenized targets, merge into regular API
        # The whole point of all this intervention is to test whether separate tokens affects perf (we hope not)
        if self.bhvr_lag_bins:
            bhvr = bhvr[..., :-self.bhvr_lag_bins, :]
            bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)

        # * Doesn't unshuffle or do any formatting
        if Output.behavior_pred in self.cfg.outputs: # Note we need to eventually implement some kind of repack, just like we do for spikes
            batch_out[f'{DataKey.covariate_space}_target'] = batch[f'{DataKey.covariate_space}_target']
            batch_out[f'{DataKey.covariate_time}_target'] = batch[f'{DataKey.covariate_time}_target']
            batch_out[f'{self.handle}_{DataKey.padding}_target'] = batch[f'{self.handle}_{DataKey.padding}_target']
            if DataKey.covariate_labels in batch:
                batch_out[DataKey.covariate_labels] = batch[DataKey.covariate_labels]
            batch_out[Output.behavior_pred] = self.simplify_logits_to_prediction(bhvr)
            if self.bhvr_mean is not None:
                batch_out[Output.behavior_pred] = batch_out[Output.behavior_pred] * self.bhvr_std + self.bhvr_mean
        if Output.behavior in self.cfg.outputs:
            batch_out[Output.behavior] = self.get_target(batch)
        if not compute_metrics:
            return batch_out

        # Compute loss
        # breakpoint()
        bhvr_tgt = self.get_target(batch)

        _, length_mask, _ = self.get_masks(
            batch, ref=bhvr_tgt,
            length_key=f'{self.handle}_{LENGTH_KEY}',
            shuffle_key=None,
            compute_channel=False,
            padding_mask=batch.get(f'{self.handle}_{DataKey.padding}_target', None),
        )
        length_mask[:, :self.bhvr_lag_bins] = False # don't compute loss for lagged out timesteps
        loss = self.compute_loss(bhvr, bhvr_tgt)
        if self.cfg.behavior_fit_thresh:
            loss_mask = length_mask & (bhvr_tgt.abs() > self.cfg.behavior_fit_thresh).any(-1)
        else:
            loss_mask = length_mask

        # blacklist
        if self.session_blacklist:
            session_mask = batch[MetaKey.session] != self.session_blacklist[0]
            for sess in self.session_blacklist[1:]:
                session_mask = session_mask & (batch[MetaKey.session] != sess)
            loss_mask = loss_mask & session_mask[:, None]
            if not session_mask.any(): # no valid sessions
                loss = torch.zeros_like(loss).mean() # don't fail
            else:
                loss = loss[loss_mask].mean()
        else:
            if len(loss[loss_mask]) == 0:
                loss = torch.zeros_like(loss).mean()
            else:
                if loss[loss_mask].mean().isnan().any():
                    breakpoint()
                if len(self.covariate_blacklist_dims) > 0:
                    # breakpoint()
                    if self.cfg.decode_tokenize_dims:
                        positions = batch[f'{DataKey.covariate_space}_target']
                        loss_mask = loss_mask & ~torch.isin(positions, self.covariate_blacklist_dims.to(device=positions.device))
                    else:
                        loss_mask = loss_mask.unsqueeze(-1).repeat(1, 1, loss.size(-1))
                        loss_mask[..., self.covariate_blacklist_dims] = False
                if not loss_mask.any():
                    logger.warning('No dims survive loss mask, kinematic loss is zero')
                    breakpoint()
                    loss = torch.zeros_like(loss).mean()
                else:
                    loss = loss[loss_mask].mean()
        r2_mask = length_mask

        batch_out['loss'] = loss
        if Metric.kinematic_r2 in self.cfg.metrics:
            valid_bhvr = bhvr[..., :bhvr_tgt.shape[-1]]
            if len(self.covariate_blacklist_dims) > 0:
                assert self.cfg.decode_tokenize_dims, "blacklist dims not implemented for non tokenized R2"
                positions: torch.Tensor = batch[f'{DataKey.covariate_space}_target']
                r2_mask = r2_mask & ~torch.isin(positions, self.covariate_blacklist_dims.to(device=positions.device))
            # breakpoint()
            valid_bhvr = self.simplify_logits_to_prediction(valid_bhvr)[r2_mask].float().detach().cpu()
            valid_tgt = bhvr_tgt[r2_mask].float().detach().cpu()
            if self.served_tokenized_covariates and not self.served_semantic_covariates: # If semantic, we don't need to reorganize
                assert len(self.covariate_blacklist_dims) == 0, "blacklist dims not implemented for non semantic R2"
                # Compute the unique covariate labels, and their repsective position indices.
                # Then pull R2 accordingly. Lord knows this isn't the most efficient, but...
                dims_per = torch.tensor([len(i) for i in batch[DataKey.covariate_labels]], device=batch[f'{DataKey.covariate_space}_target'].device).cumsum(0)
                batch_shifted_positions = batch[f'{DataKey.covariate_space}_target'] + (dims_per - dims_per[0]).unsqueeze(-1)
                flat_labels = np.array(list(itertools.chain.from_iterable(batch[DataKey.covariate_labels])))
                unique_labels, label_indices = np.unique(flat_labels, return_inverse=True)
                range_reference = np.arange(len(flat_labels))
                r2_scores = []
                batch_shifted_positions = batch_shifted_positions[r2_mask].flatten().cpu()
                for i, l in enumerate(unique_labels):
                    unique_indices = torch.as_tensor(range_reference[label_indices == i])
                    submask = torch.isin(batch_shifted_positions, unique_indices)
                    if not submask.any(): # Unlucky, shouldn't occur if we predict more.
                        # breakpoint()
                        r2_scores.append(0)
                        continue
                    r2_scores.append(r2_score(valid_tgt[submask], valid_bhvr[submask]))
                batch_out[Metric.kinematic_r2] = np.array(r2_scores)
                batch[DataKey.covariate_labels] = unique_labels
            elif self.cfg.decode_tokenize_dims:
                # extract the proper subsets according to space (for loop it) - per-dimension R2 is only relevant while dataloading maintains consistent dims (i.e. not for long) but in the meanwhile
                r2_scores = []
                positions = batch[f'{DataKey.covariate_space}_target'][r2_mask].flatten().cpu() # flatten as square full batches won't autoflatten B x T but does flatten B x T x 1
                for i in positions.unique():
                    r2_scores.append(r2_score(valid_tgt[positions == i], valid_bhvr[positions == i]))
                batch_out[Metric.kinematic_r2] = np.array(r2_scores)
            else:
                assert len(self.covariate_blacklist_dims) == 0, "blacklist dims not implemented for non tokenized R2"
                batch_out[Metric.kinematic_r2] = r2_score(valid_tgt, valid_bhvr, multioutput='raw_values')
            if batch_out[Metric.kinematic_r2].mean() < -100:
                batch_out[Metric.kinematic_r2] = np.zeros_like(batch_out[Metric.kinematic_r2])# .mean() # mute, some erratic result from near zero target skewing plots
                # print(valid_bhvr.mean().cpu().item(), valid_tgt.mean().cpu().item(), batch_out[Metric.kinematic_r2].mean())
                # breakpoint()
            # if Metric.kinematic_r2_thresh in self.cfg.metrics: # Deprecated, note to do this we'll need to recrop `position_target` as well
            #     valid_bhvr = valid_bhvr[valid_tgt.abs() > self.cfg.behavior_metric_thresh]
            #     valid_tgt = valid_tgt[valid_tgt.abs() > self.cfg.behavior_metric_thresh]
            #     batch_out[Metric.kinematic_r2_thresh] = r2_score(valid_tgt, valid_bhvr, multioutput='raw_values')
        if Metric.kinematic_acc in self.cfg.metrics:
            acc = (bhvr.argmax(1) == self.quantize(bhvr_tgt))
            batch_out[Metric.kinematic_acc] = acc[r2_mask].float().mean()
        return batch_out


class BehaviorRegression(CovariateReadout):
    r"""
        Because this is not intended to be a joint task, and backbone is expected to be tuned
        We will not make decoder fancy.
    """
    def initialize_readin(self, backbone_size):
        if self.cfg.decode_tokenize_dims: # NDT3 style
            self.inp = nn.Linear(1, backbone_size)
        else: # NDT2 style
            self.inp = nn.Linear(self.cov_dims, backbone_size)
        # No norm, that would wash out the linear

    def encode_cov(self, covariate: torch.Tensor):
        return self.inp(covariate)

    def initialize_readout(self, backbone_size):
        if self.cfg.decode_tokenize_dims:
            self.out = nn.Linear(backbone_size, 1)
        else:
            self.out = nn.Linear(backbone_size, self.cov_dims)

    def compute_loss(self, bhvr, bhvr_tgt):
        comp_bhvr = bhvr[...,:bhvr_tgt.shape[-1]]
        if self.cfg.behavior_tolerance > 0:
            # Calculate mse with a tolerance floor
            loss = torch.clamp((comp_bhvr - bhvr_tgt).abs(), min=self.cfg.behavior_tolerance) - self.cfg.behavior_tolerance
            # loss = torch.where(loss.abs() < self.cfg.behavior_tolerance, torch.zeros_like(loss), loss)
            if self.cfg.behavior_tolerance_ceil > 0:
                loss = torch.clamp(loss, -self.cfg.behavior_tolerance_ceil, self.cfg.behavior_tolerance_ceil)
            loss = loss.pow(2)
        else:
            loss = F.mse_loss(comp_bhvr, bhvr_tgt, reduction='none')
        return loss

def symlog(x: torch.Tensor):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def unsymlog(x: torch.Tensor):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class QuantizeBehavior(TaskPipeline): # Mixin
    QUANTIZE_CLASSES = 128 # coarse classes are insufficient, starting relatively fine grained (assuming large pretraining)
    # TODO bake above into config

    def __init__(self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        quantize_bound = 1.001 if not self.cfg.decode_symlog else symlog(torch.tensor(1.001))
        self.is_next_step = getattr(cfg, 'next_step_prediction', False)
        self.cov_dims = data_attrs.behavior_dim
        self.register_buffer('zscore_quantize_buckets', torch.linspace(-quantize_bound, quantize_bound, self.QUANTIZE_CLASSES + 1)) # This will produce values from 1 - self.quantize_classes, as we rule out OOB. Asymmetric as bucketize is asymmetric; on bound value is legal for left, quite safe for expected z-score range. +1 as these are boundaries, not centers

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(x != self.pad_value, x, 0) # actually redundant if padding is sensibly set to 0, but sometimes it's not
        if getattr(self.cfg, 'decode_symlog', False):
            return torch.bucketize(symlog(x), self.zscore_quantize_buckets)
        return torch.bucketize(x, self.zscore_quantize_buckets) - 1 # bucketize produces from [1, self.quantize_classes]

    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        if quantized.max() > self.zscore_quantize_buckets.shape[0]:
            raise Exception("go implement quantization clipping man")
        if getattr(self.cfg, 'decode_symlog', False):
            return unsymlog((self.zscore_quantize_buckets[quantized] + self.zscore_quantize_buckets[quantized + 1]) / 2)
        return (self.zscore_quantize_buckets[quantized] + self.zscore_quantize_buckets[quantized + 1]) / 2


class BehaviorContext(ContextPipeline, QuantizeBehavior):
    # For feeding autoregressive task
    # Simple quantizing tokenizer
    # * Actually just a reference, not actually used... this is because data must arrive embedded for the main model.
    # So either this is subsumed as
    @property
    def handle(self):
        return 'covariate'

    def get_context(self, batch: Dict[str, torch.Tensor]):
        batch[f'{self.handle}_{DataKey.padding}'] = create_token_padding_mask(
            batch[self.cfg.behavior_target], batch,
            length_key=f'{self.handle}_{LENGTH_KEY}',
        )
        breakpoint() # TODO check dims, we may not need the mean call
        return (
            self.quantize(batch[self.cfg.behavior_target]).mean(-2), # B T 1 out
            batch[DataKey.covariate_time],
            batch[DataKey.covariate_space],
            batch[f'{self.handle}_{DataKey.padding}']
        )

class ClassificationMixin(QuantizeBehavior):
    def initialize_readin(self, backbone_size): # Assume quantized readin...
        self.inp = nn.Embedding(self.QUANTIZE_CLASSES + 1, backbone_size, padding_idx=0)
        # self.inp_norm = nn.LayerNorm(backbone_size)

    def initialize_readout(self, backbone_size):
        # We use these buckets as we minmax clamp in preprocessing
        if self.cfg.decode_tokenize_dims:
            if self.is_next_step:
                self.out = nn.Linear(backbone_size, self.QUANTIZE_CLASSES)
            else:
                self.out = nn.Sequential(
                    nn.Linear(backbone_size, self.QUANTIZE_CLASSES),
                    Rearrange('b t c -> b c t')
                )
        else:
            assert not self.is_next_step, "next step not implemented for non-tokenized"
            self.out = nn.Sequential(
                nn.Linear(backbone_size, self.QUANTIZE_CLASSES * self.cov_dims),
                Rearrange('b t (c d) -> b c (t d)', c=self.QUANTIZE_CLASSES)
            )

    def encode_cov(self, covariate: torch.Tensor):
        # Note: covariate is _not_ foreseeably quantized at this point, we quantize herein during embed.
        # print(covariate.min(), covariate.max())
        # breakpoint()
        covariate = self.inp(self.quantize(covariate)) # B T Bhvr_Dims -> B T Bhvr_Dims H.
        covariate = covariate.mean(-2) # B T Bhvr_Dims H -> B T H # (Even if Bhvr_dim = 1, which is true in tokenized serving)
        # covariate = self.inp_norm(covariate)
        return covariate

    def simplify_logits_to_prediction(self, bhvr: torch.Tensor, logit_dim=1):
        return self.dequantize(bhvr.argmax(logit_dim))

    def compute_loss(self, bhvr: torch.Tensor, bhvr_tgt: torch.Tensor):
        # breakpoint()
        # print(bhvr.shape, self.quantize(bhvr_tgt).shape, self.quantize(bhvr_tgt).min(), self.quantize(bhvr_tgt).max())
        return F.cross_entropy(bhvr, self.quantize(bhvr_tgt), reduction='none', label_smoothing=self.cfg.decode_label_smooth)


class BehaviorClassification(CovariateReadout, ClassificationMixin):
    r"""
        In preparation for NDT3.
        Assumes cross-attn, spacetime path.
        Cross-attention, autoregressive classification.
    """

    def get_cov_pred(
        self, *args, **kwargs
    ) -> torch.Tensor:
        bhvr = super().get_cov_pred(*args, **kwargs)
        if not self.cfg.decode_tokenize_dims:
            bhvr = rearrange(bhvr, 'b t (c d) -> b c t d', c=self.QUANTIZE_CLASSES)
        elif self.served_tokenized_covariates:
            bhvr = rearrange(bhvr, 'b c t -> b c t 1')
        else:
            bhvr = rearrange(bhvr, 'b (t d) c -> b c t d', d=self.cov_dims)

        return bhvr

class CovariateInfill(ClassificationMixin):
    # CovariateReadout is quite overloaded; we create a simpler next step prediction covariate readout module
    # Uses classification path...
    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.served_tokenized_covariates = data_attrs.tokenize_covariates
        self.served_semantic_covariates = data_attrs.semantic_covariates
        self.reference_cov = self.cfg.behavior_target
        self.cov_dims = data_attrs.behavior_dim
        self.initialize_readin(cfg.hidden_size)
        self.initialize_readout(cfg.hidden_size)

    @property
    def handle(self):
        return 'covariate'

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode = False):
        batch[f'{self.handle}_{DataKey.padding}'] = create_token_padding_mask(
            batch[self.cfg.behavior_target], batch,
            length_key=f'{self.handle}_{LENGTH_KEY}',
        )
        return batch

    def get_context(self, batch: Dict[str, torch.Tensor]):
        enc = self.encode_cov(batch[self.cfg.behavior_target])
        return (
            enc,
            batch[DataKey.covariate_time],
            batch[DataKey.covariate_space],
            batch[f'{self.handle}_{DataKey.padding}']
        )

    def forward(
        self,
        batch,
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor,
        backbone_space: torch.Tensor,
        backbone_padding: torch.Tensor,
        compute_metrics=True,
        eval_mode=False,
    ) -> Dict[BatchKey, torch.Tensor]:
        batch_out = {}
        bhvr: torch.Tensor = self.out(backbone_features)
        if Output.behavior_pred in self.cfg.outputs: # Note we need to eventually implement some kind of repack, just like we do for spikes
            batch_out[Output.behavior_pred] = bhvr # returns logits
        bhvr_tgt = batch[self.cfg.behavior_target].flatten()
        if Output.behavior in self.cfg.outputs:
            batch_out[Output.behavior] = bhvr_tgt # Flat aspect is not ideal, watch the timestamps..
        if not compute_metrics:
            return batch_out

        # Compute loss
        loss = self.compute_loss(bhvr, bhvr_tgt)
        loss = loss[~backbone_padding].mean()
        batch_out['loss'] = loss

        r2_mask = ~backbone_padding

        if Metric.kinematic_r2 in self.cfg.metrics:
            valid_bhvr = bhvr
            # breakpoint()
            valid_bhvr = self.simplify_logits_to_prediction(valid_bhvr)[r2_mask].float().detach().cpu()
            valid_tgt = bhvr_tgt[r2_mask].float().detach().cpu()
            batch_out[Metric.kinematic_r2] = np.array([r2_score(valid_tgt, valid_bhvr)])
            # breakpoint() # Something is wildly wrong...
            if batch_out[Metric.kinematic_r2].mean() < -10000:
                # zero it out - this is a bug that occurs when the target has minimal variance (i.e. a dull batch with tiny batch size)
                # Occurs only because we can't easily full batch R2, i.e. uninteresting.
                batch_out[Metric.kinematic_r2] = np.zeros_like(batch_out[Metric.kinematic_r2])
            batch[DataKey.covariate_labels] = ['x'] # base default
        if Metric.kinematic_acc in self.cfg.metrics:
            acc = (bhvr.argmax(1) == self.quantize(bhvr_tgt))
            batch_out[Metric.kinematic_acc] = acc[r2_mask].float().mean()
        # print(batch_out[Metric.kinematic_r2])
        return batch_out

# === Utils ===

def create_token_padding_mask(
    reference: torch.Tensor | None,
    batch: Dict[str, torch.Tensor],
    length_key: str = LENGTH_KEY,
    shuffle_key: str = '',
    multiplicity: int = 1, # if reference has extra time dimensions flattened
) -> torch.Tensor:
    r"""
        Identify which features are padding or not.
        True if padding
        reference: tokens that are enumerated and must be determined whether is padding or not. Only not needed if positions are already specified in shuffle case.

        out: b x t
    """
    if shuffle_key != '':
        assert False, "Deprecated"
    if length_key not in batch: # No plausible padding, everything is square
        return torch.zeros(reference.size()[:2], device=reference.device, dtype=torch.bool)
    if shuffle_key in batch:
        # TODO deprecate this functionality, it shouldn't be relevant anymore
        # shuffle_key presence indicates reference tokens have been shuffled, and batch[shuffle_key] indicates true position. Truncation is done _outside_ of this function.
        token_position = batch[shuffle_key]
    else:
        # assumes not shuffled
        token_position = repeat(torch.arange(reference.size(1) // multiplicity, device=reference.device), 't -> (t m)', m=multiplicity)
    token_position = rearrange(token_position, 't -> () t')
    return token_position >= rearrange(batch[length_key], 'b -> b ()')

task_modules = {
    ModelTask.infill: SelfSupervisedInfill,
    ModelTask.shuffle_infill: ShuffleInfill,
    ModelTask.spike_context: SpikeContext,
    ModelTask.spike_infill: SpikeBase,
    ModelTask.next_step_prediction: NextStepPrediction,
    ModelTask.shuffle_next_step_prediction: ShuffleInfill, # yeahhhhh it's the SAME TASK WTH
    # ModelTask.shuffle_next_step_prediction: ShuffleNextStepPrediction,
    ModelTask.kinematic_decoding: BehaviorRegression,
    ModelTask.kinematic_classification: BehaviorClassification,
    ModelTask.kinematic_context: BehaviorContext, # Use classification route, mainly for tokenizing
    ModelTask.kinematic_infill: CovariateInfill,
    ModelTask.constraints: ConstraintPipeline,
    ModelTask.return_context: ReturnContext,
    ModelTask.return_infill: ReturnInfill,
}