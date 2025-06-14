from typing import Tuple, Dict, List, Optional, Any, Self, Callable
from pathlib import Path
import logging
import time
import math
import abc
import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
from einops import rearrange, repeat, reduce, einsum, pack, unpack # baby steps...
from einops.layers.torch import Rearrange
from sklearn.metrics import r2_score
from torchmetrics.text import EditDistance

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

def temporal_pool(batch: Dict[BatchKey, torch.Tensor], backbone_features: torch.Tensor, temporal_padding_mask: torch.Tensor | None, pool='mean', override_time=0, pad_timestep=0):
    r"""
        # Originally developed for behavior regression, extracted for heldoutprediction
        # Assumption is that bhvr is square! (? JY not sure what this means in retrospect)
        pad_timestep: If time is already padded, denote the timestep assigned to padding, enables creation of a smaller pool
            # TODO implement
    """
    time_key = DataKey.time.name
    if 'update_time' in batch:
        time_key = 'update_time'
    if pad_timestep:
        # pool_tgt_steps = batch[time_key].max() + 1
        pool_tgt_steps = batch[time_key][batch[time_key] != pad_timestep].max() + 1 # TODO enable
    else:
        pool_tgt_steps = batch[time_key].max() + 1
    pooled_features = torch.zeros(
        backbone_features.shape[0],
        (override_time if override_time else pool_tgt_steps) + 1, # 1 extra to send padding features to (even if there's an explicit padding timestep)
        backbone_features.shape[-1],
        device=backbone_features.device,
        dtype=backbone_features.dtype
    )
    if temporal_padding_mask is None:
        temporal_padding_mask = torch.zeros_like(batch[time_key])
    time_with_pad_marked = torch.where(
        temporal_padding_mask,
        pool_tgt_steps,
        batch[time_key]
    )
    pooled_features = pooled_features.scatter_reduce(
        src=backbone_features,
        dim=1,
        index=repeat(time_with_pad_marked, 'b t -> b t h', h=backbone_features.shape[-1]),
        reduce=pool, # * Note, pool assumes timesteps are properly marked in padding
        include_self=False
    )
    # print(torch.allclose(pooled_features[:,0,:] - backbone_features[:,:7,:].mean(1), torch.tensor(0, dtype=torch.float), atol=1e-6))
    pool_mask = torch.ones(pooled_features.size(0), pooled_features.size(1), dtype=torch.float, device=backbone_features.device)
    # Output is padding iff all contributing timepoints are padding
    pool_mask = pool_mask.scatter_reduce(
        src=temporal_padding_mask.float(),
        dim=1,
        index=time_with_pad_marked,
        reduce='prod',
        include_self=False
    ).bool()
    pooled_features = pooled_features[:,:-1] # remove padding
    pool_mask = pool_mask[:,:-1] # remove padding
    return pooled_features, pool_mask

def temporal_pool_direct(
    backbone_features: torch.Tensor,
    backbone_times: torch.Tensor,
    backbone_space: torch.Tensor,
    backbone_padding: torch.Tensor | None = None,
    pool='mean',
    pad_timestep=1500,
):
    r"""
        Reimplementation of temporal pooling ot work directly on features, times, and space, as opposed to from batch.
        Note slight implementation differences.
        Padding logic not inspected yet.
    """
    if pad_timestep:
        pool_tgt_steps = backbone_times[backbone_times != pad_timestep].max() + 1
    else:
        pool_tgt_steps = backbone_times.max() + 1
    pooled_features = torch.zeros(
        backbone_features.shape[0],
        pool_tgt_steps + 1,
        backbone_features.shape[-1],
        device=backbone_features.device,
        dtype=backbone_features.dtype
    )
    if backbone_padding is None:
        backbone_padding = torch.zeros_like(backbone_times)
    time_with_pad_marked = torch.where(
        backbone_padding,
        pool_tgt_steps,
        backbone_times
    )
    pooled_features = pooled_features.scatter_reduce(
        src=backbone_features,
        dim=1,
        index=repeat(time_with_pad_marked, 'b t -> b t h', h=backbone_features.shape[-1]),
        reduce=pool, # * Note, pool assumes timesteps are properly marked in padding
        include_self=False
    )
    pool_mask = torch.ones(pooled_features.size(0), pooled_features.size(1), dtype=torch.float, device=backbone_features.device)
    # Output is padding iff all contributing timepoints are padding
    pool_mask = pool_mask.scatter_reduce(
        src=backbone_padding.float(),
        dim=1,
        index=time_with_pad_marked,
        reduce='prod',
        include_self=False
    ).bool()
    pooled_features = pooled_features[:,:-1] # remove padding
    pool_mask = pool_mask[:,:-1] # remove padding
    return pooled_features, pool_mask

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
        else:
            breakpoint()
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
    modifies: List[str] = [] # Which DataKeys are altered in use of this Pipeline? (We check to prevent multiple subscriptions)
    # modifies: List[DataKey] = [] # Which DataKeys are altered in use of this Pipeline? (We check to prevent multiple subscriptions)

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

    def get_context(
        self,
        batch: Dict[BatchKey, torch.Tensor],
        eval_mode=False
    ) -> Tuple[torch.Tensor | List, torch.Tensor | List, torch.Tensor | List, torch.Tensor | List]:
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

    def get_conditioning_context(self, batch: Dict[BatchKey, torch.Tensor]) -> Tuple[torch.Tensor | List, torch.Tensor | List, torch.Tensor | List, torch.Tensor | List]:
        r"""
            For task specific trial _input_. (B T H)
            Same return as above.
        """
        raise NotImplementedError # TODO still not consumed in main model
        return None, None, None, None

    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        r"""
            Currently redundant with get_context - need to refactor.
            It could be that this forces a one-time modification.
            Update batch in place for modifying/injecting batch info.
        """
        return batch

    def get_trial_query(self, batch: Dict[BatchKey, torch.Tensor]):
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
            loss_mask: torch.Tensor | None = None,
            compute_metrics=True,
            eval_mode=False,
            phase: str = "train"
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
    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
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
        # TODO revisit inject constraint tokens.. what's this supposed to be
        assert self.cfg.encode_constraints, 'constraint pipeline only for encoding tokenized constraints'
        self.sparse = data_attrs.sparse_constraints
        # self.inject_constraint_tokens = data_attrs.sparse_constraints and self.cfg.encode_constraints # Injects as dimvarying context
        if self.cfg.encode_constraints:
            if self.cfg.use_constraint_cls:
                # Not obvious we actually need yet _another_ identifying cls if we're also encoding others, but can we afford a zero token if no constraints are active...?
                self.constraint_cls = nn.Parameter(torch.randn(cfg.hidden_size))
            self.constraint_dims = nn.Parameter(torch.randn(3, cfg.hidden_size))
        # self.norm = nn.LayerNorm(cfg.hidden_size) # * Actually no, don't norm the linear projection...

    def encode_constraint(self, constraint: torch.Tensor) -> torch.Tensor:
        # constraint: Out is B T H Bhvr_Dim for sparse/tokenized, or B T H if dense
        if not self.cfg.decode_tokenize_dims:
            # In the dense decode_tokenize_dims path, tokens already arrive rearranged due to crop batch. Flattening occurs outside this func
            constraint_embed = einsum(constraint, self.constraint_dims, 'b t constraint d, constraint h -> b t h d')
            if self.cfg.use_constraint_cls:
                constraint_embed = constraint_embed + rearrange(self.constraint_cls, 'h -> 1 1 h 1')
            if not self.sparse: # reduce (pretty crude - we can't tell which dim is constrained like thus)
                constraint_embed = constraint_embed.mean(-1) # B T H
        else:
            constraint_embed = einsum(constraint, self.constraint_dims, 'b t constraint, constraint h -> b t h')
            if self.sparse and self.cfg.use_constraint_cls:
                constraint_embed = constraint_embed + rearrange(self.constraint_cls, 'h -> 1 1 h')
        return constraint_embed

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        constraint = batch[DataKey.constraint.name]
        if not eval_mode and getattr(self.cfg, 'constraint_noise', 0.):
            constraint = constraint + (2 * torch.rand_like(constraint) - 1) * self.cfg.constraint_noise
            constraint = constraint.clip(min=0, max=1)
        if self.sparse:
            time = batch[DataKey.constraint_time.name]
            space = batch[DataKey.constraint_space.name]
        else:
            time = batch[DataKey.covariate_time.name]
            space = batch[DataKey.covariate_space.name]
        if self.cfg.constraint_ablate:
            constraint = torch.zeros_like(constraint[:, :1])
            constraint_embed = self.encode_constraint(constraint)
            return (
                constraint_embed,
                time[:, :1], # TODO replace with zeros
                space[:, :1], # TODO replace with zeros
                create_padding_simple(constraint, torch.ones(constraint_embed.size(0), device=constraint_embed.device, dtype=torch.long))
            )
        if self.cfg.constraint_mute:
            encode_in = torch.zeros_like(constraint)
        elif getattr(self.cfg, 'constraint_support_mute', False):
            constraint_in = constraint[:, :, :1]
            constraint_in[constraint_in < 1] = 0 # Spoof full brain control
            encode_in = torch.cat([
                constraint_in, # Brain Control Constraint
                torch.zeros_like(constraint[:, :, 1:])
            ], dim=2)
        else:
            encode_in = constraint
        constraint_embed = self.encode_constraint(encode_in) # b t h d
        padding = create_padding_simple(
            constraint, batch.get(CONSTRAINT_LENGTH_KEY, None)
        )
        return (
            constraint_embed,
            time,
            space,
            padding,
        )


# Tokenizer for fast path
class FastTokenizer:
    def __init__(
            self,
            constraint_dims: torch.Tensor | None,
            spike_readin: nn.Embedding | None,
            return_enc: nn.Embedding | None,
            reward_enc: nn.Embedding | None,
            constraint_mute: bool = False,
            constraint_support_mute: bool = False,
            return_mute: bool = False,
            reward_mute: bool = False,
            embedding_dim: int = 0,
        ):
        self.constraint_dims = constraint_dims
        self.readin = spike_readin
        self.return_enc = return_enc
        self.reward_enc = reward_enc
        self.constraint_mute = constraint_mute
        self.constraint_support_mute = constraint_support_mute
        self.return_mute = return_mute
        self.reward_mute = reward_mute
        self.embedding_dim = embedding_dim

    def encode_constraint(self, constraint: torch.Tensor) -> torch.Tensor:
        if self.constraint_dims is None:
            return None # Should be none
        # constraint: Out is B T H Bhvr_Dim for sparse/tokenized, or B T H if dense
        if self.constraint_mute:
            return torch.zeros((
                constraint.size(0),
                constraint.size(1),
                self.constraint_dims.size(-1)
            ), device=constraint.device, dtype=constraint.dtype)
        elif self.constraint_support_mute:
            constraint_in = constraint[:, :, :1]
            constraint_in[constraint_in < 1] = 0 # Spoof full brain control when model has "remotely any control"
            return constraint_in @ self.constraint_dims[:1]
        return einsum(constraint, self.constraint_dims, 'b t constraint, constraint h -> b t h')

    def encode_spikes(self, spikes: torch.Tensor) -> torch.Tensor | None:
        if self.readin is None:
            return None
        state_in = torch.as_tensor(spikes, dtype=int)
        state_in = rearrange(state_in, 'b t c h -> b t (c h)')
        state_in = self.readin(state_in.clip(max=self.readin.num_embeddings - 1))
        state_in = state_in.flatten(-2, -1)
        return state_in

    def encode_return(self, task_return: torch.Tensor) -> torch.Tensor | None:
        if self.return_enc is None:
            return None
        if self.return_mute:
            return torch.zeros((
                task_return.size(0),
                task_return.size(1),
                self.embedding_dim
            ), device=task_return.device, dtype=torch.float)
            # ), device=task_return.device, dtype=task_return.dtype)
        return self.return_enc(task_return)

    def encode_reward(self, task_reward: torch.Tensor) -> torch.Tensor | None:
        if self.reward_enc is None:
            return None
        if self.reward_mute:
            return torch.zeros((
                task_reward.size(0),
                task_reward.size(1),
                self.embedding_dim
            ), device=task_reward.device, dtype=torch.float)
            # ), device=task_reward.device, dtype=task_reward.dtype)
        return self.reward_enc(task_reward)

    def reset_cache(self):
        pass

    def allocate_cache(self, batch_size: int = 1, max_seqlen: int = 8192, dtype=torch.bfloat16):
        pass

    def push_cache(self, neural: torch.Tensor, constraint: torch.Tensor, return_reward: torch.Tensor, behavior: torch.Tensor):
        pass


class DataPipeline(TaskPipeline):
    def get_masks(
        self,
        batch: Dict[BatchKey, torch.Tensor],
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
            ref: torch.Tensor = batch[DataKey.spikes.name][..., 0]
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

        if self.cfg.spike_loss == 'poisson':
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
            self.loss = nn.PoissonNLLLoss(reduction='none', log_input=cfg.lograte)
        elif self.cfg.spike_loss == 'cross_entropy':
            self.out = RatePrediction.create_linear_head(cfg, cfg.hidden_size, cfg.neurons_per_token)
            self.loss = PoissonCrossEntropyLoss(reduction='none', soften=self.cfg.cross_ent_soften, max_count=cfg.max_neuron_count)


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
        cfg: ModelConfig, in_size: int, out_size: int, layers=1, collapse_nonpatch=True,
    ):
        assert cfg.transform_space, 'Classification heads only supported for transformed space'
        if cfg.task.spike_loss == 'poisson':
            classes = 1
        elif cfg.task.spike_loss == 'cross_entropy':
            classes = cfg.max_neuron_count+ 1
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
            if collapse_nonpatch: # For NDT3
                rearr = Rearrange('... (s c) -> ... c s', c=classes)
            else: # NDT2
                rearr = Rearrange('b t (s c) -> b c t s', c=classes)
            out_layers.append(rearr)
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
        self.max_neuron_count = cfg.max_neuron_count
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

    def encode_direct(self, spikes: torch.Tensor):
        # breakpoint()
        if self.spike_embed_style == EmbedStrat.token:
            state_in = torch.as_tensor(spikes, dtype=int)
        else:
            state_in = spikes
        state_in = rearrange(state_in, 'b t c h -> b t (c h)')
        if self.spike_embed_style == EmbedStrat.token:
            state_in = self.readin(state_in.clip(max=self.readin.num_embeddings - 1))
        elif self.spike_embed_style == EmbedStrat.project:
            state_in = self.readin(state_in.clip(max=self.max_neuron_count).float().unsqueeze(-1))
        else:
            raise NotImplementedError
        state_in = state_in.flatten(-2, -1)
        return state_in

    def encode(self, batch):
        return self.encode_direct(batch[DataKey.spikes.name])

    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        batch[DataKey.padding.name] = create_padding_simple(batch[DataKey.spikes.name], batch.get(LENGTH_KEY, None))
         # TODO fix dataloader to load LENGTH_KEY as SPIKE_LENGTH_KEY (make spikes less special)
        return batch

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        spikes = self.encode(batch)
        time = batch[DataKey.time.name]
        space = batch[DataKey.position.name]
        padding = batch[DataKey.padding.name] # Padding should be made in the `update` step
        # print(f'Spike Space range: [{space.min()}, {space.max()}]')
        return spikes, time, space, padding

class SelfSupervisedInfill(RatePrediction):
    modifies = [DataKey.spikes.name]
    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        spikes = batch[DataKey.spikes.name]
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
            DataKey.spikes.name: spikes,
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
    modifies = [DataKey.spikes.name]

    def forward(
            self,
            batch,
            backbone_features: torch.Tensor,
            backbone_times: torch.Tensor,
            backbone_space: torch.Tensor,
            backbone_padding: torch.Tensor,
            loss_mask: torch.Tensor | None = None, # comes in as `B` or flat `B` at most.
            compute_metrics=True,
            eval_mode=False,
            phase: str = "train",
    ) -> torch.Tensor:
        if not compute_metrics:
            return {}
        # ! We assume that backbone features arrives in a batch-major, time-minor format, that has already been flattened
        # We need to similarly flatten
        # Time-sorting respects original served DataKey.spikes order (this should be true, but we should check)
        target = batch[DataKey.spikes.name][..., 0].clip(max=self.max_neuron_count) # B T C 1 -> B T C -> B C
        rates = self.out(backbone_features) # B x H
        # print(rates.shape, target.min(), target.max())
        # if target.max() >= rates.shape[1]:
            # breakpoint()
        loss = self.loss(rates.flatten(0, 1) if rates.ndim == 4 else rates, target.flatten(0, 1)) # B T C P -> B' C P,  B T C -> B' x C
        comparison = repeat(torch.arange(loss.size(-1), device=loss.device), 'c -> t c', t=loss.size(0))
        if backbone_padding.ndim == 2:
            backbone_padding = backbone_padding.flatten()
        if loss_mask is not None:
            loss_mask = (loss_mask & ~backbone_padding).unsqueeze(-1) # B -> B x 1
        else:
            loss_mask = ~backbone_padding.unsqueeze(-1) # B -> B x 1
        channel_mask = (comparison < batch[CHANNEL_KEY].flatten().unsqueeze(-1))
        loss_mask = loss_mask & channel_mask
        loss = loss[loss_mask].mean()
        return { 'loss': loss }

class PerceiverSpikeContext(SpikeContext):
    r"""
        Perceiver-style spike context.
        Rough-pass implementation of POYO https://arxiv.org/pdf/2310.16046, simplified for the BCI setting.

    """
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
        assert cfg.spike_embed_style == EmbedStrat.token, "Perceiver-style spike context only supports token embedding for parity with NDT3"
        # assert cfg.assert_batch_uniform, "Perceiver-style spike context only works with uniform batch sizes in this implementation"
        # ^ is a hard requirement but we don't actually want to trigger different codepaths...
        self.num_latents = 8 # # Match https://arxiv.org/pdf/2310.16046#page15 and also roughly match 32 neurons per token as in NDT3
        # ! This is also equiv to num latents - we don't need different latents for different time (just make sure you're encoding time, which NDT3 also uses ROPE for)
        # It's rather generous, NDT usually uses 3-6 tokens per timestep

        # spike_embed_dim = round(cfg.hidden_size / cfg.neurons_per_token)

        spike_embed_dim = cfg.hidden_size
        self.readin = nn.Embedding(cfg.max_neuron_count, spike_embed_dim, padding_idx=data_attrs.pad_token if data_attrs.pad_token else None)

        self.latent_dim = cfg.hidden_size  # Dimension of each latent
        self.latents = nn.Parameter(torch.randn(self.num_latents, self.latent_dim) / torch.sqrt(torch.tensor(self.latent_dim)))
        # Redundant with main backbone position embed, but can't remove latter because it's integral to main token flow
        self.unit_embed = nn.Embedding(data_attrs.max_channel_count, self.latent_dim)

        self.cross_attention = nn.MultiheadAttention(self.latent_dim, 1, batch_first=True)

    def simple_encode(self, spikes: torch.Tensor, time: torch.Tensor, position: torch.Tensor):
        # b t c h -> b c h
        # for ndt3 slim inference
        # assumes no padding
        batch_size = spikes.size(0)
        spike_embed = self.readin(spikes.clip(max=self.readin.num_embeddings - 1).int())
        spike_embed = rearrange(spike_embed, 'b time_space one_patch one_h hidden -> b time_space (one_patch one_h hidden)')
        unit_pos = position.int() # Unreduced, since spikes aren't reduced yet
        spike_embed = spike_embed + self.unit_embed(unit_pos)
        time_steps = time.max() + 1
        latents = repeat(self.latents, 'space h -> b (t space) h', b=batch_size, t=time_steps)

        spikes_per_timestep = position.max() + 1 # Unreduced, neuron count
        within_timestep_block = torch.ones(self.num_latents, spikes_per_timestep, device=spikes.device, dtype=torch.bool)
        num_blocks = spike_embed.size(1) // spikes_per_timestep
        full_mask = ~torch.block_diag(*[within_timestep_block] * num_blocks) # False is unmasked, True is masked
        spikes, _ = self.cross_attention(latents, spike_embed, spike_embed, attn_mask=full_mask)
        return spikes

    def simple_batch_encode(self, spikes: torch.Tensor, time: torch.Tensor, space: torch.Tensor):
        # assume no padding
        space_mask = (space < self.num_latents) # This may be not-even across batches, which is the only issue.
        keep_spacetime_mask = space_mask.any(dim=0)
        return self.simple_encode(spikes, time, space), time[:, keep_spacetime_mask], space[:, keep_spacetime_mask]

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], ddpg_flow=False, eval_mode=False):
        spikes = batch[DataKey.spikes.name]
        time = batch[DataKey.time.name]
        space = batch[DataKey.position.name]
        padding = batch[DataKey.padding.name]

        space_mask = (space < self.num_latents) & (~padding) # This may be not-even across batches, which is the only issue.
        keep_spacetime_mask = space_mask.any(dim=0)

        # We want two steps
        # * a length reduction, operated across batch (since that's what determines whether the timestep exists)
        time = time[:, keep_spacetime_mask]
        space = space[:, keep_spacetime_mask]
        # * a padding update
        padding = (batch[DataKey.padding.name] & ~space_mask)[:, keep_spacetime_mask]

        batch_size, spacetime_steps, channels, _ = spikes.size()
        spike_embed = self.readin(spikes.clip(max=self.readin.num_embeddings - 1).int())
        spike_embed = rearrange(spike_embed, 'b time_space one_patch one_h hidden -> b time_space (one_patch one_h hidden)')
        unit_pos = batch[DataKey.position.name].int() # Unreduced, since spikes aren't reduced yet
        spike_embed = spike_embed + self.unit_embed(unit_pos)

        time_steps = time[~padding].max() + 1
        latents = repeat(self.latents, 'space h -> b (t space) h', b=batch_size, t=time_steps)

        # Create a causal mask
        # (L, S) is target seq length, S is source seq length - so that's query / latents size, and key / val spikes size
        # What I want is to make this block-lower-triangular, so that each self.num_latents in latent dimension cannot attend to spikes past the corresponding spikes per timestep
        # Instead currently it's block diagonal, not bad, restricts latents to looking within timestep, but not lower triangular
        # * This should be consistent across trials in a batch
        spikes_per_timestep = batch[DataKey.position.name][~batch[DataKey.padding.name]].max() + 1 # Unreduced, neuron count
        within_timestep_block = torch.ones(self.num_latents, spikes_per_timestep, device=spikes.device, dtype=torch.bool)
        num_blocks = spike_embed.size(1) // spikes_per_timestep
        full_mask = ~torch.block_diag(*[within_timestep_block] * num_blocks) # False is unmasked, True is masked

        spikes, _ = self.cross_attention(latents, spike_embed, spike_embed, attn_mask=full_mask)
        return spikes, time, space, padding


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

    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        super().update_batch(batch, eval_mode=eval_mode)
        return self.crop_batch(self.cfg.mask_ratio, batch, eval_mode=eval_mode, shuffle=True)

    def crop_batch(self, mask_ratio: float, batch: Dict[BatchKey, torch.Tensor], eval_mode=False, shuffle=True):
        r"""
            Shuffle inputs, keep only what we need for evaluation
        """
        spikes = batch[DataKey.spikes.name]
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
            nonprompt_time = (batch[DataKey.time.name] > self.cfg.context_prompt_time_thresh) # B x T mask
            shuffle = shuffle.unsqueeze(0).repeat(spikes.size(0), 1)
            nonprompt_time_shuffled = shuffle_func(nonprompt_time, shuffle).int() # bool not implemented for CUDA
            shuffle = sort_A_by_B(shuffle, nonprompt_time_shuffled) # B x T
        else:
            shuffle_func = apply_shuffle
        # Mask ratio becomes a comment on the remainder of the data
        encoder_frac = round((1 - mask_ratio) * spikes.size(1))
        # shuffle_spikes = spikes.gather(1, shuffle.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spikes.size(2), spikes.size(3)))
        for key in [DataKey.time.name, DataKey.position.name, DataKey.padding.name, CHANNEL_KEY]:
            if key in batch:
                shuffled = shuffle_func(batch[key], shuffle)
                batch.update({
                    key: shuffled[:, :encoder_frac],
                    f'{key}_target': shuffled[:, encoder_frac:],
                })
        # import pdb;pdb.set_trace()
        target = shuffle_func(target, shuffle)[:,encoder_frac:]
        batch.update({
            DataKey.spikes.name: shuffle_func(spikes, shuffle)[:,:encoder_frac],
            f'{self.handle}_target': target,
            # f'{self.handle}_encoder_frac': encoder_frac, # ! Deprecating
        })
        batch[f'{self.handle}_query'] = self.injector.make_query(target)
        return batch

    def get_loss_mask(self, batch: Dict[BatchKey, torch.Tensor], loss: torch.Tensor, padding_mask: torch.Tensor | None = None):
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
            eval_mode=False,
            phase: str = "train",
        ) -> torch.Tensor:
        batch_out = {}
        target = batch[f'{self.handle}_target'] # B T H
        if not eval_mode:
            decode_tokens = batch[f'{self.handle}_query']
            decode_time = batch[f'{DataKey.time.name}_target']
            decode_space = batch[f'{DataKey.position.name}_target']
            decode_padding = batch[f'{DataKey.padding.name}_target']
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

    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        assert False, 'deprecated. Use `next_step_prediction` modelConfig to directly specify'
        spikes = batch[DataKey.spikes]
        target = spikes[..., 0]
        breakpoint()
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
            assert reference != DataKey.spikes.name, "Decode tokenization should not run for spikes, this is for covariate exps"
        self.pad_value = data_attrs.pad_token
        self.max_space = data_attrs.max_spatial_tokens

    def make_query(self, reference: torch.Tensor):
        r"""
            Much simpler abstraction to just make a few tokens from a flat ref
        """
        b, t, *_ = reference.size() # reference should already be tokenized to desired res
        return repeat(self.cls_token, 'h -> b t h', b=b, t=t)

    def inject(self, batch: Dict[BatchKey, torch.Tensor], in_place=False, injected_time: torch.Tensor | None = None, injected_space: torch.Tensor | None = None):
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
            batch[DataKey.extra.name] = injected_tokens # B T H
            batch[DataKey.extra_time.name] = injected_time
            batch[DataKey.extra_position.name] = injected_space
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

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        if self.cfg.return_mute:
            return_embed = self.return_enc(torch.zeros_like(batch[DataKey.task_return.name]))
        else:
            return_embed = self.return_enc(batch[DataKey.task_return.name])
        if self.cfg.reward_mute:
            reward_embed = self.reward_enc(torch.zeros_like(batch[DataKey.task_reward.name]))
        else:
            reward_embed = self.reward_enc(batch[DataKey.task_reward.name])
        times = batch[DataKey.task_return_time.name]
        space = torch.zeros_like(times)
        padding = create_padding_simple(return_embed, batch.get(RETURN_LENGTH_KEY, None))
        return (
            return_embed + reward_embed,
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
        self.out = nn.Linear(backbone_out_size, self.max_return)
        # TODO - maybe merge with CovariateInfill... but I think this would be more cost than benefit.

    def predict(
            self,
            backbone_features: torch.Tensor,
    ):
        return self.out(backbone_features)

    def forward(
        self,
        batch,
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor | None = None,
        backbone_space: torch.Tensor | None = None,
        backbone_padding: torch.Tensor | None = None,
        compute_metrics=True,
        eval_mode=False,
        loss_mask=None,
        phase: str = "train",
    ) -> Dict[BatchKey, torch.Tensor]:
        has_outputs = Output.return_logits in self.cfg.outputs
        if has_outputs or compute_metrics:
            pred = self.out(backbone_features)
            target = batch[DataKey.task_return.name].flatten()
        batch_out = {}
        if has_outputs:
            batch_out[Output.return_logits] = pred
            batch_out[Output.return_target] = target
        if not compute_metrics:
            return batch_out
        loss = F.cross_entropy(pred, target, reduction='none', label_smoothing=self.cfg.decode_label_smooth)
        if loss_mask is not None:
            loss_mask = loss_mask & ~backbone_padding
        else:
            loss_mask = ~backbone_padding
        batch_out = {}
        loss = loss[loss_mask].mean()
        if not loss_mask.any():
            batch_out['loss'] = torch.zeros_like(loss)
        else:
            batch_out['loss'] = loss
        if Metric.return_acc in self.cfg.metrics:
            acc = (pred.argmax(1) == target)
            if not loss_mask.any():
                batch_out[Metric.return_acc.value] = torch.zeros_like(acc).float().mean()
            else:
                batch_out[Metric.return_acc.value] = acc[loss_mask].float().mean()
        return batch_out

class MetadataContext(ContextPipeline):
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
        self.data_attrs = data_attrs
        self.cfg = cfg # Override for now
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

        for attr in ['session', 'subject', 'task', 'array']:
            if getattr(self.cfg, f'{attr}_embed_strategy') is not EmbedStrat.none:
                assert getattr(data_attrs.context, attr), f"{attr} embedding strategy requires {attr} in data"
                if len(getattr(data_attrs.context, attr)) == 1:
                    logger.warning(f'Using {attr} embedding strategy with only one {attr}. Expected only if tuning.')

        # We write the following repetitive logic explicitly to maintain typing
        project_size = self.cfg.hidden_size

        if self.cfg.session_embed_strategy is not EmbedStrat.none:
            if self.cfg.session_embed_strategy == EmbedStrat.token and self.cfg.session_embed_token_count > 1:
                self.session_embed = nn.Parameter(torch.randn(len(data_attrs.context.session), self.cfg.session_embed_token_count, self.cfg.session_embed_size) / math.sqrt(self.cfg.session_embed_size))
                self.session_flag = nn.Parameter(torch.randn(self.cfg.session_embed_token_count, self.cfg.session_embed_size) / math.sqrt(self.cfg.session_embed_size))
            else:
                self.session_embed = nn.Embedding(len(data_attrs.context.session), self.cfg.session_embed_size)
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
                self.subject_embed = nn.Parameter(torch.randn(len(data_attrs.context.subject), self.cfg.subject_embed_token_count, self.cfg.subject_embed_size) / math.sqrt(self.cfg.subject_embed_size))
                self.subject_flag = nn.Parameter(torch.randn(self.cfg.subject_embed_token_count, self.cfg.subject_embed_size) / math.sqrt(self.cfg.subject_embed_size))
            else:
                self.subject_embed = nn.Embedding(len(data_attrs.context.subject), self.cfg.subject_embed_size)
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
                len(data_attrs.context.array),
                self.cfg.array_embed_size,
                padding_idx=data_attrs.context.array.index('') if '' in data_attrs.context.array else None
            )
            self.array_embed.weight.data.fill_(0) # Don't change by default
            if self.cfg.array_embed_strategy == EmbedStrat.concat:
                project_size += self.cfg.array_embed_size
            elif self.cfg.array_embed_strategy == EmbedStrat.token:
                assert self.cfg.array_embed_size == self.cfg.hidden_size
                if self.cfg.init_flags:
                    self.array_flag = nn.Parameter(torch.randn(data_attrs.max_arrays, self.cfg.array_embed_size) / math.sqrt(self.cfg.array_embed_size))
                else:
                    self.array_flag = nn.Parameter(torch.zeros(data_attrs.max_arrays, self.cfg.array_embed_size))

        if self.cfg.task_embed_strategy is not EmbedStrat.none:
            if self.cfg.task_embed_strategy == EmbedStrat.token and self.cfg.task_embed_token_count > 1:
                self.task_embed = nn.Parameter(torch.randn(len(data_attrs.context.task), self.cfg.task_embed_token_count, self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
                self.task_flag = nn.Parameter(torch.randn(self.cfg.task_embed_token_count, self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
            else:
                self.task_embed = nn.Embedding(len(data_attrs.context.task), self.cfg.task_embed_size)
                if self.cfg.task_embed_strategy == EmbedStrat.concat:
                    project_size += self.cfg.task_embed_size
                elif self.cfg.task_embed_strategy == EmbedStrat.token:
                    assert self.cfg.task_embed_size == self.cfg.hidden_size
                    if self.cfg.init_flags:
                        self.task_flag = nn.Parameter(torch.randn(self.cfg.task_embed_size) / math.sqrt(self.cfg.task_embed_size))
                    else:
                        self.task_flag = nn.Parameter(torch.zeros(self.cfg.task_embed_size))

    def try_transfer(self, module_name: str, transfer_module: Any = None, transfer_data_attrs: Optional[DataAttrs] = None):
        if (module := getattr(self, module_name, None)) is not None:
            if transfer_module is not None:
                if isinstance(module, nn.Parameter):
                    assert module.data.shape == transfer_module.data.shape
                    # Currently will fail for array flag transfer, no idea what the right policy is right now
                    module.data = transfer_module.data
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
        transfer_embed: nn.Embedding | nn.Parameter,
    ) -> nn.Embedding | nn.Parameter:
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

    def transfer_weights(self, transfer_model: Self, transfer_data_attrs: Optional[DataAttrs] = None):
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


    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        # TODO phase these out given re-generation of PittCO
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

class CovariateReadout(DataPipeline, ConstraintPipeline):
    r"""
        Base class for decoding (regression/classification) of covariates.
        Constraints may be packed in here because the encoding is fused with behavior in the non-sparse case, but we may want to refactor that out.
    """
    modifies = [DataKey.bhvr_vel.name, DataKey.constraint.name, DataKey.constraint_time.name]

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
        self.reference_cov = self.cfg.behavior_target.name
        self.injector = TemporalTokenInjector(
            cfg,
            data_attrs,
            None, # deprecating reference while trying ot clean up terminology # self.cfg.behavior_target.name if not self.encodes else f'{self.handle}_target',
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

    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode = False):
        batch[f'{self.handle}_{DataKey.padding.name}'] = create_padding_simple(
            batch[self.cfg.behavior_target.name],
            batch.get(f'{self.handle}_{LENGTH_KEY}', None)
        )
        return self.crop_batch(self.cfg.covariate_mask_ratio, batch, eval_mode=eval_mode) # Remove encode

    def crop_batch(self, mask_ratio: float, batch: Dict[BatchKey, torch.Tensor], eval_mode=False, shuffle=True):
        covariates = batch[self.cfg.behavior_target.name] # B (T Cov_Dims) 1 if tokenized, else  B x T x Cov_Dims,
        if DataKey.covariate_time.name not in batch:
            cov_time = torch.arange(covariates.size(1), device=covariates.device)
            if self.cfg.decode_tokenize_dims:
                cov_time = repeat(cov_time, 't -> b (t d)', b=covariates.size(0), d=self.cov_dims)
            else:
                cov_time = repeat(cov_time, 't -> b t', b=covariates.size(0))
            batch[DataKey.covariate_time.name] = cov_time
        if DataKey.covariate_space.name not in batch:
            if self.cfg.decode_tokenize_dims: # Here in is the implicit padding for space position, to fix.
                cov_space = repeat(
                    torch.arange(covariates.size(2), device=covariates.device),
                    'd -> b (t d)', b=covariates.size(0), t=covariates.size(1)
                )
            else:
                # Technically if data arrives as b t* 1, we can still use above if-case circuit
                cov_space = torch.zeros_like(batch[DataKey.covariate_time.name])
            batch[DataKey.covariate_space.name] = cov_space

        if not self.encodes: # Just make targets, exit
            batch[f'{DataKey.covariate_time.name}_target'] = batch[DataKey.covariate_time.name]
            batch[f'{DataKey.covariate_space.name}_target'] = batch[DataKey.covariate_space.name]
            batch[f'{self.handle}_{DataKey.padding.name}_target'] = batch[f'{self.handle}_{DataKey.padding.name}']
        else:
            if self.cfg.decode_tokenize_dims and not self.served_tokenized_covariates:
                covariates = rearrange(covariates, 'b t bhvr_dim -> b (t bhvr_dim) 1')
                if self.cfg.encode_constraints:
                    batch[DataKey.constraint.name] = rearrange(batch[DataKey.constraint.name], 'b t constraint bhvr_dim -> b (t bhvr_dim) constraint')
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
            if self.cfg.context_prompt_time_thresh:
                shuffle_func = apply_shuffle_2d
                nonprompt_time = (batch[DataKey.covariate_time.name] >= self.cfg.context_prompt_time_thresh) # B x T mask
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
                DataKey.covariate_time.name,
                DataKey.covariate_space.name,
                f'{self.handle}_{DataKey.padding.name}',
            ]:
                shuffle_key(key)
            if self.cfg.encode_constraints and not self.inject_constraint_tokens:
                for key in [
                    DataKey.constraint.name,
                    DataKey.constraint_time.name,
                ]:
                    shuffle_key(key)
            splits = [encoder_frac, covariates.size(1) - encoder_frac]
            enc, target = torch.split(shuffle_func(covariates, shuffle), splits, dim=1)
            batch.update({
                self.cfg.behavior_target.name: enc,
                f'{self.handle}_target': target,
                f'{self.handle}_encoder_frac': encoder_frac,
            })
        batch[f'{self.handle}_query'] = self.injector.make_query(self.get_target(batch))
        return batch

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        if self.cfg.covariate_mask_ratio == 1.0:
            # return super().get_context(batch)
            return [], [], [], []
        enc = self.encode_cov(batch[self.cfg.behavior_target.name])
        if self.cfg.encode_constraints and not self.inject_constraint_tokens:
            constraint = self.encode_constraint(batch[DataKey.constraint.name]) # B T H Bhvr_Dim. Straight up not sure how to collapse non-losslessly - we just mean pool for now.
            enc = enc + constraint
        return (
            enc,
            batch[DataKey.covariate_time.name],
            batch[DataKey.covariate_space.name],
            batch[f'{self.handle}_{DataKey.padding.name}']
        )

    @abc.abstractmethod
    def encode_cov(self, covariate: torch.Tensor) -> torch.Tensor: # B T Bhvr_Dims or possibly B (T Bhvr_Dims)
        raise NotImplementedError

    def get_cov_pred(
        self,
        batch: Dict[BatchKey, torch.Tensor],
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor,
        backbone_space: torch.Tensor,
        backbone_padding: torch.Tensor,
        eval_mode=False,
        batch_out={},
    ) -> torch.Tensor:
        r"""
            returns: flat seq of predictions, B T' H' (H' is readout dim, regression) or B C T' (classification)
        """
        if self.cfg.decode_separate:
            if self.cfg.decode_time_pool: # B T H -> B T H
                assert False, "Deprecated, currently would pool across modalities... but time is available if you still wanna try"
                backbone_features, backbone_padding = temporal_pool(batch, backbone_features, backbone_padding, pool=self.cfg.decode_time_pool)
                if Output.pooled_features in self.cfg.outputs:
                    batch_out[Output.pooled_features] = backbone_features.detach()

            decode_tokens = batch[f'{self.handle}_query']
            decode_time = batch[f'{DataKey.covariate_time.name}_target']
            decode_space = batch[f'{DataKey.covariate_space.name}_target']
            decode_padding = batch[f'{self.handle}_{DataKey.padding.name}_target']
            if not self.inject_constraint_tokens and self.cfg.encode_constraints:
                decode_tokens = decode_tokens + self.encode_constraint(
                    batch[f'{DataKey.constraint.name}_target'],
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

    def get_target(self, batch: Dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        if self.cfg.covariate_mask_ratio == 1.0:
            tgt = batch[self.cfg.behavior_target.name]
        else:
            tgt = batch[f'{self.handle}_target']
        if self.bhvr_mean is not None:
            tgt = tgt - self.bhvr_mean
            tgt = tgt / self.bhvr_std
        return tgt

    def simplify_logits_to_prediction(self, bhvr: torch.Tensor, *args, **kwargs):
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
        eval_mode=False,
        temperature=0.,
        phase: str = "train",
    ) -> torch.Tensor:
        batch_out = {}
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
            batch_out[f'{DataKey.covariate_space.name}_target'] = batch[f'{DataKey.covariate_space.name}_target']
            batch_out[f'{DataKey.covariate_time.name}_target'] = batch[f'{DataKey.covariate_time.name}_target']
            batch_out[f'{self.handle}_{DataKey.padding.name}_target'] = batch[f'{self.handle}_{DataKey.padding.name}_target']
            if DataKey.covariate_labels.name in batch:
                batch_out[DataKey.covariate_labels.name] = batch[DataKey.covariate_labels.name]
            batch_out[Output.behavior_pred] = self.simplify_logits_to_prediction(bhvr, temperature=temperature)
            if self.bhvr_mean is not None:
                batch_out[Output.behavior_pred] = batch_out[Output.behavior_pred] * self.bhvr_std + self.bhvr_mean
        if Output.behavior in self.cfg.outputs:
            batch_out[Output.behavior] = self.get_target(batch)
        if not compute_metrics:
            return batch_out

        bhvr_tgt = self.get_target(batch)

        _, length_mask, _ = self.get_masks(
            batch, ref=bhvr_tgt,
            length_key=f'{self.handle}_{LENGTH_KEY}',
            shuffle_key=None,
            compute_channel=False,
            padding_mask=batch.get(f'{self.handle}_{DataKey.padding.name}_target', None),
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
                    if self.cfg.decode_tokenize_dims:
                        positions = batch[f'{DataKey.covariate_space.name}_target']
                        loss_mask = loss_mask & ~torch.isin(positions, self.covariate_blacklist_dims.to(device=positions.device))
                    else:
                        loss_mask = loss_mask.unsqueeze(-1).repeat(1, 1, loss.size(-1))
                        loss_mask[..., self.covariate_blacklist_dims] = False
                if not loss_mask.any():
                    logger.warning('No dims survive loss mask, kinematic loss is zero')
                    # breakpoint()
                    loss = torch.zeros_like(loss).mean()
                else:
                    loss = loss[loss_mask].mean()
        r2_mask = length_mask

        batch_out['loss'] = loss
        if Metric.kinematic_r2 in self.cfg.metrics:
            valid_bhvr = bhvr[..., :bhvr_tgt.shape[-1]]
            if len(self.covariate_blacklist_dims) > 0:
                assert self.cfg.decode_tokenize_dims, "blacklist dims not implemented for non tokenized R2"
                positions: torch.Tensor = batch[f'{DataKey.covariate_space.name}_target']
                r2_mask = r2_mask & ~torch.isin(positions, self.covariate_blacklist_dims.to(device=positions.device))
            valid_bhvr = self.simplify_logits_to_prediction(valid_bhvr)[r2_mask].float().detach().cpu()
            valid_tgt = bhvr_tgt[r2_mask].float().detach().cpu()
            if self.served_tokenized_covariates and not self.served_semantic_covariates: # If semantic, we don't need to reorganize
                assert len(self.covariate_blacklist_dims) == 0, "blacklist dims not implemented for non semantic R2"
                # Compute the unique covariate labels, and their repsective position indices.
                # Then pull R2 accordingly. Lord knows this isn't the most efficient, but...
                dims_per = torch.tensor([len(i) for i in batch[DataKey.covariate_labels.name]], device=batch[f'{DataKey.covariate_space.name}_target'].device).cumsum(0)
                batch_shifted_positions = batch[f'{DataKey.covariate_space.name}_target'] + (dims_per - dims_per[0]).unsqueeze(-1)
                flat_labels = np.array(list(itertools.chain.from_iterable(batch[DataKey.covariate_labels.name])))
                unique_labels, label_indices = np.unique(flat_labels, return_inverse=True)
                range_reference = np.arange(len(flat_labels))
                r2_scores = []
                batch_shifted_positions = batch_shifted_positions[r2_mask].flatten().cpu()
                for i, l in enumerate(unique_labels):
                    unique_indices = torch.as_tensor(range_reference[label_indices == i])
                    submask = torch.isin(batch_shifted_positions, unique_indices)
                    if not submask.any(): # Unlucky, shouldn't occur if we predict more.
                        r2_scores.append(0)
                        continue
                    r2_scores.append(r2_score(valid_tgt[submask], valid_bhvr[submask]))
                batch_out[Metric.kinematic_r2.value] = np.array(r2_scores)
                batch[DataKey.covariate_labels.name] = unique_labels
            elif self.cfg.decode_tokenize_dims:
                # extract the proper subsets according to space (for loop it) - per-dimension R2 is only relevant while dataloading maintains consistent dims (i.e. not for long) but in the meanwhile
                r2_scores = []
                positions = batch[f'{DataKey.covariate_space.name}_target'][r2_mask].flatten().cpu() # flatten as square full batches won't autoflatten B x T but does flatten B x T x 1
                for i in positions.unique():
                    r2_scores.append(r2_score(valid_tgt[positions == i], valid_bhvr[positions == i]))
                batch_out[Metric.kinematic_r2.value] = np.array(r2_scores)
            else:
                assert len(self.covariate_blacklist_dims) == 0, "blacklist dims not implemented for non tokenized R2"
                batch_out[Metric.kinematic_r2.value] = r2_score(valid_tgt, valid_bhvr, multioutput='raw_values')
            if batch_out[Metric.kinematic_r2.value].mean() < getattr(self.cfg, 'clip_r2_min', -10000):
                batch_out[Metric.kinematic_r2.value] = np.zeros_like(batch_out[Metric.kinematic_r2.value])# .mean() # mute, some erratic result from near zero target skewing plots
                # print(valid_bhvr.mean().cpu().item(), valid_tgt.mean().cpu().item(), batch_out[Metric.kinematic_r2].mean())
                # breakpoint()
            # if Metric.kinematic_r2_thresh in self.cfg.metrics: # Deprecated, note to do this we'll need to recrop `position_target` as well
            #     valid_bhvr = valid_bhvr[valid_tgt.abs() > self.cfg.behavior_metric_thresh]
            #     valid_tgt = valid_tgt[valid_tgt.abs() > self.cfg.behavior_metric_thresh]
            #     batch_out[Metric.kinematic_r2_thresh] = r2_score(valid_tgt, valid_bhvr, multioutput='raw_values')
            batch_out[Metric.kinematic_r2.value] = torch.as_tensor(batch_out[Metric.kinematic_r2.value])
        if Metric.kinematic_acc in self.cfg.metrics:
            acc = (bhvr.argmax(1) == self.quantize(bhvr_tgt))
            batch_out[Metric.kinematic_acc.value] = acc[r2_mask].float().mean()
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
        quantize_bound = 1.001 #  if not self.cfg.decode_symlog else symlog(torch.tensor(1.001))
        self.is_next_step = cfg.next_step_prediction
        self.cov_dims = data_attrs.behavior_dim
        self.register_buffer('zscore_quantize_buckets', torch.linspace(-quantize_bound, quantize_bound, self.cfg.decode_quantize_classes + 1)) # This will produce values from 1 - self.quantize_classes, as we rule out OOB. Asymmetric as bucketize is asymmetric; on bound value is legal for left, quite safe for expected z-score range. +1 as these are boundaries, not centers
        self.bin_width = self.zscore_quantize_buckets[1] - self.zscore_quantize_buckets[0]

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(x != self.pad_value, x, 0) # actually redundant if padding is sensibly set to 0, but sometimes it's not
        # if self.cfg.decode_symlog:
            # return torch.bucketize(symlog(x), self.zscore_quantize_buckets)
        return torch.bucketize(x, self.zscore_quantize_buckets) - 1 # bucketize produces from [1, self.quantize_classes]

    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        # if quantized.max() > self.zscore_quantize_buckets.shape[0]:
            # raise Exception("go implement quantization clipping man")
        # if self.cfg.decode_symlog:
            # return unsymlog((self.zscore_quantize_buckets[quantized] + self.zscore_quantize_buckets[quantized + 1]) / 2)
        return (self.zscore_quantize_buckets[quantized] + self.zscore_quantize_buckets[quantized + 1]) / 2

class QuantizeSimple:
#     # For torch.compile, remove forking paths
    def __init__(self,
        zscore_quantize_buckets: torch.Tensor,
        pad_value: int = 0,
    ):
        super().__init__()
        self.pad_value = pad_value
        self.zscore_quantize_buckets = zscore_quantize_buckets

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.where(x != self.pad_value, x, 0) # actually redundant if padding is sensibly set to 0, but sometimes it's not
        return torch.bucketize(x, self.zscore_quantize_buckets) - 1 # bucketize produces from [1, self.quantize_classes]

    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        return (self.zscore_quantize_buckets[quantized] + self.zscore_quantize_buckets[quantized + 1]) / 2


class BehaviorContext(ContextPipeline, QuantizeBehavior):
    # For feeding autoregressive task
    # Simple quantizing tokenizer
    # * Actually just a reference, not actually used... this is because data must arrive embedded for the main model.
    # So either this is subsumed as
    @property
    def handle(self):
        return 'covariate'

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        batch[f'{self.handle}_{DataKey.padding.name}'] = create_padding_simple(
            batch[self.cfg.behavior_target.name],
            batch.get(f'{self.handle}_{LENGTH_KEY}', None),
        )
        # breakpoint() # TODO check dims, we may not need the mean call
        return (
            self.quantize(batch[self.cfg.behavior_target.name]).mean(-2), # B T 1 out
            batch[DataKey.covariate_time],
            batch[DataKey.covariate_space],
            batch[f'{self.handle}_{DataKey.padding.name}']
        )

class ClassificationMixin(QuantizeBehavior):
    def initialize_readin(self, backbone_size): # Assume quantized readin...
        self.inp = nn.Embedding(getattr(self.cfg, 'decode_quantize_classes', 128) + 1, backbone_size, padding_idx=0)
        # self.inp_norm = nn.LayerNorm(backbone_size)

    def initialize_readout(self, backbone_size):
        # We use these buckets as we minmax clamp in preprocessing
        if self.cfg.decode_tokenize_dims:
            if self.is_next_step:
                self.out = nn.Linear(backbone_size, self.cfg.decode_quantize_classes)
            else:
                self.out = nn.Sequential(
                    nn.Linear(backbone_size, self.cfg.decode_quantize_classes),
                    Rearrange('b t c -> b c t')
                )
        else:
            assert not self.is_next_step, "next step not implemented for non-tokenized"
            self.out = nn.Sequential(
                nn.Linear(backbone_size, self.cfg.decode_quantize_classes * self.cov_dims),
                Rearrange('b t (c d) -> b c (t d)', c=self.cfg.decode_quantize_classes)
            )

    def encode_cov(self, covariate: torch.Tensor):
        # Note: covariate is _not_ foreseeably quantized at this point, we quantize herein during embed.
        r"""
            covariate: B T Bhvr_Dims. Bhvr_Dims=1 for most cases, ie any flat path
            returns:
                B T H
        """
        covariate = self.inp(self.quantize(covariate)) # B T Bhvr_Dims -> B T Bhvr_Dims H.
        covariate = covariate.mean(-2) # B T Bhvr_Dims H -> B T H # (Even if Bhvr_dim = 1, which is true in tokenized serving)
        # covariate = self.inp_norm(covariate)
        return covariate

    def simplify_logits_to_prediction(self, logits: torch.Tensor, logit_dim=1, temperature=0.): # 0. -> argmax
        # breakpoint()
        if temperature > 0:
            batched = logits.ndim == 3
            if batched:
                b, d, c = logits.shape # batch, kin dim, class
                logits = rearrange(logits, 'b d c -> (b d) c')
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=logit_dim) # TODO deprecate logit_dim
            choice = torch.multinomial(probabilities, 1)
            if batched:
                choice = rearrange(choice, '(b d) 1 -> b d', b=b, d=d)
        else:
            choice = logits.argmax(logit_dim)
        return self.dequantize(choice)

    def compute_loss(self, bhvr: torch.Tensor, bhvr_tgt: torch.Tensor):
        if getattr(self.cfg, 'decode_hl_gauss_sigma_bin_ratio', 0.):
            # https://arxiv.org/pdf/2403.03950v1.pdf Stop Regressing
            sigma = self.bin_width * self.cfg.decode_hl_gauss_sigma_bin_ratio
            cdf_evals = torch.special.erf(
                (self.zscore_quantize_buckets - bhvr_tgt.unsqueeze(-1))
                / (torch.sqrt(torch.tensor(2.0)) * sigma)
            )
            z = cdf_evals[..., -1] - cdf_evals[..., 0]
            bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
            probs = bin_probs / z.unsqueeze(-1)
            return F.cross_entropy(bhvr, probs, reduction='none')
        # breakpoint()
        # print(bhvr.shape, self.quantize(bhvr_tgt).shape, self.quantize(bhvr_tgt).min(), self.quantize(bhvr_tgt).max())
        tgts = self.quantize(bhvr_tgt)
        return F.cross_entropy(bhvr, tgts, reduction='none', label_smoothing=self.cfg.decode_label_smooth)


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
            bhvr = rearrange(bhvr, 'b t (c d) -> b c t d', c=self.cfg.decode_quantize_classes)
        elif self.served_tokenized_covariates:
            bhvr = rearrange(bhvr, 'b c t -> b c t 1')
        else:
            bhvr = rearrange(bhvr, 'b (t d) c -> b c t d', d=self.cov_dims)

        return bhvr

class CovariateInfill(ClassificationMixin):
    r"""
        Primary covariate decoding class for NDT3.
        # CovariateReadout is quite overloaded; we create a simpler next step prediction covariate readout module
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
        self.served_tokenized_covariates = data_attrs.tokenize_covariates
        self.served_semantic_covariates = data_attrs.semantic_covariates
        self.reference_cov = self.cfg.behavior_target.name
        self.cov_dims = data_attrs.behavior_dim
        self.initialize_readin(cfg.hidden_size)
        self.initialize_readout(cfg.hidden_size)
        if Metric.kinematic_r2 in self.cfg.metrics:
            import torchmetrics
            self.train_r2_score = torchmetrics.R2Score()
            self.val_r2_score = torchmetrics.R2Score()
            self.eval_r2_score = torchmetrics.R2Score()

    @property
    def handle(self):
        return 'covariate'

    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode = False):
        batch[f'{self.handle}_{DataKey.padding.name}'] = create_padding_simple(
            batch[self.cfg.behavior_target.name],
            batch.get(f'{self.handle}_{LENGTH_KEY}', None),
        )
        return batch

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        enc = self.encode_cov(batch[self.cfg.behavior_target.name])
        return (
            enc,
            batch[DataKey.covariate_time.name],
            batch[DataKey.covariate_space.name],
            batch[f'{self.handle}_{DataKey.padding.name}']
        )

    def predict(self, backbone_features: torch.Tensor, temperature=0.):
        class_feats = self.out(backbone_features)
        pred = self.simplify_logits_to_prediction(class_feats, temperature=temperature)
        if temperature > 0:
            pred = pred.squeeze(-1)
        return pred

    def get_r2_metric(self, prefix: str):
        if prefix == 'train':
            return self.train_r2_score
        elif prefix == 'val':
            return self.val_r2_score
        elif prefix == 'eval':
            return self.eval_r2_score

    def forward(
        self,
        batch,
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor | None = None,
        backbone_space: torch.Tensor | None = None,
        backbone_padding: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None, # subset loss if provided
        compute_metrics=True,
        eval_mode=False,
        temperature=0.,
        phase: str = "train",
    ) -> Dict[BatchKey, torch.Tensor]:
        batch_out = {}
        bhvr: torch.Tensor = self.out(backbone_features)
        if Output.behavior_logits in self.cfg.outputs:
            batch_out[Output.behavior_logits] = bhvr
        if Output.behavior_pred in self.cfg.outputs: # Note we need to eventually implement some kind of repack, just like we do for spikes
            batch_out[Output.behavior_pred] = self.simplify_logits_to_prediction(bhvr, logit_dim=-1, temperature=temperature) # returns logits
        if DataKey.bhvr_mask.name in batch:
            batch_out[Output.behavior_mask] = batch[DataKey.bhvr_mask.name]
        bhvr_tgt = batch[self.cfg.behavior_target.name].flatten()
        if Output.behavior in self.cfg.outputs:
            batch_out[Output.behavior] = bhvr_tgt # Flat aspect is not ideal, watch the timestamps..
        if not compute_metrics:
            return batch_out

        # Compute loss
        loss = self.compute_loss(bhvr, bhvr_tgt) # We don't _have_ to predict this for non-padding or at all if there's no targets, but it's convenient.
        # breakpoint()
        if loss_mask is not None:
            loss_mask = loss_mask & ~backbone_padding
        else:
            loss_mask = ~backbone_padding
        if DataKey.bhvr_mask in batch:
            loss_mask = loss_mask & batch[DataKey.bhvr_mask].flatten()
        elif DataKey.bhvr_mask.name in batch:
            loss_mask = loss_mask & batch[DataKey.bhvr_mask.name].flatten()

        if not loss_mask.any(): # ! This really shouldn't trigger and will cause unused parameters for DDP.
            # FALCON note - will trigger for M1 with long intertrial masked stretches
            # ! This shouldn't trigger because we take care to have a concluding kinematic padding token in each sequence in worst case - but it still seems to be triggering...?
            loss = (loss * 0).mean()
        else:
            loss = loss[loss_mask].mean()
        batch_out['loss'] = loss
        if Metric.kinematic_r2 in self.cfg.metrics:
            valid_bhvr = bhvr
            valid_bhvr = self.simplify_logits_to_prediction(valid_bhvr)[loss_mask].float().detach().cpu()
            valid_tgt = bhvr_tgt[loss_mask].float().detach().cpu()
            # print(f"Reporting r2 on {valid_tgt.shape} of {bhvr.shape} timesteps")
            # check for empty comparison
            if len(valid_bhvr) <= 1:
                batch_out[Metric.kinematic_r2.value] = torch.tensor([0.])
            else:
                if phase == 'train':
                    batch_out[Metric.kinematic_r2.value] = self.train_r2_score(valid_bhvr, valid_tgt)
                    if batch_out[Metric.kinematic_r2.value].mean() < getattr(self.cfg, 'clip_r2_min', -10000):
                        # zero it out - this is a bug that occurs when the target has minimal variance (i.e. a dull batch with tiny batch size)
                        # Occurs only because we can't easily full batch R2, i.e. uninteresting.
                        batch_out[Metric.kinematic_r2.value].zero_()
                elif phase == 'val':
                    self.val_r2_score.update(valid_bhvr, valid_tgt)
                elif phase == 'eval':
                    self.eval_r2_score.update(valid_bhvr, valid_tgt)
        if Metric.kinematic_acc in self.cfg.metrics:
            acc = (bhvr.argmax(1) == self.quantize(bhvr_tgt))
            if not loss_mask.any():
                batch_out[Metric.kinematic_acc.value] = torch.zeros_like(acc).float().mean()
            else:
                batch_out[Metric.kinematic_acc.value] = acc[loss_mask].float().mean()
        if Metric.kinematic_mse in self.cfg.metrics:
            mse = F.mse_loss(self.simplify_logits_to_prediction(bhvr), bhvr_tgt, reduction='none')
            if not loss_mask.any():
                batch_out[Metric.kinematic_mse.value] = torch.zeros_like(mse).mean()
            else:
                mse_mask = mse[loss_mask].mean()
            batch_out[Metric.kinematic_mse.value] = mse_mask
        return batch_out

    def act(
        self,
        backbone_features: torch.Tensor,
        *args,
        **kwargs
    ) -> D.distribution.Distribution:
        r"""
            Satisfy agent interface.
            backbone_features: seqeunce of state embeddings from backbone, filtered for action prediction, as in NDT3 pretraining.
            - That is, for 6D actions, we produce 6 1D logit distributions, from 6 backbone features
        """
        logits = self.out(backbone_features)
        return D.categorical.Categorical(logits=logits)

class CovariateLinear(TaskPipeline):
    r"""
        Simple linear class for comparison with decoding.
        Old classes overloaded, this one copies CovariateInfill and linearizes.
        Only standard path tested.
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
        self.served_tokenized_covariates = data_attrs.tokenize_covariates
        self.served_semantic_covariates = data_attrs.semantic_covariates
        self.reference_cov = self.cfg.behavior_target.name
        self.cov_dims = data_attrs.behavior_dim
        self.initialize_readin(cfg.hidden_size)
        self.initialize_readout(cfg.hidden_size)
        assert self.served_tokenized_covariates, "Linear covariate readout only supports tokenized covariates"
        assert cfg.next_step_prediction, "Linear covariate readout only supports next step prediction"
        if Metric.kinematic_r2 or Metric.kinematic_r2_var in self.cfg.metrics:
            r"""
                Note on dimensionality.
                `cov_dims` does not need to be set unless using variance weighted, which also requires homogeneous dimensionality across training data.
                By default, all covariate dimensions are flattened, so r2 is given a single Nx1 vector.
                Variance-weighted R2 path will attempt to reconstruct the original NxK path, even K pred/truth pairs to compute R2.
                This will conflict if K != cov_dims - 1.
                Uniform r2 will broadcast the Nx1 to cov_dims-1, and return what is effectively the same r2 (average over K identical computations).
                A bit wasteful, but torchmetrics gives us accumulation over batches for free.
            """
            import torchmetrics
            multioutput = 'variance_weighted' if Metric.kinematic_r2_var in self.cfg.metrics else 'uniform_average'
            self.train_r2_score = torchmetrics.R2Score() # num_outputs=1) # No multioutput...
            self.val_r2_score = torchmetrics.R2Score(multioutput=multioutput) # num_outputs=self.cov_dims - 1, multioutput=multioutput) # -1 for padding
            self.eval_r2_score = torchmetrics.R2Score(multioutput=multioutput) # num_outputs=self.cov_dims - 1, multioutput=multioutput) # -1 for padding

    @property
    def handle(self):
        return 'covariate'

    def initialize_readin(self, backbone_size):
        self.inp = nn.Linear(1, backbone_size)

    def initialize_readout(self, backbone_size):
        self.out = nn.Sequential(
            nn.Linear(backbone_size, 1),
            Rearrange('... 1 -> ...') # squeeze out
        )

    def get_r2_metric(self, prefix: str):
        if prefix == 'train':
            return self.train_r2_score
        elif prefix == 'val':
            return self.val_r2_score
        elif prefix == 'eval':
            return self.eval_r2_score

    def encode_cov(self, covariate: torch.Tensor):
        # Note: covariate is _not_ foreseeably quantized at this point, we quantize herein during embed.
        r"""
            covariate: B T Bhvr_Dims. Bhvr_Dims=1 for most cases, ie any flat path
            returns:
                B T H
        """
        return self.inp(covariate) # B T Bhvr_Dims -> B T Bhvr_Dims H.

    def compute_loss(self, bhvr: torch.Tensor, bhvr_tgt: torch.Tensor):
        return F.mse_loss(bhvr, bhvr_tgt, reduction='none')

    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode = False):
        batch[f'{self.handle}_{DataKey.padding.name}'] = create_padding_simple(
            batch[self.cfg.behavior_target.name],
            batch.get(f'{self.handle}_{LENGTH_KEY}', None),
        )
        return batch

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        enc = self.encode_cov(batch[self.cfg.behavior_target.name])
        return (
            enc,
            batch[DataKey.covariate_time.name],
            batch[DataKey.covariate_space.name],
            batch[f'{self.handle}_{DataKey.padding.name}']
        )

    def predict(self, backbone_features: torch.Tensor, temperature=0.):
        return self.out(backbone_features)

    def forward(
        self,
        batch,
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor | None = None,
        backbone_space: torch.Tensor | None = None,
        backbone_padding: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None, # subset loss if provided
        compute_metrics=True,
        eval_mode=False,
        temperature=0.,
        phase: str = "train",
    ) -> Dict[BatchKey, torch.Tensor]:
        batch_out = {}
        bhvr: torch.Tensor = self.out(backbone_features)
        if Output.behavior_pred in self.cfg.outputs: # Note we need to eventually implement some kind of repack, just like we do for spikes
            batch_out[Output.behavior_pred] = bhvr
        if DataKey.bhvr_mask.name in batch:
            batch_out[Output.behavior_mask] = batch[DataKey.bhvr_mask.name]
        bhvr_tgt = batch[self.cfg.behavior_target.name].flatten()
        if Output.behavior in self.cfg.outputs:
            batch_out[Output.behavior] = bhvr_tgt # Flat aspect is not ideal, watch the timestamps..
        if not compute_metrics:
            return batch_out

        # Compute loss
        loss = self.compute_loss(bhvr, bhvr_tgt) # We don't _have_ to predict this for non-padding or at all if there's no targets, but it's convenient.
        # breakpoint()
        if loss_mask is not None:
            loss_mask = loss_mask & ~backbone_padding
        else:
            loss_mask = ~backbone_padding
        if DataKey.bhvr_mask in batch:
            loss_mask = loss_mask & batch[DataKey.bhvr_mask].flatten()
        elif DataKey.bhvr_mask.name in batch:
            loss_mask = loss_mask & batch[DataKey.bhvr_mask.name].flatten()

        if not loss_mask.any(): # ! This really shouldn't trigger and will cause unused parameters for DDP.
            # FALCON note - will trigger for M1 with long intertrial masked stretches
            # ! This shouldn't trigger because we take care to have a concluding kinematic padding token in each sequence in worst case - but it still seems to be triggering...?
            loss = (loss * 0).mean()
        else:
            loss = loss[loss_mask].mean()
        batch_out['loss'] = loss
        # breakpoint()
        if Metric.kinematic_r2 in self.cfg.metrics or Metric.kinematic_r2_var in self.cfg.metrics:
            r2_key = Metric.kinematic_r2 if Metric.kinematic_r2 in self.cfg.metrics else Metric.kinematic_r2_var
            valid_bhvr = bhvr[loss_mask].float().detach().cpu()
            valid_tgt = bhvr_tgt[loss_mask].float().detach().cpu()
            # print(f"Reporting r2 on {valid_tgt.shape} of {bhvr.shape} timesteps")

            # check for empty comparison
            if len(valid_bhvr) <= 1:
                batch_out[r2_key.value] = torch.tensor([0.])
            else:
                if phase == 'train':
                    batch_out[r2_key.value] = self.train_r2_score(valid_bhvr, valid_tgt)
                    if batch_out[r2_key.value].mean() < getattr(self.cfg, 'clip_r2_min', -10000):
                        # zero it out - this is a bug that occurs when the target has minimal variance (i.e. a dull batch with tiny batch size)
                        # Occurs only because we can't easily full batch R2, i.e. uninteresting.
                        batch_out[r2_key.value].zero_()
                elif phase == 'val':
                    if Metric.kinematic_r2_var in self.cfg.metrics: # only implemented here for falcon m1 parity...
                        unique_dims = len(batch[DataKey.covariate_labels.name][0]) # assuming consistent dims
                        dim_annotations = batch[DataKey.covariate_space.name].flatten()[loss_mask]
                        # assuming simple flattening, we just reshape
                        valid_bhvr = valid_bhvr.view(-1, unique_dims)
                        valid_tgt = valid_tgt.view(-1, unique_dims)
                        # print(f'Timepoints minibatch: {valid_bhvr.shape}')
                        # breakpoint()
                        dim_annotations = dim_annotations.view(-1, unique_dims)
                        assert dim_annotations.unique(dim=0).shape == (1, unique_dims), f"Failed dim consistency check: {dim_annotations.unique(dim=0)} {dim_annotations.shape} {unique_dims}"
                        og_device = self.val_r2_score.device

                        if self.val_r2_score.device != valid_bhvr.device:
                            self.val_r2_score = self.val_r2_score.to(valid_bhvr.device)
                        self.val_r2_score.update(valid_bhvr, valid_tgt)
                        self.val_r2_score.to(device=og_device)
                    else:
                        self.val_r2_score.update(valid_bhvr, valid_tgt)
                elif phase == 'eval':
                    self.eval_r2_score.update(valid_bhvr, valid_tgt)
        if Metric.kinematic_mse in self.cfg.metrics:
            mse = F.mse_loss(valid_bhvr, valid_tgt, reduction='none')
            if not loss_mask.any():
                batch_out[Metric.kinematic_mse.value] = torch.zeros_like(mse).mean()
            else:
                mse_mask = mse[loss_mask].mean()
            batch_out[Metric.kinematic_mse.value] = mse_mask
        return batch_out

    def act(
        self,
        backbone_features: torch.Tensor,
        *args,
        **kwargs
    ) -> D.distribution.Distribution:
        raise NotImplementedError("Linear readout does not support act atm.")

class CovariateProbe(CovariateLinear):
    r"""
        Mean pool linear probe for simpler decoding from models not pretrained with kin loss.
        Also more aligned with standard representation learning probing.
        Implemented while assuming neural inputs are only modality in input.
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
        self.max_trial_length = cfg.transformer.max_trial_length # Max timestep for padding
        del self.inp

    def update_batch(self, batch: Dict[BatchKey, torch.Tensor], eval_mode = False):
        return batch

    def get_context(self, batch: Dict[BatchKey, torch.Tensor], eval_mode=False):
        return [], [], [], []

    def predict(self, backbone_features: torch.Tensor, temperature=0.):
        return self.out(backbone_features)

    def forward(
        self,
        batch,
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor,
        backbone_space: torch.Tensor,
        backbone_padding: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None, # subset loss if provided
        compute_metrics=True,
        eval_mode=False,
        temperature=0.,
        phase: str = "train",
    ) -> Dict[BatchKey, torch.Tensor]:
        batch_out = {}
        backbone_features, backbone_padding = temporal_pool_direct(backbone_features, backbone_times, backbone_space, backbone_padding, pool='mean', pad_timestep=self.max_trial_length)
        bhvr: torch.Tensor = self.out(backbone_features)
        if Output.behavior_pred in self.cfg.outputs: # Note we need to eventually implement some kind of repack, just like we do for spikes
            batch_out[Output.behavior_pred] = bhvr
        if DataKey.bhvr_mask.name in batch:
            batch_out[Output.behavior_mask] = batch[DataKey.bhvr_mask.name]
        bhvr_tgt = batch[self.cfg.behavior_target.name].flatten(1) # Not flattened
        if Output.behavior in self.cfg.outputs:
            batch_out[Output.behavior] = bhvr_tgt # Flat aspect is not ideal, watch the timestamps..
        if not compute_metrics:
            return batch_out

        loss = self.compute_loss(bhvr, bhvr_tgt) # We don't _have_ to predict this for non-padding or at all if there's no targets, but it's convenient.
        if loss_mask is not None:
            loss_mask = loss_mask & ~backbone_padding
        else:
            loss_mask = ~backbone_padding
        if DataKey.bhvr_mask in batch:
            loss_mask = loss_mask & batch[DataKey.bhvr_mask].flatten()
        elif DataKey.bhvr_mask.name in batch:
            loss_mask = loss_mask & batch[DataKey.bhvr_mask.name].flatten()

        if not loss_mask.any(): # ! This really shouldn't trigger and will cause unused parameters for DDP.
            # FALCON note - will trigger for M1 with long intertrial masked stretches
            # ! This shouldn't trigger because we take care to have a concluding kinematic padding token in each sequence in worst case - but it still seems to be triggering...?
            loss = (loss * 0).mean()
        else:
            loss = loss[loss_mask].mean()
        batch_out['loss'] = loss
        if Metric.kinematic_r2 in self.cfg.metrics or Metric.kinematic_r2_var in self.cfg.metrics:
            r2_key = Metric.kinematic_r2 if Metric.kinematic_r2 in self.cfg.metrics else Metric.kinematic_r2_var
            valid_bhvr = bhvr[loss_mask].float().detach().cpu()
            valid_tgt = bhvr_tgt[loss_mask].float().detach().cpu()
            # print(f"Reporting r2 on {valid_tgt.shape} of {bhvr.shape} timesteps")

            # check for empty comparison
            if len(valid_bhvr) <= 1:
                batch_out[r2_key.value] = torch.tensor([0.])
            else:
                if phase == 'train':
                    batch_out[r2_key.value] = self.train_r2_score(valid_bhvr, valid_tgt)
                    if batch_out[r2_key.value].mean() < getattr(self.cfg, 'clip_r2_min', -10000):
                        # zero it out - this is a bug that occurs when the target has minimal variance (i.e. a dull batch with tiny batch size)
                        # Occurs only because we can't easily full batch R2, i.e. uninteresting.
                        batch_out[r2_key.value].zero_()
                elif phase == 'val':
                    if Metric.kinematic_r2_var in self.cfg.metrics: # only implemented here for falcon m1 parity...
                        unique_dims = len(batch[DataKey.covariate_labels.name][0]) # assuming consistent dims
                        dim_annotations = batch[DataKey.covariate_space.name][loss_mask]
                        # assuming simple flattening, we just reshape
                        valid_bhvr = valid_bhvr.view(-1, unique_dims)
                        valid_tgt = valid_tgt.view(-1, unique_dims)
                        dim_annotations = dim_annotations.view(-1, unique_dims)
                        assert dim_annotations.unique(dim=0).shape == (1, unique_dims), f"Failed dim consistency check: {dim_annotations.unique(dim=0)} {dim_annotations.shape} {unique_dims}"
                        og_device = self.val_r2_score.device

                        if self.val_r2_score.device != valid_bhvr.device:
                            self.val_r2_score = self.val_r2_score.to(valid_bhvr.device)
                        self.val_r2_score.update(valid_bhvr, valid_tgt)
                        self.val_r2_score.to(device=og_device)
                    else:
                        self.val_r2_score.update(valid_bhvr, valid_tgt)
                elif phase == 'eval':
                    self.eval_r2_score.update(valid_bhvr, valid_tgt)
        if Metric.kinematic_mse in self.cfg.metrics:
            mse = F.mse_loss(valid_bhvr, valid_tgt, reduction='none')
            if not loss_mask.any():
                batch_out[Metric.kinematic_mse.value] = torch.zeros_like(mse).mean()
            else:
                mse_mask = mse[loss_mask].mean()
            batch_out[Metric.kinematic_mse.value] = mse_mask
        return batch_out

class BehaviorSequenceDecoding(TaskPipeline):
    r"""
        Discrete classification for Seq2Seq (i.e. outputs don't match input length).
        In Seq2Seq, the entire input is seen before output is produced.
        Consequently there are no "behavior tokens;" only pooled neural tokens.
    """

    def __init__(
        self, backbone_out_size: int, channel_count: int, cfg: ModelConfig, data_attrs: DataAttrs,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        self.backbone_out_size = backbone_out_size
        self.time_pad = cfg.transformer.max_trial_length
        self.classes = self.cfg.decode_quantize_classes
        self.out = nn.Linear(backbone_out_size, self.classes)
        self.loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True) # handwriting

        self.edit_train = EditDistance()
        self.edit_val = EditDistance()
        self.edit_test = EditDistance()

    @property
    def handle(self):
        return 'covariate'

    # No token injection
    def get_metric(self, prefix: str):
        if prefix == 'train':
            return self.edit_train
        elif prefix == 'val':
            return self.edit_val
        elif prefix == 'eval':
            return self.edit_test

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        backbone_features: torch.Tensor,
        backbone_times: torch.Tensor | None = None,
        backbone_space: torch.Tensor | None = None,
        backbone_padding: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None, # subset loss if provided
        compute_metrics=True,
        eval_mode=False,
        temperature=0.,
        phase: str = "train",
    ) -> Dict[BatchKey, torch.Tensor]:
        batch_out = {}
        # Pool it down!
        temporal_padding_mask = create_padding_simple(backbone_features, batch[LENGTH_KEY])
        backbone_features, temporal_padding_mask = temporal_pool(batch, backbone_features, temporal_padding_mask, pool=self.cfg.decode_time_pool, pad_timestep=self.time_pad)
        bhvr = self.out(backbone_features).log_softmax(-1) # B T C

        pool_factor = batch[LENGTH_KEY].max() / bhvr.size(1) # equivalent to batch[DataKey.position.name].max(), but avoids risk of picking up padding on latter
        bhvr_timesteps = (batch[LENGTH_KEY] / pool_factor).int()
        if Output.behavior_pred in self.cfg.outputs:
            decodedSeqs = []
            for iterIdx in range(bhvr.shape[0]): # batchIdx
                decodedSeq = torch.argmax(
                    bhvr[iterIdx, 0 : bhvr_timesteps[iterIdx] , :],
                    dim=-1,
                )  # [num_seq,]
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq[decodedSeq != 0].cpu()
                decodedSeqs.append(decodedSeq)
            batch_out[Output.behavior_pred] = decodedSeqs
        if Output.behavior in self.cfg.outputs:
            tgt_lengths = batch[COVARIATE_LENGTH_KEY]
            batch_out[Output.behavior] = [tgt[:tgt_lengths[i]] for i, tgt in enumerate(batch[self.cfg.behavior_target.name])]
        if not compute_metrics:
            return batch_out
        # No special mask allowed, only lengths
        bhvr_tgt = batch[self.cfg.behavior_target.name]
        bhvr_tgt = rearrange(bhvr_tgt, 'b s 1 -> b s')
        loss = self.loss(rearrange(bhvr, 'b t c -> t b c'),
                        bhvr_tgt,
                        bhvr_timesteps,
                        batch[COVARIATE_LENGTH_KEY])
        batch_out['loss'] = loss
        if Metric.cer in self.cfg.metrics:
            # Convert to concrete predictions
            pred = bhvr
            adjustedLens = bhvr_timesteps
            # https://github.com/cffan/neural_seq_decoder/blob/master/src/neural_decoder/neural_decoder_trainer.py#L175
            decodedSeqs = []
            trueSeqs = []
            for iterIdx in range(bhvr.shape[0]): # batchIdx
                decodedSeq = torch.argmax(
                    pred[iterIdx, 0 : adjustedLens[iterIdx], :],
                    dim=-1,
                )  # [num_seq,]
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq[decodedSeq != 0].cpu()
                # decodedSeq = decodedSeq.cpu().detach().numpy()
                # decodedSeq = np.array([i for i in decodedSeq if i != 0])
                trueSeq = bhvr_tgt[iterIdx][0 : batch[COVARIATE_LENGTH_KEY][iterIdx]].cpu()

                decodedSeq = ''.join([chr(i) for i in decodedSeq])
                trueSeq = ''.join([chr(i) for i in trueSeq])
                decodedSeqs.append(decodedSeq)
                trueSeqs.append(trueSeq)
                # trueSeq = np.array(
                    # y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                # )

                # matcher = SequenceMatcher(
                #     a=trueSeq.tolist(), b=decodedSeq.tolist()
                # )
                # total_edit_distance += matcher.distance()
                # total_seq_length += len(trueSeq)
            if phase == 'train':
                pass
                # self.edit_train.update(decodedSeqs, trueSeqs) # Not logging.
            elif phase == 'val':
                self.edit_val.update(decodedSeqs, trueSeqs)
            elif phase == 'test':
                self.edit_test.update(decodedSeqs, trueSeqs)
        return batch_out


# === Utils ===
def simplify_logits_to_prediction(
        dequantize: Callable[[torch.Tensor], torch.Tensor],
        logits: torch.Tensor, # batch x class
        logit_dim=1,
        temperature=0.): # 0. -> argmax. Assumes batched, i.e. shape =
    # returns: batch
    if temperature > 0:
        logits = logits / temperature
        probabilities = torch.softmax(logits, dim=logit_dim) # TODO deprecate logit_dim
        choice = torch.multinomial(probabilities, 1)
    else:
        choice = logits.argmax(logit_dim)
    return dequantize(choice)

def create_padding_simple(
    reference: torch.Tensor,
    lengths: torch.Tensor | None,
): # Simplified for .compile
    if lengths is None:
        return torch.zeros(reference.size()[:2], device=reference.device, dtype=torch.bool)
    token_position = torch.arange(reference.size(1), device=reference.device)
    # token_position = rearrange(token_position, 't -> () t')
    return token_position.unsqueeze(0) >= lengths.unsqueeze(-1)

def create_token_padding_mask(
    reference: torch.Tensor | None,
    batch: Dict[BatchKey, torch.Tensor],
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

# These heads don't need to provide any updates - current the input is manually created and fed in
class VFunction(TaskPipeline):
    r"""
        For V-learning abstraction
        Takes state stream, produces evaluation
    """
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
        self.out = nn.Linear(backbone_out_size, 1)

    def forward(
        self,
        obs: torch.Tensor, # B T H (TODO give B H)
    ):
        r"""
            return: Same # of states, value evaluation
        """
        return self.out(obs)

class QFunction(TaskPipeline):
    r"""
        For Q-learning abstraction
        Takes state / action streams.
    """
    def __init__(self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
        bottleneck_factor: int = 1 # Learning seems to become unstable with larger bottleneck factors, which is bad for uninform ckpt selection (OTOH we may just be overfitting)
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        # TODO Flash it
        x_attn_layer = nn.TransformerDecoderLayer(
            d_model=backbone_out_size // bottleneck_factor,
            nhead=1,
            dim_feedforward=cfg.hidden_size // bottleneck_factor, # reduce capacity
            # dim_feedforward=cfg.hidden_size,
            dropout=cfg.dropout,
            batch_first=True,
            activation='gelu',
        )
        hidden = backbone_out_size // bottleneck_factor
        self.is_linear = cfg.rl.transition_building == "flatten"
        if self.is_linear:
            self.core = nn.Linear(backbone_out_size * 2, hidden)
        else:
            self.q_query = nn.Parameter(torch.randn(hidden) / np.sqrt(hidden))
            self.readin = nn.Linear(backbone_out_size, hidden)
            self.core = nn.TransformerDecoder(x_attn_layer, num_layers=1)
        self.out = nn.Linear(hidden, 1)
        self.pos_enc = nn.Parameter(torch.randn(cfg.max_spatial_position, hidden) / np.sqrt(hidden))

    def forward(
        self,
        obs: torch.Tensor, # (B) T H
        action: torch.Tensor, # (B) T K H
        action_padding: torch.Tensor | None = None, # B K, True if padding
        position: torch.Tensor | None = None,
    ):
        r"""
            return: Same # of outputs
        """
        action_padding = None
        in_shape = obs.shape
        if position is None:
            pos_encoding = self.pos_enc[:action.size(-2)]
        else:
            pos_encoding = self.pos_enc[position]
        action = action + pos_encoding
        transitions = torch.cat([
            rearrange(obs, '... h -> ... 1 h'),
            action
        ], dim=-2)
        if transitions.ndim == 4:
            transitions = rearrange(transitions, 'b t k h -> (b t) k h')

        if self.is_linear:
            # flatten k h
            transitions = rearrange(transitions, '... k h -> ... (k h)')
            q = self.core(transitions)
        else:
            transitions = self.readin(transitions)
            query = repeat(self.q_query, 'h -> b 1 h', b=transitions.shape[0])
            if action_padding is not None:
                memory_padding = torch.cat([
                    torch.zeros(action_padding.size(0), 1, dtype=bool, device=action_padding.device),
                    action_padding
                ], dim=-1)
            else:
                memory_padding = None
            q = self.core(query, transitions, memory_key_padding_mask=memory_padding).squeeze(-2) # remove time dimension (only one query)
        out = self.out(q)
        if len(in_shape) == 3:
            return rearrange(out, '(b t) 1-> b t 1', b=in_shape[0])
        return out

class QFunctionTarget(QFunction):
    r"""
        Polyak of QFunction
    """
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
        for p in self.parameters():
            p.requires_grad = False

    def soft_update_to(
        self,
        q_function: QFunction,
        polyak=1e-2, # default taken from iql trainer
    ):
        for p, p_target in zip(q_function.parameters(), self.parameters()):
            p_target.data.copy_(p_target.data * (1.0 - polyak) + p.data * polyak)

task_modules = {
    ModelTask.infill: SelfSupervisedInfill,
    ModelTask.shuffle_infill: ShuffleInfill,
    ModelTask.spike_context: SpikeContext,
    ModelTask.spike_infill: SpikeBase,
    ModelTask.perceiver_spike_context: PerceiverSpikeContext,
    ModelTask.next_step_prediction: NextStepPrediction,
    ModelTask.shuffle_next_step_prediction: ShuffleInfill, # yeahhhhh it's the SAME TASK WTH
    # ModelTask.shuffle_next_step_prediction: ShuffleNextStepPrediction,
    ModelTask.kinematic_decoding: BehaviorRegression,
    ModelTask.kinematic_classification: BehaviorClassification,
    ModelTask.kinematic_linear: CovariateLinear,
    ModelTask.kinematic_context: BehaviorContext, # Use classification route, mainly for tokenizing
    ModelTask.kinematic_infill: CovariateInfill,
    ModelTask.constraints: ConstraintPipeline,
    ModelTask.return_context: ReturnContext,
    ModelTask.return_infill: ReturnInfill,
    ModelTask.metadata_context: MetadataContext,
    ModelTask.v_function: VFunction,
    ModelTask.q_function: QFunction,
    ModelTask.q_function_target: QFunctionTarget,
    ModelTask.seq_decoding: BehaviorSequenceDecoding,
    ModelTask.kinematic_probe: CovariateProbe,
}
