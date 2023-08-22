from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import abc
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange, repeat, reduce, pack, unpack # baby steps...
from einops.layers.torch import Rearrange
from sklearn.metrics import r2_score
import logging

logger = logging.getLogger(__name__)

from context_general_bci.config import (
    ModelConfig, ModelTask, Metric, Output, EmbedStrat, DataKey, MetaKey,
)

from context_general_bci.dataset import DataAttrs, LENGTH_KEY, CHANNEL_KEY, HELDOUT_CHANNEL_KEY, COVARIATE_LENGTH_KEY, COVARIATE_CHANNEL_KEY
from context_general_bci.contexts import context_registry, ContextInfo
from context_general_bci.subjects import subject_array_registry, SortedArrayInfo

from context_general_bci.components import SpaceTimeTransformer

# It's not obvious that augmentation will actually help - might hinder feature tracking, which is consistent
# through most of data collection (certainly good if we aggregate sensor/sessions)
SHUFFLE_KEY = "shuffle"

def logsumexp(x):
    c = x.max()
    return c + (x - c).exp().sum().log()

def apply_shuffle(item: torch.Tensor, shuffle: torch.Tensor):
    # item: B T *
    # shuffle: T
    return item.transpose(1, 0)[shuffle].transpose(1, 0)

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
    does_update_root = False
    unique_space = False # accept unique space as input?

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

    def get_masks(self, batch, channel_key=CHANNEL_KEY, length_key=LENGTH_KEY, ref: torch.Tensor | None = None, compute_channel=True):
        r"""
            length_key: token-level padding info
            channel_key: intra-token padding info
        """
        # loss_mask: b t *
        if ref is None:
            ref: torch.Tensor = batch[DataKey.spikes][..., 0]
        loss_mask = torch.ones(ref.size(), dtype=torch.bool, device=ref.device)

        length_mask = create_temporal_padding_mask(ref, batch, length_key=length_key)
        length_mask = ~(length_mask & torch.isnan(ref).any(-1))

        loss_mask = loss_mask & rearrange(length_mask, 'b t -> b t 1')

        if channel_key in batch and compute_channel: # only some of b x a x c are valid
            channels = batch[channel_key] # b x a of ints < c (or b x t)
            if channels.ndim == 1:
                channels = channels.unsqueeze(1)
            assert ref.ndim >= 3 # Channel dimension assumed as dim 2
            # Note no shuffling occurs here because 1. channel_key shuffle is done when needed earlier
            # 2. no spatial shuffling occurs so we do need to apply_shuffle(torch.arange(c))
            comparison = repeat(torch.arange(ref.size(2), device=ref.device), 'c -> 1 1 c')
            channel_mask = comparison < rearrange(channels, 'b t -> b t 1') # dim 2 is either arrays (base case) or tokens (flat)
            loss_mask = loss_mask & channel_mask
        else:
            loss_mask = loss_mask[..., 0] # don't specify channel dim if not used, saves HELDOUT case
            channel_mask = None
        return loss_mask, length_mask, channel_mask

    def get_context(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        r"""
            Context for covariates that should be embedded.
            (e.g. behavior, stimuli, ICMS)
            JY co-opting to also just track separate covariates that should possibly be reoncstructed (but main model doesn't know to do this atm, may need to signal.)
            returns:
            - a sequence of embedded tokens (B T H)
            - their associated timesteps. (B T)
            - their associated space steps (B T)
        """
        return None, None, None

    def get_trial_context(self, batch: Dict[str, torch.Tensor]):
        r"""
            For task specific trial _input_. (B H)
        """
        raise NotImplementedError # Nothing in main model to use this
        return []

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

    def forward(self, batch, backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        r"""
            By default only return outputs. (Typically used in inference)
            - compute_metrics: also return metrics.
            - eval_mode: Run IO in eval mode (e.g. no masking)
        """
        raise NotImplementedError

class RatePrediction(TaskPipeline):
    def __init__(
        self,
        backbone_out_size: int,
        channel_count: int,
        cfg: ModelConfig,
        data_attrs: DataAttrs,
        decoder: Optional[nn.Module] = None,
    ):
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs
        )
        if self.serve_tokens_flat:
            assert Metric.bps not in self.cfg.metrics, "bps metric not supported for flat tokens"
        if decoder is not None:
            self.out = decoder
        else:
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

class SelfSupervisedInfill(RatePrediction):
    does_update_root = True
    unique_space = True
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

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
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

class ShuffleInfill(RatePrediction):
    r"""
        Technical design decision note:
        - JY instinctively decided to split up inputs and just carry around split tensors rather than the splitting metadata.
        - This is somewhat useful in the end (rather than the unshuffling solution) as we can simply collect the masked crop
        - However the code is pretty dirty and this may eventually change

    """
    does_update_root = True

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
            context_integration=getattr(cfg.transformer, 'context_integration', 'in_context'),
            embed_space=cfg.transformer.embed_space,
        )
        self.max_spatial = data_attrs.max_spatial_tokens
        self.causal = cfg.causal
        # import pdb;pdb.set_trace()
        # ! TODO re-enable
        # self.out = RatePrediction.create_linear_head(cfg, cfg.hidden_size, cfg.neurons_per_token, layers=1 if self.cfg.linear_head else 2)
        self.out = RatePrediction.create_linear_head(cfg, cfg.hidden_size, cfg.neurons_per_token)
        if getattr(cfg, 'force_zero_mask', False):
            self.register_buffer('mask_token', torch.zeros(cfg.hidden_size))
        else:
            self.mask_token = nn.Parameter(torch.randn(cfg.hidden_size))

        # TODO JY delete?
        # * Redundant implementation with decode_separate: False path in HeldoutPrediction... scrap at some point
        self.joint_heldout = self.cfg.query_heldout and not ModelTask.heldout_decoding in self.cfg.tasks
        if self.joint_heldout:
            self.heldout_mask_token = nn.Parameter(torch.randn(cfg.hidden_size))
            self.query_readout = RatePrediction.create_linear_head(cfg, cfg.hidden_size, self.cfg.query_heldout)
        # import pdb;pdb.set_trace()

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode=False):
        return self.shuffle_crop_batch(self.cfg.mask_ratio, batch, eval_mode=eval_mode)

    @staticmethod
    def shuffle_crop_batch(mask_ratio: float, batch: Dict[str, torch.Tensor], eval_mode=False):
        r"""
            Shuffle inputs, keep only what we need for evaluation
        """
        spikes = batch[DataKey.spikes]
        target = spikes[..., 0]
        if eval_mode:
            # manipulate keys so that we predict for all steps regardless of masking status (definitely hacky)
            batch.update({
                SHUFFLE_KEY: torch.arange(spikes.size(1), device=spikes.device),
                'spike_target': target,
                'encoder_frac': spikes.size(1),
                # f'{DataKey.time}_target': batch[DataKey.time],
                # f'{DataKey.position}_target': batch[DataKey.position],
            })
            return batch
        # spikes: B T S H or B T H (no array support)
        # TODO (low-pri) also support spacetime shuffle
        shuffle = torch.randperm(spikes.size(1), device=spikes.device)
        encoder_frac = int((1 - mask_ratio) * spikes.size(1))
        # shuffle_spikes = spikes.gather(1, shuffle.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spikes.size(2), spikes.size(3)))
        for key in [DataKey.time, DataKey.position, CHANNEL_KEY]:
            if key in batch:
                shuffled = apply_shuffle(batch[key], shuffle)
                batch.update({
                    key: shuffled[:, :encoder_frac],
                    f'{key}_target': shuffled[:, encoder_frac:],
                })
        # import pdb;pdb.set_trace()
        batch.update({
            DataKey.spikes: apply_shuffle(spikes, shuffle)[:,:encoder_frac],
            'spike_target': apply_shuffle(target, shuffle)[:,encoder_frac:],
            'encoder_frac': encoder_frac,
            SHUFFLE_KEY: shuffle, # seems good to keep around...
        })
        return batch

    def get_masks(self, batch: Dict[str, torch.Tensor], loss: torch.Tensor):
        # get_masks
        loss_mask = torch.ones(loss.size(), device=loss.device, dtype=torch.bool)
        # note LENGTH_KEY and CHANNEL_KEY are for padding tracking
        # while DataKey.time and DataKey.position are for content
        if LENGTH_KEY in batch:
            token_position = rearrange(batch[SHUFFLE_KEY][batch['encoder_frac']:], 't -> () t')
            length_mask = token_position < rearrange(batch[LENGTH_KEY], 'b -> b ()')
            loss_mask = loss_mask & length_mask.unsqueeze(-1)
        if CHANNEL_KEY in batch:
            # CHANNEL_KEY padding tracking has already been shuffled
            # And within each token, we just have c channels to track, always in order
            comparison = repeat(torch.arange(loss.size(-1), device=loss.device), 'c -> 1 t c', t=loss.size(1)) # ! assuming flat - otherwise we need the space dimension as well.
            channel_mask = comparison < batch[f'{CHANNEL_KEY}_target'].unsqueeze(-1) # unsqueeze the channel dimension
            loss_mask = loss_mask & channel_mask
        return loss_mask

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        # B T
        batch_out = {}
        target = batch['spike_target']
        if target.ndim == 5:
            raise NotImplementedError("cannot even remember what this should look like")
            # decoder_mask_tokens = repeat(self.mask_token, 'h -> b t s h', b=target.size(0), t=target.size(1), s=target.size(2))
        else:
            if not eval_mode:
                decoder_mask_tokens = repeat(self.mask_token, 'h -> b t h', b=target.size(0), t=target.size(1))
                decoder_input = torch.cat([backbone_features, decoder_mask_tokens], dim=1)
                times = torch.cat([batch[DataKey.time], batch[f'{DataKey.time}_target']], 1)
                positions = torch.cat([batch[DataKey.position], batch[f'{DataKey.position}_target']], 1)
            else:
                decoder_input = backbone_features
                times = batch[DataKey.time]
                positions = batch[DataKey.position]
            if self.joint_heldout:
                time_seg = torch.arange(0, times.max()+1, device=times.device, dtype=times.dtype)
                decoder_input = torch.cat([
                    decoder_input,
                    repeat(self.heldout_mask_token, 'h -> b t h', b=target.size(0), t=time_seg.size(0)) # Assumes square
                ], 1)
                times = torch.cat([
                    times, repeat(time_seg, 't -> b t', b=times.size(0))
                ], 1)
                positions = torch.cat([
                    positions,
                    torch.full((positions.size(0), time_seg.size(0)), self.max_spatial - 1, device=positions.device, dtype=positions.dtype),
                ], 1)
            trial_context = []
            for key in ['session', 'subject', 'task']:
                if key in batch and batch[key] is not None:
                    ctx = batch[key] # unsup always gets fastpath
                    # ctx = batch[key].detach() if getattr(self.cfg, 'detach_decode_context') else batch[key]
                    trial_context.append(ctx)
            temporal_padding_mask = create_temporal_padding_mask(None, batch, truncate_shuffle=False)
            if self.joint_heldout:
                temporal_padding_mask = torch.cat([
                    temporal_padding_mask,
                    torch.full((temporal_padding_mask.size(0), time_seg.size(0)), False, device=temporal_padding_mask.device, dtype=temporal_padding_mask.dtype),
                ], 1)
            if DataKey.extra in batch:
                # Someone's querying (assuming `HeldoutPrediction`, integrate here)
                decoder_input = torch.cat([
                    decoder_input,
                    batch[DataKey.extra]
                ], 1)
                times = torch.cat([
                    times,
                    batch[DataKey.extra_time]
                ], 1)
                positions = torch.cat([
                    positions,
                    batch[DataKey.extra_position]
                ], 1)
                temporal_padding_mask = torch.cat([
                    temporal_padding_mask,
                    create_temporal_padding_mask(batch[DataKey.extra], batch, length_key=COVARIATE_LENGTH_KEY)
                ], 1)
            reps: torch.Tensor = self.decoder(
                decoder_input,
                trial_context=trial_context,
                temporal_padding_mask=temporal_padding_mask,
                space_padding_mask=None, # TODO implement (low pri)
                causal=self.causal,
                times=times,
                positions=positions
            )
            if self.joint_heldout:
                heldout_reps, reps = torch.split(reps, [time_seg.size(0), reps.size(1)-time_seg.size(0)], dim=1)
                heldout_rates = self.query_readout(heldout_reps)
            if DataKey.extra in batch:
                # remove extraneous
                reps, batch[DataKey.extra] = torch.split(reps, [reps.size(1)-batch[DataKey.extra].size(1), batch[DataKey.extra].size(1)], dim=1)
            if getattr(self.cfg, 'decode_use_shuffle_backbone', False):
                batch_out['update_features'] = reps
                batch_out['update_time'] = times
            reps = reps[:, -target.size(1):]
            rates = self.out(reps)
        if Output.logrates in self.cfg.outputs:
            # out is B T C, we want B T' C, and then to unshuffle
            if eval_mode:
                # we're doing a full query for qualitative eval
                unshuffled = rates
            else:
                all_tokens = torch.cat([
                    torch.full(batch[DataKey.spikes].size()[:-1], float('-inf'), device=rates.device),
                    rates
                ], dim=1)
                unshuffled = apply_shuffle(all_tokens, batch[SHUFFLE_KEY].argsort())
            batch_out[Output.logrates] = unshuffled  # unflattening occurs outside
        if Output.heldout_logrates in self.cfg.outputs and self.joint_heldout:
            batch_out[Output.heldout_logrates] = heldout_rates
        if not compute_metrics:
            return batch_out
        loss: torch.Tensor = self.loss(rates, target) # b t' c
        loss_mask = self.get_loss_mask(batch, loss) # shuffle specific
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        if self.joint_heldout:
            heldout_target = batch[DataKey.heldout_spikes][...,0]
            heldout_rates = heldout_rates[..., :heldout_target.size(-1)] # Crop to relevant max length
            heldout_loss = self.loss(heldout_rates, heldout_target)
            heldout_loss_mask, heldout_length_mask, heldout_channel_mask = self.get_masks(
                batch, ref=heldout_target,
                length_key=COVARIATE_LENGTH_KEY,
                channel_key=COVARIATE_CHANNEL_KEY
            )
            heldout_loss = heldout_loss[heldout_loss_mask]
            batch_out['loss'] = batch_out['loss'] + heldout_loss.mean()
            if Metric.co_bps in self.cfg.metrics:
                batch_out[Metric.co_bps] = self.bps(
                    heldout_rates.unsqueeze(-2), heldout_target.unsqueeze(-2),
                    length_mask=heldout_length_mask,
                    channel_mask=heldout_channel_mask
                )
            if Metric.block_co_bps in self.cfg.metrics:
                batch_out[Metric.block_co_bps] = self.bps(
                    heldout_rates.unsqueeze(-2), heldout_target.unsqueeze(-2),
                    length_mask=heldout_length_mask,
                    channel_mask=heldout_channel_mask,
                    block=True
                )
        return batch_out


class NextStepPrediction(RatePrediction):
    r"""
        One-step-ahead modeling prediction. Teacher-forced (we don't use force self-consistency, to save on computation)
        Note while pretraining necesarily should be causal (no way of preventing ctx bleed across layers)
        We can still use a semi-causal decoder (however much context we can afford).
    """
    does_update_root = True
    def __init__(self, backbone_out_size: int, channel_count: int, cfg: ModelConfig, data_attrs: DataAttrs, **kwargs):
        super().__init__(backbone_out_size, channel_count, cfg, data_attrs, **kwargs)
        self.start_token = nn.Parameter(torch.randn(cfg.hidden_size))
        assert not data_attrs.serve_tokens_flat, "not implemented, try ShuffleNextStepPrediction"

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
        batch: Dict[str, torch.Tensor],
        backbone_features: torch.Tensor,
        compute_metrics=True,
        eval_mode=False,
    ) -> torch.Tensor:
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
        if getattr(self.cfg, 'decode_tokenize_dims', False):
            # this logic is for covariate decode, not heldout neurons
            assert reference != DataKey.spikes, "Decode tokenization should not run for spikes, this is for covariate exps"
            self.decode_dims = data_attrs.behavior_dim
            self.cls_token = nn.Parameter(torch.randn(data_attrs.behavior_dim, cfg.hidden_size))
        else:
            self.cls_token = nn.Parameter(torch.randn(cfg.hidden_size))
        self.pad_value = data_attrs.pad_token
        self.max_space = data_attrs.max_spatial_tokens

    def inject(self, batch: Dict[str, torch.Tensor], in_place=False, injected_time: torch.Tensor | None = None, injected_space: torch.Tensor | None = None):
        # Implement injection
        # Assumption is that behavior time == spike time (i.e. if spike is packed, so is behavior), and there's no packing
        b, t = batch[self.reference].size()[:2]
        if injected_time is None:
            injected_time = torch.arange(t, device=batch[self.reference].device)
        if injected_space is None:
            injected_space = torch.arange(self.decode_dims, device=batch[self.reference].device)
        if self.cfg.decode_tokenize_dims:
            injected_tokens = repeat(self.cls_token, 'd h -> b (t d) h',
                b=b,
                t=t, # Time (not _token_, i.e. in spite of flat serving)
                d=self.decode_dims,
            )
            injected_time = repeat(injected_time, 't -> b (t d)', b=b, d=self.decode_dims)
            injected_space = repeat(injected_space, 'd -> b (t d)', b=b, t=t)
        else:
            injected_tokens = repeat(self.cls_token, 'h -> b t h',
                b=b,
                t=t, # Time (not _token_, i.e. in spite of flat serving)
            )
            injected_time = repeat(injected_time, 't -> b t', b=b)
            # TODO I no longer understand the below flow, really
            injected_space = torch.full(
                (b, t),
                self.max_space - 1, # ! For heldout prediction path, we want to inject a clearly distinguished space token from regular neural data space tokens
                # ! This means - 1. make sure `max_channels` is configured high enough that this heldout token is unique (since we will _not_ always add a mask token)
                # self.pad_value, # There is never more than one injected space token
                device=batch[self.reference].device
            )
        # I want to inject padding tokens for space so nothing actually gets added on that dimension
        if in_place:
            batch[DataKey.extra] = injected_tokens # B T H
            batch[DataKey.extra_time] = injected_time
            batch[DataKey.extra_position] = injected_space
        return injected_tokens, injected_time, injected_space

    def extract(self, batch: Dict[str, torch.Tensor], features: torch.Tensor) -> torch.Tensor:
        if DataKey.extra in batch: # in-place path assumed
            return batch[DataKey.extra]
        return features[:, -batch[self.reference].size(1):] # assuming queries stitched to back

class HeldoutPrediction(RatePrediction):
    r"""
        Regression for co-smoothing
    """
    def __init__(
        self, backbone_out_size: int, channel_count: int, cfg: ModelConfig, data_attrs: DataAttrs,
    ):
        self.spacetime = cfg.transform_space
        self.concatenating = cfg.transform_space and not data_attrs.serve_tokens
        if cfg.task.decode_strategy == EmbedStrat.project:
            if data_attrs.serve_tokens_flat:
                assert cfg.task.decode_time_pool == 'mean', "Only mean pooling is supported for flat serving"
                decoder = RatePrediction.create_linear_head(cfg, backbone_out_size, cfg.task.query_heldout, layers=1 if cfg.task.linear_head else 2)
            else:
                assert not data_attrs.serve_tokens, 'not implemented'
                if self.concatenating:
                    decoder = nn.Identity() # dummy
                else:
                    backbone_out_size = backbone_out_size * data_attrs.max_arrays
                decoder = None
        elif cfg.task.decode_strategy == EmbedStrat.token:
            # TODO vet this path (for spacetime), add the spacetime transformer decoder
            decoder = nn.Identity()
        super().__init__(
            backbone_out_size=backbone_out_size,
            channel_count=channel_count,
            cfg=cfg,
            data_attrs=data_attrs,
            decoder=decoder
        )
        if self.cfg.decode_strategy == EmbedStrat.token:
            assert self.spacetime, "Only spacetime transformer is supported for token decoding"
            self.time_pad = cfg.transformer.max_trial_length
            if self.cfg.decode_separate:
                self.decoder = SpaceTimeTransformer(
                    cfg.transformer,
                    max_spatial_tokens=0, # Assume pooling
                    n_layers=cfg.decoder_layers,
                    allow_embed_padding=True,
                    # TODO update these, deprecate allow_embed_padding
                    context_integration=getattr(cfg.transformer, 'context_integration', 'in_context'),
                    embed_space=cfg.transformer.embed_space,
                )
            self.injector = TemporalTokenInjector(cfg, data_attrs, DataKey.heldout_spikes, force_zero_mask=cfg.force_zero_mask)
            self.out = RatePrediction.create_linear_head(cfg, cfg.hidden_size, cfg.task.query_heldout)

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode = False):
        if self.cfg.decode_strategy != EmbedStrat.token or self.cfg.decode_separate:
            return batch
        self.injector.inject(batch, in_place=True)
        return batch

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        if self.cfg.decode_strategy == EmbedStrat.token:
            # Copied verbatim from BehaviorRegression
            # crop out injected tokens, -> B T H
            if self.cfg.decode_separate:
                # import pdb;pdb.set_trace()
                temporal_padding_mask = create_temporal_padding_mask(backbone_features, batch)
                if self.cfg.decode_time_pool: # B T H -> B T H
                    backbone_features, temporal_padding_mask = temporal_pool(batch, backbone_features, temporal_padding_mask, pool=self.cfg.decode_time_pool)
                    if Output.pooled_features in self.cfg.outputs:
                        batch_out[Output.pooled_features] = backbone_features.detach()
                decode_tokens, decode_time, decode_space = self.injector.inject(batch)
                if self.cfg.decode_time_pool:
                    src_time = decode_time
                    src_space = torch.zeros_like(decode_space) # space is now uninformative (we still have cls tokens in our injection to distinguish novel queries)
                    if backbone_features.size(1) < src_time.size(1):
                        # We pooled, but encoding doesn't have all timesteps. Pad to match
                        backbone_features = F.pad(backbone_features, (0, 0, 0, src_time.size(1) - backbone_features.size(1)), value=0)
                        temporal_padding_mask = F.pad(temporal_padding_mask, (0, src_time.size(1) - temporal_padding_mask.size(1)), value=True)
                else:
                    src_time = batch.get(DataKey.time, None)
                    if DataKey.position not in batch:
                        src_space = None
                    else:
                        src_space = torch.zeros_like(batch[DataKey.position])
                decoder_input = torch.cat([backbone_features, decode_tokens], dim=1)
                times = torch.cat([src_time, decode_time], dim=1)
                positions = torch.cat([src_space, torch.zeros_like(decode_space)], dim=1) # Kill any accidental padding, space is no-op here
                if temporal_padding_mask is not None:
                    extra_padding_mask = create_temporal_padding_mask(decode_tokens, batch, length_key=COVARIATE_LENGTH_KEY)
                    temporal_padding_mask = torch.cat([temporal_padding_mask, extra_padding_mask], dim=1)

                trial_context = []
                for key in ['session', 'subject', 'task']:
                    if key in batch and batch[key] is not None:
                        trial_context.append(batch[key].detach()) # Provide context, but hey, let's not make it easier for the model to steer the unsupervised-calibrated context
                backbone_features: torch.Tensor = self.decoder(
                    decoder_input,
                    temporal_padding_mask=temporal_padding_mask,
                    trial_context=trial_context,
                    times=times,
                    positions=positions,
                    space_padding_mask=None, # (low pri)
                    causal=False,
                )
            backbone_features = self.injector.extract(batch, backbone_features)
        elif self.cfg.decode_strategy == EmbedStrat.project and not self.concatenating:
            if self.spacetime:
                temporal_padding_mask = create_temporal_padding_mask(backbone_features, batch, truncate_shuffle=not self.cfg.decode_use_shuffle_backbone)
                if self.cfg.decode_time_pool and DataKey.time in batch:
                    backbone_features, temporal_padding_mask = temporal_pool(batch, backbone_features, temporal_padding_mask, pool=self.cfg.decode_time_pool)
            else:
                backbone_features = rearrange(backbone_features.clone(), 'b t a c -> b t (a c)')
        rates: torch.Tensor = self.out(backbone_features)
        batch_out = {}
        if Output.heldout_logrates in self.cfg.outputs:
            batch_out[Output.heldout_logrates] = rates
        if not compute_metrics:
            return batch_out
        spikes = batch[DataKey.heldout_spikes][..., 0]
        if self.cfg.query_heldout:
            rates = rates[..., :spikes.size(-1)]
        loss: torch.Tensor = self.loss(rates, spikes)
        # re-expand array dimension to match API expectation for array dim
        if self.cfg.query_heldout:
            loss_mask, length_mask, channel_mask = self.get_masks(
                batch, ref=spikes,
                length_key=COVARIATE_LENGTH_KEY,
                channel_key=COVARIATE_CHANNEL_KEY
            )
        else:
            loss = rearrange(loss, 'b t c -> b t 1 c')
            loss_mask, length_mask, channel_mask = self.get_masks(
                batch,
                compute_channel=False
            ) # channel_key expected to be no-op since we don't provide this mask
        loss = loss[loss_mask]
        batch_out['loss'] = loss.mean()
        if Metric.co_bps in self.cfg.metrics:
            batch_out[Metric.co_bps] = self.bps(
                rates.unsqueeze(-2), spikes.unsqueeze(-2),
                length_mask=length_mask,
                channel_mask=channel_mask
            )
        if Metric.block_co_bps in self.cfg.metrics:
            batch_out[Metric.block_co_bps] = self.bps(
                rates.unsqueeze(-2), spikes.unsqueeze(-2),
                length_mask=length_mask,
                channel_mask=channel_mask,
                block=True
            )

        return batch_out

class CovariateReadout(TaskPipeline):
    r"""
        Base class for decoding (regression/classification) of covariates.
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
        assert self.cfg.decode_strategy == EmbedStrat.token, 'Non-token decoding deprecated'
        self.decode_cross_attn = getattr(cfg, 'decoder_context_integration', 'in_context') == 'cross_attn'
        self.injector = TemporalTokenInjector(
            cfg,
            data_attrs,
            self.cfg.behavior_target if self.cfg.covariate_mask_ratio == 1.0 else 'covariate_target',
            force_zero_mask=self.decode_cross_attn and not getattr(self.cfg, 'decode_tokenize_dims', False)
        )
        self.time_pad = cfg.transformer.max_trial_length
        if self.cfg.decode_separate: # If we need additional cross-attention to decode. Not needed if mask tokens are procssed earlier.
            self.decoder = SpaceTimeTransformer(
                cfg.transformer,
                max_spatial_tokens=0,
                n_layers=cfg.decoder_layers,
                allow_embed_padding=True,
                context_integration=getattr(cfg, 'decoder_context_integration', 'in_context'),
                embed_space=not self.decode_cross_attn
            )
        self.cov_dims = data_attrs.behavior_dim

        self.causal = cfg.causal
        self.spacetime = cfg.transform_space
        assert self.spacetime, "Only spacetime transformer is supported for token decoding"
        self.bhvr_lag_bins = round(self.cfg.behavior_lag / data_attrs.bin_size_ms)
        assert self.bhvr_lag_bins >= 0, "behavior lag must be >= 0, code not thought through otherwise"

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

        if getattr(self.cfg, 'decode_normalizer', ''):
            # See `data_kin_global_stat`
            zscore_path = Path(self.cfg.decode_normalizer)
            assert zscore_path.exists(), f'normalizer path {zscore_path} does not exist'
            self.register_buffer('bhvr_mean', torch.load(zscore_path)['mean'])
            self.register_buffer('bhvr_std', torch.load(zscore_path)['std'])
        else:
            self.bhvr_mean = None
            self.bhvr_std = None
        self.initialize_readin(cfg.hidden_size)
        self.initialize_readout(cfg.hidden_size)

    @abc.abstractmethod
    def initialize_readin(self):
        raise NotImplementedError

    @abc.abstractmethod
    def initialize_readout(self):
        raise NotImplementedError

    def update_batch(self, batch: Dict[str, torch.Tensor], eval_mode = False):
        if self.cfg.covariate_mask_ratio < 1.0:
            batch = self.crop_batch(self.cfg.covariate_mask_ratio, batch, eval_mode=eval_mode) # Remove encode
        if self.cfg.decode_strategy != EmbedStrat.token or self.cfg.decode_separate:
            return batch
        logger.warning('Legacy injection path - should this be running?')
        breakpoint()
        self.injector.inject(batch, in_place=True)
        return batch

    def crop_batch(self, mask_ratio: float, batch: Dict[str, torch.Tensor], eval_mode=False):
        covariates = batch[self.cfg.behavior_target] # Assume B x T x Cov_Dims
        # i.e. this codeflow assumes that dataloader doesn't tokenize bhvr, and does some tokenizing logic in here
        target = covariates
        if eval_mode:
            batch.update({
                f'covariate_{SHUFFLE_KEY}': torch.arange(covariates.size(1), device=covariates.device),
                f'covariate_{DataKey.time}': torch.arange(covariates.size(1), device=covariates.device),
                'covariate_target': target,
                'covariate_encoder_frac': covariates.size(1)
            })
        # TODO this will be tricky if we are tokenizing the covariate H dimension (then flatten T x H to TH x 1)
        shuffle = torch.randperm(covariates.size(1), device=covariates.device)
        encoder_frac = int((1 - mask_ratio) * covariates.size(1))
        if f'covariate_{DataKey.time}' not in batch:
            batch[f'covariate_{DataKey.time}'] = torch.arange(covariates.size(1), device=covariates.device)
        if f'covariate_{DataKey.position}' not in batch:
            if self.cfg.decode_tokenize_dims:
                batch[f'covariate_{DataKey.position}'] = torch.arange(self.cov_dims, device=covariates.device)
            else:
                batch[f'covariate_{DataKey.position}'] = torch.zeros_like(batch[f'covariate_{DataKey.time}'])
        for key in [f'covariate_{DataKey.time}', f'covariate_{DataKey.position}']:
            if key in batch:
                shuffled = apply_shuffle(batch[key], shuffle)
                batch.update({
                    key: shuffled[:, :encoder_frac],
                    f'{key}_target': shuffled[:, encoder_frac:],
                })
        batch.update({
            self.cfg.behavior_target: apply_shuffle(covariates, shuffle)[:, :encoder_frac],
            'covariate_target': apply_shuffle(target, shuffle)[:, encoder_frac:],
            'covariate_encoder_frac': encoder_frac,
            f'covariate_{SHUFFLE_KEY}': shuffle,
        })
        return batch

    def get_context(self, batch: Dict[str, torch.Tensor]):
        if self.cfg.covariate_mask_ratio == 1.0:
            return None, None
        cov = self.encode_cov(batch[self.cfg.behavior_target])
        time = batch[f'covariate_{DataKey.time}']
        space = batch[f'covariate_{DataKey.position}']
        if self.cfg.decode_tokenize_dims:
            # JY isn't clear yet whether (T Bhvr_Dims) will be together or apart, so we try to support both
            if cov.ndim > 3:
                cov = rearrange(cov, 'b t bhvr_dim h -> b (t bhvr_dim) h')
                time = repeat(time, 'b t -> b (t bhvr_dim)', bhvr_dim=self.cov_dims)
                space = repeat(space, 'b t -> b (t bhvr_dim)', bhvr_dim=self.cov_dims)
        return cov, time, space

    @abc.abstractmethod
    def encode_cov(self, covariate: torch.Tensor) -> torch.Tensor: # B T Bhvr_Dims or possibly B (T Bhvr_Dims)
        raise NotImplementedError

    def get_cov_pred(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, eval_mode=False, batch_out={}) -> torch.Tensor:
        if self.cfg.decode_separate:
            temporal_padding_mask = create_temporal_padding_mask(backbone_features, batch)
            if self.cfg.decode_time_pool: # B T H -> B T H
                backbone_features, temporal_padding_mask = temporal_pool(batch, backbone_features, temporal_padding_mask, pool=self.cfg.decode_time_pool)
                if Output.pooled_features in self.cfg.outputs:
                    batch_out[Output.pooled_features] = backbone_features.detach()
            # * This "injection" step only really needs to happen if we've not already injected covariates as inputs; in that case, we need to retrieve previously injected info (see `crop_batch`)
            decode_tokens, decode_time, decode_space = self.injector.inject(
                batch,
                injected_time=batch[f'covariate_{DataKey.time}_target'] if self.cfg.covariate_mask_ratio < 1.0 else None,
                injected_space=batch[f'covariate_{DataKey.position}_target'] if self.cfg.covariate_mask_ratio < 1.0 else None
            )
            if not self.cfg.decode_tokenize_dims:
                # Clip decode space to 0, space isn't used for this decoder
                decode_space = torch.zeros_like(decode_space)

            # Re-extract src time and space. Only time is always needed to dictate attention for causality, but current implementation will re-embed time. JY doesn't want to asymmetrically re-embed only time, so space is retrieved. Really, we need larger refactor to just pass in time/space embeddings externally.
            if self.cfg.decode_time_pool:
                assert not self.cfg.covariate_mask_ratio < 1.0, "not implemented"
                assert not self.cfg.decode_tokenize_dims, 'time pool not implemented with tokenized dims'
                src_time = decode_time
                src_space = torch.zeros_like(decode_space)
                if backbone_features.size(1) < src_time.size(1):
                    # We want to pool, but encoding doesn't necessarily have all timesteps. Pad to match
                    backbone_features = F.pad(backbone_features, (0, 0, 0, src_time.size(1) - backbone_features.size(1)), value=0)
                    temporal_padding_mask = F.pad(temporal_padding_mask, (0, src_time.size(1) - temporal_padding_mask.size(1)), value=True)
            else:
                src_time = batch.get(DataKey.time, None)
                # Space only used in non-cross attn path. JY doesn't remember why I zero-ed this out, but skipping over for tokenize dim decode
                src_space = torch.zeros_like(batch[DataKey.position]) if DataKey.position in batch else None

            # allow looking N-bins of neural data into the "future"; we back-shift during the actual loss comparison.
            if self.causal and self.cfg.behavior_lag_lookahead:
                decode_time = decode_time + self.bhvr_lag_bins

            if self.decode_cross_attn:
                times = decode_time
                positions = decode_space
                other_kwargs = {
                    'memory': backbone_features,
                    'memory_times': batch.get(DataKey.time, None),
                    'memory_padding_mask': temporal_padding_mask,
                }
                if temporal_padding_mask is not None:
                    temporal_padding_mask = create_temporal_padding_mask(
                        decode_tokens,
                        batch,
                        length_key=COVARIATE_LENGTH_KEY,
                        duplicity=self.cov_dims if self.cfg.decode_tokenize_dims else 1
                    )
            else:
                assert not self.cfg.decode_tokenize_dims, 'non-cross attn not implemented with tokenized dims'
                decode_tokens = torch.cat([backbone_features, decode_tokens], dim=1)
                times = torch.cat([src_time, decode_time], dim=1)
                positions = torch.cat([src_space, decode_space], dim=1)
                if temporal_padding_mask is not None:
                    extra_padding_mask = create_temporal_padding_mask(decode_tokens, batch, length_key=COVARIATE_LENGTH_KEY)
                    temporal_padding_mask = torch.cat([temporal_padding_mask, extra_padding_mask], dim=1)
                other_kwargs = {}
            trial_context = []
            for key in ['session', 'subject', 'task']:
                if key in batch and batch[key] is not None:
                    trial_context.append(batch[key].detach()) # Provide context, but hey, let's not make it easier for the decoder to steer the unsupervised-calibrated context

            backbone_features: torch.Tensor = self.decoder(
                decode_tokens,
                temporal_padding_mask=temporal_padding_mask,
                trial_context=trial_context,
                times=times,
                positions=positions,
                space_padding_mask=None, # (low pri, only needed if space queries are heterogeneous. # TODO Needed for tokenized bhvr in NDT3)
                causal=self.causal,
                **other_kwargs
            )
        # crop out injected tokens, -> B T H
        if not self.decode_cross_attn:
            backbone_features = self.injector.extract(batch, backbone_features)
        return self.out(backbone_features)

    def get_target(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.cfg.covariate_mask_ratio == 1.0:
            tgt = batch[self.cfg.behavior_target]
        else:
            tgt = batch['covariate_target']
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

    def forward(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, compute_metrics=True, eval_mode=False) -> torch.Tensor:
        batch_out = {}
        bhvr = self.get_cov_pred(batch, backbone_features, eval_mode=eval_mode, batch_out=batch_out) # Comes out non-flat (B T D) or (B C T D)
        # At this point (computation and beyond) it is easiest to just restack tokenized targets, merge into regular API
        # The whole point of all this intervention is to test whether separate tokens affects perf (we hope not)

        if self.bhvr_lag_bins:
            bhvr = bhvr[..., :-self.bhvr_lag_bins, :]
            bhvr = F.pad(bhvr, (0, 0, self.bhvr_lag_bins, 0), value=0)

        if Output.behavior_pred in self.cfg.outputs: # Note we need to eventually implement some kind of repack, just like we do for spikes
            batch_out[Output.behavior_pred] = self.simplify_logits_to_prediction(bhvr)
            if self.bhvr_mean is not None:
                batch_out[Output.behavior_pred] = batch_out[Output.behavior_pred] * self.bhvr_std + self.bhvr_mean
        if Output.behavior in self.cfg.outputs:
            batch_out[Output.behavior] = batch[self.cfg.behavior_target]
        if not compute_metrics:
            return batch_out

        # Compute loss
        bhvr_tgt = self.get_target(batch)

        # TODO I'm highly certain that masks no longer work with cropped flow
        # Hm, how can I know whether a given length is illegitimate/just padding? Need this from shuffle...
        _, length_mask, _ = self.get_masks(
            batch, ref=bhvr_tgt,
            length_key=COVARIATE_LENGTH_KEY,
            compute_channel=False
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
                loss = loss[loss_mask].mean()

        r2_mask = length_mask

        batch_out['loss'] = loss
        if Metric.kinematic_r2 in self.cfg.metrics:
            valid_bhvr = bhvr[..., :bhvr_tgt.shape[-1]]
            valid_bhvr = self.simplify_logits_to_prediction(valid_bhvr)[r2_mask]
            valid_tgt = bhvr_tgt[r2_mask]
            # breakpoint()
            batch_out[Metric.kinematic_r2] = r2_score(valid_tgt.float().detach().cpu(), valid_bhvr.float().detach().cpu(), multioutput='raw_values')
            if batch_out[Metric.kinematic_r2].mean() < -10:
                batch_out[Metric.kinematic_r2] = np.zeros_like(batch_out[Metric.kinematic_r2])# .mean() # mute, some erratic result from near zero target
                # print(valid_bhvr.mean().cpu().item(), valid_tgt.mean().cpu().item(), batch_out[Metric.kinematic_r2].mean())
                # breakpoint()
            if Metric.kinematic_r2_thresh in self.cfg.metrics:
                valid_bhvr = valid_bhvr[valid_tgt.abs() > self.cfg.behavior_metric_thresh]
                valid_tgt = valid_tgt[valid_tgt.abs() > self.cfg.behavior_metric_thresh]
                batch_out[Metric.kinematic_r2_thresh] = r2_score(valid_tgt.float().detach().cpu(), valid_bhvr.float().detach().cpu(), multioutput='raw_values')
        if Metric.kinematic_acc in self.cfg.metrics:
            acc = (bhvr.argmax(1) == self.quantize(bhvr_tgt))
            batch_out[Metric.kinematic_acc] = acc[r2_mask].float().mean()
        return batch_out


class BehaviorRegression(CovariateReadout):
    r"""
        Because this is not intended to be a joint task, and backbone is expected to be tuned
        We will not make decoder fancy.
    """
    def initialize_readin(self):
        if self.cfg.decode_tokenize_dims: # NDT3 style
            self.inp = nn.Linear(1, self.cfg.hidden_size)
        else: # NDT2 style
            self.inp = nn.Linear(self.data_attrs.behavior_dim, self.cfg.hidden_size)

    def encode_cov(self, covariate: torch.Tensor):
        return self.inp(covariate)

    def initialize_readout(self, backbone_size):
        if getattr(self.cfg, 'decode_tokenize_dims', False):
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

    def get_cov_pred(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, eval_mode=False, batch_out={}) -> torch.Tensor:
        bhvr = super().get_cov_pred(batch, backbone_features, eval_mode, batch_out)
        if self.cfg.decode_tokenize_dims:
            bhvr = rearrange(bhvr, 'b (t d) 1 -> b t d', d=self.cov_dims)
        return bhvr

def symlog(x: torch.Tensor):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def unsymlog(x: torch.Tensor):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

class BehaviorClassification(CovariateReadout):
    r"""
        In preparation for NDT3.
        Assumes cross-attn, spacetime path.
        Cross-attention, autoregressive classification.
    """
    QUANTIZE_CLASSES = 128 # coarse classes are insufficient, starting relatively fine grained (assuming large pretraining)
    def initialize_readin(self, backbone_size): # Assume quantized readin...
        self.inp = nn.Embedding(self.QUANTIZE_CLASSES, backbone_size)

    def initialize_readout(self, backbone_size):
        if self.cfg.decode_tokenize_dims:
            self.out = nn.Linear(backbone_size, self.QUANTIZE_CLASSES)
        else:
            self.out = nn.Linear(backbone_size, self.QUANTIZE_CLASSES * self.cov_dims)
        self.register_buffer('zscore_quantize_buckets', torch.linspace(-2., 2., self.QUANTIZE_CLASSES + 1)) # quite safe for expected z-score range. +1 as these are boundaries, not centers
        assert self.spacetime, "BehaviorClassification requires spacetime path"
        assert self.cfg.decode_separate, "BehaviorClassification requires decode_separate"
        assert not self.cfg.behavior_lag, "BehaviorClassification does not support behavior_lag"

    def encode_cov(self, covariate: torch.Tensor):
        # Note: covariate is _not_ foreseeably quantized at this point, we quantize herein during embed.
        covariate = self.inp(self.quantize(covariate)) # B T Bhvr_Dims -> B T Bhvr_Dims H.
        if not self.cfg.decode_tokenize_dims:
            covariate = covariate.sum(-2) # B T Bhvr_Dims H -> B T H
        return covariate

    def quantize(self, x: torch.Tensor):
        x = torch.where(x != self.pad_value, x, 0)
        return torch.bucketize(symlog(x), self.zscore_quantize_buckets)

    def dequantize(self, quantized: torch.Tensor):
        if quantized.max() > self.zscore_quantize_buckets.shape[0]:
            raise Exception("go implement quantization clipping man")
        return unsymlog((self.zscore_quantize_buckets[quantized] + self.zscore_quantize_buckets[quantized + 1]) / 2)

    def get_cov_pred(self, batch: Dict[str, torch.Tensor], backbone_features: torch.Tensor, eval_mode=False, batch_out={}) -> torch.Tensor:
        bhvr = super().get_cov_pred(batch, backbone_features, eval_mode, batch_out)
        if self.cfg.decode_tokenize_dims:
            bhvr = rearrange(bhvr, 'b (t d) c -> b c t d', d=self.cov_dims)
        else:
            bhvr = rearrange(bhvr, 'b t (c d) -> b c t d', c=self.QUANTIZE_CLASSES)
        return bhvr

    def simplify_logits_to_prediction(self, bhvr: torch.Tensor):
        return self.dequantize(bhvr.argmax(1))

    def compute_loss(self, bhvr, bhvr_tgt):
        comp_bhvr = bhvr[...,:bhvr_tgt.shape[-1]]
        return F.cross_entropy(comp_bhvr, self.quantize(bhvr_tgt), reduction='none', label_smoothing=self.cfg.decode_label_smooth)

# === Utils ===

def create_temporal_padding_mask(
    reference: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    length_key: str = LENGTH_KEY,
    truncate_shuffle: bool = True,
    duplicity=1,
) -> torch.Tensor:
    r"""
        Identify which features are padding or not. TODO needs update if we support encoder output of behavior
        # temporal_padding refers to general length padding in `serve_tokens_flat` case
        True if padding
    """

    if length_key not in batch:
        return torch.zeros(reference.size()[:2], device=reference.device, dtype=torch.bool)
    if length_key == LENGTH_KEY and SHUFFLE_KEY in batch:
        token_position = batch[SHUFFLE_KEY]
        # If we had extra injected, the padd
        if truncate_shuffle:
            token_position = token_position[:batch['encoder_frac']]
    else:
        token_position = repeat(torch.arange(reference.size(1) // duplicity, device=reference.device), 't -> (t d)', d=duplicity)
    token_position = rearrange(token_position, 't -> () t')
    return token_position >= rearrange(batch[length_key], 'b -> b ()')

task_modules = {
    ModelTask.infill: SelfSupervisedInfill,
    ModelTask.shuffle_infill: ShuffleInfill,
    ModelTask.next_step_prediction: NextStepPrediction,
    ModelTask.shuffle_next_step_prediction: ShuffleInfill, # yeahhhhh it's the SAME TASK WTH
    # ModelTask.shuffle_next_step_prediction: ShuffleNextStepPrediction,
    ModelTask.heldout_decoding: HeldoutPrediction,
    ModelTask.kinematic_decoding: BehaviorRegression,
    ModelTask.kinematic_classification: BehaviorClassification,
}