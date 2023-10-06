from typing import Optional, List, Any, Dict, Mapping
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from einops import rearrange, pack, unpack, repeat, reduce
import logging

from context_general_bci.config import TransformerConfig, ModelConfig
from context_general_bci.dataset import DataAttrs, MetaKey

logger = logging.getLogger(__name__)

class FlippedDecoderLayer(nn.TransformerDecoderLayer):
    r"""
        We perform cross-attn then self-attn rather than self-attn then cross-attn.
        Intuition is that the initial self-attn in a non-sequential decode is useless (no information to change...)
        And these decoder layers are often thin in our experiments.
        So don't waste like 25 or 50% of the parameters.
    """
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, cfg: TransformerConfig, input_times: bool = False):
        super().__init__()
        self.input_times = input_times
        position = torch.arange(0, cfg.max_trial_length, dtype=torch.float).unsqueeze(1)
        self.learnable = cfg.learnable_position and not getattr(cfg, 'debug_force_nonlearned_position', False)
        # if self.learnable:
        #     self.register_buffer('pe', position.long())
        #     self.pos_embedding = nn.Embedding(cfg.max_trial_length, cfg.n_state)
        # else:
        pe = torch.zeros(cfg.max_trial_length, cfg.n_state)
        div_term = torch.exp(torch.arange(0, cfg.n_state, 2).float() * (-math.log(10000.0) / cfg.n_state))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) # t x 1 x d
        self.register_buffer('pe', pe)
        if self.learnable:
            self.pe = nn.Parameter(self.pe)

    def forward(self, x: torch.Tensor, batch_first=True):
        if self.input_times:
            pos_embed = self.pe[x].squeeze(2)
        else:
            pos_embed = self.pe[:x.size(1 if batch_first else 0), :]
            pos_embed = pos_embed.transpose(0, 1) if batch_first else pos_embed
        return pos_embed
        # return rearrange(pos_embed, 't b d -> b t 1 d' if batch_first else 't b d -> t b 1 d')

class SpaceTimeTransformer(nn.Module):
    r"""
        This model transforms temporal sequences of population arrays.
        - There's a spatial component. In early experiments, this was an array dimension.
            - This is still the input shape for now, but we'll likely refactor data to provide tokens.
            - i.e. data stream as <SUBJECT> <ARRAY1> <group 1> <group 2> ... <group N1> <ARRAY2> <group 1> <group 2> ... <group N2> ...
        - We will now refactor into a more generic space dimension.
    """
    def __init__(
        self,
        config: TransformerConfig,
        max_spatial_tokens: int = 0,
        # Several of these later parameters are here bc they are different in certain decode flows
        n_layers: int = 0, # override
        allow_embed_padding=False,
        debug_override_dropout_in=False,
        debug_override_dropout_out=False,
        context_integration='in_context',
        embed_space=True,
    ):
        super().__init__()
        self.cfg = config
        layer_cls = nn.TransformerEncoderLayer if context_integration == 'in_context' else FlippedDecoderLayer
        enc_cls = nn.TransformerEncoder if context_integration == 'in_context' else nn.TransformerDecoder
        self.cross_attn_enabled = context_integration == 'cross_attn'
        assert self.cfg.flat_encoder, "Nonflat (array-based) encoder deprecated"
        enc_layer = layer_cls(
            self.cfg.n_state,
            self.cfg.n_heads,
            dim_feedforward=int(self.cfg.n_state * self.cfg.feedforward_factor),
            dropout=self.cfg.dropout,
            batch_first=True,
            activation=self.cfg.activation,
            norm_first=self.cfg.pre_norm,
        )
        # Always on, for .compile
        # if self.cfg.pre_norm and self.cfg.final_norm: # Note, this would be equally accomplished with `norm=True` on the encoder.
            # self.final_norm = nn.LayerNorm(self.cfg.n_state) # per Kaiming's MAE for vision
        self.final_norm = nn.LayerNorm(self.cfg.n_state) # per Kaiming's MAE for vision
        n_layers = n_layers or self.cfg.n_layers
        if self.cfg.factorized_space_time:
            assert enc_cls == nn.TransformerEncoder, "Factorized space time only supported with encoder"
            assert not self.cfg.flat_encoder, "Flat encoder not supported with factorized space time"
            self.space_transformer_encoder = nn.TransformerEncoder(enc_layer, round(n_layers / 2))
            self.time_transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers - round(n_layers / 2))
        else:
            self.transformer_encoder = enc_cls(enc_layer, n_layers)
        # if not getattr(self.cfg, 'debug_force_nonlearned_position', False) and (self.cfg.flat_encoder or self.cfg.learnable_position):
        if allow_embed_padding:
            self.time_encoder = nn.Embedding(self.cfg.max_trial_length + 1, self.cfg.n_state, padding_idx=self.cfg.max_trial_length)
        else:
            self.time_encoder = nn.Embedding(self.cfg.max_trial_length, self.cfg.n_state)
        # else:
            # self.time_encoder = PositionalEncoding(self.cfg, input_times=self.cfg.transform_space)
        if debug_override_dropout_in:
            self.dropout_in = nn.Identity()
        else:
            self.dropout_in = nn.Dropout(self.cfg.dropout)
        if debug_override_dropout_out:
            self.dropout_out = nn.Identity()
        else:
            self.dropout_out = nn.Dropout(self.cfg.dropout)
        self.embed_space = embed_space
        if self.cfg.transform_space and self.embed_space:
            n_space = max_spatial_tokens if max_spatial_tokens else self.cfg.max_spatial_tokens
            self.n_space = n_space
            if allow_embed_padding:
                self.space_encoder = nn.Embedding(n_space + 1, self.cfg.n_state, padding_idx=n_space)
            else:
                self.space_encoder = nn.Embedding(n_space, self.cfg.n_state)

    @property
    def out_size(self):
        return self.cfg.n_state

    @staticmethod
    def generate_square_subsequent_mask_from_times(times: torch.Tensor, ref_times: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
            Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).

            times: B x Token

            out: B x T x T
        """
        if ref_times is None:
            ref_times = times
        return times[:, :, None] < ref_times[:, None, :]
        # return times[:, :, None] >= ref_times[:, None, :]
        # return torch.where(
        #     times[:, :, None] >= ref_times[:, None, :],
        #     0.0, float('-inf')
        # )

    def forward(
        self,
        src: torch.Tensor, # B T H, already embedded. (Flat spacetime path now asserted, can't deal with heterogeneity otherwise (need to implement hierarchy carefully again if so).)
        padding_mask: Optional[torch.Tensor] = None, # B T
        causal: bool=True,
        autoregressive: bool = False, # Only allow next step (disregards `times`) prediction; uses a triangular mask
        times: Optional[torch.Tensor] = None, # for flat spacetime path, B x Token
        positions: Optional[torch.Tensor] = None, # for flat spacetime path
        memory: Optional[torch.Tensor] = None, # memory as other context if needed for covariate decoder flow
        memory_times: Optional[torch.Tensor] = None, # solely for causal masking, not for re-embedding
        memory_padding_mask: Optional[torch.Tensor] = None,
        materialize_causal: bool = True, # For some reason, fastpath warns about materializing src at inference, but errors without materialized src on train. Bruh.
    ) -> torch.Tensor: # T B H
        r"""
            Each H is a token to be transformed with other T x A tokens.
            Additional T' and T tokens from context are included as well.

            We assume that the provided trial and temporal context is consistently shaped. i.e. any context provided is provided for all samples.
            (So attention masks do not vary across batch)
        """
        # breakpoint()
        # if times.max() > self.cfg.max_trial_length:
            # raise ValueError(f'Trial length {times.max()} exceeds max trial length {self.cfg.max_trial_length}')
        # if positions is not None and positions.max() > self.n_space:
            # raise ValueError(f'Space length {positions.max()} exceeds max space length {self.n_space}')
        # print(f'Debug: Time: {times.unique()} Space: {positions.unique()}')
        # print(f'Debug: Space: {positions.unique()}')
        # breakpoint()
        src = self.dropout_in(src)
        # === Embeddings ===
        src = src + self.time_encoder(times)
        if self.embed_space:
            src = src + self.space_encoder(positions)
        if not materialize_causal:
            assert False
            # https://github.com/pytorch/pytorch/issues/96941
            # ! Apparently is_causal is just a type hint and won't actually materialize the mask, so this is bad code to run.
            # i.e. the encoder call, with our materialized mask, succeeds, regardless of is_causal, and produces a different result than no mask, is_causal=True, which has undefined behavior.
            # * Annoyingly, pytorch casts the mask to float during checks and then whine about the mask being float ... we'll just have to live with nn.TransformerEncoderLayer warnings for now, unless we adopt SDPA directly
            # https://github.com/pytorch/pytorch/issues/97532
            src_mask = None
        elif autoregressive:
            src_mask = torch.triu(torch.ones(src.size(1), src.size(1)), diagonal=1).bool()
            src_mask = src_mask.to(src.device)
        elif causal:
            src_mask = SpaceTimeTransformer.generate_square_subsequent_mask_from_times(times)
            if src_mask.ndim == 3: # expand along heads
                src_mask = repeat(src_mask, 'b t1 t2 -> (b h) t1 t2', h=self.cfg.n_heads)
        else:
            src_mask = None

        # if padding_mask is None:
        #     padding_mask = torch.zeros(src.size()[:2], dtype=torch.bool, device=src.device)

        if self.cross_attn_enabled and memory is not None:
            if memory_times is None: # No mask needed for trial-context only, unless specified
                memory_mask = None
            else: # This is the covariate decode path
                memory_mask = SpaceTimeTransformer.generate_square_subsequent_mask_from_times(
                    times, memory_times
                )
                memory_mask = repeat(memory_mask, 'b t1 t2 -> (b h) t1 t2', h=self.cfg.n_heads)
            if padding_mask is not None:
                # ! Allow attention if full sequence is padding - no loss will be computed...
                padding_mask[padding_mask.all(1)] = False
            # breakpoint()
            output = self.transformer_encoder(
                src,
                memory,
                tgt_mask=src_mask,
                tgt_key_padding_mask=padding_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_padding_mask
            )
            if output.isnan().any():
                raise ValueError('NaN in output')
                breakpoint()
        else:
            # Flash attn, context manager is extra debug guard
            # with torch.backends.cuda.sdp_kernel(
            #     enable_flash=True,
            #     enable_math=False,
            #     enable_mem_efficient=False
            # ):
            output = self.transformer_encoder(
                src,
                src_mask,
                src_key_padding_mask=padding_mask, # should be none in flash/autoregressive path
                is_causal=causal, # Flash Attn hint (token causality, not time causality)
            )
        output = self.dropout_out(output)
        # if self.cfg.pre_norm and self.cfg.final_norm:
        output = self.final_norm(output) # Always on, for .compile
        return output