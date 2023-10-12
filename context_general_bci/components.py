from typing import Optional, List, Any, Dict, Mapping
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import math
from einops import rearrange, pack, unpack, repeat, reduce
import logging
from functools import partial

from rotary_embedding_torch import RotaryEmbedding
from flash_attn.modules.block import Block
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import (
    GatedMlp,
    Mlp,
)
try:
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
except ImportError:
    dropout_add_layer_norm = None

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

r"""
    The following streamlined blocks are pulled from Flash Attn flash_attn.models.gpt, modified lightly
"""

def create_mixer_cls(config: TransformerConfig, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    head_dim = config.n_state // config.n_heads
    softmax_scale = head_dim ** (-0.5)
    if getattr(config, 'scale_attn_by_inverse_layer_idx', False):
        assert layer_idx is not None
        softmax_scale /= float(layer_idx + 1)
    dwconv = getattr(config, "attn_dwconv", False)
    qkv_proj_bias = out_proj_bias = config.use_attn_biases
    if config.rotary_position:
        rotary_emb_dim = head_dim
        rotary_emb_base = getattr(config, "rotary_emb_base", 10000.0)
        rotary_emb_scale_base = getattr(config, "rotary_emb_scale_base", None)
        rotary_emb_interleaved = getattr(config, "rotary_emb_interleaved", False)
    else:
        rotary_emb_dim = 0
        rotary_emb_base = 10000.0
        rotary_emb_scale_base = None
        rotary_emb_interleaved = False
    fused_bias_fc = getattr(config, "fused_bias_fc", False)
    mixer_cls = partial(
        MHA,
        num_heads=config.n_heads, # JY: Note to self -- Grouped MQA is available here
        qkv_proj_bias=qkv_proj_bias,
        out_proj_bias=out_proj_bias,
        dropout=config.dropout,
        softmax_scale=softmax_scale,
        causal=True,
        layer_idx=layer_idx,
        rotary_emb_dim=rotary_emb_dim,
        rotary_emb_base=rotary_emb_base,
        rotary_emb_scale_base=rotary_emb_scale_base,
        rotary_emb_interleaved=rotary_emb_interleaved,
        use_flash_attn=True,
        fused_bias_fc=fused_bias_fc,
        dwconv=dwconv,
        **factory_kwargs,
    )
    return mixer_cls

def create_mlp_cls(config: TransformerConfig, layer_idx=None, device=None, dtype=None):
    r"""
        vs. the one in flash_attn.models.gpt, we remove fused path
    """
    factory_kwargs = {"device": device, "dtype": dtype}
    mlp_fc1_bias = config.use_biases
    mlp_fc2_bias = config.use_biases
    assert config.activation in [
        "gelu",
        "gelu_new",
        "gelu_fast",
        "gelu_approx",
        "gelu_pytorch_tanh",
        "relu",
        "glu",
        "swiglu",
        "geglu",
    ]
    if config.activation in ["glu", "swiglu", "geglu"]:
        activation = (
            F.sigmoid
            if config.activation == "glu"
            else (F.silu if config.activation == "swiglu" else F.gelu)
        )
        mlp_cls = GatedMlp
        mlp_cls = partial(
            mlp_cls,
            hidden_features=int(config.n_state * config.feedforward_factor),
            activation=activation,
            bias1=mlp_fc1_bias,
            bias2=mlp_fc2_bias,
            **factory_kwargs,
        )
    else:
        if config.activation == "relu":
            activation = partial(F.relu, inplace=True)
        else:
            approximate = (
                "tanh"
                if config.activation
                in ["gelu_new", "gelu_fast", "gelu_approx", "gelu_pytorch_tanh"]
                else "none"
            )
            activation = partial(F.gelu, approximate=approximate)
        mlp_cls = Mlp
        mlp_cls = partial(
            mlp_cls,
            hidden_features=int(config.n_state * config.feedforward_factor),
            activation=activation,
            bias1=mlp_fc1_bias,
            bias2=mlp_fc2_bias,
            **factory_kwargs,
        )
    return mlp_cls

def create_block(config: TransformerConfig, layer_idx=None, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = create_mixer_cls(config, layer_idx, **factory_kwargs)
    mlp_cls = create_mlp_cls(config, layer_idx, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm,
        elementwise_affine=config.learnable_norm,
        bias=config.use_biases,
        **factory_kwargs,
    )
    # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
    residual_in_fp32 = getattr(config, "residual_in_fp32", False)
    block = Block(
        config.n_state,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        prenorm=config.pre_norm,
        resid_dropout1=config.dropout,
        resid_dropout2=config.dropout,
        fused_dropout_add_ln=getattr(config, "fused_dropout_add_ln", False),
        residual_in_fp32=residual_in_fp32,
        sequence_parallel=False,
        mark_shared_params=False,
    )
    block.layer_idx = layer_idx
    return block

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * n_layer))

def check_trues_at_end(tensor):
    cumsum = torch.cumsum(tensor, dim=1)
    return torch.all(cumsum == cumsum[:, -1].unsqueeze(1) * tensor)

class StreamlinedTransformer(nn.Module):
    r"""
        We follow FlashAttn's GPT example, swapping pieces out to support
        our explicit time + space embeddings (over flat sequences).

        Compared to SpaceTimeTransformer, this should add support for:
        - Rotary position encoding
        - SwiGLU
        - Removed biases/simplify norms
        - FlashAttn v2

        We remove the Model/Tensor/SequenceParallel optimizations from FlashAttn for simplicity.
    """
    @property
    def out_size(self):
        return self.cfg.n_state

    def __init__(
        self,
        config: TransformerConfig,
        max_spatial_tokens: int = 0,
        allow_embed_padding=True,
        device=None,
        dtype=None,
        process_group=None,
        **kwargs
        # Missing: process_group, device, dtype
    ):
        super().__init__()
        self.cfg = config
        logger.info(f"Streamlined path ignoring kwargs: {kwargs}")
        if not self.cfg.rotary_position: # hits inner mechanisms
            if allow_embed_padding:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length + 1, self.cfg.n_state, padding_idx=self.cfg.max_trial_length)
            else:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length, self.cfg.n_state)

        self.n_space = max_spatial_tokens if max_spatial_tokens else self.cfg.max_spatial_tokens
        if allow_embed_padding:
            self.space_encoder = nn.Embedding(self.n_space + 1, self.cfg.n_state, padding_idx=self.n_space)
        else:
            self.space_encoder = nn.Embedding(self.n_space, self.cfg.n_state)

        # Begin FlashAttn copy-path
        factory_kwargs = {"device": device, "dtype": dtype}
        assert not process_group, "TensorParallel not supported"
        self.sequence_parallel = getattr(config, "sequence_parallel", True)
        assert self.cfg.activation in [
            "gelu",
            "gelu_new",
            "gelu_fast",
            "gelu_approx",
            "gelu_pytorch_tanh",
            "relu",
            "sqrelu",
            "glu",
            "swiglu",
            "geglu",
        ]

        # TD [2022-07-30]: Force residual in fp32, seems to make fp16 training more stable
        self.residual_in_fp32 = getattr(config, "residual_in_fp32", False)
        self.prenorm = self.cfg.pre_norm

        # We change the order of dropout, residual and layer norm:
        # Instead of LN -> Attn / MLP -> Dropout -> Add, we do:
        # Dropout -> Add -> LN -> Attn / MLP, returning both the residual branch (output of Add) and
        # the main branch (output of MLP). The model definition is unchanged, but the mapping of the
        # nn.Dropout probabilities are changed.
        # This is for performance reason: we can fuse dropout + add + layer_norm.
        self.layers = nn.ModuleList(
            [
                create_block(config, layer_idx=i, **factory_kwargs)
                for i in range(self.cfg.n_layers)
            ]
        )

        self.fused_dropout_add_ln = getattr(config, "fused_dropout_add_ln", False)
        if self.fused_dropout_add_ln:
            if dropout_add_layer_norm is None:
                raise ImportError("dropout_layer_norm is not installed")
        if self.prenorm:
            self.drop_f = nn.Dropout(self.cfg.dropout)
            self.ln_f = nn.LayerNorm(
                self.cfg.n_state,
                elementwise_affine=self.cfg.learnable_norm,
                bias=self.cfg.use_biases,
                # **factory_kwargs
            )

        self.apply(
            partial(
                _init_weights,
                n_layer=self.cfg.n_layers,
                initializer_range=self.cfg.initializer_range,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        hidden_states, # (batch, seq_len, hidden)
        times: torch.Tensor, # for flat spacetime path, B x Token
        positions: torch.Tensor, # for flat spacetime path
        inference_params=None
    ):
        r"""
            Assumes autoregressive, causal mask.
            Assumes self-attention, not cross-attention.
            Assumes times and positions are provided
            Out: (batch, seq_len, hidden)
        """

        if not self.cfg.rotary_position:
            hidden_states = hidden_states + self.time_encoder(times)
        hidden_states = hidden_states + self.space_encoder(positions)

        # TODO use times for rotary path, use padding_mask

        residual = None
        mixer_kwargs = {}
        if inference_params is not None:
            mixer_kwargs["inference_params"] = inference_params
        for layer in self.layers:
            if self.prenorm:
                hidden_states, residual = layer(
                    hidden_states, residual, mixer_kwargs=mixer_kwargs
                )
            else:
                hidden_states = layer(hidden_states, mixer_kwargs=mixer_kwargs)
        if self.prenorm:
            if not self.fused_dropout_add_ln or dropout_add_layer_norm is None:
                dropped = self.drop_f(hidden_states)
                residual = (dropped + residual) if residual is not None else dropped
                hidden_states = self.ln_f(residual.to(dtype=self.ln_f.weight.dtype))
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = dropout_add_layer_norm(
                    hidden_states,
                    residual,
                    self.ln_f.weight,
                    self.ln_f.bias,
                    self.drop_f.p if self.training else 0.0,
                    self.ln_f.eps,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
        return hidden_states


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

        if getattr(self.cfg, 'rotary_position', False):
            raise NotImplementedError('Rotary position not supported for Pytorch native ')
        else:
            if allow_embed_padding:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length + 1, self.cfg.n_state, padding_idx=self.cfg.max_trial_length)
            else:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length, self.cfg.n_state)

        self.dropout_in = nn.Dropout(self.cfg.dropout)
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