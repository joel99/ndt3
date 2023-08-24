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


class ReadinMatrix(nn.Module):
    r"""
        Linear projection to transform input population to canonical (probably PC-related) input.
        Optional rank bottleneck (`readin_compress`)
    """
    def __init__(self, in_count: int, out_count: int, data_attrs: DataAttrs, cfg: ModelConfig):
        super().__init__()
        self.contexts = data_attrs.context.session # ! currently assuming no session overlap
        self.compress = cfg.readin_compress
        self.unique_readin = nn.Parameter(
            init.kaiming_uniform_(
                torch.empty(len(self.contexts), in_count, cfg.readin_dim if self.compress else out_count),
                a=math.sqrt(5)
            )
        )
        if self.compress:
            self.project = nn.Parameter(
                init.kaiming_uniform_(
                    torch.empty(cfg.readin_dim, out_count),
                )
            )


    def forward(self, state_in: torch.Tensor, batch: Dict[str, torch.Tensor], readin=True):
        r"""
            state_in: B T A H
        """
        # use session (in lieu of context) to index readin parameter (b x in x out)
        readin_matrix = torch.index_select(self.unique_readin, 0, batch[MetaKey.session])
        if readin:
            state_in = torch.einsum('btai,bih->btah', state_in, readin_matrix)
            if self.compress:
                state_in = torch.matmul(state_in, self.project)
        else: # readout
            readin_matrix = rearrange(readin_matrix, 'b i h -> b h i')
            if self.compress:
                state_in = torch.matmul(state_in, self.project.T) # b t a h x h readin
            state_in = torch.einsum('btah,bhi->btai', state_in, readin_matrix)
        return state_in

    def load_state_dict(self, transfer_state_dict: Mapping[str, Any], transfer_attrs: DataAttrs):
        state_dict = {}
        if self.compress:
            state_dict['project'] = transfer_state_dict['project']
        num_reassigned = 0
        current_state = self.state_dict()['unique_readin']
        for n_idx, target in enumerate(self.contexts):
            if target in transfer_attrs.context.session:
                s_idx = transfer_attrs.context.session.index(target)
                current_state[n_idx] = transfer_state_dict[f'unique_readin'][s_idx]
                num_reassigned += 1
        logger.info(f'Loaded {num_reassigned} of {len(self.contexts)} readin matrices.')
        state_dict['unique_readin'] = current_state # unnecessary?
        return super().load_state_dict(state_dict)

class ReadinCrossAttentionV2(nn.Module):
    r"""
        Motivation: Sorted datasets and DANDI high firing rate is not well integrated. Stitching sort of fixes, but this is expensive (many parameters)
        - We kind of just want to find an alternative to stitching that is not as expensive _per ctx_ because neural data inherently has many ctxs
        - e.g. if 1 session produces ~50 trials, this totally doesn't warrant 16K parameters (e.g. 40 sessions @ 128 channels to 128 hidden size = ~0.6M parameters which is larger than base NDT for ~2K trials),
        - And if 1 session produces 2K trials (as in Monkeys/native), well, we're much better in that case, but this is implausible for human tuning. (still potentially workable with multi-subject transfer but totally empirical)

        Issue with below; projecting each channel to a full value is expensive; using a full key is also unnecessary. (we can have a smaller key to negotiate this initial projection).

        Idea:
        - Still use cross attention with R learned queries ("R" for rank)
        - Use ctx embeddings to update queries.
        - Learn fixed position embeddings of size H (no need to make huge, at most R)
        - Make the value projection just a scalar refactor (for normalization).
        - Max memory consumption will be the B T A C H in Q @ K.
        - In the end we will assemble B T A R, and project into B T A H.
        TODO unclear why we should try this over e.g. just a bottlenecked (compressed) projection; but that's still ~128 x 128 = 16K params per ctx.
        - We can surely learn to identify the "high firing" channels vs the "low firing" channels in more efficient manners.
    """
    def __init__(self, in_count: int, out_count: int, data_attrs: DataAttrs, cfg: ModelConfig):
        raise NotImplementedError

    def forward(self, state_in: torch.Tensor, batch: Dict[str, torch.Tensor], readin=True):
        raise NotImplementedError

class ReadinCrossAttention(nn.Module):
    r"""
        A linear projection (`ReadinMatrix`) is disadvantaged in two ways:
        - couples value with relevance
        - requires a separate projection (~Channel x Hidden) for each context
            - extra parameters may be statistically inefficient (though computational footprint is negligible)

        We thus apply a cross-attention readin strategy, which outsources context-specific parameters to the context embeddings.
        Hopefully they have enough capacity. Also, this module has high memory costs (several GB) due to hidden-vector per channel in value computation
        # ! Actually, wayy too much memory. We actually go over 14G just scaling on Indy.
        # Also, initial testing shows little promise in small-scale Maze nlb.

        - individual channels get learned position embeddings
        - position embedding + context specific embedding (e.g. session embed) concat and project to
            - key
            - value
            - TODO add more context embeds (subject, array)
            - TODO decouple these context embeds with global step context embed
        - global query vectors of size H
            - TODO develop task-queries

        Readout strategies:
        - take backbone, pos embed and context embed, project to H
        - TODO should readout include task embed? (maybe as a global pre-transform, to be symmetric)
    """
    def __init__(self, in_count: int, out_count: int, data_attrs: DataAttrs, cfg: ModelConfig):
        super().__init__()
        self.query = nn.Parameter(torch.randn(cfg.readin_dim))
        self.channel_position_embeds = nn.Parameter(init.kaiming_uniform_(torch.empty(in_count, cfg.readin_dim), a=math.sqrt(5)))
        self.scale = math.sqrt(cfg.readin_dim)
        self.key_project = nn.Sequential(
            nn.Linear(cfg.readin_dim + cfg.session_embed_size, cfg.readin_dim),
            nn.GELU(),
            nn.Linear(cfg.readin_dim, cfg.readin_dim)
        )
        self.value_project = nn.Sequential(
            nn.Linear(1 + cfg.readin_dim + cfg.session_embed_size, cfg.readin_dim),
            nn.GELU(),
            nn.Linear(cfg.readin_dim, out_count)
        )

    def forward(self, state_in: torch.Tensor, session: torch.Tensor, subject: torch.Tensor, array: torch.Tensor):
        r"""
            state_in: B T A C
            session: B x H
            subject: B x H
            array: B x A x H
        """
        b, t, a, c = state_in.size()
        h = session.size(-1)
        r = self.channel_position_embeds.size(-1)
        keys = self.key_project(torch.cat([
            rearrange(self.channel_position_embeds, 'c r -> 1 c r').expand(b, c, r), # add batch dim
            rearrange(session, 'b h -> b 1 h').expand(b, c, h) # add input-channel dim
        ], dim=-1)) # b c r
        values = self.value_project(torch.cat([
            rearrange(self.channel_position_embeds, 'c r -> 1 1 1 c r').expand(b, t, a, c, r), # add B T A
            rearrange(session, 'b h -> b 1 1 1 h').expand(b, t, a, c, h), # add T A C dim
            state_in.unsqueeze(-1),
        ], dim=-1)) # b t a c h

        # perform cross attention
        scores = torch.einsum('bcr, r->bc', keys, self.query) / self.scale
        normalized_scores = F.softmax(scores, dim=-1)

        state_in = torch.einsum('bc, btach -> btah', normalized_scores, values) # b q c x b t a c h
        return state_in

class ContextualMLP(nn.Module):
    def __init__(self, in_count: int, out_count: int, cfg: ModelConfig):
        super().__init__()
        # self.channel_position_embeds = nn.Parameter(init.kaiming_uniform_(torch.empty(out_count, cfg.readin_dim), a=math.sqrt(5)))
        self.readout_project = nn.Sequential(
            nn.Linear(in_count + cfg.session_embed_size * cfg.session_embed_token_count, cfg.readin_dim),
            nn.GELU(),
            nn.Linear(cfg.readin_dim, out_count)
            # nn.Linear(cfg.readin_dim, cfg.readin_dim)
        )

    def forward(self, state_in: torch.Tensor, batch: Dict[str, torch.Tensor]):
        r"""
            state_in: B T A H
            session: B x H
            subject: B x H
            array: B x A x H

            out: B T A H (or C)
        """
        session_embed = rearrange(
            batch['session'], 'b h -> b 1 1 h' if batch['session'].ndim == 2 else 'b k h -> b 1 1 (k h)'
        )
        return self.readout_project(torch.cat([
        # queries = self.readout_project(torch.cat([
            state_in,
            session_embed.expand(*state_in.size()[:-1], -1),
        ], -1))
        # To show up in a given index, a query must have a high score against that index embed
        # return torch.einsum('btah, ch -> btac', queries, self.channel_position_embeds)



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
        if self.cfg.pre_norm and self.cfg.final_norm: # Note, this would be equally accomplished with `norm=True` on the encoder.
            self.final_norm = nn.LayerNorm(self.cfg.n_state) # per Kaiming's MAE for vision
        n_layers = n_layers or self.cfg.n_layers
        if self.cfg.factorized_space_time:
            assert enc_cls == nn.TransformerEncoder, "Factorized space time only supported with encoder"
            assert not self.cfg.flat_encoder, "Flat encoder not supported with factorized space time"
            self.space_transformer_encoder = nn.TransformerEncoder(enc_layer, round(n_layers / 2))
            self.time_transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers - round(n_layers / 2))
        else:
            self.transformer_encoder = enc_cls(enc_layer, n_layers)
        if not getattr(self.cfg, 'debug_force_nonlearned_position', False) and (self.cfg.flat_encoder or self.cfg.learnable_position):
            if allow_embed_padding:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length + 1, self.cfg.n_state, padding_idx=self.cfg.max_trial_length)
            else:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length, self.cfg.n_state)
        else:
            self.time_encoder = PositionalEncoding(self.cfg, input_times=self.cfg.transform_space)
        if debug_override_dropout_in:
            self.dropout_in = nn.Identity()
        else:
            self.dropout_in = nn.Dropout(self.cfg.dropout)
        if debug_override_dropout_out:
            self.dropout_out = nn.Identity()
        else:
            self.dropout_out = nn.Dropout(self.cfg.dropout)
        # And implement token level etc.
        # if self.cfg.fixup_init:
        #     self.fixup_initialization()
        self.embed_space = embed_space
        if self.cfg.transform_space and self.embed_space:
            n_space = max_spatial_tokens if max_spatial_tokens else self.cfg.max_spatial_tokens
            if allow_embed_padding:
                self.space_encoder = nn.Embedding(n_space + 1, self.cfg.n_state, padding_idx=n_space)
            else:
                self.space_encoder = nn.Embedding(n_space, self.cfg.n_state)

    def fixup_initialization(self):
        r"""
        http://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        temp_state_dic = {}
        en_layers = self.cfg.n_layers

        for l in self.encoder.layers:
            for name, param in l.named_parameters():
                if name in ["linear1.weight",
                            "linear2.weight",
                            "self_attn.out_proj.weight",
                            ]:
                    temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * param
                elif name in ["self_attn.v_proj.weight",]:
                    temp_state_dic[name] = (0.67 * (en_layers) ** (- 1. / 4.)) * (param * (2**0.5))
            l.load_state_dict(temp_state_dic, strict=False)

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
        return torch.where(
            times[:, :, None] >= ref_times[:, None, :],
            0.0, float('-inf')
        )

    # === Masks ===
    def make_src_mask(self, src: torch.Tensor, temporal_context: Optional[torch.Tensor], trial_context: Optional[torch.Tensor], times: torch.Tensor, t: int, s: int=1, causal=True):
        r"""
            args:
                temporal_context: b t temp_c h
                trial_context: b trial_c h

            Produces time major (T*S + TempCtx + TrialCtx, T*S + TempCtx + TrialCtx)
            Use t=1 to produce a space only mask, s=1 to produce a time only mask
        """
        if causal:
            src_mask = SpaceTimeTransformer.generate_square_subsequent_mask_from_times(times)
        else:
            src_mask = None
        # Update src mask for context. Note that row is attender, col is attended.
        # (For simplicity in construction)
        # Temporal Context is allowed to attend Trial acausally and self causally (if causal), but not to src
        # ? Why restrict? Well, we should test in acausal settings, but it's restricted so causal info doesn't bleed through it
        # Trial Context is allowed to attend to self acausally, but that's it.
        # Somewhat redundant code structure is to play nice with typing
        if temporal_context is not None: # introduce t * context_num tokens
            assert not self.cross_attn_enabled, "Temporal context not supported in cross attention"
            assert not self.cfg.flat_encoder, "Temporal context not supported in flat encoder"
            if src_mask is None:
                src_mask = torch.zeros((t * s, t * s), dtype=torch.float, device=src.device) # all attending
            # Since temporal context is expected to be used in a causal cases (ICMS)
            # We provide causal masks; technically there may be a case where spikes should attend all temporal context but can only be achieved indirectly in this setup.
            temporal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=src.device)
            context_num = temporal_context.size(-2)
            temporal_mask = rearrange(
                repeat(temporal_mask, 't1 t2 -> t1 t2 c1 c2', c1=s + context_num, c2=context_num),
                't1 t2 c1 c2 -> (t1 c1) (t2 c2)'
            )
            src_mask = F.pad(src_mask, (0, 0, 0, t * context_num), value=float('-inf'))
            src_mask = torch.cat([src_mask, temporal_mask], dim=1)
        if trial_context is not None and not self.cross_attn_enabled:
            if src_mask is None:
                src_mask = torch.zeros((t * s, t * s), dtype=torch.float, device=src.device) # all attending
            src_mask = F.pad(src_mask, (0, 0, 0, trial_context.size(1)), value=float('-inf'))
            src_mask = F.pad(src_mask, (0, trial_context.size(1)), value=0)

        if src_mask is not None and src_mask.ndim == 3: # expand along heads
            src_mask = repeat(src_mask, 'b t1 t2 -> (b h) t1 t2', h=self.cfg.n_heads)
        return src_mask

    def forward(
        self,
        src: torch.Tensor, # B T H, already embedded. (Flat spacetime path now asserted, can't deal with heterogeneity otherwise (need to implement hierarchy carefully again if so).)
        trial_context: torch.Tensor | List[torch.Tensor] = [], # T' [B H]
        padding_mask: Optional[torch.Tensor] = None, # B T
        temporal_context: List[torch.Tensor] = [], # cov_types [B TC H]
        temporal_times: List[torch.Tensor] = [], # cov_types [B TC]
        causal: bool=True,
        times: Optional[torch.Tensor] = None, # for flat spacetime path, B x Token
        positions: Optional[torch.Tensor] = None, # for flat spacetime path
        memory: Optional[torch.Tensor] = None, # memory as other context if needed for covariate decoder flow
        memory_times: Optional[torch.Tensor] = None, # solely for causal masking, not for re-embedding
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor: # T B H
        r"""
            Each H is a token to be transformed with other T x A tokens.
            Additional T' and T tokens from context are included as well.

            We assume that the provided trial and temporal context is consistently shaped. i.e. any context provided is provided for all samples.
            (So attention masks do not vary across batch)
        """
        # import pdb;pdb.set_trace()
        # breakpoint()
        src = self.dropout_in(src)
        # === Embeddings ===
        src = src + self.time_encoder(times)
        if self.embed_space:
            src = src + self.space_encoder(positions)
        b, t, s = src.size(0), src.size(1), 1 # it's implied that codepaths get that "space" is not really 1, but it's not used

        if len(temporal_context) > 0:
            assert False, "codepath abandoned, requisite covariate embed implemented in `model.py`"
            # TODO fix this...
            temporal_context_stack, temporal_context_pieces = pack(temporal_context, 'b * h')
            temporal_times_stack, temporal_times_pieces = pack(temporal_times, 'b *')
        else:
            temporal_context = None

        if len(trial_context) > 0:
            trial_context, _ = pack(trial_context, 'b * h')
        else:
            trial_context = None
        # === Transform ===
        contextualized_src = [src]
        if not self.cross_attn_enabled: # If cross attn is enabled, context goes into memory. Otherwise, it's in src.
            if temporal_context:
                contextualized_src.append(temporal_context)
            if trial_context is not None:
                contextualized_src.append(trial_context)
        contextualized_src, ps = pack(contextualized_src, 'b * h') # b [(t a) + (t n) + t'] h
        src_mask = self.make_src_mask(src, temporal_context, trial_context, times, t, s, causal=causal) # TODO

        if padding_mask is None:
            padding_mask = torch.zeros((b, t), dtype=torch.bool, device=src.device)

        # Trial context is never padded
        if trial_context is not None and not self.cross_attn_enabled:
            padding_mask = F.pad(padding_mask, (0, trial_context.size(-2)), value=False)

        # TODO temporal context padding

        # import pdb;pdb.set_trace()
        # print(t, s, contextualized_src.size(), src_mask.size(), padding_mask.size())
        # ! Bug patch note
        # There's an edge case that nans out outputs. It occurs when a padding token can't attend to anything.
        # This occurs specifically due to confluence of:
        # 1. padding specified by `src_key_padding_mask` can't be attended to (no self attending)
        # 2. we use shuffle on heterogeneous datasets. So we may get datapoints that have no real data in given timesteps.
        # 3. pad value set to 0, so padding gets marked at time 0
        # 4. all timestep 0 tokens are padding (no other tokens in sequence that can be attended to)
        # Suggested fix: pad value set to something nonzero. (IDR why we didn't set that in the first place, I think concerns about attending to sub-chunk padding?)
        # import pdb;pdb.set_trace()
        if self.cross_attn_enabled:
            # cross_ctx = [i for i in [memory, trial_context] if i is not None] # verbose for torchscript
            cross_ctx = []
            if memory is not None:
                cross_ctx.append(memory)
            if trial_context is not None:
                cross_ctx.append(trial_context)
            memory = torch.cat(cross_ctx, dim=1)
            if memory_times is None: # No mask needed for trial-context only, unless specified
                memory_mask = None
            else: # This is the covariate decode path
                memory_mask = SpaceTimeTransformer.generate_square_subsequent_mask_from_times(
                    times, memory_times
                )
                if trial_context is not None:
                    memory_mask = torch.cat([memory_mask, torch.zeros(
                        (memory_mask.size(0), contextualized_src.size(1), trial_context.size(1)),
                        dtype=torch.float, device=memory_mask.device
                    )], dim=-1)
                if temporal_context is not None: # P sure last dim is the attended to dim
                    breakpoint()
                    memory_mask = F.pad(memory_mask, (0, temporal_context.size(-2)), value=float('-inf')) # ! No! We want to be able to attend to this, precisely using temporal_times
                memory_mask = repeat(memory_mask, 'b t1 t2 -> (b h) t1 t2', h=self.cfg.n_heads)
                if memory_padding_mask is not None:
                    if trial_context is not None:
                        memory_padding_mask = F.pad(memory_padding_mask, (0, trial_context.size(1)), value=False)
                    if temporal_context is not None:
                        memory_padding_mask = F.pad(memory_padding_mask, (0, temporal_context.size(1)), value=False)

            # convert padding masks to float to suppress torch 2 warnings
            if torch.__version__.startswith('2.0'): # Need float for 2.0 and higher
                if padding_mask is not None:
                    padding_mask = torch.where(padding_mask, float('-inf'), 0.0)
                if memory_padding_mask is not None:
                    memory_padding_mask = torch.where(memory_padding_mask, float('-inf'), 0.0)
            output = self.transformer_encoder(
                contextualized_src,
                memory,
                tgt_mask=src_mask,
                tgt_key_padding_mask=padding_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_padding_mask
            )
        else:
            if padding_mask is not None:
                if torch.__version__.startswith('2.0'): # Need float for 2.0 and higher
                    padding_mask = torch.where(padding_mask, float('-inf'), 0.0)
            output = self.transformer_encoder(
                contextualized_src,
                src_mask,
                src_key_padding_mask=padding_mask
            )
        output, *_ = unpack(output, ps, 'b * h')

        output = self.dropout_out(output)
        if self.cfg.pre_norm and self.cfg.final_norm:
            output = self.final_norm(output)
        return output



r"""
    Torchscript barebones below
"""



class PositionalEncodingScript(nn.Module):
    # lock in batch first
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

    def forward(self, x: torch.Tensor):
        if self.input_times:
            pos_embed = self.pe[x].squeeze(2)
        else:
            pos_embed = self.pe[:x.size(1), :]
            pos_embed = pos_embed.transpose(0, 1)
        return pos_embed
        # return rearrange(pos_embed, 't b d -> b t 1 d' if batch_first else 't b d -> t b 1 d')


class SpaceTimeTransformerDecoderScript(nn.Module):
    r"""
        Thin spacetime copy for scripting.
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

        # verbose config for torchscript
        self.n_state = self.cfg.n_state
        self.n_heads = self.cfg.n_heads
        self.pre_norm = self.cfg.pre_norm
        self.has_final_norm = self.cfg.final_norm

        layer_cls = FlippedDecoderLayer
        enc_cls = nn.TransformerDecoder

        enc_layer = layer_cls(
            self.cfg.n_state,
            self.cfg.n_heads,
            dim_feedforward=int(self.cfg.n_state * self.cfg.feedforward_factor),
            dropout=self.cfg.dropout,
            batch_first=True,
            activation=self.cfg.activation,
            norm_first=self.cfg.pre_norm,
        )
        if self.pre_norm and self.final_norm: # Note, this would be equally accomplished with `norm=True` on the encoder.
            self.final_norm = nn.LayerNorm(self.cfg.n_state) # per Kaiming's MAE for vision

        n_layers = n_layers or self.cfg.n_layers
        self.transformer_encoder = enc_cls(enc_layer, n_layers)
        if not getattr(self.cfg, 'debug_force_nonlearned_position', False) and (self.cfg.flat_encoder or self.cfg.learnable_position):
            if allow_embed_padding:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length + 1, self.cfg.n_state, padding_idx=self.cfg.max_trial_length)
            else:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length, self.cfg.n_state)
        else:
            self.time_encoder = PositionalEncodingScript(self.cfg, input_times=self.cfg.transform_space)

    @property
    def out_size(self):
        # verbose for torchscript
        return self.n_state
        # return self.cfg.n_state

    # === Masks ===
    def make_src_mask(self, src: torch.Tensor, trial_context: torch.Tensor, times: torch.Tensor, t: int, s: int=1, causal: bool=True):
        r"""
            args:
                temporal_context: b t temp_c h
                trial_context: b trial_c h

            Produces time major (T*S + TempCtx + TrialCtx, T*S + TempCtx + TrialCtx)
            Use t=1 to produce a space only mask, s=1 to produce a time only mask
        """
        if not causal:
            return None
        src_mask = generate_square_subsequent_mask_from_times(times)
        return src_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).view(-1, src_mask.size(1), src_mask.size(2))

    def forward(
        self,
        src: torch.Tensor, # B T A H - embedded already (or possibly B T A S_A H), or B Token H
        times: torch.Tensor, # for flat spacetime path, B x Token
        trial_context: torch.Tensor, # B T' H
        # trial_context: List[torch.Tensor], # T' [B H]
        memory: torch.Tensor, # memory as other context if needed for covariate decoder flow
        memory_times: Optional[torch.Tensor] = None, # solely for causal masking, not for re-embedding
        padding_mask: Optional[torch.Tensor] = None, # B T
        memory_padding_mask: Optional[torch.Tensor] = None,
        causal: bool=True,
    ) -> torch.Tensor: # T B H
        r"""
            Simplified to assume `flat_encoder` and not `factorized_space_time`.
        """
        # import pdb;pdb.set_trace()
        # Embeddings
        src = src + self.time_encoder(times)
        b, t, s = src.size(0), src.size(1), 1 # it's implied that codepaths get that "space" is not really 1, but it's not used

        src_mask = self.make_src_mask(src, trial_context, times, t, s, causal=causal)

        if padding_mask is None:
            padding_mask = torch.zeros((b, t), dtype=torch.bool, device=src.device)

        # Trial context is never padded
        memory = torch.cat([memory, trial_context], dim=1)
        memory_mask = generate_square_subsequent_mask_from_times(
            times, memory_times
        )
        memory_mask = torch.cat([memory_mask, torch.zeros(
            (memory_mask.size(0), src.size(1), trial_context.size(1)),
            dtype=torch.float, device=memory_mask.device
        )], dim=-1)
        memory_mask = memory_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).view(-1, memory_mask.size(1), memory_mask.size(2))
        # memory_mask = repeat(memory_mask, 'b t1 t2 -> (b h) t1 t2', h=self.cfg.n_heads)
        # memory_padding_mask = torch.cat([
        #     memory_padding_mask,
        #     torch.zeros((memory_padding_mask.size(0), trial_context.size(1)), dtype=torch.bool, device=memory_padding_mask.device)
        # ], 1)
        output = self.transformer_encoder(
            src,
            memory,
            tgt_mask=src_mask,
            tgt_key_padding_mask=padding_mask,
            memory_mask=memory_mask,
            # memory_key_padding_mask=memory_padding_mask
        )[:, : t * s]

        if self.pre_norm and self.final_norm:
            output = self.final_norm(output)
        return output

class SpaceTimeTransformerEncoderScript(nn.Module):
    r"""
        Thin spacetime copy for scripting.
        Many breaking diffs, note there's no pre_norm/final norm here
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

        # verbose config for torchscript
        self.n_state = self.cfg.n_state
        self.n_heads = self.cfg.n_heads
        self.pre_norm = self.cfg.pre_norm
        self.has_final_norm = self.cfg.final_norm

        layer_cls = nn.TransformerEncoderLayer
        enc_cls = nn.TransformerEncoder

        enc_layer = layer_cls(
            self.cfg.n_state,
            self.cfg.n_heads,
            dim_feedforward=int(self.cfg.n_state * self.cfg.feedforward_factor),
            dropout=self.cfg.dropout,
            batch_first=True,
            activation=self.cfg.activation,
            norm_first=self.cfg.pre_norm,
        )

        n_layers = n_layers or self.cfg.n_layers
        if self.cfg.factorized_space_time:
            assert enc_cls == nn.TransformerEncoder, "Factorized space time only supported with encoder"
            assert not self.cfg.flat_encoder, "Flat encoder not supported with factorized space time"
            self.space_transformer_encoder = nn.TransformerEncoder(enc_layer, round(n_layers / 2))
            self.time_transformer_encoder = nn.TransformerEncoder(enc_layer, n_layers - round(n_layers / 2))
        else:
            self.transformer_encoder = enc_cls(enc_layer, n_layers)
        if not getattr(self.cfg, 'debug_force_nonlearned_position', False) and (self.cfg.flat_encoder or self.cfg.learnable_position):
            if allow_embed_padding:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length + 1, self.cfg.n_state, padding_idx=self.cfg.max_trial_length)
            else:
                self.time_encoder = nn.Embedding(self.cfg.max_trial_length, self.cfg.n_state)
        else:
            self.time_encoder = PositionalEncodingScript(self.cfg, input_times=self.cfg.transform_space)
        if debug_override_dropout_in:
            self.dropout_in = nn.Identity()
        else:
            self.dropout_in = nn.Dropout(self.cfg.dropout)
        if debug_override_dropout_out:
            self.dropout_out = nn.Identity()
        else:
            self.dropout_out = nn.Dropout(self.cfg.dropout)
        self.embed_space = embed_space

        # if self.cfg.transform_space and self.embed_space:
        n_space = max_spatial_tokens if max_spatial_tokens else self.cfg.max_spatial_tokens
        self.space_encoder = nn.Embedding(n_space, self.cfg.n_state)

    @property
    def out_size(self):
        return self.n_state

    # === Masks ===
    def make_src_mask(self, src: torch.Tensor, trial_context: torch.Tensor, times: torch.Tensor, t: int, s: int=1, causal: bool=True):
        r"""
            args:
                temporal_context: b t temp_c h
                trial_context: b trial_c h

            Produces time major (T*S + TempCtx + TrialCtx, T*S + TempCtx + TrialCtx)
            Use t=1 to produce a space only mask, s=1 to produce a time only mask
        """
        if causal:
            src_mask = generate_square_subsequent_mask_from_times(times)
        else:
            src_mask = None
        if src_mask is None:
            src_mask = torch.zeros((t * s, t * s), dtype=torch.float, device=src.device) # all attending
        src_mask = F.pad(src_mask, (0, 0, 0, trial_context.size(1)), value=float('-inf'))
        src_mask = F.pad(src_mask, (0, trial_context.size(1)), value=0.)

        if src_mask is not None and src_mask.ndim == 3: # expand along heads
            src_mask = src_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).view(-1, src_mask.size(1), src_mask.size(2))
        return src_mask

    def forward(
        self,
        src: torch.Tensor, # B T A H - embedded already (or possibly B T A S_A H), or B Token H
        times: torch.Tensor, # for flat spacetime path, B x Token
        positions: torch.Tensor, # for flat spacetime path
        trial_context: torch.Tensor, # B T' H
        causal: bool=True,
    ) -> torch.Tensor: # T B H
        r"""
            Simplified to assume `flat_encoder` and not `factorized_space_time`.
        """
        src = src + self.time_encoder(times) + self.space_encoder(positions)
        b, t, s = src.size(0), src.size(1), 1 # it's implied that codepaths get that "space" is not really 1, but it's not used

        contextualized_src = torch.cat([src, trial_context], dim=1)
        # contextualized_src, ps = pack(contextualized_src, 'b * h') # b [(t a) + (t n) + t'] h

        src_mask = self.make_src_mask(src, trial_context, times, t, s, causal=causal)

        # padding_mask = torch.zeros((b, t), dtype=torch.bool, device=src.device)
        # padding_mask = torch.cat([
        #     padding_mask,
        #     torch.zeros((b, trial_context.size(-2)), dtype=torch.bool, device=src.device)
        # ], dim=-1)

        output = self.transformer_encoder(contextualized_src, src_mask)
        # output = self.transformer_encoder(contextualized_src, src_mask, src_key_padding_mask=padding_mask)
        return output[:, : t * s]

def generate_square_subsequent_mask_from_times(times: torch.Tensor, ref_times: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        times: B x Token

        out: B x T x T
    """
    if ref_times is None:
        ref_times = times
    # mask = (times[:, :, None] < ref_times[:, None, :]) # Note sign flip vs where expression
    return torch.where(
        times[:, :, None] >= ref_times[:, None, :],
        0.0, float('-inf')
    ).float()
    # return mask.masked_fill(mask, float('-inf'))