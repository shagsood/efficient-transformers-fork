# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Callable, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.olmo3.modeling_olmo3 import (
    Olmo3Attention,
    Olmo3Config,
    Olmo3DecoderLayer,
    Olmo3ForCausalLM,
    Olmo3Model,
    Olmo3RotaryEmbedding,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffSlidingWindowCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffOlmo3RotaryEmbedding(Olmo3RotaryEmbedding):
    """
    QEff version of OLMo3 rotary embedding with static sin/cos precomputation.

    The parent class handles YaRN RoPE scaling: its __init__ calls
    ROPE_INIT_FUNCTIONS["yarn"] to compute the modified inv_freq and
    attention_scaling factor. We just precompute cos/sin from that inv_freq
    so the forward pass becomes a simple index lookup.
    """

    def __init__(self, config: Olmo3Config, device=None, rope_type=None):
        super().__init__(config=config, device=device, rope_type=rope_type)
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Uses precomputed cos/sin indexed by position_ids instead of computing
    on every forward pass.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    **kwargs,
):
    """FP16-safe eager attention using torch.where instead of -inf masking."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
        )
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class QEffOlmo3Attention(Olmo3Attention):
    """
    QEff OLMo3 attention with KV cache, continuous batching, and per-layer
    sliding window support.

    Each layer knows if it's sliding or full attention via self.is_sliding
    (derived from config.layer_types[layer_idx]). Sliding layers pass
    is_sliding=True to QEffSlidingWindowCache so the cache uses modulo
    arithmetic for the circular buffer.

    OLMo3 uses separate RoPE per attention type — sliding layers get standard
    RoPE, full attention layers get YaRN-scaled RoPE. The correct cos/sin
    is selected at the model level and passed down.
    """

    def __qeff_init__(self):
        self.is_sliding = self.attention_type == "sliding_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states))
        key_states = self.k_norm(self.k_proj(hidden_states))
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        query_states, key_states = qeff_apply_rotary_pos_emb(
            query_states, key_states, cos_cached, sin_cached, position_ids
        )

        if past_key_values is not None:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids,
                "is_sliding": self.is_sliding,
                "sliding_window_pattern": self.config.sliding_window_pattern,
                "sliding_window": past_key_values.sliding_window_len,
            }
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffOlmo3DecoderLayer(Olmo3DecoderLayer):
    """
    QEff OLMo3 decoder layer with per-layer causal mask creation.

    Each layer creates its own mask based on whether it's a sliding window
    or full attention layer:
    - Sliding layers: mask with target_length = sliding_window (circular buffer)
    - Full attention layers: mask with target_length = full context length
    """

    def __qeff_init__(self):
        # Store config reference for mask creation (DecoderLayer doesn't store it by default)
        self.config = self.self_attn.config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ) -> torch.Tensor:
        # Create per-layer mask based on attention type
        if self.self_attn.is_sliding:
            attention_mask = _create_causal_mask(
                position_ids=position_ids,
                target_length=past_key_value.sliding_window_len,
                sliding_window=past_key_value.sliding_window_len,
            )
        else:
            # Full attention: use the cache shape of the first full-attention layer
            first_full_idx = self.config.sliding_window_pattern - 1
            attention_mask = _create_causal_mask(
                position_ids=position_ids,
                target_length=past_key_value.key_cache[first_full_idx].shape[-2],
            )

        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class QEffOlmo3Model(Olmo3Model):
    """
    QEff OLMo3 model with sliding window cache and precomputed RoPE.

    Uses QEffSlidingWindowCache (not QEffDynamicCache) to support per-layer
    cache sizes — sliding layers get smaller caches (sliding_window tokens),
    full attention layers get the complete context length.

    RoPE is precomputed once per attention type and passed to each layer:
    - Sliding layers get standard RoPE (rope_type="default")
    - Full attention layers get YaRN-scaled RoPE (config.rope_scaling)
    """

    def __qeff_init__(self):
        # Derive sliding_window_pattern from layer_types
        # Default OLMo3: 3 sliding + 1 full → pattern = 4
        for i, lt in enumerate(self.config.layer_types):
            if lt == "full_attention":
                self.config.sliding_window_pattern = i + 1
                break
        else:
            # All layers are sliding — no full attention layers
            self.config.sliding_window_pattern = self.config.num_hidden_layers

        # OLMo3 uses SEPARATE RoPE for sliding vs full attention layers:
        #   - Sliding: standard RoPE (rope_type="default", no YaRN scaling)
        #   - Full: YaRN-scaled RoPE (uses config.rope_scaling)
        rotary_emb_sliding = QEffOlmo3RotaryEmbedding(config=self.config, rope_type="default")
        self.sin_cached_sliding = torch.nn.Parameter(
            rotary_emb_sliding.sin_cached * rotary_emb_sliding.attention_scaling
        )
        self.cos_cached_sliding = torch.nn.Parameter(
            rotary_emb_sliding.cos_cached * rotary_emb_sliding.attention_scaling
        )

        rotary_emb_full = QEffOlmo3RotaryEmbedding(config=self.config)
        self.sin_cached_full = torch.nn.Parameter(rotary_emb_full.sin_cached * rotary_emb_full.attention_scaling)
        self.cos_cached_full = torch.nn.Parameter(rotary_emb_full.cos_cached * rotary_emb_full.attention_scaling)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffSlidingWindowCache.from_legacy_cache(
                config=self.config, past_key_values=past_key_values
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Masks are created per-layer in QEffOlmo3DecoderLayer.forward()
        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Select RoPE based on layer's attention type (sliding=default, full=YaRN)
            if decoder_layer.self_attn.is_sliding:
                layer_cos = self.cos_cached_sliding
                layer_sin = self.sin_cached_sliding
            else:
                layer_cos = self.cos_cached_full
                layer_sin = self.sin_cached_full

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=None,  # Created per-layer
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                sin_cached=layer_sin,
                cos_cached=layer_cos,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if use_cache:
            next_cache = past_key_values.to_legacy_cache()

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class QEffOlmo3ForCausalLM(Olmo3ForCausalLM):
    """
    QEff OLMo3 for causal language modeling.

    Key changes from HF:
    - INT32 cast for logit extraction (ONNX-safe)
    - get_submodules_for_export for subfunction extraction
    - get_dummy_pkv_cache for per-layer cache sizing (sliding vs full)
    """

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffOlmo3DecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        # Cast to INT32 to avoid issue while running in ONNXRT
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        """Create dummy past key values with per-layer sizes for sliding vs full attention."""
        n_heads = config.num_key_value_heads
        d_head = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        global_cache_shape = [batch_size, n_heads, seq_len, d_head]
        sliding_cache_shape = [batch_size, n_heads, min(config.sliding_window, seq_len), d_head]

        past_key_values = []
        for i in range(config.num_hidden_layers):
            is_sliding = config.layer_types[i] == "sliding_attention"
            cache_shape = sliding_cache_shape if is_sliding else global_cache_shape
            new_layer_key_cache = torch.zeros(cache_shape, dtype=torch.float32)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=torch.float32)
            past_key_values.append((new_layer_key_cache, new_layer_value_cache))
        return past_key_values
