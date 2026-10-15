# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.cohere2_moe.modeling_cohere2_moe import (
    Cohere2MoeAttention,
    Cohere2MoeConfig,
    Cohere2MoeDecoderLayer,
    Cohere2MoeExperts,
    Cohere2MoeForCausalLM,
    Cohere2MoeModel,
    Cohere2MoeRotaryEmbedding,
    Cohere2MoeSparseMoeBlock,
    Cohere2MoeTopKRouter,
    repeat_kv,
    rotate_half,
)

from QEfficient.transformers.cache_utils import QEffGPTOSSHybridCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.moe import (
    MoEFlavour,
    MoEProfile,
    MoEWeights,
    QEffMoEBlockMixin,
    build_canonical_expert_weights,
    delete_module_attrs,
    silu_glu_mlp,
)
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffCohere2MoeRotaryEmbedding(Cohere2MoeRotaryEmbedding):
    """Static interleaved RoPE tables shared by every decoder layer."""

    def __init__(self, config: Cohere2MoeConfig, device=None):
        super().__init__(config=config, device=device)
        dtype = getattr(config, "torch_dtype", None) or torch.get_default_dtype()
        positions = torch.arange(self.original_max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        embedding = torch.stack((freqs, freqs), dim=-1).reshape(self.original_max_seq_len, config.head_dim)
        self.register_buffer("cos_cached", (embedding.cos() * self.attention_scaling).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (embedding.sin() * self.attention_scaling).to(dtype), persistent=False)


def qeff_apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype = query_states.dtype
    query_states = query_states.float()
    key_states = key_states.float()
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return query_states.to(query_dtype), key_states.to(query_dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask,
            torch.full_like(attn_weights, MIN_MASKED_ATTENTION_VALUE, dtype=attn_weights.dtype),
            attn_weights,
        )
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class QEffCohere2MoeExperts(Cohere2MoeExperts):
    def __qeff_init__(self):
        self.weights_transformed = False

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        self.moe_weights = build_canonical_expert_weights(
            gate_up=self.gate_up_proj,
            down=self.down_proj,
            fused=True,
            fused_split_dim=1,
            transpose_gate_up=True,
            transpose_down=True,
            clone=True,
        )
        delete_module_attrs(self, "gate_up_proj", "down_proj")
        self.weights_transformed = True
        return self.moe_weights


class QEffCohere2MoeTopKRouter(Cohere2MoeTopKRouter):
    def forward(self, hidden_states: torch.Tensor):
        router_logits = F.linear(hidden_states, self.weight)
        router_scores, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        if self.expert_selection_fn == "softmax":
            router_scores = F.softmax(router_scores, dim=1, dtype=torch.float)
        elif self.expert_selection_fn == "sigmoid":
            router_scores = torch.sigmoid(router_scores)
            if self.norm_topk_prob:
                router_scores = router_scores / torch.einsum("bk->b", router_scores).unsqueeze(-1)
        else:
            raise ValueError("`expert_selection_fn` can only be `softmax` or `sigmoid`")
        return router_logits, router_scores.to(hidden_states.dtype), selected_experts


class QEffCohere2MoeSparseMoeBlock(QEffMoEBlockMixin, Cohere2MoeSparseMoeBlock):
    supported_moe_flavours = (
        MoEFlavour.SIMPLE_LOOP,
        MoEFlavour.DECODE_BMM,
        MoEFlavour.EXPERT_PARALLEL,
    )

    def __qeff_init__(self):
        super().__qeff_init__()
        self.top_k = getattr(self.gate, "top_k", None)
        self.norm_topk_prob = getattr(self.gate, "norm_topk_prob", False)
        self.num_experts = getattr(self.gate, "num_experts", getattr(self.experts, "num_experts", None))

    def transform_weights(self) -> MoEWeights:
        if getattr(self, "weights_transformed", False):
            return self.moe_weights
        self.moe_weights = self.experts.transform_weights()
        self.weights_transformed = True
        return self.moe_weights

    @property
    def moe_profile(self) -> MoEProfile:
        return MoEProfile(expert_mlp=partial(silu_glu_mlp, act_fn=self.experts.act_fn))

    def route(self, x: torch.Tensor):
        _, top_w, top_i = self.gate(x)
        return (top_i, top_w.to(x.dtype)), None

    def apply_shared_experts(self, out: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(residual)
            if self.shared_expert_combination_strategy == "sum":
                return out + shared_output
            elif self.shared_expert_combination_strategy == "average":
                return (out + shared_output) / 2
            else:
                raise ValueError("`shared_expert_combination_strategy` can only be `sum` or `average`")
        return out


class QEffCohere2MoePrefillOnlyMoeMLP(QEffCohere2MoeSparseMoeBlock):
    supported_moe_flavours = QEffCohere2MoeSparseMoeBlock.supported_moe_flavours


class QEffCohere2MoePrefillChunkedMoeMLP(QEffCohere2MoePrefillOnlyMoeMLP):
    supported_moe_flavours = QEffCohere2MoeSparseMoeBlock.supported_moe_flavours


class QEffCohere2MoeAttention(Cohere2MoeAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        query_states = (
            self.q_proj(hidden_states)
            .view(batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.sliding_window is not None or self.force_rope:
            cos, sin = position_embeddings
            query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {
                "position_ids": position_ids,
                "batch_index": batch_index,
                "is_sliding": self.sliding_window is not None,
                "sliding_window": self.sliding_window,
            }
            if comp_ctx_lengths is not None and self.sliding_window is None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, self.scaling
        )
        attn_output = attn_output.reshape(batch_size, sequence_length, self.config.num_attention_heads * self.head_dim)
        return self.o_proj(attn_output), attn_weights


class QEffCohere2MoeDecoderLayer(Cohere2MoeDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states_attention, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states_mlp = self.mlp(hidden_states)
        return residual + hidden_states_attention + hidden_states_mlp


def _cache_layer_length(cache: QEffGPTOSSHybridCache | None, layer_idx: int, fallback: int) -> int:
    if cache is not None and layer_idx < len(cache.layers):
        keys = getattr(cache.layers[layer_idx], "keys", None)
        if keys is not None:
            return keys.shape[2]
    return fallback


class QEffCohere2MoeModel(Cohere2MoeModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffCohere2MoeRotaryEmbedding(config=self.config)
        self.cos_cached = nn.Parameter(self.rotary_emb.cos_cached, requires_grad=False)
        self.sin_cached = nn.Parameter(self.rotary_emb.sin_cached, requires_grad=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = past_key_values is not None and not isinstance(past_key_values, Cache)
        if isinstance(past_key_values, QEffGPTOSSHybridCache):
            qeff_cache = past_key_values
        elif isinstance(past_key_values, Cache):
            qeff_cache = QEffGPTOSSHybridCache.from_cache(self.config, past_key_values)
        elif past_key_values is not None:
            qeff_cache = QEffGPTOSSHybridCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache:
            # shortcut: standalone eager prefill cache is prompt-sized, use preallocated retained state for decode.
            qeff_cache = QEffGPTOSSHybridCache(config=self.config)
        else:
            qeff_cache = None

        past_seen_tokens = qeff_cache.get_seq_length() if qeff_cache is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        full_layer = next(
            (i for i, layer_type in enumerate(self.config.layer_types) if layer_type == "full_attention"), 0
        )
        sliding_layer = next(
            (i for i, layer_type in enumerate(self.config.layer_types) if layer_type == "sliding_attention"), full_layer
        )
        fallback_length = position_ids.shape[-1]
        full_length = _cache_layer_length(qeff_cache, full_layer, fallback_length)
        sliding_length = _cache_layer_length(
            qeff_cache,
            sliding_layer,
            min(self.config.sliding_window or fallback_length, fallback_length),
        )
        masks = attention_mask if isinstance(attention_mask, dict) else None
        if masks is None:
            full_mask = _create_causal_mask(position_ids, target_length=full_length)
            sliding_mask = _create_causal_mask(
                position_ids, target_length=sliding_length, sliding_window=self.config.sliding_window
            )
        else:
            full_mask = masks["full_attention"]
            sliding_mask = masks["sliding_attention"]

        hidden_states = inputs_embeds
        sin = self.sin_cached[position_ids].to(device=hidden_states.device)
        cos = self.cos_cached[position_ids].to(device=hidden_states.device)
        position_embeddings = (cos, sin)
        for layer_idx, decoder_layer in enumerate(self.layers):
            layer_mask = full_mask if self.config.layer_types[layer_idx] == "full_attention" else sliding_mask
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=qeff_cache,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        if return_legacy_cache and qeff_cache is not None:
            qeff_cache = qeff_cache.to_legacy_cache()
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=qeff_cache if use_cache else None,
        )


class QEffCohere2MoeForCausalLM(Cohere2MoeForCausalLM):
    def get_submodules_for_export(self) -> type[nn.Module]:
        return {QEffCohere2MoeDecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            cache_position=cache_position,
            **kwargs,
        )
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state.gather(
            1, logit_index.unsqueeze(-1).expand(-1, -1, outputs.last_hidden_state.shape[-1])
        )
        logits = self.lm_head(hidden_states).float() * self.logit_scale
        return MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len, dtype=None, retain_full_kv=False):
        dtype = dtype or getattr(config, "torch_dtype", None) or torch.float32
        cache = []
        for layer_type in config.layer_types:
            layer_len = (
                seq_len
                if retain_full_kv or layer_type == "full_attention"
                else min(config.sliding_window or seq_len, seq_len)
            )
            shape = (batch_size, config.num_key_value_heads, layer_len, config.head_dim)
            cache.append((torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)))
        return cache

    def get_pkv_dynamic_axes(self, retain_full_kv=False, continuous_batching=False):
        batch_axis = "full_batch_size" if continuous_batching else "batch_size"
        return [
            {0: batch_axis, 2: "ctx_len" if retain_full_kv or layer_type == "full_attention" else "sliding_window"}
            for layer_type in self.config.layer_types
        ]

    def get_specializations(self, batch_size: int, prefill_seq_len: int, ctx_len: int, **kwargs):
        if kwargs.get("prefill_only") and not kwargs.get("enable_chunking"):
            ctx_len = prefill_seq_len
        return [
            {
                "batch_size": batch_size or 1,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": self.config.sliding_window,
            },
            {
                "batch_size": batch_size or 1,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "sliding_window": self.config.sliding_window,
            },
        ]
