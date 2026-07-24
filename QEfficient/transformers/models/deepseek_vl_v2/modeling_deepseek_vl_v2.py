# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEfficient modeling for the text decoder of ``deepseek-ai/DeepSeek-OCR-2``
(HF class ``DeepseekOCR2ForCausalLM``, ``model_type: deepseek_vl_v2``).

This checkpoint is a DeepSeek-V2-family MoE decoder with **multi-latent attention
disabled** (``use_mla=False``): attention is plain Llama-style multi-head attention
and the FFN is a DeepSeek MoE block (64 routed + 2 shared experts, top-6, softmax
gate, no top-k renorm, routed scaling 1.0; layer 0 is a dense MLP).

The modeling is *self-contained* rather than a set of method-patches over an HF class,
because neither reference class is usable on transformers 5.x:
  * the ``trust_remote_code`` reference is pinned to transformers 4.46.3 (imports the
    removed ``LlamaFlashAttention2`` / ``is_torch_fx_available``, never threads
    ``position_embeddings``, and returns the legacy 3-tuple from attention); and
  * transformers-native ``deepseek_v2`` is MLA-only and its strict config rejects this
    checkpoint's null LoRA ranks.

Attention/RoPE/KV-cache reuse the same QEfficient primitives as ``modeling_llama``; the
MoE FFN reuses the fused-expert ``bmm`` pattern from ``deepseek_v3``. The vision tower
(SAM ViT-B -> Qwen2-as-encoder -> linear projector -> ``masked_scatter_``) is NOT part
of this file; the text decoder is validated on its own (``validated-cpu-simplified``).
"""

from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.transformers.models.llama.modeling_llama import (
    eager_attention_forward,
    qeff_apply_rotary_pos_emb,
)

from .configuration_deepseek_vl_v2 import DeepseekVLV2Config


class QEffDeepseekVLV2RMSNorm(nn.Module):
    """RMSNorm mapped to the compiler-known custom op (numerically identical to the
    fp32-upcast RMSNorm used by the reference DeepseekV2RMSNorm)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return CustomRMSNormFunc.apply(hidden_states, self.weight, self.variance_epsilon)


class QEffDeepseekVLV2RotaryEmbedding(LlamaRotaryEmbedding):
    """Llama-style rotary embedding with static cos/sin caches (as in modeling_llama)."""

    def __init__(self, config: DeepseekVLV2Config, device=None):
        super().__init__(config=config)
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=config.torch_dtype
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class QEffDeepseekVLV2MLP(nn.Module):
    """Dense SwiGLU MLP (used by layer 0 and by the shared-expert branch)."""

    def __init__(self, config: DeepseekVLV2Config, intermediate_size: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QEffDeepseekVLV2MoEGate(nn.Module):
    """Routed-expert gate: softmax scoring, greedy top-k, no top-k renorm, routed
    scaling factor applied (matches the reference MoEGate for this checkpoint's config:
    ``scoring_func='softmax'``, ``topk_method='greedy'``, ``norm_topk_prob=False``)."""

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.gating_dim)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class QEffDeepseekVLV2MoE(nn.Module):
    """MoE FFN with fused routed experts (bmm over the top-k selected experts) plus a
    shared-expert branch, following the deepseek_v3 fused-expert pattern."""

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList(
            [
                QEffDeepseekVLV2MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = QEffDeepseekVLV2MoEGate(config)
        if config.n_shared_experts is not None:
            self.shared_experts = QEffDeepseekVLV2MLP(
                config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
            )

    def __qeff_init__(self):
        """Fuse per-expert weights into stacked tensors for a batched-bmm MoE."""
        self.all_gate_proj = nn.Parameter(
            torch.cat([exp.gate_proj.weight.T.unsqueeze(0) for exp in self.experts], dim=0)
        )
        self.all_up_proj = nn.Parameter(torch.cat([exp.up_proj.weight.T.unsqueeze(0) for exp in self.experts], dim=0))
        self.all_down_proj = nn.Parameter(
            torch.cat([exp.down_proj.weight.T.unsqueeze(0) for exp in self.experts], dim=0)
        )
        self.act_fn = self.experts[0].act_fn

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        seq_len, _ = hidden_states.shape
        gate_proj = self.all_gate_proj[topk_indices.flatten()]
        up_proj = self.all_up_proj[topk_indices.flatten()]
        down_proj = self.all_down_proj[topk_indices.flatten()]
        expert_in = (
            hidden_states.unsqueeze(1).expand(-1, self.gate.top_k, -1).contiguous().view(-1, 1, self.config.hidden_size)
        )
        gate_out = torch.bmm(expert_in, gate_proj)
        up_out = torch.bmm(expert_in, up_proj)
        hidden = self.act_fn(gate_out) * up_out
        expert_output = torch.bmm(hidden, down_proj)
        experts_out = expert_output.view(seq_len, self.gate.top_k, self.config.hidden_size)
        experts_out = experts_out * topk_weights.unsqueeze(-1)
        final_hidden_states = torch.einsum("abc->ac", experts_out)
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class QEffDeepseekVLV2Attention(nn.Module):
    """Plain multi-head attention (Llama-style) with QEff RoPE + KV-cache handling."""

    def __init__(self, config: DeepseekVLV2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        if past_key_value is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffDeepseekVLV2DecoderLayer(nn.Module):
    """Decoder layer: dense MLP for layer < first_k_dense_replace, MoE otherwise."""

    def __init__(self, config: DeepseekVLV2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QEffDeepseekVLV2Attention(config, layer_idx)
        self.mlp = (
            QEffDeepseekVLV2MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else QEffDeepseekVLV2MLP(config)
        )
        self.input_layernorm = QEffDeepseekVLV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = QEffDeepseekVLV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        sin_cached=None,
        cos_cached=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class QEffDeepseekVLV2Model(nn.Module):
    """Bare decoder: embeddings + stack of decoder layers + final norm."""

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [QEffDeepseekVLV2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = QEffDeepseekVLV2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __qeff_init__(self):
        self.rotary_emb = QEffDeepseekVLV2RotaryEmbedding(config=self.config)
        self.sin_cached = nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)
        for layer in self.layers:
            if isinstance(layer.mlp, QEffDeepseekVLV2MoE):
                layer.mlp.__qeff_init__()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        causal_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        sin = self.sin_cached[position_ids].unsqueeze(1)
        cos = self.cos_cached[position_ids].unsqueeze(1)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                sin_cached=sin,
                cos_cached=cos,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class QEffDeepseekVLV2ForCausalLM(nn.Module):
    """Causal-LM head over the deepseek_vl_v2 text decoder."""

    def __init__(self, config: DeepseekVLV2Config):
        super().__init__()
        self.config = config
        self.model = QEffDeepseekVLV2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def __qeff_init__(self):
        self.model.__qeff_init__()

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDeepseekVLV2DecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
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
        )
