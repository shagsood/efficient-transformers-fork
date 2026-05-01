# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import math
from typing import List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeAttention,
    Qwen3_5MoeDecoderLayer,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
    Qwen3_5MoeSparseMoeBlock,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeTextRotaryEmbedding,
    apply_rotary_pos_emb,
    l2norm,
    repeat_kv,
    rotate_half,
)

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffDynamicLayer
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


# ---------------------------------------------------------------------------
# RMSNorm with gated fusion (for Gated Delta Net norm layer)
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeGatedDeltaNetCustomRMSNormAIC(nn.Module):
    def forward(self, hidden_states, gate):
        return (
            CustomRMSNormFunc.apply(
                hidden_states, self.weight, self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps
            )
        ) * F.silu(gate.to(torch.float32))


# ---------------------------------------------------------------------------
# Hybrid cache: KV pairs (full_attention) + conv/recurrent (linear_attention)
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeDynamicCache(Cache):
    def __init__(self, config):
        super().__init__(layers=[])
        self.config = config
        self.layer_types = list(config.layer_types)
        self.transformer_layers = [i for i, lt in enumerate(self.layer_types) if lt == "full_attention"]
        self.last_linear_layer = next(
            (i for i in range(len(self.layer_types) - 1, -1, -1) if self.layer_types[i] == "linear_attention"),
            None,
        )
        self.kv_layers = [
            QEffDynamicLayer() if lt == "full_attention" else None for lt in self.layer_types
        ]
        self.conv_states = [None for _ in self.layer_types]
        self.recurrent_states = [None for _ in self.layer_types]

    @classmethod
    def from_legacy_cache(cls, config, past_key_values):
        cache = cls(config)
        if past_key_values is None:
            return cache
        idx = 0
        for layer_idx, lt in enumerate(config.layer_types):
            if lt == "full_attention":
                if idx < len(past_key_values):
                    key_states, value_states = past_key_values[idx]
                    cache.kv_layers[layer_idx] = QEffDynamicLayer()
                    cache.kv_layers[layer_idx]._seen_tokens = key_states.shape[2]
                    cache.kv_layers[layer_idx].keys = key_states
                    cache.kv_layers[layer_idx].values = value_states
                idx += 1
            else:
                if idx < len(past_key_values):
                    conv_state, recurrent_state = past_key_values[idx]
                    cache.conv_states[layer_idx] = conv_state
                    cache.recurrent_states[layer_idx] = recurrent_state
                idx += 1
        return cache

    def to_legacy_cache(self):
        legacy = []
        for layer_idx, lt in enumerate(self.layer_types):
            if lt == "full_attention":
                kv = self.kv_layers[layer_idx]
                if kv is not None and kv.keys is not None:
                    legacy.append((kv.keys, kv.values))
                else:
                    legacy.append((torch.empty(0), torch.empty(0)))
            else:
                conv_state = self.conv_states[layer_idx]
                rec_state = self.recurrent_states[layer_idx]
                if conv_state is None:
                    conv_state = torch.empty(0)
                if rec_state is None:
                    rec_state = torch.empty(0)
                legacy.append((conv_state, rec_state))
        return legacy

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        kv = self.kv_layers[layer_idx]
        if kv is None:
            kv = QEffDynamicLayer()
            self.kv_layers[layer_idx] = kv
        return kv.update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx=0):
        for i in self.transformer_layers:
            kv = self.kv_layers[i]
            if kv is not None and kv.keys is not None:
                return kv.keys.shape[2]
        return 0

    def reorder_cache(self, beam_idx):
        beam_idx_device = beam_idx.to(self.kv_layers[0].keys.device if self.kv_layers[0] else "cpu")
        for layer_idx, lt in enumerate(self.layer_types):
            if lt == "full_attention":
                kv = self.kv_layers[layer_idx]
                if kv is not None and kv.keys is not None:
                    kv.keys = kv.keys.index_select(0, beam_idx_device)
                    kv.values = kv.values.index_select(0, beam_idx_device)
            else:
                if self.conv_states[layer_idx] is not None:
                    self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx_device)
                if self.recurrent_states[layer_idx] is not None:
                    self.recurrent_states[layer_idx] = self.recurrent_states[layer_idx].index_select(0, beam_idx_device)

    def has_previous_state(self, layer_idx):
        lt = self.layer_types[layer_idx]
        if lt == "full_attention":
            kv = self.kv_layers[layer_idx]
            return kv is not None and kv.key_cache is not None
        return self.conv_states[layer_idx] is not None

    def get_layer_state(self, layer_idx):
        lt = self.layer_types[layer_idx]
        if lt == "full_attention":
            return None
        return (self.conv_states[layer_idx], self.recurrent_states[layer_idx])


# ---------------------------------------------------------------------------
# Rotary embedding with precomputed cache
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeTextRotaryEmbedding(Qwen3_5MoeTextRotaryEmbedding):
    def __init__(self, config=None):
        super().__init__(config)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
            self.sin_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
        )


# ---------------------------------------------------------------------------
# Helpers: MRoPE, attention, conv1d update
# ---------------------------------------------------------------------------


def qeff_apply_interleaved_mrope(freqs, mrope_section):
    half_shape = freqs[0].shape[-1] // 2
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
        offset += half_shape
        length += half_shape
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids, mrope_section, unsqueeze_dim=1):
    cos = cos[position_ids]
    sin = sin[position_ids]

    cos = qeff_apply_interleaved_mrope(cos, mrope_section)
    sin = qeff_apply_interleaved_mrope(sin, mrope_section)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[:, :, :, :rotary_dim], q[:, :, :, rotary_dim:]
    k_rot, k_pass = k[:, :, :, :rotary_dim], k[:, :, :, rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def eager_attention_forward(module, query, key, value, attention_mask, scaling):
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


def qeff_torch_causal_conv1d_update(hidden_states, conv_state, weight, position_ids, bias=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    idx = position_ids[0].flatten()
    zeros = torch.zeros(state_len, dtype=idx.dtype, device=idx.device)
    out = torch.cat([zeros, idx], dim=0)
    order = torch.argsort(out)
    last4_positions = order[-state_len:]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    updated_conv_state = hidden_states_new.index_select(2, last4_positions.long())

    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:]).to(hidden_states.dtype)
    return out, updated_conv_state


# ---------------------------------------------------------------------------
# Full-attention layer with KV cache
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeAttention(Qwen3_5MoeAttention):
    def __qeff_init__(self):
        self.rotary_emb = QEffQwen3_5MoeTextRotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[QEffQwen3_5MoeDynamicCache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q_out = self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_out, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids[0]}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Linear-attention: Gated Delta Rule with binary-lifting chunk algorithm
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeGatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    def __qeff_init__(self):
        self.chunk_gated_delta_rule = self.torch_chunk_gated_delta_rule_qeff
        chunk_size = 64

        mask_causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool)
        for i in range(chunk_size):
            for j in range(i + 1):
                mask_causal[i, j] = False
        self.register_buffer("_mask_causal", mask_causal, persistent=False)

        mask_strict = torch.zeros(chunk_size, chunk_size, dtype=torch.bool)
        for i in range(chunk_size):
            for j in range(i + 1, chunk_size):
                mask_strict[i, j] = True
        self.register_buffer("_mask_strict", mask_strict, persistent=False)

        ones_lower = torch.zeros(chunk_size, chunk_size)
        for i in range(chunk_size):
            for j in range(i + 1):
                ones_lower[i, j] = 1.0
        self.register_buffer("_ones_lower", ones_lower, persistent=False)

        self.register_buffer("_eye", torch.eye(chunk_size), persistent=False)

    def torch_chunk_gated_delta_rule_qeff(
        self,
        query, key, value, g, beta,
        position_ids,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        mask_causal=None,
        mask_strict=None,
        ones_lower=None,
        eye=None,
    ):
        initial_dtype = query.dtype
        if use_qk_l2norm_in_kernel:
            query = l2norm(query, dim=-1, eps=1e-6)
            key = l2norm(key, dim=-1, eps=1e-6)
        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
        ]

        mask = (position_ids[0] != -1).unsqueeze(1)
        zeros = torch.zeros(g.shape, dtype=g.dtype, device=g.device)
        g = torch.where(mask, g, zeros)

        qkv_zeros = torch.zeros(key.shape, dtype=key.dtype, device=key.device)
        key = torch.where(mask.unsqueeze(-1), key, qkv_zeros)

        batch_size, num_heads, sequence_length, k_head_dim = key.shape
        v_head_dim = value.shape[-1]
        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
        query = F.pad(query, (0, 0, 0, pad_size))
        key = F.pad(key, (0, 0, 0, pad_size))
        value = F.pad(value, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        g = F.pad(g, (0, pad_size))
        total_sequence_length = sequence_length + pad_size
        scale = 1 / (query.shape[-1] ** 0.5)
        query = query * scale

        v_beta = value * beta.unsqueeze(-1)
        k_beta = key * beta.unsqueeze(-1)
        query, key, value, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (query, key, value, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

        # CumSum replacement: matmul with lower-triangular ones
        L_cs = g.size(-1)
        idx = torch.arange(L_cs, device=g.device)
        mask_g = (idx.unsqueeze(1) >= idx.unsqueeze(0)).to(g.dtype)
        g = g @ mask_g.T

        # Decay mask (replaces tril().exp())
        diff = g.unsqueeze(-1) - g.unsqueeze(-2)
        diff = diff * (~mask_strict).float()
        decay_mask = diff.exp().float()
        decay_mask = decay_mask * (~mask_strict).float()

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

        # Binary lifting: (I - A)^{-1} via log2(K) steps
        eye = torch.eye(chunk_size, device=attn.device, dtype=attn.dtype)
        L_bl = eye.clone()
        Apow = attn
        K = 32
        for _ in range(int(math.log2(K))):
            L_bl = L_bl @ (eye + Apow)
            Apow = Apow @ Apow

        attn = L_bl

        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        last_recurrent_state = (
            torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
            if initial_state is None
            else initial_state.to(value)
        )
        core_attn_out = torch.zeros_like(value)
        mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

        for i in range(0, total_sequence_length // chunk_size):
            q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
            attn_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
            v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
            v_new = v_i - v_prime
            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
            core_attn_out[:, :, i] = attn_inter + attn_chunk @ v_new
            last_recurrent_state = (
                last_recurrent_state * g[:, :, i, -1, None, None].exp()
                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
            )

        if not output_final_state:
            last_recurrent_state = None
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
        )
        core_attn_out = core_attn_out[:, :, :sequence_length]
        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
        return core_attn_out, last_recurrent_state

    def _recurrent_step_batched(self, query, key, value, g, beta, recurrent_state):
        dtype = query.dtype
        q = query.float().transpose(1, 2)
        k = key.float().transpose(1, 2)
        v = value.float().transpose(1, 2)
        b = beta.transpose(1, 2).float().unsqueeze(-1)
        decay = g.transpose(1, 2).float().exp()
        decay = decay.unsqueeze(-1).unsqueeze(-1)

        scale = 1.0 / (q.shape[-1] ** 0.5)
        q = l2norm(q, dim=-1, eps=1e-6) * scale
        k = l2norm(k, dim=-1, eps=1e-6)

        S = recurrent_state.float()
        S_decayed = S * decay[:, :, 0]
        kv_mem = (S_decayed * k[:, :, 0].unsqueeze(-1)).sum(dim=-2)
        delta = (v[:, :, 0] - kv_mem) * b[:, :, 0]
        S_new = S_decayed + k[:, :, 0].unsqueeze(-1) * delta.unsqueeze(-2)
        out = (S_new * q[:, :, 0].unsqueeze(-1)).sum(dim=-2)

        out = out.unsqueeze(2).transpose(1, 2).to(dtype)
        return out, S_new.to(recurrent_state.dtype)

    def forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta = self.in_proj_b(hidden_states).sigmoid()
        g = -self.A_log.float().exp() * F.softplus(self.in_proj_a(hidden_states).float() + self.dt_bias)

        if cache_params is not None and cache_params.conv_states[self.layer_idx] is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
            mixed_qkv, new_conv_state = qeff_torch_causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), position_ids, self.conv1d.bias,
            )
            cache_params.conv_states[self.layer_idx] = new_conv_state
        else:
            recurrent_state = None
            mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if cache_params is not None and recurrent_state is not None:
            # Compute BOTH paths; torch.where selects — single ONNX, hardware predicates
            recurrent_out, recurrent_S = self._recurrent_step_batched(query, key, value, g, beta, recurrent_state)

            chunk_out, chunk_S = self.chunk_gated_delta_rule(
                query, key, value,
                g=g, beta=beta,
                position_ids=position_ids,
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                mask_causal=self._mask_causal,
                mask_strict=self._mask_strict,
                ones_lower=self._ones_lower,
                eye=self._eye,
            )

            is_decode = hidden_states.shape[1] == torch.tensor(1)
            core_attn_out = torch.where(is_decode, recurrent_out, chunk_out)
            last_recurrent_state = torch.where(is_decode, recurrent_S, chunk_S)
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state
        else:
            core_attn_out, _ = self.chunk_gated_delta_rule(
                query, key, value,
                g=g, beta=beta,
                position_ids=position_ids,
                initial_state=recurrent_state,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                mask_causal=self._mask_causal,
                mask_strict=self._mask_strict,
                ones_lower=self._ones_lower,
                eye=self._eye,
            )

        core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))

    @staticmethod
    def apply_mask_to_padding_states(hidden_states, attention_mask):
        if attention_mask is not None and attention_mask.shape[1] > 1:
            dtype = hidden_states.dtype
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder layer — dispatches on layer_type
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeDecoderLayer(Qwen3_5MoeDecoderLayer):
    def __qeff_init__(self):
        if self.layer_type == "linear_attention":
            self.linear_attn.__class__ = QEffQwen3_5MoeGatedDeltaNet
            self.linear_attn.__qeff_init__()
        elif self.layer_type == "full_attention":
            self.self_attn.__class__ = QEffQwen3_5MoeAttention
            self.self_attn.__qeff_init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[QEffQwen3_5MoeDynamicCache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        del use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        hidden_states = residual + hidden_states
        return hidden_states


# ---------------------------------------------------------------------------
# ONNX-safe MoE: static loop over ALL experts (correctness-first)
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeSparseMoeBlock(Qwen3_5MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, H = hidden_states.shape
        T = B * S
        x = hidden_states.view(T, H)

        shared_out = self.shared_expert(x)
        shared_out = F.sigmoid(self.shared_expert_gate(x)) * shared_out

        router_logits, routing_weights, top_i = self.gate(x)

        expert_out = torch.zeros_like(x)
        for e in range(self.experts.num_experts):
            expert_mask = (top_i == e).to(routing_weights.dtype)
            w = (routing_weights * expert_mask).sum(dim=-1, keepdim=True)
            gate_up = F.linear(x, self.experts.gate_up_proj[e])
            gate, up = gate_up.chunk(2, dim=-1)
            h = self.experts.act_fn(gate) * up
            out = F.linear(h, self.experts.down_proj[e])
            expert_out = expert_out + out * w

        return (expert_out + shared_out).view(B, S, H)


# ---------------------------------------------------------------------------
# Text model backbone
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeTextModel(Qwen3_5MoeTextModel):
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[QEffQwen3_5MoeDynamicCache, Tuple[Tuple[torch.FloatTensor, ...], ...]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_legacy_cache = False
        if past_key_values is not None and not isinstance(past_key_values, QEffQwen3_5MoeDynamicCache):
            return_legacy_cache = True
            past_key_values = QEffQwen3_5MoeDynamicCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffQwen3_5MoeDynamicCache(self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if isinstance(attention_mask, torch.Tensor):
            target_length = attention_mask.shape[-1]
        elif past_key_values is not None:
            target_length = past_key_values.get_seq_length()
        else:
            target_length = inputs_embeds.shape[1]
        causal_mask = _create_causal_mask(
            position_ids=position_ids[0], target_length=target_length, sliding_window=None
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids[1:])

        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


# ---------------------------------------------------------------------------
# CausalLM head with ONNX retained state specs
# ---------------------------------------------------------------------------


class QEffQwen3_5MoeForCausalLM(Qwen3_5MoeForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3_5MoeDecoderLayer}

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
        return past_key_values

    def _iter_retained_state_names(self) -> List[str]:
        names = []
        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                names.extend([f"past_key.{layer_idx}", f"past_value.{layer_idx}"])
            else:
                names.extend([f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"])
        return names

    def get_retained_state_names(self) -> List[str]:
        return self._iter_retained_state_names()

    def get_onnx_retained_state_specs(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache_shape: List[int],
        continuous_batching: bool = False,
        retain_full_kv: bool = False,
    ) -> dict:
        del seq_len, retain_full_kv
        batch_axis_name = "full_batch_size" if continuous_batching else "batch_size"
        specs = {
            "past_key_values": [],
            "input_names": [],
            "output_names": [],
            "dynamic_axes": {},
        }

        for layer_idx, layer_type in enumerate(self.config.layer_types):
            if layer_type == "full_attention":
                layer_names = [f"past_key.{layer_idx}", f"past_value.{layer_idx}"]
                layer_tensors = [
                    torch.zeros(tuple(kv_cache_shape), dtype=torch.float32),
                    torch.zeros(tuple(kv_cache_shape), dtype=torch.float32),
                ]
                layer_axes = [
                    {0: batch_axis_name, 2: "ctx_len"},
                    {0: batch_axis_name, 2: "ctx_len"},
                ]
            else:
                layer = self.model.layers[layer_idx].linear_attn
                conv_shape = (batch_size, layer.conv_dim, layer.conv_kernel_size)
                recurrent_shape = (batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim)
                layer_names = [f"conv_state.{layer_idx}", f"recurrent_state.{layer_idx}"]
                layer_tensors = [
                    torch.zeros(conv_shape, dtype=torch.float32),
                    torch.zeros(recurrent_shape, dtype=torch.float32),
                ]
                layer_axes = [{0: batch_axis_name}, {0: batch_axis_name}]

            specs["past_key_values"].append(layer_tensors)
            for name, axes in zip(layer_names, layer_axes):
                specs["input_names"].append(name)
                specs["output_names"].append(f"{name}_RetainedState")
                specs["dynamic_axes"][name] = axes

        return specs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[QEffQwen3_5MoeDynamicCache, Tuple[Tuple[torch.FloatTensor, ...], ...]]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del logits_to_keep
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        if position_ids is None:
            hidden_states = outputs.last_hidden_state[:, -1:, :]
        else:
            text_position_ids = position_ids[0] if position_ids.ndim == 3 else position_ids
            logit_index = text_position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = outputs.last_hidden_state[
                torch.arange(text_position_ids.shape[0]).view(-1, 1), logit_index
            ]

        logits = self.lm_head(hidden_states).float()
        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
