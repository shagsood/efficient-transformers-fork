# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Optional, Type

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.hy_v3.modeling_hy_v3 import (
    HYV3Attention,
    HYV3DecoderLayer,
    HYV3ForCausalLM,
    HYV3Model,
    HYV3MoE,
    HYV3RotaryEmbedding,
    HYV3TopKRouter,
    repeat_kv,
    rotate_half,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.customop.ctx_scatter_gather import (
    CtxGatherFunc3DGeneralized,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffHYV3RotaryEmbedding(HYV3RotaryEmbedding):
    """
    Copied from HYV3RotaryEmbedding: https://github.com/huggingface/transformers/blob/main/src/transformers/models/hy_v3/modeling_hy_v3.py
    The only differences are:
    - Add static sin/cos computations.
    """

    def __init__(self, config, device=None):
        super().__init__(config=config)

        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
            self.sin_cached[:seq_len].to(dtype=x.dtype) * self.attention_scaling,
        )


def qeff_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze cos[position_ids] and sin[position_ids] for broadcasting.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def qeff_apply_precomputed_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
):
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    half_dim = rotary_dim // 2

    q_half = torch.cat((-q_rot[..., half_dim:], q_rot[..., :half_dim]), dim=-1)
    k_half = torch.cat((-k_rot[..., half_dim:], k_rot[..., :half_dim]), dim=-1)

    q_embed = (q_rot * cos) + (q_half * sin)
    k_embed = (k_rot * cos) + (k_half * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=key_states.dtype), attn_weights
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=key_states.dtype).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def _build_matched_idx_from_cumsum(T2Ei: torch.Tensor):
    """Build a packed-row to original-token index table for active expert rows.

    Returns:
        matched_idx: [batch_size, seq_len] int32 — packed-row → original-token map.
        valid_rows:  [batch_size, 1] int32     — count of active rows per batch,
                                                 taken from the last column of the
                                                 cumsum this function already builds.
                                                 (Returned alongside so callers don't
                                                 need a separate `.sum(dim=1)` which
                                                 lowers to a dynamic-axes ONNX
                                                 ReduceSum that qaic-compile rejects.)
    """
    batch_size, seq_len = T2Ei.shape
    int32_max = torch.iinfo(torch.int32).max
    int32_max_scalar = torch.tensor(int32_max, dtype=torch.int32, device=T2Ei.device)
    token_idx = torch.arange(seq_len, dtype=torch.int32, device=T2Ei.device).unsqueeze(0).expand(batch_size, -1)
    valid_prefix = torch.cumsum(T2Ei.to(torch.int32), dim=1)
    valid_rows = valid_prefix[:, -1:]  # [batch_size, 1] — total active-row count per batch
    valid_dest = valid_prefix - 1
    scatter_pos = torch.where(T2Ei, valid_dest, int32_max_scalar)
    matched_idx = torch.full_like(token_idx, int32_max)
    matched_idx = CtxScatterFunc3DInt.apply(
        matched_idx.unsqueeze(-1),
        scatter_pos,
        token_idx.unsqueeze(-1),
    ).squeeze(-1)
    return matched_idx, valid_rows


def _cumsum_scatter_gather_update_expert_blocked(
    x: torch.Tensor,
    T2Ei: torch.Tensor,
    W_g: torch.Tensor,
    W_u: torch.Tensor,
    W_d: torch.Tensor,
    routing_weight: torch.Tensor,
    expert_out: torch.Tensor,
    act_fn,
    packed_chunk_size: int,
) -> torch.Tensor:
    batch_size, seq_len = T2Ei.shape
    packed_chunk_size = max(1, min(packed_chunk_size, seq_len))

    # `_build_matched_idx_from_cumsum` already computes
    # `valid_prefix = T2Ei.to(int32).cumsum(dim=1)` and its last column IS
    # the per-batch count of active rows. Reuse it — this replaces the
    # earlier `.sum(dim=1)` (which lowered to an ONNX ReduceSum with
    # dynamic axes and was rejected by qaic-compile as
    # `ReduceSum_XXXX: Non-constant axes tensor not supported` on
    # gridsdca job 4992668) and also avoids the fp16-ones-vector matmul
    # I first tried (which failed at compile with QAIC backend assertion
    # `Unexpected DDR output buffer ... kind: StaticConstantDDR` on
    # gridsdca job 5284955 — the constant ones-vector caused the QAIC
    # backend to lift the MatMul output to constant-DDR memory kind,
    # which downstream torch.clamp/torch.where can't consume). Slicing
    # the last column of the cumsum reuses an op that's already in the
    # graph and lowers to a plain `Slice`, which QAIC handles cleanly.
    matched_idx, valid_rows = _build_matched_idx_from_cumsum(T2Ei)
    row_range = torch.arange(packed_chunk_size, dtype=torch.int32, device=x.device).unsqueeze(0)
    x_expanded = x.unsqueeze(0).expand(batch_size, -1, -1)

    for packed_start in range(0, seq_len, packed_chunk_size):
        packed_stop = packed_start + packed_chunk_size
        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]

        x_chunk = CtxGatherFunc3DGeneralized.apply(x_expanded, chunk_matched_idx)
        gate_prime = x_chunk @ W_g
        up_prime = x_chunk @ W_u
        down_chunk = (up_prime * act_fn(gate_prime)) @ W_d

        rw_chunk = CtxGatherFunc3DGeneralized.apply(routing_weight, chunk_matched_idx)
        down_chunk = down_chunk * rw_chunk

        expert_out_chunk = CtxGatherFunc3DGeneralized.apply(expert_out, chunk_matched_idx)
        updated_chunk = expert_out_chunk + down_chunk

        chunk_valid_rows = torch.clamp(
            valid_rows - packed_start,
            min=torch.zeros_like(valid_rows),
            max=torch.full_like(valid_rows, packed_chunk_size),
        )
        updated_chunk = torch.where(
            (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
        )
        expert_out = CtxScatterFunc3DGeneralized.apply(expert_out, chunk_matched_idx, updated_chunk)

    return expert_out


class QEffHYV3Attention(HYV3Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __qeff_init__(self):
        self.rotary_emb = QEffHYV3RotaryEmbedding(config=self.config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if sin_cached is not None and cos_cached is not None:
            sin, cos = sin_cached, cos_cached
            rotary_dim = int(self.rotary_emb.cos_cached.shape[-1])
            query_states, key_states = qeff_apply_precomputed_rotary_pos_emb(
                query_states, key_states, cos, sin, rotary_dim
            )
        else:
            kv_seq_len = (
                past_key_value.get_seq_length(self.layer_idx, cache_position)
                if past_key_value is not None
                else key_states.shape[-2]
            )
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            past_seen_tokens = past_key_value.get_seq_length(self.layer_idx) if past_key_value is not None else 0
            blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
            use_blocking = blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)
            if use_blocking:
                attn_output, attn_weights = generic_blocked_attention_interface(
                    module=self,
                    query=query_states,
                    key=key_states,
                    value=value_states,
                    attention_mask=attention_mask,
                    scaling=self.scaling,
                    layer_idx=self.layer_idx,
                    past_key_value=past_key_value,
                    blocking_config=blocking_config,
                    comp_ctx_lengths=comp_ctx_lengths,
                    batch_index=batch_index,
                    position_ids=position_ids,
                    past_seen_tokens=past_seen_tokens,
                )
            else:
                key_states, value_states, attention_mask, _ = past_key_value_update(
                    module=self,
                    key=key_states,
                    value=value_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    comp_ctx_lengths=comp_ctx_lengths,
                    batch_index=batch_index,
                    position_ids=position_ids,
                )
                attn_output, attn_weights = eager_attention_forward(
                    self,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling,
                    **kwargs,
                )
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffHYV3TopKRouter(HYV3TopKRouter):
    def forward(
        self,
        hidden_states: torch.Tensor,
        e_score_correction_bias: torch.Tensor,
    ) -> tuple:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = nn.functional.linear(hidden_states.float(), self.weight.float())
        routing_weights = torch.sigmoid(router_logits)

        scores_for_choice = routing_weights + e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)

        # Sum top_k_weights along the top_k dim for renormalization. This was
        # originally `torch.einsum("ab->a", top_k_weights)` — a workaround for
        # a subfunction-time break with `.sum(dim=1)` on an earlier SDK — but
        # SDK 1.22.1.12's `qaic-compile` rejects the resulting dynamic-axes
        # ReduceSum:
        #   [Operator-'ReduceSum_382'] : ReduceSum: Non-constant axes tensor not supported
        # (fired on gridsdca job 4970559, 2026-07-13, prefill compile). Unroll
        # across the known-constant top_k count — lowers to a chain of
        # constant-shape Adds, no ReduceSum. Same pattern applied to the top_k
        # sum in `QEffHYV3MoE.moe()` (see the block there for the full rationale).
        denominator = top_k_weights[:, 0]
        for k in range(1, self.top_k):
            denominator = denominator + top_k_weights[:, k]
        denominator = denominator.unsqueeze(-1) + 1e-20
        top_k_weights = top_k_weights / denominator
        top_k_weights = top_k_weights * self.router_scaling_factor

        return router_logits, top_k_weights, top_k_index


class QEffHYV3MoE(HYV3MoE):
    """
    MoE block. Fuses expert weights into stacked tensors at transform time and dispatches
    top-k experts via batched-BMM (M3: num_experts=192 > 32, a static per-expert loop hangs the compiler).
    """

    def __qeff_init__(self):
        # Cache plain-attribute references only — do NOT register new
        # `nn.Parameter`s over the fused HF weights. Registering a Parameter
        # promotes the alias into the model's state_dict, and at
        # `torch.jit._get_trace_graph` time each Parameter is materialized as
        # its own tensor in the traced graph. For a 192-expert / 79-layer MoE
        # (tencent/Hy3, 293B) that's 79 × 3 × ~4.8 GB = ~1.15 TB of transient
        # trace-time materialization on top of the ~275 GB fused-weight
        # resident — exactly what OOMed grid jobs 4582747 / 4861874 / 4866404 /
        # 4960374 at ~2.0 TB against the san-qp200 2.05 TB host cap. Instead
        # follow the qwen3_moe reference pattern: keep the fused source
        # unchanged, split/transpose lazily inside `_split_expert_weights`.
        self.gate_up_proj_w = self.experts.gate_up_proj  # [E, 2I, H] fused HF layout
        self.down_proj_w = self.experts.down_proj  # [E, H, I] fused HF layout
        self.expert_dim = self.gate_up_proj_w.shape[1] // 2
        self.act_fn = self.experts.act_fn

    def _split_expert_weights(self):
        """Return `(gate_proj_w, up_proj_w, down_proj_w)` — strided views over
        the fused HF weights in `[E, H, I]` layout suitable for BMM(x, W).

        All three returned tensors share storage with `self.experts.gate_up_proj`
        / `self.experts.down_proj` — zero materialized copies. Called once per
        forward instead of held as persistent Parameters (see `__qeff_init__`
        for why).
        """
        d = self.expert_dim
        gate_proj_w = self.gate_up_proj_w[:, :d, :].transpose(1, 2)  # [E, H, I]
        up_proj_w = self.gate_up_proj_w[:, d:, :].transpose(1, 2)  # [E, H, I]
        down_proj_w = self.down_proj_w.transpose(1, 2)  # [E, I, H]
        return gate_proj_w, up_proj_w, down_proj_w

    def moe(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        hidden_dim = self.gate.hidden_dim

        gate_proj_w, up_proj_w, down_proj_w = self._split_expert_weights()
        gate_proj = gate_proj_w[top_k_index.flatten()]
        up_proj = up_proj_w[top_k_index.flatten()]
        down_proj = down_proj_w[top_k_index.flatten()]

        expert_in = hidden_states.unsqueeze(1).expand(-1, self.top_k, -1).contiguous().view(-1, 1, hidden_dim)
        gate_out = torch.bmm(expert_in, gate_proj)
        up_out = torch.bmm(expert_in, up_proj)
        hidden = self.act_fn(gate_out) * up_out
        expert_output = torch.bmm(hidden, down_proj)

        experts_out = expert_output.view(num_tokens, self.top_k, hidden_dim)
        experts_out = experts_out * top_k_weights.unsqueeze(-1)
        # Sum the top_k contributions along dim=1. This was originally
        # `torch.einsum("abc->ac", experts_out)` — a workaround for a
        # subfunction-time break with `.sum(dim=1)` on an earlier SDK — but the
        # einsum lowers to an ONNX ReduceSum with a dynamic axes tensor, and
        # SDK 1.22.1.12's `qaic-compile` rejects that:
        #   [Operator-'ReduceSum_2308'] : ReduceSum: Non-constant axes tensor not supported
        # (fired on gridsdca job 4866404, 2026-07-12, prefill compile). Slice-
        # and-add across a known-constant top_k count lowers to a chain of
        # constant-shape Adds — no ReduceSum, no dynamic axes, and works cleanly
        # both with and without `use_onnx_subfunctions=True`.
        final_hidden_states = experts_out[:, 0]
        for k in range(1, self.top_k):
            final_hidden_states = final_hidden_states + experts_out[:, k]

        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        _, top_k_weights, top_k_index = self.gate(hidden_states, self.e_score_correction_bias)
        routed_output = self.moe(hidden_states, top_k_index, top_k_weights)

        if self.enable_moe_fp32_combine:
            hidden_states = (routed_output.float() + self.shared_experts(hidden_states).float()).to(
                hidden_states.dtype
            )
        else:
            hidden_states = routed_output + self.shared_experts(hidden_states)

        return hidden_states.reshape(batch_size, seq_len, hidden_dim)


class QEffPrefillChunkedHYV3MoE(QEffHYV3MoE):
    """
    Prefill-only MoE dispatch: reads all 192 experts once via NSP-blocked
    cumsum-scatter-gather (see `_cumsum_scatter_gather_update_expert_blocked`)
    instead of top-k batched-BMM. Registered via `PrefillOnlyChunkedTransform`
    for the disaggregated prefill QPC (`use_onnx_subfunctions=True`).
    """

    supports_moe_prefill_blocking = True

    def _forward_expert_blocked(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        num_experts = self.gate.num_experts
        num_nsp = self.expert_blocking_num_nsp
        if num_experts % num_nsp != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by expert_blocking_num_nsp ({num_nsp})")

        routing_weights = hidden_states.new_zeros((num_tokens, num_experts))
        routing_weights.scatter_(1, top_k_index, top_k_weights)

        local_experts = num_experts // num_nsp
        # Strided-view reshape, no `.contiguous()` — see `__qeff_init__` on
        # QEffHYV3MoE for why avoiding `nn.Parameter` aliases (and any
        # `.contiguous()` copies of the fused weights) is critical for the
        # huge-tier grid compile. `torch.matmul` inside
        # `_cumsum_scatter_gather_update_expert_blocked` accepts strided
        # operands; the per-slot dim-1 indexing (`W_g[:, slot]`) on a strided
        # 4-D view returns a strided sub-view — bit-identical to the same
        # slot's sub-view of a contiguous build (verified on tiny-random).
        gate_proj_w, up_proj_w, down_proj_w = self._split_expert_weights()
        rw = routing_weights.transpose(0, 1).view(local_experts, num_nsp, num_tokens).transpose(0, 1)
        W_g = gate_proj_w.view(local_experts, num_nsp, hidden_dim, -1).transpose(0, 1)
        W_u = up_proj_w.view(local_experts, num_nsp, hidden_dim, -1).transpose(0, 1)
        W_d = down_proj_w.view(local_experts, num_nsp, -1, hidden_dim).transpose(0, 1)
        expert_out = hidden_states.new_zeros((num_nsp, num_tokens, hidden_dim))
        routing_weights_unsqueezed = rw.unsqueeze(-1)

        for slot in range(local_experts):
            expert_out = _cumsum_scatter_gather_update_expert_blocked(
                x=hidden_states,
                T2Ei=rw[:, slot, :] > 0,
                W_g=W_g[:, slot],
                W_u=W_u[:, slot],
                W_d=W_d[:, slot],
                routing_weight=routing_weights_unsqueezed[:, slot],
                expert_out=expert_out,
                act_fn=self.act_fn,
                packed_chunk_size=self.expert_blocking_packed_chunk_size,
            )

        # Sum expert_out along the num_nsp axis. `.sum(dim=0)` lowers to
        # ONNX ReduceSum with a dynamic axes tensor which qaic-compile
        # rejects — same class as the other two ReduceSum sites fixed
        # in this file. num_nsp is `self.expert_blocking_num_nsp`,
        # a Python-int attribute set at model init, so unrolling across
        # it lowers to a chain of constant-shape Adds.
        final = expert_out[0]
        for i in range(1, num_nsp):
            final = final + expert_out[i]
        return final

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        _, top_k_weights, top_k_index = self.gate(hidden_states, self.e_score_correction_bias)
        routed_output = self._forward_expert_blocked(hidden_states, top_k_index, top_k_weights)

        if self.enable_moe_fp32_combine:
            hidden_states = (routed_output.float() + self.shared_experts(hidden_states).float()).to(
                hidden_states.dtype
            )
        else:
            hidden_states = routed_output + self.shared_experts(hidden_states)

        return hidden_states.reshape(batch_size, seq_len, hidden_dim)


class QEffHYV3DecoderLayer(HYV3DecoderLayer):
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
        position_embeddings: Optional[tuple] = None,  # necessary, but kept here for BC
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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


class QEffHYV3Model(HYV3Model):
    def __qeff_init__(self):
        self.rotary_emb = QEffHYV3RotaryEmbedding(config=self.config)
        self.sin_cached = torch.nn.Parameter(self.rotary_emb.sin_cached * self.rotary_emb.attention_scaling)
        self.cos_cached = torch.nn.Parameter(self.rotary_emb.cos_cached * self.rotary_emb.attention_scaling)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        attention_mask = _create_causal_mask(position_ids=position_ids, target_length=past_seen_tokens)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        sin = self.sin_cached[position_ids].unsqueeze(1)
        cos = self.cos_cached[position_ids].unsqueeze(1)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
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

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class QEffHYV3ForCausalLM(HYV3ForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        """
        Return the set of class used as the repeated layer across the model for subfunction extraction.
        """
        return {QEffHYV3DecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs: MoeModelOutputWithPast = self.model(
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

        hidden_states = outputs.last_hidden_state
        # INT32 gather index (U2) — ONNX Gather is strict about index dtype
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).to(hidden_states.dtype)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
