# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import os
from typing import List, Optional, Type

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaDecoderModel,
    DiffusionGemmaDecoderTextAttention,
    DiffusionGemmaDecoderTextLayer,
    DiffusionGemmaEncoderTextAttention,
    DiffusionGemmaEncoderTextLayer,
    DiffusionGemmaEncoderTextModel,
    DiffusionGemmaForBlockDiffusion,
    DiffusionGemmaRMSNorm,
    DiffusionGemmaTextExperts,
    DiffusionGemmaTextRouter,
    apply_rotary_pos_emb,
)

from QEfficient.customop.ctx_scatter_gather import CtxScatterFunc
from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffGemma4DynamicCache, QEffGemma4DynamicLayer
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_MASKED_ATTENTION_VALUE = -1e9  # finite under FP16; never -inf inside torch.where
_FP16_CLAMP_MIN = -65504.0
_FP16_CLAMP_MAX = 65504.0

EXPERT_BLOCKING_NUM_NSP = int(os.environ.get("EXPERT_BLOCKING_NUM_NSP", "16"))


# ---------------------------------------------------------------------------
# No-gather cache layer for the encoder prefill.
# The standard QEffDynamicLayer.update() appends a CtxGather with INT32_MAX indices
# for positions past gather_limit (needed for AR decode), which is illegal in ORT and
# mishandled by the QPC for the single-pass bulk prefill write. A one-shot prefill
# reads the full scatter buffer (positions 0..N all valid), so we skip the gather.
# ---------------------------------------------------------------------------
class _NoGatherCacheLayer(QEffGemma4DynamicLayer):
    """Cache layer that scatters KV at position_ids and returns the raw buffer (no CtxGather)."""

    def update(self, key_states, value_states, cache_kwargs=None):
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
            self._mark_initialized(self.keys)
            return self.keys, self.values
        self._mark_initialized(self.keys)
        position_ids = cache_kwargs.get("position_ids") if cache_kwargs else None
        if position_ids is not None:
            self.keys = CtxScatterFunc.apply(self.keys, position_ids, key_states)
            self.values = CtxScatterFunc.apply(self.values, position_ids, value_states)
        return self.keys, self.values


def _is_onnx_export() -> bool:
    return torch.onnx.is_in_onnx_export()


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Dynamic-shape-safe repeat_kv: uses -1 in reshape to avoid baking seq_len."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, _, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, -1, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, -1, head_dim)


def _clamp_to_fp16_range(t: torch.Tensor) -> torch.Tensor:
    if not _is_onnx_export():
        return t
    return t.clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX)


def _saturating_residual_add(residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """Mirrors gemma4's helper — fp32 sum then clamp back to fp16 range.

    Required for the canvas-decode path: without this clamp the decoder's residual
    stream can exceed fp16 ±65504 on hardware, producing inf in subsequent attn
    matmuls and NaN softmax rows. The encoder's layer-entry clamp kept its residual
    bounded; the decoder has no such guard.
    """
    if not _is_onnx_export():
        return residual + hidden_states
    return (residual.float() + hidden_states.float()).clamp(_FP16_CLAMP_MIN, _FP16_CLAMP_MAX).to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Custom RMSNorm — same as Gemma4 variant (with_scale=False support)
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaRMSNorm(DiffusionGemmaRMSNorm):
    """Export-safe RMSNorm.

    For ``with_scale=True`` modules, exports as the AIC ``CustomRMSNorm`` op
    (the proven compiler path). For ``with_scale=False`` modules (v_norm,
    self_conditioning.post_norm, router.norm with no parent-registered unit
    buffer), use the HF parent forward directly — it produces basic
    ``Pow/ReduceMean/Sqrt/Mul`` ONNX ops that the compiler handles correctly.

    The alternative (``CustomRMSNormFunc.apply(x, new_ones(dim), eps)``) exports a
    runtime ``Shape→Gather→Concat→ConstantOfShape`` chain feeding the CustomRMSNorm
    weight input, which computes the wrong norm on hardware. Bypassing the custom
    op when ``with_scale=False`` matches the eager forward exactly.
    """

    def __qeff_init__(self):
        if not getattr(self, "with_scale", True) and hasattr(self, "weight") and not hasattr(self, "_qeff_unit_weight"):
            self.register_buffer("_qeff_unit_weight", torch.ones_like(self.weight))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not _is_onnx_export():
            return super().forward(hidden_states)
        if getattr(self, "with_scale", True):
            return CustomRMSNormFunc.apply(hidden_states, self.weight, self.eps)
        # with_scale=False: the parent registered _qeff_unit_weight via Router
        # init. Use the AIC custom op when that buffer is real (compiler path
        # we've validated). Otherwise fall through to plain RMSNorm (no weight)
        # to avoid the broken ConstantOfShape-fed CustomRMSNorm chain.
        weight = getattr(self, "_qeff_unit_weight", None)
        if weight is not None and weight.shape[-1] != 1:
            return CustomRMSNormFunc.apply(hidden_states, weight, self.eps)
        return super().forward(hidden_states)


# ---------------------------------------------------------------------------
# Router — same as gemma4 (drop router_probabilities return, compile-safe topk)
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaTextRouter(DiffusionGemmaTextRouter):
    def __qeff_init__(self):
        if (
            hasattr(self, "norm")
            and not getattr(self.norm, "with_scale", True)
            and not hasattr(self.norm, "_qeff_unit_weight")
        ):
            self.norm.register_buffer("_qeff_unit_weight", torch.ones(self.hidden_size))

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size

        router_probabilities = nn.functional.softmax(self.proj(hidden_states), dim=-1)
        top_k_weights, top_k_index = torch.topk(
            router_probabilities,
            k=self.config.top_k_experts,
            dim=-1,
        )
        top_k_weights = top_k_weights / torch.einsum("bk->b", top_k_weights).unsqueeze(-1)
        top_k_weights = top_k_weights * self.per_expert_scale[top_k_index]
        return router_probabilities, top_k_weights, top_k_index


# ---------------------------------------------------------------------------
# Experts — batched BMM (same structure as QEffGemma4TextExperts)
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaTextExperts(DiffusionGemmaTextExperts):
    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        gate_up_proj_t = self.gate_up_proj.transpose(1, 2)
        gate_up_out = torch.matmul(hidden_states, gate_up_proj_t).permute(1, 0, 2)
        gate, up = gate_up_out.chunk(2, dim=-1)
        activated = self.act_fn(gate) * up

        down_proj_t = self.down_proj.transpose(1, 2)
        experts_out = torch.matmul(activated.permute(1, 0, 2), down_proj_t).permute(1, 0, 2)
        # Avoid scatter_add_ which traces to ScatterElements(reduction='add') in ONNX
        # and compiles incorrectly on AI 100 (large per-layer cosine error compounding
        # over 30 layers). Use broadcast equality + weighted sum instead (no scatter).
        # top_k_index: [tokens, top_k], top_k_weights: [tokens, top_k]
        # one_hot[t,k,e] = (top_k_index[t,k] == e)
        expert_ids = torch.arange(self.num_experts, device=top_k_index.device, dtype=top_k_index.dtype)
        one_hot = (top_k_index.unsqueeze(-1) == expert_ids.view(1, 1, -1)).to(top_k_weights.dtype)
        # expert_weights[t, e] = sum_k(one_hot[t,k,e] * top_k_weights[t,k])
        expert_weights = torch.einsum("tke,tk->te", one_hot, top_k_weights)
        weighted_experts = experts_out.transpose(1, 2)  # [tokens, hidden, num_experts]
        combine_weights = expert_weights.to(experts_out.dtype).unsqueeze(-1)  # [tokens, num_experts, 1]
        return torch.bmm(weighted_experts, combine_weights).squeeze(-1)


# ---------------------------------------------------------------------------
# Helpers for building QEff-safe attention masks
# ---------------------------------------------------------------------------


def _build_diffusion_encoder_additive_mask(
    position_ids: torch.Tensor,
    target_length: int,
    dtype: torch.dtype,
    sliding_window: Optional[int] = None,
    mm_token_type_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Export-safe additive attention mask for the encoder.

    When mm_token_type_ids is provided and the sequence is not a single decode
    step, builds a bidirectional mask for image-token groups (mirrors Gemma4's
    _build_bidirectional_vision_attention_mask logic).

    Mask value: HF uses ``torch.finfo(dtype).min`` so softmax of masked positions is
    exactly 0; we match that on CPU eager (HF==QEffPT parity). For ONNX export / QPC we
    use ``-1e4`` instead, because ``finfo.min`` overflows fp16 and causes attention NaNs
    on hardware.
    """
    mask_value = -1e4 if _is_onnx_export() else torch.finfo(dtype).min
    base_mask = _create_causal_mask(
        position_ids=position_ids,
        target_length=target_length,
        sliding_window=sliding_window,
    )
    if mm_token_type_ids is None or position_ids.shape[1] == 1:
        return base_mask.to(dtype=dtype) * mask_value

    # Build bidirectional attention within each contiguous image-token block
    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.to(torch.int64), dim=1) - 1
    vision_group_ids = torch.where(is_vision, vision_group_ids, torch.full_like(vision_group_ids, -1))

    kv_indices = torch.arange(target_length, device=vision_group_ids.device, dtype=torch.int64).view(1, -1)
    seq_len_limit = torch.full_like(kv_indices, vision_group_ids.shape[1] - 1)
    safe_kv_indices = torch.minimum(kv_indices, seq_len_limit)
    kv_group_ids = torch.gather(vision_group_ids, 1, safe_kv_indices.expand(vision_group_ids.shape[0], -1))
    kv_group_ids = torch.where(kv_indices < vision_group_ids.shape[1], kv_group_ids, torch.full_like(kv_group_ids, -1))

    same_group = (vision_group_ids.unsqueeze(-1) == kv_group_ids.unsqueeze(1)) & (vision_group_ids.unsqueeze(-1) >= 0)
    attention_mask = base_mask & ~same_group.unsqueeze(1)
    return attention_mask.to(dtype=dtype) * mask_value


def _build_diffusion_decoder_additive_mask(
    canvas_length: int,
    encoder_kv_length: int,
    dtype: torch.dtype,
    batch_size: int,
    device: torch.device,
    encoder_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Bidirectional mask for the diffusion decoder canvas.

    Canvas tokens attend bidirectionally to (a) the *real* encoder KV positions
    and (b) all other canvas tokens. They must NOT attend to padded/empty encoder
    cache slots: the encoder cache is sized to ctx_len, but only the first
    ``seq_len`` slots hold real prompt/image K/V — the remainder is junk that was
    never meant to be read. HF's ``create_diffusion_decoder_attention_mask`` masks
    exactly these slots (mllama/whisper/t5 do the same for cross-attention caches).
    Leaving them unmasked lets each junk slot dilute the softmax over the real K/V.

    ``encoder_attention_mask`` is the [batch, encoder_kv_length] 1/0 signal
    (1 = real, 0 = pad). When ``None`` the mask is all-attend (legacy behaviour, kept so
    the single-QPC / top-level paths that don't yet thread the signal are unchanged).

    Mask value follows the encoder convention: ``torch.finfo(dtype).min`` on CPU
    eager (exact 0 softmax weight, HF==QEffPT parity) and ``-1e4`` for ONNX/QPC
    (fp16-overflow-safe).

    EXPORT SAFETY: the returned mask keeps the query axis as size 1 and derives the
    KV axis purely from the ``encoder_attention_mask`` tensor's own (dynamic) shape —
    NO Python ``canvas_length``/``encoder_kv_length`` ints are baked into ``expand``.
    The additive mask ``[B, 1, 1, ctx+canvas]`` broadcasts over the canvas query axis
    inside ``attn_weights + mask``; baking those ints freezes the mask at the
    export-dummy width and broadcast-mismatches the real runtime KV width.

    Shape: [batch_size, 1, 1, encoder_kv_length + canvas_length]  (query axis broadcasts)
    """
    if encoder_attention_mask is None:
        # Legacy all-attend mask (no masking). Query axis size-1 (broadcasts).
        total_kv_len = encoder_kv_length + canvas_length
        return torch.zeros(batch_size, 1, 1, total_kv_len, dtype=dtype, device=device)

    mask_value = -1e4 if _is_onnx_export() else torch.finfo(dtype).min
    # Padded encoder slots -> mask_value; real slots -> 0. KV width follows the input
    # tensor's dynamic shape (no Python-int expand). [B, 1, 1, ctx]
    enc_additive = (1 - encoder_attention_mask.to(dtype)) * mask_value
    enc_part = enc_additive[:, None, None, :]
    # Canvas always fully attended (bidirectional). [B, 1, 1, canvas]
    canvas_part = torch.zeros(enc_part.shape[0], 1, 1, canvas_length, dtype=dtype, device=enc_part.device)
    return torch.cat([enc_part, canvas_part], dim=-1)


# ---------------------------------------------------------------------------
# Encoder attention — causal + KV cache update (QEff-patched)
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaEncoderTextAttention(DiffusionGemmaEncoderTextAttention):
    """Encoder attention with QEff-compatible eager forward."""

    def __qeff_init__(self):
        if hasattr(self, "v_norm") and not getattr(self.v_norm, "with_scale", True):
            if not hasattr(self.v_norm, "_qeff_unit_weight"):
                self.v_norm.register_buffer(
                    "_qeff_unit_weight",
                    torch.ones(self.head_dim, dtype=torch.float32),
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"position_ids": position_ids}
            )

        key_states_for_attn = _repeat_kv(key_states, self.num_key_value_groups)
        value_states_for_attn = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) * self.scaling
        if self.config.final_logit_softcapping is not None:
            pass  # softcapping is on logits only, not on attn_weights for this arch
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Decoder attention — bidirectional + read-only encoder KV cache
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaDecoderTextAttention(DiffusionGemmaDecoderTextAttention):
    """Decoder attention: bidirectional over canvas+encoder KV, no cache update."""

    def __qeff_init__(self):
        if hasattr(self, "v_norm") and not getattr(self.v_norm, "with_scale", True):
            if not hasattr(self.v_norm, "_qeff_unit_weight"):
                self.v_norm.register_buffer(
                    "_qeff_unit_weight",
                    torch.ones(self.head_dim, dtype=torch.float32),
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

        if past_key_values is not None:
            # Read encoder KV cache without updating it (bidirectional cross-attend)
            layer = past_key_values.layers[self.layer_idx]
            encoder_key_states = layer.keys
            encoder_value_states = layer.values
            key_states = torch.cat([encoder_key_states, key_states], dim=2)
            value_states = torch.cat([encoder_value_states, value_states], dim=2)

        key_states_for_attn = _repeat_kv(key_states, self.num_key_value_groups)
        value_states_for_attn = _repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states_for_attn.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            # Mask is pre-sized per layer_type by QEffDiffusionGemmaDecoderModel.forward.
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states_for_attn)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Encoder text layer — wraps MoE + attention with QEff-patched sub-modules
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaEncoderTextLayer(DiffusionGemmaEncoderTextLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = _clamp_to_fp16_range(hidden_states)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = self.router(hidden_states_flat)
        hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = _saturating_residual_add(hidden_states_1, hidden_states_2)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)
        hidden_states *= self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder text layer
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaDecoderTextLayer(DiffusionGemmaDecoderTextLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Clamp to fp16 range at layer entry + saturating residual adds (mirrors the
        # encoder layer). Without these the canvas-decode residual stream overflows
        # fp16 → softmax NaN.
        hidden_states = _clamp_to_fp16_range(hidden_states)
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)

        hidden_states_flat = residual.reshape(-1, residual.shape[-1])
        _, top_k_weights, top_k_index = self.router(hidden_states_flat)
        hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
        hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
        hidden_states_2 = hidden_states_2.reshape(residual.shape)
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)

        hidden_states = hidden_states_1 + hidden_states_2
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = _saturating_residual_add(residual, hidden_states)
        hidden_states *= self.layer_scalar
        return hidden_states


# ---------------------------------------------------------------------------
# Encoder text model
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaEncoderTextModel(DiffusionGemmaEncoderTextModel):
    """
    QEff-patched encoder text model.

    Replaces HF's dynamic mask creation with static QEff-friendly masks;
    uses QEffGemma4DynamicCache for the encoder KV cache.
    """

    def _precomputed_rope_gather(self, position_ids: torch.Tensor, layer_type: str, dtype: torch.dtype):
        """Gather-based RoPE for full_attention layers.

        Precomputes the cos/sin table at max_position_embeddings once, then indexes
        by position_ids via Gather. This avoids the runtime MatMul(inv_freq, positions)
        which mis-compiles when inv_freq has trailing zeros (partial_rotary_factor < 1).
        """
        inv_freq = getattr(self.rotary_emb, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self.rotary_emb, f"{layer_type}_attention_scaling")
        max_pos = min(self.config.max_position_embeddings, 4096)
        all_pos = torch.arange(max_pos, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(all_pos, inv_freq.float())  # [max_pos, D]
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_pos, 2D]
        cos_table = (emb.cos() * attention_scaling).to(dtype)  # [max_pos, 2D]
        sin_table = (emb.sin() * attention_scaling).to(dtype)
        # Gather: position_ids [B, S] → cos [B, S, 2D]
        pos_clamped = position_ids.clamp(min=0, max=max_pos - 1)
        cos = cos_table[pos_clamped]  # [B, S, 2D]
        sin = sin_table[pos_clamped]
        return cos, sin

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        # convert legacy cache to QEffGemma4DynamicCache at model entry
        if use_cache and isinstance(past_key_values, Cache) and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_cache(self.config, past_key_values)
        elif use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffGemma4DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            ).unsqueeze(0)

        hidden_states = inputs_embeds

        # Precompute RoPE once per layer-type. For full_attention, use a precomputed
        # Gather-from-table instead of the runtime MatMul(inv_freq, position_ids): the
        # full_attention inv_freq has 192/256 trailing zeros (partial_rotary_factor=0.25),
        # which mxfp6-quantize to tiny nonzeros → wrong angles at large positions. Storing
        # exact cos/sin as a buffer avoids this.
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            if layer_type == "full_attention" and _is_onnx_export():
                position_embeddings[layer_type] = self._precomputed_rope_gather(
                    position_ids, layer_type, hidden_states.dtype
                )
            else:
                position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        for i, encoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_type = self.config.layer_types[i]
            is_sliding = layer_type == "sliding_attention"
            sliding_window = self.config.sliding_window if is_sliding else None

            # Compute target_length from cached KV
            if past_key_values is not None and len(past_key_values.layers) > i:
                layer_keys = past_key_values.layers[i].keys
                if layer_keys is not None and layer_keys.numel() > 0:
                    target_length = layer_keys.shape[-2]
                else:
                    target_length = (
                        min(self.config.sliding_window, self.config.max_position_embeddings)
                        if is_sliding
                        else inputs_embeds.shape[1]
                    )
            else:
                target_length = (
                    min(self.config.sliding_window, self.config.max_position_embeddings)
                    if is_sliding
                    else inputs_embeds.shape[1]
                )

            layer_attention_mask = _build_diffusion_encoder_additive_mask(
                position_ids=position_ids,
                target_length=target_length,
                dtype=hidden_states.dtype,
                sliding_window=sliding_window,
                mm_token_type_ids=mm_token_type_ids if inputs_embeds.shape[1] != 1 else None,
            )

            hidden_states = encoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        next_cache = past_key_values.to_legacy_cache() if use_cache else None
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache)


# ---------------------------------------------------------------------------
# Decoder model — bidirectional diffusion denoiser
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaDecoderModel(DiffusionGemmaDecoderModel):
    """
    QEff-patched decoder model.

    The decoder reads the encoder's KV cache (read-only) and processes a
    fixed-length canvas with bidirectional attention over canvas+encoder KV.
    No KV cache update — the decoder is stateless across diffusion steps.
    """

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        prev_tokens: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # wrap legacy cache
        if past_key_values is not None and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(self.text_config, past_key_values)

        inputs_embeds = self.embed_tokens(decoder_input_ids)

        if prev_tokens is not None:
            # Fused-sampler path: feed the previous step's accepted token embedding as
            # self-conditioning (replaces softmax(prev_logits) @ embed_tokens.weight).
            soft_embeddings = self.embed_tokens(prev_tokens) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
        elif self_conditioning_logits is not None:
            soft_embeddings = torch.matmul(
                self_conditioning_logits.softmax(dim=-1, dtype=torch.float32).to(self.embed_tokens.weight.dtype),
                self.embed_tokens.weight,
            ) * self.embed_tokens.embed_scale.to(inputs_embeds.dtype)
        else:
            soft_embeddings = torch.zeros_like(inputs_embeds)
        inputs_embeds = self.self_conditioning(inputs_embeds, soft_embeddings)

        canvas_length = inputs_embeds.shape[1]
        if decoder_position_ids is None:
            cache_seq_length = past_key_values.get_seq_length(layer_idx=0) if past_key_values is not None else 0
            decoder_position_ids = torch.arange(
                cache_seq_length,
                cache_seq_length + canvas_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            ).unsqueeze(0)

        # Build one decoder mask per layer_type. sliding_attention layers have KV
        # width = sliding_window; full_attention layers have KV width = ctx_len.
        # A single-width mask can't cover both because `attn_weights + mask`
        # broadcasts on dim -1 (kv_length). We derive each type's mask width from
        # a representative layer of that type's key tensor via `torch.ones_like` on
        # a projection — this makes the KV axis a live shape-op in ONNX (Shape /
        # Gather on the key tensor), which the compiler resolves to sliding_window
        # vs ctx_len correctly per specialization.
        #
        # Caller-supplied encoder_attention_mask is ignored: slicing it per-type at
        # runtime traces as `aicdynamicrawslice`, which is not on-chip mappable at
        # unified-graph scale.
        mask_mapping = {}
        for layer_type in self.unique_layer_types:
            rep_layer_key = None
            if past_key_values is not None:
                for idx, lt in enumerate(self.text_config.layer_types):
                    if lt == layer_type and idx < len(past_key_values.layers):
                        candidate = past_key_values.layers[idx].keys
                        if candidate is not None:
                            rep_layer_key = candidate
                            break

            if rep_layer_key is not None:
                # ones_like on a [B, 1, KV, 1] slice reshaped to [B, KV] — KV axis
                # tracks rep_layer_key's dynamic dim -2 (sliding_window or ctx_len).
                kv_proj = rep_layer_key[:, 0:1, :, 0:1].reshape(rep_layer_key.shape[0], -1)
                per_type_mask = torch.ones_like(kv_proj, dtype=torch.int64)
            else:
                per_type_mask = None

            mask_mapping[layer_type] = _build_diffusion_decoder_additive_mask(
                canvas_length=canvas_length,
                encoder_kv_length=rep_layer_key.shape[-2] if rep_layer_key is not None else 0,
                dtype=inputs_embeds.dtype,
                batch_size=inputs_embeds.shape[0],
                device=inputs_embeds.device,
                encoder_attention_mask=per_type_mask,
            )

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, decoder_position_ids, layer_type)

        for i, decoder_layer in enumerate(self.layers[: self.text_config.num_hidden_layers]):
            layer_type = self.text_config.layer_types[i]
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=mask_mapping[layer_type],
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(_clamp_to_fp16_range(hidden_states))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)


# ---------------------------------------------------------------------------
# Vision encoder wrapper (for dual-QPC support)
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaVisionEncoderWrapper(nn.Module):
    """
    Standalone vision encoder wrapper for dual-QPC export.

    Runs vision_tower + embed_vision from the encoder model and clips
    outputs to FP16 range.
    """

    def __init__(self, model: "QEffDiffusionGemmaForBlockDiffusion"):
        super().__init__()
        self.model = model
        self.mm_tokens_per_image = getattr(model.config, "mm_tokens_per_image", 256)

    def get_submodules_for_export(self) -> Type[nn.Module]:
        encoder_model = self.model.model.encoder
        return {encoder_model.vision_tower.encoder.layers[0].__class__}

    def forward(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
        encoder_model = self.model.model.encoder
        vision_tower = encoder_model.vision_tower
        padding_positions = (image_position_ids == -1).all(dim=-1)

        inputs_embeds = vision_tower.patch_embedder(pixel_values, image_position_ids, padding_positions)
        valid_tokens = ~padding_positions
        vision_attention_mask = (~valid_tokens).unsqueeze(1).unsqueeze(2).to(dtype=inputs_embeds.dtype)
        vision_attention_mask = vision_attention_mask * torch.finfo(inputs_embeds.dtype).min
        vision_attention_mask = vision_attention_mask.expand(-1, 1, inputs_embeds.shape[1], -1)

        hidden_states = inputs_embeds
        position_embeddings = vision_tower.encoder.rotary_emb(hidden_states, image_position_ids)
        for layer in vision_tower.encoder.layers[: vision_tower.encoder.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=vision_attention_mask,
                position_embeddings=position_embeddings,
                position_ids=image_position_ids,
            )

        output_length = getattr(vision_tower.config, "default_output_length", None)
        if output_length is None:
            output_length = pixel_values.shape[-2] // (
                vision_tower.config.pooling_kernel_size * vision_tower.config.pooling_kernel_size
            )
        hidden_states, _ = vision_tower.pooler(
            hidden_states=hidden_states,
            pixel_position_ids=image_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        if vision_tower.config.standardize:
            hidden_states = (hidden_states - vision_tower.std_bias) * vision_tower.std_scale

        vision_embeds = encoder_model.embed_vision(inputs_embeds=hidden_states)
        if vision_embeds.dim() == 2:
            vision_embeds = vision_embeds.unsqueeze(0)

        # clamp vision projector output to FP16 range
        vision_embeds = vision_embeds.clamp(-60000.0, 60000.0)
        return vision_embeds[:, : self.mm_tokens_per_image, :]


# ---------------------------------------------------------------------------
# Language decoder wrapper (for dual-QPC support and single-QPC combined path)
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaUnifiedWrapper(nn.Module):
    """Single-QPC wrapper: runs encoder-prefill and canvas-decode child wrappers
    and dispatches between their outputs on a shape-derived predicate."""

    # Non-autoregressive: driven directly via QAICInferenceSession, not generate().
    supports_autoregressive_generate = False

    def __init__(self, model: "QEffDiffusionGemmaForBlockDiffusion"):
        super().__init__()
        self.model = model
        self.config = model.config
        self.text_config = model.config.text_config
        self.encoder_prefill = QEffDiffusionGemmaEncoderPrefillWrapper(model)
        self.canvas_decode = QEffDiffusionGemmaCanvasDecodeWrapper(model)

    def get_submodules_for_export(self):
        return {QEffDiffusionGemmaEncoderTextLayer, QEffDiffusionGemmaDecoderTextLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        image_idx: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        del batch_index, kwargs

        canvas_len = decoder_input_ids.shape[1] if decoder_input_ids is not None else 1
        is_encode = canvas_len == torch.tensor(1, dtype=torch.int64, device=decoder_input_ids.device)

        # Canvas decode runs first so it reads the input KV before encoder-prefill mutates it.
        dec_out = self.canvas_decode(
            decoder_input_ids=decoder_input_ids,
            decoder_position_ids=decoder_position_ids,
            self_conditioning_logits=self_conditioning_logits,
            past_key_values=past_key_values,
            encoder_attention_mask=encoder_attention_mask,
        )
        dec_logits = dec_out[0] if isinstance(dec_out, tuple) else dec_out

        enc_logits, next_image_idx, enc_pkv = self.encoder_prefill(
            input_ids=input_ids,
            position_ids=position_ids,
            vision_embeds=vision_embeds,
            image_idx=image_idx,
            mm_token_type_ids=mm_token_type_ids,
            past_key_values=past_key_values,
        )

        enc_canvas_logits = enc_logits.expand(-1, canvas_len, -1)
        canvas_logits = torch.where(is_encode.view(1, 1, 1), enc_canvas_logits, dec_logits)
        gated_image_idx = torch.where(is_encode.view(1, 1), next_image_idx, image_idx)

        return canvas_logits, gated_image_idx, enc_pkv

    def get_dummy_inputs(self, **kwargs):
        # Start from encoder-prefill dummies and override decoder-side keys with
        # canvas_len=1 (Prefill-shaped) so the wrapper traces both branches on the
        # same shape.
        enc_di = self.encoder_prefill.get_dummy_inputs()
        merged = {**enc_di}
        bs = enc_di["input_ids"].shape[0]
        vocab = self.text_config.vocab_size
        merged["decoder_input_ids"] = torch.zeros((bs, 1), dtype=torch.int64)
        merged["decoder_position_ids"] = torch.zeros((bs, 1), dtype=torch.int64)
        merged["self_conditioning_logits"] = torch.zeros((bs, 1, vocab), dtype=torch.float32)
        # encoder_attention_mask is optional (canvas_decode builds a fallback when None).
        # Leave it out of the export dummy to keep the decoder's fallback path traced.
        return merged

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        canvas_length: Optional[int] = None,
        **compiler_options,
    ):
        # Union: Prefill spec sized from encoder_prefill; Decode spec sized from canvas_decode.
        # Both carry vision_batch_size/vision_tokens for the encoder branch's inputs which
        # are still fed on the Decode spec (both branches evaluate — compiler folds).
        prefill_seq_len = prefill_seq_len or 32
        ctx_len = ctx_len or constants.INTERN_CTX_LEN
        canvas_length = canvas_length or getattr(self.config, "canvas_length", 256)
        text_cfg = self.config.text_config
        mm_tokens_per_image = self.model._get_mm_tokens_per_image()
        specs = [
            {
                "_graph_name": "Prefill",
                "batch_size": batch_size,
                "seq_len": prefill_seq_len,
                "canvas_len": 1,
                "ctx_len": ctx_len,
                "sliding_window": text_cfg.sliding_window,
                "vision_batch_size": batch_size,
                "vision_tokens": mm_tokens_per_image,
            },
            {
                "_graph_name": "Decode",
                "batch_size": batch_size,
                "seq_len": 1,
                "canvas_len": canvas_length,
                "ctx_len": ctx_len,
                "sliding_window": text_cfg.sliding_window,
                "vision_batch_size": batch_size,
                "vision_tokens": mm_tokens_per_image,
            },
        ]
        return specs, compiler_options

    def get_onnx_dynamic_axes(self, **kwargs):
        # Union of both children's dynamic axes.
        text_cfg = self.config.text_config
        axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_tokens"},
            "mm_token_type_ids": {0: "batch_size", 1: "seq_len"},
            "decoder_input_ids": {0: "batch_size", 1: "canvas_len"},
            "decoder_position_ids": {0: "batch_size", 1: "canvas_len"},
            "self_conditioning_logits": {0: "batch_size", 1: "canvas_len"},
        }
        for i, layer_type in enumerate(text_cfg.layer_types):
            ctx_axis = {0: "batch_size", 2: "sliding_window" if layer_type == "sliding_attention" else "ctx_len"}
            for kv in ("key", "value"):
                axes[f"past_{kv}.{i}"] = ctx_axis
        return axes

    def get_output_names(self, **kwargs):
        # Encoder branch writes KV as retained-state outputs (its own
        # EncoderPrefillWrapper naming is `past_*_out`; on the unified graph we
        # bind them as retained-state so qaic-compile keeps one on-device
        # allocation across the two specializations of the same QPC).
        text_cfg = self.config.text_config
        names = ["canvas_logits", "image_idx_output"]
        for i in range(text_cfg.num_hidden_layers):
            for kv in ("key", "value"):
                names.append(f"past_{kv}.{i}_RetainedState")
        return names


# ---------------------------------------------------------------------------
# Disaggregated dual-QPC wrappers
# ---------------------------------------------------------------------------
# Two independent single-graph QPCs:
#   1. Encoder-prefill: input_ids + vision -> writes KV -> past_*_RetainedState
#   2. Canvas-decode:   canvas + encoder KV (read-only input) -> canvas_logits
# The runner host-copies past_*.{i}_RetainedState from encoder to past_*.{i} on
# the decoder each diffusion step (examples/disagg_serving/* pattern).


class QEffDiffusionGemmaEncoderPrefillWrapper(nn.Module):
    """Standalone encoder-prefill QPC: prompt(+vision) → filled KV cache.

    No decoder path, no is_encode gate. The encoder writes the KV cache via
    past_key_values.update(); we return the filled KV as _RetainedState outputs.
    """

    def __init__(self, model: "QEffDiffusionGemmaForBlockDiffusion"):
        super().__init__()
        self.model = model
        self.config = model.config
        self.text_config = model.config.text_config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDiffusionGemmaEncoderTextLayer}

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        vision_embeds: Optional[torch.Tensor] = None,
        image_idx: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ):
        del kwargs
        text_cfg = self.config.text_config
        if past_key_values is not None and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(text_cfg, past_key_values)

        # Force ALL cache layers non-sliding for the encoder prefill. The sliding-window
        # rolling gather corrupts KV under bulk single-pass prefill + padding (gather_limit
        # produces INT32_MAX gather indices for padding positions). Non-sliding falls through
        # to the parent's plain CtxScatter+CtxGather, which handles padding correctly. Valid
        # because the encoder prefills the full prompt (seq_len < sliding_window) in one pass.
        if past_key_values is not None:
            for layer in past_key_values.layers:
                layer.is_sliding = False

        # Replace cache layers with _NoGatherCacheLayer: the standard update() traces a
        # CtxGather with INT32_MAX sentinel indices (illegal in ORT, corrupt in the QPC).
        # A single-pass prefill reads the full scatter buffer, so the gather is unneeded.
        if past_key_values is not None:
            for i, layer in enumerate(past_key_values.layers):
                new_layer = _NoGatherCacheLayer(is_sliding=False)
                new_layer.keys = layer.keys
                new_layer.values = layer.values
                if layer.keys is not None:
                    new_layer._mark_initialized(layer.keys)
                past_key_values.layers[i] = new_layer

        # Inject vision into the text embeddings, then run the encoder language model
        # (writes the KV cache in-place via past_key_values.update — no gate, no clone).
        inputs_embeds, next_image_idx = self.model._inject_vision_embeds(input_ids, vision_embeds, image_idx)

        # Pass mm_token_type_ids=None so we do NOT apply the bidirectional-vision attention
        # override over image-token blocks. HF's encoder forward also skips it on the standard
        # path (it calls create_masks_for_generate(...) but discards the result, running the
        # language_model with attention_mask=None). Mirroring that gives bit-identical HF↔QEff
        # parity; building the bidirectional mask instead diverges QEff from HF across layers.
        enc_outputs = self.model.model.encoder.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            mm_token_type_ids=None,
        )

        # Emit a small "enc_logits" output (encoder's last-position logit) as a graph-liveness
        # anchor: without a non-trivial output consumer the compiler dead-eliminates the
        # KV-write chain (fp16 retained-state outputs get folded away). The runner does not
        # read it.
        hidden_states = enc_outputs.last_hidden_state
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        last_hidden = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        enc_logits = self.model._apply_logit_softcapping(self.model.lm_head(last_hidden).float())

        pkv = [
            (past_key_values.layers[i].keys, past_key_values.layers[i].values)
            for i in range(text_cfg.num_hidden_layers)
        ]
        return enc_logits, next_image_idx, pkv

    # -- export metadata (single Prefill specialization) --

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        canvas_length: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len or 32
        ctx_len = ctx_len or constants.INTERN_CTX_LEN
        text_cfg = self.config.text_config
        mm_tokens_per_image = self.model._get_mm_tokens_per_image()
        spec = {
            "batch_size": batch_size,
            "seq_len": prefill_seq_len,
            "ctx_len": ctx_len,
            "sliding_window": text_cfg.sliding_window,
            "vision_batch_size": batch_size,
            "vision_tokens": mm_tokens_per_image,
        }
        return [spec], compiler_options

    def get_onnx_dynamic_axes(self, **kwargs):
        text_cfg = self.config.text_config
        axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_tokens"},
            "mm_token_type_ids": {0: "batch_size", 1: "seq_len"},
        }
        for i, layer_type in enumerate(text_cfg.layer_types):
            ctx_axis = {0: "batch_size", 2: "sliding_window" if layer_type == "sliding_attention" else "ctx_len"}
            for kv in ("key", "value"):
                axes[f"past_{kv}.{i}"] = ctx_axis
        return axes

    def get_output_names(self, **kwargs):
        text_cfg = self.config.text_config
        # enc_logits is a graph-liveness anchor (not consumed by the runner) — see forward().
        # past_*_keyout / past_*_valout are the encoder-filled KV emitted as REGULAR outputs
        # (not _RetainedState). Single-spec encoder QPCs have their _RetainedState pathway
        # dead-elimed by qaic-compile (no in-graph consumer); regular outputs are user-visible
        # and survive. The runner host-copies them into the decoder QPC's past_*.{i} inputs.
        names = ["enc_logits", "image_idx_output"]
        for i in range(text_cfg.num_hidden_layers):
            for kv in ("key", "value"):
                names.append(f"past_{kv}.{i}_out")
        return names

    def get_dummy_inputs(self, **kwargs):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        mm_tokens_per_image = self.model._get_mm_tokens_per_image()
        text_cfg = self.config.text_config
        seq_len = max(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, mm_tokens_per_image + 32)

        input_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        mm_token_type_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        text_prefix_len = min(5, seq_len)
        image_start = text_prefix_len
        image_end = min(image_start + mm_tokens_per_image, seq_len)
        input_ids[:, image_start:image_end] = self.config.image_token_id
        mm_token_type_ids[:, image_start:image_end] = 1

        return {
            "input_ids": input_ids,
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "vision_embeds": torch.zeros((bs, mm_tokens_per_image, text_cfg.hidden_size), dtype=torch.float32),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "mm_token_type_ids": mm_token_type_ids,
            "past_key_values": self.model.get_dummy_pkv_cache(config=text_cfg, batch_size=bs, seq_len=seq_len),
        }


class QEffDiffusionGemmaCanvasDecodeWrapper(nn.Module):
    """Standalone canvas-decode QPC: canvas + encoder KV (read-only) → canvas_logits.

    No encoder path, no is_encode gate. past_key_values are graph INPUTS (the
    encoder's filled KV), read by the decoder cross-attention without update — so
    they are NOT emitted as _RetainedState outputs (the runner feeds them fresh
    each diffusion step from the encoder QPC's capture).

    Optional fused-sampler mode (`fuse_sampler=True`): instead of returning the
    [B,canvas,vocab] canvas_logits (which forces the host through a vocab-wide
    softmax + Gumbel-max per step), the QPC emits two compact outputs —
    `denoiser_tokens [B,canvas]` and `token_entropy [B,canvas]` — computed
    on-device. The host shrinks to a length-canvas argsort+cumsum acceptance.
    Uses the canonical QEff sampler pattern (top-k Gumbel-max with a host-fed
    `random_numbers [B,canvas,max_top_k]` seed); see
    `QEfficient/transformers/sampler/sampler.py`.
    """

    def __init__(
        self,
        model: "QEffDiffusionGemmaForBlockDiffusion",
        fuse_sampler: bool = False,
        max_top_k: int = 64,
    ):
        super().__init__()
        self.model = model
        self.lm_head = model.lm_head
        self.config = model.config
        self.text_config = model.config.text_config
        self.fuse_sampler = fuse_sampler
        self.max_top_k = max_top_k

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDiffusionGemmaDecoderTextLayer}

    def forward(
        self,
        decoder_input_ids: torch.LongTensor,
        decoder_position_ids: torch.LongTensor,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
        random_numbers: Optional[torch.Tensor] = None,
        prev_tokens: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        del kwargs
        text_cfg = self.config.text_config
        if past_key_values is not None and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(text_cfg, past_key_values)

        # Fused mode uses token-feedback self-conditioning (cheap [B,256] input) and must NOT
        # also receive self_conditioning_logits — feeding both would double-apply feedback.
        dec_out = self.model.model.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=None if self.fuse_sampler else self_conditioning_logits,
            decoder_position_ids=decoder_position_ids,
            encoder_attention_mask=encoder_attention_mask,
            prev_tokens=prev_tokens if self.fuse_sampler else None,
        )
        canvas_logits = self.model._apply_logit_softcapping(self.lm_head(dec_out.last_hidden_state).float())

        if not self.fuse_sampler:
            return (canvas_logits,)

        # Fused on-device sampler. temperature [B,1,1] broadcasts over [B,canvas,vocab];
        # random_numbers [B,canvas,k] in (0,1) are drawn on the host. The 1e-20 clamps
        # avoid log(0) and match the host Gumbel implementation.
        lt = canvas_logits / temperature
        # entropy via log_softmax (numerically stable; matches host _denoising_step bit-close)
        shifted = lt - lt.amax(dim=-1, keepdim=True)
        log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1, keepdim=True))
        log_softmax = shifted - log_sum_exp
        token_entropy = -(torch.exp(log_softmax) * log_softmax).sum(dim=-1)  # [B, canvas]

        # top-k Gumbel-max denoiser: tiny RNG seed [B,canvas,k], not 256 MB
        topk_values, topk_indices = torch.topk(lt, k=self.max_top_k, dim=-1)  # [B,canvas,k]
        gumbel = -torch.log(-torch.log(random_numbers + 1e-20) + 1e-20)
        win = (topk_values + gumbel).argmax(dim=-1, keepdim=True)  # [B,canvas,1]
        denoiser_tokens = topk_indices.gather(dim=-1, index=win).squeeze(-1)  # [B, canvas]

        return (denoiser_tokens, token_entropy)

    # -- export metadata (single Decode specialization) --

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        canvas_length: Optional[int] = None,
        **compiler_options,
    ):
        ctx_len = ctx_len or constants.INTERN_CTX_LEN
        canvas_length = canvas_length or getattr(self.config, "canvas_length", 256)
        text_cfg = self.config.text_config
        spec = {
            "batch_size": batch_size,
            "canvas_len": canvas_length,
            "ctx_len": ctx_len,
            "sliding_window": text_cfg.sliding_window,
        }
        if self.fuse_sampler:
            spec["max_top_k"] = self.max_top_k
        return [spec], compiler_options

    def get_onnx_dynamic_axes(self, **kwargs):
        text_cfg = self.config.text_config
        axes = {
            "decoder_input_ids": {0: "batch_size", 1: "canvas_len"},
            "decoder_position_ids": {0: "batch_size", 1: "canvas_len"},
            "encoder_attention_mask": {0: "batch_size", 1: "ctx_len"},
        }
        if self.fuse_sampler:
            # temperature is [B,1,1] (broadcast scalar per batch) — only batch dim is dynamic.
            # random_numbers is [B,canvas,max_top_k] — k is a fixed spec symbol so no axis here.
            # prev_tokens is [B,canvas] — the previous step's denoiser_tokens fed back for the
            # decoder's self-conditioning path (replaces the 256 MB self_conditioning_logits).
            axes["temperature"] = {0: "batch_size"}
            axes["random_numbers"] = {0: "batch_size", 1: "canvas_len"}
            axes["prev_tokens"] = {0: "batch_size", 1: "canvas_len"}
        else:
            axes["self_conditioning_logits"] = {0: "batch_size", 1: "canvas_len"}
        for i, layer_type in enumerate(text_cfg.layer_types):
            ctx_axis = {0: "batch_size", 2: "sliding_window" if layer_type == "sliding_attention" else "ctx_len"}
            for kv in ("key", "value"):
                axes[f"past_{kv}.{i}"] = ctx_axis
        return axes

    def get_output_names(self, **kwargs):
        if self.fuse_sampler:
            return ["denoiser_tokens", "token_entropy"]
        return ["canvas_logits"]

    def get_dummy_inputs(self, **kwargs):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        text_cfg = self.config.text_config
        canvas_length = getattr(self.config, "canvas_length", 256)
        # KV is sized to ctx so the decoder cross-attends the full encoder cache.
        seq_len = max(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, self.model._get_mm_tokens_per_image() + 32)
        # encoder_attention_mask: per-slot 1=real / 0=pad over the encoder KV cache.
        # Dummy marks all ctx slots real (worst case for graph liveness); the runner
        # feeds the true mask (1 up to the real prompt length, 0 beyond) at inference.
        enc_attn_mask = torch.ones((bs, seq_len), dtype=torch.int64)
        inputs = {
            "decoder_input_ids": torch.zeros((bs, canvas_length), dtype=torch.int64),
            "decoder_position_ids": torch.arange(canvas_length, dtype=torch.int64).view(1, canvas_length).repeat(bs, 1),
            "encoder_attention_mask": enc_attn_mask,
            "past_key_values": self.model.get_dummy_pkv_cache(config=text_cfg, batch_size=bs, seq_len=seq_len),
        }
        if self.fuse_sampler:
            # Plausible non-degenerate dummies so the graph traces a meaningful softmax/topk.
            inputs["temperature"] = torch.full((bs, 1, 1), 0.7, dtype=torch.float32)
            inputs["random_numbers"] = torch.rand((bs, canvas_length, self.max_top_k), dtype=torch.float32) * 0.998 + 1e-3
            # prev_tokens: arbitrary valid token IDs in [0, vocab). Pretty trace-safe at zero too.
            inputs["prev_tokens"] = torch.zeros((bs, canvas_length), dtype=torch.int64)
        else:
            # Soft self-conditioning logits — only the un-fused path consumes this 256MB tensor.
            inputs["self_conditioning_logits"] = torch.zeros((bs, canvas_length, text_cfg.vocab_size), dtype=torch.float32)
        return inputs
        return inputs


# ---------------------------------------------------------------------------
# Top-level model class — registered to AutoModelForImageTextToText
# ---------------------------------------------------------------------------


class QEffDiffusionGemmaForBlockDiffusion(DiffusionGemmaForBlockDiffusion):
    """
    QEff-patched DiffusionGemmaForBlockDiffusion.

    Registered to AutoModelForImageTextToText.

    Supports:
      - Single-QPC: encoder prefill + decoder canvas-denoise in one compiled graph
      - Dual-QPC  : separate vision and language QPCs (kv_offload=True)
    """

    def get_qeff_vision_encoder(self) -> QEffDiffusionGemmaVisionEncoderWrapper:
        return QEffDiffusionGemmaVisionEncoderWrapper(self)

    def get_qeff_language_decoder(self) -> QEffDiffusionGemmaUnifiedWrapper:
        return QEffDiffusionGemmaUnifiedWrapper(self)

    def get_qeff_unified_wrapper(self) -> QEffDiffusionGemmaUnifiedWrapper:
        """Single-QPC unified wrapper (encoder-prefill + canvas-decode in one QPC)."""
        return QEffDiffusionGemmaUnifiedWrapper(self)

    def get_qeff_encoder_prefill(self) -> QEffDiffusionGemmaEncoderPrefillWrapper:
        """Disaggregated dual-QPC: standalone encoder-prefill QPC."""
        return QEffDiffusionGemmaEncoderPrefillWrapper(self)

    def get_qeff_canvas_decode(
        self,
        fuse_sampler: bool = False,
        max_top_k: int = 64,
    ) -> QEffDiffusionGemmaCanvasDecodeWrapper:
        """Disaggregated dual-QPC: build the standalone canvas-decode wrapper.

        Set ``fuse_sampler=True`` to emit ``denoiser_tokens`` / ``token_entropy``
        on-device instead of the full ``[B, canvas, vocab]`` ``canvas_logits``.
        """
        return QEffDiffusionGemmaCanvasDecodeWrapper(self, fuse_sampler=fuse_sampler, max_top_k=max_top_k)

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffDiffusionGemmaEncoderTextLayer}

    def _get_mm_tokens_per_image(self) -> int:
        return getattr(
            self.config.vision_config,
            "default_output_length",
            getattr(self.config, "mm_tokens_per_image", 256),
        )

    def _get_vision_max_patches(self) -> int:
        pooling_kernel_size = getattr(self.config.vision_config, "pooling_kernel_size", 3)
        default_output_length = getattr(self.config.vision_config, "default_output_length", 280)
        return default_output_length * pooling_kernel_size * pooling_kernel_size

    def get_dummy_pkv_cache(self, config, batch_size: int, seq_len: int):
        past_key_values = []
        for layer_type in config.layer_types:
            if layer_type == "sliding_attention":
                n_heads = config.num_key_value_heads
                d_head = config.head_dim
                layer_seq_len = min(config.sliding_window, seq_len)
            else:
                n_heads = config.num_global_key_value_heads or config.num_key_value_heads
                d_head = getattr(config, "global_head_dim", None) or config.head_dim
                layer_seq_len = seq_len
            cache_shape = [batch_size, n_heads, layer_seq_len, d_head]
            past_key_values.append(
                (torch.zeros(cache_shape, dtype=torch.float32), torch.zeros(cache_shape, dtype=torch.float32))
            )
        return past_key_values

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        canvas_length: Optional[int] = None,
        img_size: Optional[int] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len or 32
        ctx_len = ctx_len or constants.INTERN_CTX_LEN
        canvas_length = canvas_length or getattr(self.config, "canvas_length", 256)
        mm_tokens_per_image = self._get_mm_tokens_per_image()
        max_patches = self._get_vision_max_patches()
        text_cfg = self.config.text_config

        vision = [{"batch_size": batch_size, "max_patches": max_patches}]

        def build_encoder_prefill_spec(comp_ctx_lengths=None):
            spec = {
                "_graph_name": "Prefill",
                "batch_size": 1 if continuous_batching else batch_size,
                # seq_len=prefill_seq_len uniquely identifies the encoder specialization
                "seq_len": prefill_seq_len,
                "canvas_len": canvas_length,
                "ctx_len": ctx_len,
                "sliding_window": text_cfg.sliding_window,
                "vision_batch_size": batch_size,
                "vision_tokens": mm_tokens_per_image,
                "is_encode": 1,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size or batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size or batch_size
            return spec

        def build_decoder_canvas_spec(comp_ctx_lengths=None):
            # seq_len=1 for decoder: the compiler uses this shape change to dispatch
            # (seq_len differs from encoder's prefill_seq_len, uniquely identifying this spec)
            spec = {
                "_graph_name": "Decode",
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": 1,
                "canvas_len": canvas_length,
                "ctx_len": ctx_len,
                "sliding_window": text_cfg.sliding_window,
                "vision_batch_size": batch_size,
                "vision_tokens": mm_tokens_per_image,
                "is_encode": 0,
            }
            if comp_ctx_lengths is not None:
                spec["comp_ctx_lengths"] = comp_ctx_lengths
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size or batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size or batch_size
            return spec

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            lang = [build_encoder_prefill_spec(ccl) for ccl in comp_ctx_lengths_prefill]
            lang.extend(build_decoder_canvas_spec(ccl) for ccl in comp_ctx_lengths_decode)
        else:
            lang = [build_encoder_prefill_spec(), build_decoder_canvas_spec()]

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options
        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        text_cfg = self.config.text_config

        vision_dynamic_axes = {
            "pixel_values": {0: "batch_size", 1: "max_patches"},
            "image_position_ids": {0: "batch_size", 1: "max_patches"},
        }
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "decoder_input_ids": {0: "batch_size", 1: "canvas_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_tokens"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "mm_token_type_ids": {0: "batch_size", 1: "seq_len"},
            "decoder_position_ids": {0: "batch_size", 1: "canvas_len"},
            "self_conditioning_logits": {0: "batch_size", 1: "canvas_len"},
        }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        for i, layer_type in enumerate(text_cfg.layer_types):
            if layer_type == "sliding_attention":
                ctx_axis = {
                    0: "full_batch_size" if continuous_batching else "batch_size",
                    2: "sliding_window",
                }
            else:
                ctx_axis = {
                    0: "full_batch_size" if continuous_batching else "batch_size",
                    2: "ctx_len",
                }
            for kv in ("key", "value"):
                lang_dynamic_axes[f"past_{kv}.{i}"] = ctx_axis

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}
        return {**vision_dynamic_axes, **lang_dynamic_axes}

    def get_output_names(self, kv_offload: bool = False):
        text_cfg = self.config.text_config
        vision_output_names = ["vision_embeds"]
        # Unified output names for both encoder (canvas_len=1) and decoder (canvas_len=canvas_length).
        # Encoder: canvas_logits[bs,1,vocab] holds the first-token logit (TTFT).
        # Decoder: canvas_logits[bs,canvas_length,vocab] holds the denoised token logits.
        lang_output_names = [
            "canvas_logits",
            "vision_embeds_RetainedState",
            "image_idx_output",
        ]
        for i in range(text_cfg.num_hidden_layers):
            for kv in ("key", "value"):
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")
        if kv_offload:
            return {"vision": vision_output_names, "lang": lang_output_names}
        return lang_output_names

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS
        mm_tokens_per_image = self._get_mm_tokens_per_image()
        max_patches = self._get_vision_max_patches()
        canvas_length = getattr(self.config, "canvas_length", 256)
        text_cfg = self.config.text_config
        seq_len = max(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, mm_tokens_per_image + 32)
        patch_dim = getattr(self.config.vision_config, "patch_size", 16) ** 2 * 3

        # Build image_position_ids
        image_position_ids = torch.full((bs, max_patches, 2), -1, dtype=torch.int64)
        pooled_side = int(mm_tokens_per_image**0.5)
        patch_side = pooled_side * getattr(self.config.vision_config, "pooling_kernel_size", 3)
        xs = torch.arange(patch_side, dtype=torch.int64).view(1, -1).expand(patch_side, -1).reshape(-1)
        ys = torch.arange(patch_side, dtype=torch.int64).view(-1, 1).expand(-1, patch_side).reshape(-1)
        valid_positions = torch.stack((xs, ys), dim=-1)
        image_position_ids[:, : valid_positions.shape[0], :] = valid_positions.unsqueeze(0)

        input_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        mm_token_type_ids = torch.zeros((bs, seq_len), dtype=torch.int64)
        text_prefix_len = min(5, seq_len)
        image_start = text_prefix_len
        image_end = min(image_start + mm_tokens_per_image, seq_len)
        input_ids[:, image_start:image_end] = self.config.image_token_id
        mm_token_type_ids[:, image_start:image_end] = 1

        vision_inputs = {
            "pixel_values": torch.zeros((bs, max_patches, patch_dim), dtype=torch.float32),
            "image_position_ids": image_position_ids,
        }
        # is_encode=1 (encoder sentinel) for the export trace.
        # Both encoder AND decoder paths are executed during torch.onnx.export because
        # is_encode flows through the graph as a real tensor input (not a Python bool).
        # The compiler constant-folds is_encode=1/0 per specialization.
        lang_inputs = {
            "input_ids": input_ids,
            "vision_embeds": torch.zeros((bs, mm_tokens_per_image, text_cfg.hidden_size), dtype=torch.float32),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "mm_token_type_ids": mm_token_type_ids,
            "decoder_input_ids": torch.zeros((bs, canvas_length), dtype=torch.int64),
            "decoder_position_ids": torch.arange(canvas_length, dtype=torch.int64).view(1, canvas_length).repeat(bs, 1),
            "self_conditioning_logits": torch.zeros((bs, canvas_length, text_cfg.vocab_size), dtype=torch.float32),
            "is_encode": torch.ones(1, dtype=torch.int64),
            "past_key_values": self.get_dummy_pkv_cache(
                config=text_cfg,
                batch_size=fbs if continuous_batching else bs,
                seq_len=seq_len,
            ),
        }
        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int8)
        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        return {**vision_inputs, **lang_inputs}

    def _apply_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits

    def _inject_vision_embeds(
        self,
        input_ids: torch.LongTensor,
        vision_embeds: Optional[torch.Tensor],
        image_idx: Optional[torch.Tensor],
    ):
        """Inject vision features into the text embeddings at image-token positions.

        Shared by all three export paths (single-QPC forward, dual-QPC LangWrapper,
        disaggregated encoder-prefill). Uses torch.where instead of masked_scatter and
        clamps the gather index to avoid -1, both for export safety.
        """
        encoder_model = self.model.encoder
        lang_model = encoder_model.language_model
        text_cfg = self.config.text_config

        special_image_mask = input_ids == self.config.image_token_id
        llm_input_ids = input_ids.clone()
        llm_input_ids[special_image_mask] = text_cfg.pad_token_id
        inputs_embeds = lang_model.embed_tokens(llm_input_ids)

        next_image_idx = image_idx
        if input_ids.shape[1] != 1 and special_image_mask.any() and vision_embeds is not None:
            if vision_embeds.dim() == 2:
                vision_embeds = vision_embeds.unsqueeze(0)
            if next_image_idx is None:
                next_image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)

            indices1 = special_image_mask.to(torch.int64).cumsum(1) - 1
            indices1 = torch.where(
                indices1 >= 0,
                indices1 + next_image_idx.to(indices1.device),
                indices1,
            )
            indices0 = torch.arange(special_image_mask.shape[0], device=inputs_embeds.device).view(-1, 1)
            safe_indices1 = indices1.clamp(min=0)
            gathered_vision_embeds = vision_embeds[indices0, safe_indices1]
            inputs_embeds = torch.where(special_image_mask.unsqueeze(-1), gathered_vision_embeds, inputs_embeds)
            next_image_idx = (indices1.max() + 1).reshape(1, 1)

        if next_image_idx is None:
            next_image_idx = torch.zeros((1, 1), dtype=torch.int64, device=inputs_embeds.device)

        return inputs_embeds, next_image_idx

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        image_idx: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_position_ids: Optional[torch.LongTensor] = None,
        self_conditioning_logits: Optional[torch.FloatTensor] = None,
        is_encode: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        is_encode: scalar int64 tensor, 1 = encoder prefill, 0 = canvas decode.
        Passed as a real graph input so torch.where(is_encode, ...) is preserved
        in the ONNX graph and constant-folded per specialization by the compiler.
        """
        del attention_mask, inputs_embeds, labels, use_cache, batch_index, kwargs

        # convert legacy cache to QEffGemma4DynamicCache
        text_cfg = self.config.text_config
        if past_key_values is not None and not isinstance(past_key_values, QEffGemma4DynamicCache):
            past_key_values = QEffGemma4DynamicCache.from_legacy_cache(text_cfg, past_key_values)

        if is_encode is None:
            is_encode = torch.ones(1, dtype=torch.int64, device=input_ids.device)

        is_enc_bool = is_encode.bool()

        # Clone KV before encoder to preserve the original ONNX nodes.
        # ONNX in-place mutation (CtxScatterFunc) rebinds the Python tensor object to
        # the ScatterND output — without clone, orig_keys would alias the post-scatter node.
        orig_keys = [layer.keys.clone() for layer in past_key_values.layers]
        orig_vals = [layer.values.clone() for layer in past_key_values.layers]

        # ---- Encoder path (always traced) ----
        enc_inputs_embeds, next_image_idx = self._inject_vision_embeds(input_ids, vision_embeds, image_idx)
        enc_outputs = self.model.encoder.language_model(
            inputs_embeds=enc_inputs_embeds,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            mm_token_type_ids=mm_token_type_ids,
        )
        enc_keys = [past_key_values.layers[i].keys for i in range(text_cfg.num_hidden_layers)]
        enc_vals = [past_key_values.layers[i].values for i in range(text_cfg.num_hidden_layers)]

        # INT32 logit gather index
        hidden_states = enc_outputs.last_hidden_state
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        enc_hidden = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        enc_logits = self._apply_logit_softcapping(self.lm_head(enc_hidden).float())
        canvas_length = decoder_input_ids.shape[1]
        enc_canvas_logits = enc_logits.expand(-1, canvas_length, -1)

        # Gate the KV for the decoder to read:
        # Prefill (is_encode=1) → decoder reads encoder-filled KV (enc_keys).
        # Decode  (is_encode=0) → decoder reads retained KV from past_key.i inputs (orig_keys).
        # Write the gated values back into the cache so the decoder reads them.
        is_f4 = is_enc_bool.view(1, 1, 1, 1)
        for i in range(text_cfg.num_hidden_layers):
            past_key_values.layers[i].keys = torch.where(is_f4, enc_keys[i], orig_keys[i])
            past_key_values.layers[i].values = torch.where(is_f4, enc_vals[i], orig_vals[i])

        # ---- Decoder path (always traced) ----
        dec_outputs = self.model.decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=self_conditioning_logits,
            decoder_position_ids=decoder_position_ids,
        )
        dec_logits = self._apply_logit_softcapping(self.lm_head(dec_outputs.last_hidden_state).float())

        # ---- Gate logits output ----
        is_f1 = is_enc_bool.view(1, 1, 1)
        canvas_logits = torch.where(is_f1, enc_canvas_logits, dec_logits)

        is_f_idx = is_enc_bool.view(1, 1)
        gated_image_idx = torch.where(is_f_idx, next_image_idx, image_idx)

        # KV retained state: already gated above; read the final values from the cache.
        gated_pkv = [
            (past_key_values.layers[i].keys, past_key_values.layers[i].values)
            for i in range(text_cfg.num_hidden_layers)
        ]

        return canvas_logits, vision_embeds, gated_image_idx, gated_pkv
