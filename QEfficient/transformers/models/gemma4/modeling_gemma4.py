# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
QEff Gemma4 modeling file — text backbone for Cloud AI 100.

Gemma 4 differs from Gemma 3 in several significant ways:
  1. Hybrid per-layer attention config:
     - sliding_attention layers use head_dim=256, 16 KV heads (GQA 2:1)
     - full_attention layers use global_head_dim=512, num_global_key_value_heads=4,
       with `attention_k_eq_v=True` so v_proj is dropped and V == K.
  2. Per-layer-type RoPE:
     - sliding: default rope, theta=10000
     - full: `proportional` rope type, partial_rotary_factor=0.25, theta=1000000
       (only the first 25% of head_dim receives rotary; the rest passes through)
  3. Norm scheme: sandwich LayerNorm (input, post-attn, pre-ffn, post-ffn), plus
     per-head q_norm, k_norm, v_norm (RMSNorm without scale on v).
  4. Optional MoE block per layer (enable_moe_block). For the 31B-it dense variant
     this is False, so we implement the dense path first.
  5. Optional per_layer_input gating (hidden_size_per_layer_input).
     Disabled for 31B (=0), so we omit for now.
  6. KV sharing across layers via `num_kv_shared_layers`. Disabled for 31B (=0).

Reference: gemma3 modeling file. This implementation inherits the same
sliding-window hybrid pattern and per-layer-type RoPE dispatch.
"""

from typing import List, Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextConfig,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4TextRotaryEmbedding,
    repeat_kv,
    rotate_half,
)

from QEfficient.customop.rms_norm import CustomRMSNorm
from QEfficient.transformers.cache_utils import QEffSlidingWindowCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


# ---------------------------------------------------------------------------
# Custom RMSNorm for Gemma4 — same `1 + weight` convention as Gemma3
# ---------------------------------------------------------------------------


class GemmaRMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        div_first = hidden_states * torch.rsqrt(torch.tensor(hidden_states.shape[-1], dtype=hidden_states.dtype))
        variance = div_first.pow(2).sum(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)


class QEffGemma4CustomRMSNormAIC(nn.Module):
    """
    Gemma4 RMSNorm: `weight` is initialized to zeros and applied as `(1 + weight)`.
    Same convention as Gemma3. Uses custom AIC RMSNorm op.

    Note: Gemma4's `Gemma4RMSNorm` also has a `with_scale=False` variant (used for v_norm);
    that variant has no `weight` parameter. We route that case through the standard
    `CustomRMSNormAIC` by checking `with_scale` at transform time.
    """

    def forward(self, hidden_states):
        if hasattr(self, "weight") and self.weight is not None:
            weight = self.weight.to(hidden_states.dtype) + 1.0
        else:
            # with_scale=False: effective weight is just 1s
            weight = torch.ones(hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)
        out = GemmaRMSNormFunc.apply(
            hidden_states,
            weight,
            self.variance_epsilon if hasattr(self, "variance_epsilon") else self.eps,
        )
        return out.to(hidden_states.dtype)


# ---------------------------------------------------------------------------
# Rotary embedding — per-layer-type inv_freq with partial rotary support
# ---------------------------------------------------------------------------


def qeff_apply_rotary_pos_emb_full(x, cos, sin, unsqueeze_dim=1):
    """
    Apply rotary pos emb to a single tensor (q or k).

    In TF5.5 Gemma4, apply_rotary_pos_emb takes one tensor (not q,k pair) and
    `cos`/`sin` are already indexed by position_ids upstream. We simply broadcast
    and apply the rotation.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def qeff_apply_rotary_pos_emb_partial(x, cos, sin, partial_rotary_factor, unsqueeze_dim=1):
    """
    Apply rotary to only the first `partial_rotary_factor * head_dim` dims.

    Used for Gemma4 full_attention layers where `rope_type=proportional` and
    `partial_rotary_factor=0.25`: only 25% of head_dim receives rotary encoding,
    the remaining 75% passes through unchanged.
    """
    head_dim = x.shape[-1]
    rotary_dim = int(head_dim * partial_rotary_factor)
    # Split
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    # Apply rotary to rotary portion only
    cos_r = cos[..., :rotary_dim].unsqueeze(unsqueeze_dim)
    sin_r = sin[..., :rotary_dim].unsqueeze(unsqueeze_dim)
    x_rot = (x_rot * cos_r) + (rotate_half(x_rot) * sin_r)
    # Concatenate back
    return torch.cat((x_rot, x_pass), dim=-1)


# ---------------------------------------------------------------------------
# Attention — handles both sliding (standard GQA) and full (K=V, partial RoPE)
# ---------------------------------------------------------------------------


class QEffGemma4TextAttention(Gemma4TextAttention):
    """
    QEff Gemma4 attention. Inherits from upstream `Gemma4TextAttention` which
    already configures per-layer-type head_dim, num_kv_heads, and v_proj=None
    for full attention.

    Overrides forward() to:
      - Replace native apply_rotary_pos_emb with QEff variants (handle partial RoPE)
      - Use QEffDynamicCache/QEffSlidingWindowCache for on-device KV retention
      - Add `batch_index`, `comp_ctx_lengths`, `cache_position` for CB / CCL
      - Replace -inf masking with torch.where + -1e9 sentinel (FP16 safe)
    """

    def __qeff_init__(self):
        # rotary_emb is shared on the model (TextModel.rotary_emb); nothing to do here.
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        # Q projection + Q-norm + RoPE
        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        if not self.is_sliding and getattr(self.config, "rope_parameters", {}).get("full_attention", {}).get(
            "rope_type"
        ) == "proportional":
            # full_attention: partial RoPE (25% of head_dim)
            partial = self.config.rope_parameters["full_attention"].get("partial_rotary_factor", 0.25)
            query_states = qeff_apply_rotary_pos_emb_partial(query_states, cos, sin, partial, unsqueeze_dim=2)
        else:
            query_states = qeff_apply_rotary_pos_emb_full(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # K projection + K-norm + RoPE
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        key_states = self.k_norm(key_states)
        if not self.is_sliding and getattr(self.config, "rope_parameters", {}).get("full_attention", {}).get(
            "rope_type"
        ) == "proportional":
            partial = self.config.rope_parameters["full_attention"].get("partial_rotary_factor", 0.25)
            key_states = qeff_apply_rotary_pos_emb_partial(key_states, cos, sin, partial, unsqueeze_dim=2)
        else:
            key_states = qeff_apply_rotary_pos_emb_full(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        # V projection (or alias to K for full_attention with attention_k_eq_v)
        if self.v_proj is not None:
            value_states = self.v_proj(hidden_states).view(hidden_shape)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)
        else:
            # K=V: use the K states (pre-RoPE would be cleaner but matches upstream)
            value_states = key_states

        # KV cache update
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "batch_index": batch_index,
                "position_ids": position_ids,
                "is_sliding": self.is_sliding,
                "sliding_window": getattr(past_key_values, "sliding_window_len", None),
            }
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Standard scaled-dot-product attention with FP16-safe masking
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = torch.where(
                attention_mask, torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=torch.float32), attn_weights
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, None


# ---------------------------------------------------------------------------
# Decoder layer — sandwich LayerNorm scheme (dense variant for 31B)
# ---------------------------------------------------------------------------


class QEffGemma4TextDecoderLayer(Gemma4TextDecoderLayer):
    """
    QEff Gemma4 decoder layer. For the 31B dense model (`enable_moe_block=False`),
    the layer is a standard transformer block with sandwich LayerNorm:
      input_norm → self_attn → post_attn_norm → residual
      pre_ffn_norm → mlp → post_ffn_norm → residual
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        last_cache_position: int = 0,
        per_layer_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Build the layer-type-specific causal mask here (the model forward passes None).
        if self.self_attn.is_sliding:
            target_length = past_key_value.sliding_window_len
            attention_mask = _create_causal_mask(
                position_ids=position_ids,
                target_length=target_length,
                sliding_window=target_length,
            )
        else:
            # Full-attention layer: use the global KV length from the cache.
            # For Gemma4 31B with no sliding-window sharing, this is simply the current cache length.
            full_len = past_key_value.key_cache[0].shape[-2] if len(past_key_value.key_cache) > 0 else 0
            attention_mask = _create_causal_mask(position_ids=position_ids, target_length=full_len)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Dense FFN path (no MoE for 31B)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # layer_scalar (registered buffer, typically 1.0)
        hidden_states = hidden_states * self.layer_scalar

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ---------------------------------------------------------------------------
# Text model — manages rotary_emb per layer_type, dispatches to decoder layers
# ---------------------------------------------------------------------------


class QEffGemma4TextModel(Gemma4TextModel):
    """
    QEff Gemma4 text model. Computes per-layer-type rotary embeddings once and
    routes each decoder layer to the appropriate `(cos, sin)` tuple based on
    `config.layer_types[i]`.
    """

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
        last_cache_position: int = 0,
        **flash_attn_kwargs,
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

        # Convert legacy list cache to QEffSlidingWindowCache
        if use_cache and not isinstance(past_key_values, Cache):
            # Gemma4 uses `layer_types` directly (no `_sliding_window_pattern`);
            # derive the pattern period (e.g. 6 for `[s,s,s,s,s,full,...]`) so the cache
            # can index the first full-attention layer as the "pattern boundary".
            if not hasattr(self.config, "_sliding_window_pattern") or self.config._sliding_window_pattern is None:
                lt = self.config.layer_types
                # Find the first full_attention layer — that marks the pattern period.
                for i, t in enumerate(lt):
                    if t == "full_attention":
                        object.__setattr__(self.config, "_sliding_window_pattern", i + 1)
                        break
                else:
                    object.__setattr__(self.config, "_sliding_window_pattern", len(lt))
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

        hidden_states = inputs_embeds

        # Compute per-layer-type RoPE (sliding + full)
        position_embeddings = {}
        if hasattr(self, "rotary_emb"):
            for layer_type in set(self.config.layer_types):
                position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_type = self.config.layer_types[i]
            layer_pos_emb = position_embeddings.get(layer_type)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=layer_pos_emb,
                attention_mask=None,  # built inside the decoder layer per layer_type
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                last_cache_position=last_cache_position,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values.to_legacy_cache() if use_cache else None

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


# ---------------------------------------------------------------------------
# Text Causal LM wrapper
# ---------------------------------------------------------------------------


class QEffGemma4ForCausalLM(Gemma4ForCausalLM):
    """
    Text-only causal LM. Used directly when the model is loaded as `AutoModelForCausalLM`
    without the vision branch.
    """

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
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

        hidden_states = outputs[0]
        # INT32 position_ids.argmax for ONNX-safe last-token indexing
        logit_idx = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = hidden_states[torch.arange(position_ids.shape[0]).view(-1, 1), logit_idx]
        logits = self.lm_head(hidden_states).float()
        # final_logit_softcapping (Gemma4 uses 30.0)
        if getattr(self.config, "final_logit_softcapping", None) is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------------------------------------------------------------------------
# VLM wrappers — delegate to QEffGemma3's pattern for vision encoder + decoder
# ---------------------------------------------------------------------------


class QEffGemma4EncoderWrapper(nn.Module):
    """Vision encoder wrapper — extracts image features for ONNX export."""

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        # Gemma4VisionModel is at self.model.vision_tower; alias for Gemma3-style API compat
        self.model.vision_model = self.model.vision_tower

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.vision_tower.encoder.layers[0].__class__}

    def forward(self, pixel_values):
        image_features = self.model.get_image_features(pixel_values=pixel_values)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        return image_features


class QEffGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Combined VLM wrapper — vision encoder + language decoder."""

    def __qeff_init__(self):
        # Post-transform alias so existing call sites referencing self.language_model work.
        self.language_model = self.model.language_model

    def get_qeff_vision_encoder(self):
        return QEffGemma4EncoderWrapper(self)

    def forward(
        self,
        input_ids,
        position_ids,
        pixel_values,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        image_features = self.get_image_features(pixel_values=pixel_values)
        if hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        inputs_embeds = self.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape
        selected = input_ids == self.config.image_token_index
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = image_features.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)

        if past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = QEffSlidingWindowCache.from_legacy_cache(
                config=self.language_model.config, past_key_values=past_key_values
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
        )
        # Next-token position + softcapping (same as CausalLM path)
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()
        if getattr(self.language_model.config, "final_logit_softcapping", None) is not None:
            cap = self.language_model.config.final_logit_softcapping
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap

        present = outputs.past_key_values
        if isinstance(present, Cache):
            if hasattr(present, "to_legacy_cache"):
                present = present.to_legacy_cache()
            elif hasattr(present, "layers"):
                legacy_cache = ()
                for layer in present.layers:
                    legacy_cache += ((getattr(layer, "keys", None), getattr(layer, "values", None)),)
                present = legacy_cache
        return logits, pixel_values, image_idx, present
