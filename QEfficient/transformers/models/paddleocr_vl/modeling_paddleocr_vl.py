# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPast, BaseModelOutputWithPooling
from transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
    PaddleOCRAttention,
    PaddleOCRDecoderLayer,
    PaddleOCRProjector,
    PaddleOCRRotaryEmbedding,
    PaddleOCRTextModel,
    PaddleOCRVisionAttention,
    PaddleOCRVisionEmbeddings,
    PaddleOCRVisionEncoder,
    PaddleOCRVisionTransformer,
    PaddleOCRVLForConditionalGeneration,
    apply_rotary_pos_emb_vision,
    repeat_kv,
    rotate_half,
)

from QEfficient.blocking.attention_blocking import (
    AttentionBlockingConfig,
    BlockingMode,
    generic_blocked_attention_interface,
    past_key_value_update,
)
from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE

# Fixed test resolution for the first onboarding pass (392x392 -> 28x28 patches -> 196
# vision tokens after the 2x2 projector merge), rather than compiling for the full
# dynamic smart_resize range (384px-1536px). See research-wiki-draft.md.
_TEST_GRID = 28
_SEQ_LEN = 512
_CTX_LEN = 1024


def qeff_prepare_mrope_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    mrope_section: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index precomputed cos/sin by M-RoPE position_ids and apply Qwen-style split-half selection.

    Args:
        cos, sin: (max_seq_len, head_dim) -- from QEffPaddleOCRTextModel.cos_cached / sin_cached
        position_ids: (3, batch, seq_len) -- temporal/height/width positions
        mrope_section: [t, h, w] sizes (each in half-head_dim units, summing to head_dim // 2)
    Returns:
        cos, sin: (batch, 1, seq_len, head_dim) -- ready for q/k multiplication
    """
    half = cos.shape[-1] // 2
    cos = cos[..., :half][position_ids]  # (3, bs, seq, half)
    sin = sin[..., :half][position_ids]
    cos = torch.cat([chunk[i % 3] for i, chunk in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
    sin = torch.cat([chunk[i % 3] for i, chunk in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)
    return cos, sin


def qeff_apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class QEffPaddleOCRVisionEmbeddings(PaddleOCRVisionEmbeddings):
    """Single image per vision forward call (matches get_specializations' vision_batch_size).

    Replaces HF's Python loop over `grid_thw` rows (one row per image) with a direct read of
    row 0 -- the loop degenerates to a single iteration under our one-image-per-call spec.
    """

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Same math as HF, but `dim`/`sqrt_num_positions` come from config-constant Python ints
        (self.embed_dim, self.num_positions) instead of HF's `embeddings.shape[-1]` /
        `position_embedding.weight.shape[0]` -- those tensor-shape reads trace into a
        non-constant ONNX Reshape shape tensor that qaic-compile rejects (U7), even though ORT
        constant-folds it fine.
        """
        dim = self.embed_dim
        sqrt_num_positions = int(self.num_positions**0.5)
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(height, width), mode="bilinear", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.FloatTensor, grid_thw: torch.LongTensor) -> torch.Tensor:
        t_val, h_val, w_val = int(grid_thw[0, 0].item()), int(grid_thw[0, 1].item()), int(grid_thw[0, 2].item())
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # (num_patches, hidden, 1, 1)
        embeddings = patch_embeds.flatten(-2).squeeze(-1)  # (num_patches, hidden_size)

        position_embedding = self.interpolate_pos_encoding(embeddings, h_val, w_val).squeeze(0).repeat(t_val, 1)
        embeddings = embeddings + position_embedding
        return embeddings


class QEffPaddleOCRVisionAttention(PaddleOCRVisionAttention):
    """Replaces the dynamic cu_seqlens-split loop with a static block-diagonal mask."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_length = hidden_states.shape[0]

        query_states = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Build a static block-diagonal bidirectional mask from cu_seqlens: positions within
        # the same image attend to each other; cross-image (or cross-video-frame) is masked.
        min_val = torch.finfo(query_states.dtype).min
        rows = torch.arange(seq_length, device=hidden_states.device).view(1, -1)
        cols = torch.arange(seq_length, device=hidden_states.device).view(-1, 1)
        start = cu_seqlens[:-1].view(-1, 1, 1)
        end = cu_seqlens[1:].view(-1, 1, 1)
        row_mask = (rows >= start) & (rows < end)
        col_mask = (cols >= start) & (cols < end)
        block_mask = row_mask & col_mask

        final_mask = torch.ones((seq_length, seq_length), dtype=torch.float32, device=hidden_states.device)
        final_mask[block_mask.any(dim=0)] = 0.0
        final_mask = torch.where(final_mask == 1.0, min_val, final_mask)
        attention_mask = final_mask.unsqueeze(0)

        query_states = query_states.transpose(0, 1).unsqueeze(0)  # (1, heads, seq, head_dim)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


class QEffPaddleOCRVisionEncoder(PaddleOCRVisionEncoder):
    """Vectorises HF's position-id / cu_seqlens construction (which relies on shared
    `vision_utils` helpers with `.tolist()`-driven Python loops) with a single-image inline
    version -- mirrors the pattern already proven safe for GLM-OCR/Qwen2.5-VL in this fork.
    """

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        t_val, h_val, w_val = int(grid_thw[0, 0].item()), int(grid_thw[0, 1].item()), int(grid_thw[0, 2].item())
        device = inputs_embeds.device

        # merge_size=1: PaddleOCR-VL merges patches in the projector (after the encoder),
        # unlike Qwen2.5-VL which merges inside the encoder -- rotary positions are plain
        # (row, col), no merge-block reordering needed.
        hpos_ids = torch.arange(h_val, device=device).unsqueeze(1).expand(h_val, w_val).flatten()
        wpos_ids = torch.arange(w_val, device=device).unsqueeze(0).expand(h_val, w_val).flatten()
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1)  # (h*w, 2)
        if t_val > 1:
            pos_ids = pos_ids.repeat(t_val, 1)

        seq_len = h_val * w_val * t_val
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        rotary_embeddings = self.rotary_pos_emb(pos_ids)
        rotary_embeddings = rotary_embeddings.repeat(1, 2)
        position_embeddings = (rotary_embeddings.cos(), rotary_embeddings.sin())

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


class QEffPaddleOCRVisionTransformer(PaddleOCRVisionTransformer):
    """Strips the `@merge_with_config_defaults`/`@capture_outputs` decorators from HF's
    forward -- output-capturing machinery that isn't needed for export and isn't verified
    ONNX-safe."""

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.embeddings(pixel_values, grid_thw=grid_thw)
        encoder_outputs = self.encoder(inputs_embeds=hidden_states, grid_thw=grid_thw, attention_mask=attention_mask)
        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=None)


class QEffPaddleOCRProjector(PaddleOCRProjector):
    """Single-image spatial merge (drops HF's `.split()` + Python loop over multiple images)
    plus the FP16 overflow clamp on the projector's output (V1 invariant)."""

    def forward(self, image_features: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        t_val, h_val, w_val = (
            int(image_grid_thw[0, 0].item()),
            int(image_grid_thw[0, 1].item()),
            int(image_grid_thw[0, 2].item()),
        )
        m1, m2 = self.merge_kernel_size
        d = image_features.shape[-1]
        h_block, w_block = h_val // m1, w_val // m2

        image_features = self.pre_norm(image_features)
        image_features = image_features.reshape(t_val, h_block, m1, w_block, m2, d)
        image_features = image_features.transpose(2, 3)
        image_features = image_features.reshape(t_val * h_block * w_block, m1 * m2 * d)

        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        # FP16 overflow clamp (V1) -- projector activations can exceed float16 max (65504).
        hidden_states = hidden_states.clamp(-60000, 60000)
        return hidden_states


class QEffPaddleOCRRotaryEmbedding(PaddleOCRRotaryEmbedding):
    """Precomputes sin/cos at init so torch.jit.trace sees static buffers."""

    def __init__(self, config, device=None):
        super().__init__(config=config)
        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * module.scaling
    if attention_mask is not None:
        # -1e9, not -inf: FP16 cannot represent -inf safely (U1).
        attn_weights = torch.where(
            attention_mask,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=query.dtype),
            attn_weights,
        )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffPaddleOCRAttention(PaddleOCRAttention):
    """KV cache + precomputed M-RoPE (passed down from QEffPaddleOCRTextModel)."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        blocking_config = getattr(self, "attn_blocking_config", AttentionBlockingConfig())
        use_blocking = blocking_config is not None and (blocking_config.mode != BlockingMode.NONE)

        if use_blocking:
            past_seen_tokens = past_key_values.get_seq_length(self.layer_idx) if past_key_values is not None else 0
            attn_output, attn_weights = generic_blocked_attention_interface(
                module=self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                layer_idx=self.layer_idx,
                past_key_value=past_key_values,
                blocking_config=blocking_config,
                comp_ctx_length=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids[0],
                past_seen_tokens=past_seen_tokens,
            )
        else:
            key_states, value_states, attention_mask, _ = past_key_value_update(
                module=self,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                position_ids=position_ids[0],
            )
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
            )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_values


class QEffPaddleOCRDecoderLayer(PaddleOCRDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        cos_cached: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            sin_cached=sin_cached,
            cos_cached=cos_cached,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class QEffPaddleOCRTextModel(PaddleOCRTextModel):
    def __qeff_init__(self):
        """Called by ModuleMappingTransform after class replacement."""
        self.rotary_emb = QEffPaddleOCRRotaryEmbedding(config=self.config)
        attention_scaling = getattr(self.rotary_emb, "attention_scaling", 1.0)
        self.cos_cached = nn.Parameter(self.rotary_emb.cos_cached * attention_scaling, requires_grad=False)
        self.sin_cached = nn.Parameter(self.rotary_emb.sin_cached * attention_scaling, requires_grad=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
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
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            past_key_values = QEffDynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        # PaddleOCR-VL position_ids: (4, batch, seq_len)
        #   [0]  = text 1D positions for the causal mask
        #   [1:] = 3D M-RoPE (T, H, W) positions
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            mrope_position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids if position_ids.ndim == 2 else position_ids[0]
            mrope_position_ids = (
                position_ids[1:] if position_ids.ndim == 3 else position_ids.unsqueeze(0).expand(3, -1, -1)
            )

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens
        causal_mask = _create_causal_mask(
            position_ids=text_position_ids,
            target_length=target_length,
            sliding_window=None,
        )

        hidden_states = inputs_embeds

        # Precomputed M-RoPE (faster than re-computing per layer)
        mrope_section = self.config.rope_parameters["mrope_section"]
        cos, sin = qeff_prepare_mrope_cos_sin(self.cos_cached, self.sin_cached, mrope_position_ids, mrope_section)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                sin_cached=sin,
                cos_cached=cos,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QEffPaddleOCRVLEncoderWrapper(nn.Module):
    def __init__(self, model: "QEffPaddleOCRVLForConditionalGeneration"):
        super().__init__()
        self.model = model
        self.model.vision_model = self.model.model.visual

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.model.visual.vision_model.encoder.layers[0].__class__}

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        image_embeds = self.model.model.visual(pixel_values, grid_thw=image_grid_thw).last_hidden_state
        image_embeds = self.model.model.projector(image_embeds, image_grid_thw)
        # Normalize rank [total_tokens, H] -> [1, total_tokens, H] for the language session
        # handoff (V15) -- vision_batch_size is always 1 image per vision forward call.
        image_embeds = image_embeds.unsqueeze(0)
        return image_embeds


class QEffPaddleOCRVLDecoderWrapper(nn.Module):
    def __init__(self, model: "QEffPaddleOCRVLForConditionalGeneration"):
        super().__init__()
        self.model = model
        self.language_model = model.model.language_model

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffPaddleOCRDecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor,
        vision_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
        image_idx: torch.LongTensor,
        past_key_values,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape

        # masked_scatter is ONNX-unfriendly; replace with cumsum-indexed gather (V4).
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(B, device=input_ids.device).view(-1, 1)
        total_tokens = vision_embeds.shape[0] * vision_embeds.shape[1]
        image_features_expanded = vision_embeds.reshape(total_tokens, C).unsqueeze(0)[indices0, indices1.clamp(min=0)]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # Decode step (seq_len == 1) has no image tokens; skip the merge (V10).
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(B, device=input_ids.device).view(-1, 1), logit_index]
        logits = self.model.lm_head(hidden_states)
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        return logits, vision_embeds, image_idx, outputs.past_key_values


class QEffPaddleOCRVLForConditionalGeneration(PaddleOCRVLForConditionalGeneration):
    """QEff wrapper for PaddleOCR-VL.

    Single-QPC (kv_offload=False): combined vision+language ONNX graph.
    Dual-QPC  (kv_offload=True):   separate vision/language sessions via
                                    get_qeff_vision_encoder / get_qeff_language_decoder.
    """

    def get_qeff_vision_encoder(self) -> QEffPaddleOCRVLEncoderWrapper:
        return QEffPaddleOCRVLEncoderWrapper(self)

    def get_qeff_language_decoder(self) -> QEffPaddleOCRVLDecoderWrapper:
        return QEffPaddleOCRVLDecoderWrapper(self)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_idx: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
    ):
        image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw).last_hidden_state
        image_embeds = self.model.projector(image_embeds, image_grid_thw)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape

        # masked_scatter is ONNX-unfriendly; replace with cumsum-indexed gather (V4).
        selected = input_ids == self.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(B, device=input_ids.device).view(-1, 1)
        image_features_expanded = image_embeds.unsqueeze(0)[indices0, indices1.clamp(min=0)]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        # Decode step (seq_len == 1) has no image tokens; skip the merge (V10).
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)

        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        logit_index = position_ids[0].to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(B, device=input_ids.device).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)
        # pixel_values is not a retained state: the compiler DCEs the vision encoder in
        # the decode spec, so pixel_values is re-uploaded from host on every decode call.
        return logits, image_idx, outputs.past_key_values

    # ------------------------------------------------------------------
    # QEff interface: dummy inputs, specializations, dynamic axes, etc.
    # ------------------------------------------------------------------

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ) -> Dict:
        bs: int = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs: int = constants.ONNX_EXPORT_EXAMPLE_FBS

        vision_cfg = self.config.vision_config
        patch_size = vision_cfg.patch_size
        spatial_merge_size = vision_cfg.spatial_merge_size
        grid_h = grid_w = _TEST_GRID
        n_patches = grid_h * grid_w
        vision_size = n_patches // (spatial_merge_size**2)

        text_cfg = self.config.text_config
        num_layers = text_cfg.num_hidden_layers

        vision_inputs = {
            "pixel_values": torch.zeros((n_patches, 3, patch_size, patch_size), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64),
        }

        kv_shape = get_padding_shape_from_config(
            config=text_cfg,
            batch_size=fbs if continuous_batching else bs,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        past_kv = [[torch.zeros(kv_shape, dtype=torch.float32) for _ in range(2)] for _ in range(num_layers)]

        lang_inputs = {
            "input_ids": torch.zeros((bs, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN), dtype=torch.int64),
            "vision_embeds": torch.zeros((bs, vision_size, text_cfg.hidden_size), dtype=torch.float32),
            # position_ids: (4, batch, seq_len) -- first dim is text 1D, rest are M-RoPE
            "position_ids": (
                torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
                .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
                .repeat(bs, 1)
                .unsqueeze(0)
                .repeat(4, 1, 1)
            ),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
            "past_key_values": past_kv,
        }

        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        else:
            lang_inputs.pop("vision_embeds")
            return {**vision_inputs, **lang_inputs}

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        comp_ctx_lengths_prefill = compiler_options.pop("comp_ctx_lengths_prefill", None)
        comp_ctx_lengths_decode = compiler_options.pop("comp_ctx_lengths_decode", None)

        vision_cfg = self.config.vision_config
        patch_size = vision_cfg.patch_size
        spatial_merge_size = vision_cfg.spatial_merge_size

        if height is not None and width is not None:
            grid_h, grid_w = height // patch_size, width // patch_size
        elif img_size is not None:
            grid_h = grid_w = img_size // patch_size
        else:
            grid_h = grid_w = _TEST_GRID

        n_patches = grid_h * grid_w
        vision_size = n_patches // (spatial_merge_size**2)
        grid_height = n_patches * batch_size
        grid_width = 3 * patch_size * patch_size

        prefill_seq_len = prefill_seq_len or _SEQ_LEN
        ctx_len = ctx_len or _CTX_LEN

        vision = [
            {
                "batch_size": batch_size,
                "vision_size": vision_size,
                "grid_height": grid_height,
                "grid_width": grid_width,
                "grid_h": grid_h,
                "grid_w": grid_w,
            }
        ]

        def _lang_prefill(extra=None):
            spec = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
                "grid_height": grid_height,
                "grid_width": grid_width,
            }
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                spec["full_batch_exec_size"] = full_batch_size
            if extra:
                spec.update(extra)
            return spec

        def _lang_decode(extra=None):
            spec = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": 1,
                "ctx_len": ctx_len,
                "vision_size": vision_size,
                "vision_batch_size": batch_size,
                "grid_height": grid_height,
                "grid_width": grid_width,
            }
            if continuous_batching:
                spec["full_batch_size"] = kv_cache_batch_size
            else:
                spec["batch_size"] = kv_cache_batch_size
            if extra:
                spec.update(extra)
            return spec

        if comp_ctx_lengths_prefill is not None:
            lang = []
            for ccl in comp_ctx_lengths_prefill:
                lang.append(_lang_prefill({"comp_ctx_lengths": ccl}))
            for ccl in comp_ctx_lengths_decode:
                lang.append(_lang_decode({"comp_ctx_lengths": ccl}))
        else:
            lang = [_lang_prefill(), _lang_decode()]

        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options
        else:
            # Vision dims must stay in every lang spec so the compiler can resolve ONNX
            # symbolic dims from the fused vision+language graph (V9).
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ) -> Dict:
        num_layers = self.config.text_config.num_hidden_layers

        vision_dynamic_axes = {
            "pixel_values": {0: "grid_height"},
            "image_grid_thw": {0: "batch_size"},
        }

        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {1: "batch_size", 2: "seq_len"},
            "vision_embeds": {0: "vision_batch_size", 1: "vision_size"},
        }

        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        for i in range(num_layers):
            batch_dim = "full_batch_size" if continuous_batching else "batch_size"
            lang_dynamic_axes[f"past_key.{i}"] = {0: batch_dim, 2: "ctx_len"}
            lang_dynamic_axes[f"past_value.{i}"] = {0: batch_dim, 2: "ctx_len"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}
        else:
            merged = {**vision_dynamic_axes, **lang_dynamic_axes}
            merged.pop("vision_embeds", None)
            # pixel_values and image_grid_thw shapes are baked as ONNX constants (via
            # .item() in QEffPaddleOCRVision{Embeddings,Encoder}) -- removing them from
            # dynamic axes prevents the "Inconsistent retained state" compiler error.
            merged.pop("pixel_values", None)
            merged.pop("image_grid_thw", None)
            return merged

    def get_output_names(self, kv_offload: bool = False) -> Union[List[str], Dict[str, List[str]]]:
        vision_outputs = ["vision_embeds"]
        lang_outputs = ["logits"]
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_outputs.append(f"past_{kv}.{i}_RetainedState")

        if kv_offload:
            lang_outputs.insert(1, "vision_embeds_RetainedState")
            lang_outputs.insert(2, "image_idx_output")
            return {"vision": vision_outputs, "lang": lang_outputs}
        else:
            # Single-QPC: no pixel_values_RetainedState (compiler DCE causes inconsistency)
            lang_outputs.insert(1, "image_idx_output")
            return lang_outputs

    def prepare_inputs_for_generation(self, inputs, prefill_seq_len=128, batch_size=1):
        input_ids_length = inputs["input_ids"].shape[1]
        text_position_ids = torch.arange(input_ids_length).view(1, 1, input_ids_length).expand(-1, batch_size, -1)

        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros_like(inputs["input_ids"], dtype=torch.int32)
            mm_token_type_ids = mm_token_type_ids.masked_fill(inputs["input_ids"] == self.config.image_token_id, 1)
            mm_token_type_ids = mm_token_type_ids.masked_fill(inputs["input_ids"] == self.config.video_token_id, 2)

        mrope_position_ids, rope_deltas = self.model.get_rope_index(
            input_ids=inputs["input_ids"],
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs["attention_mask"],
        )
        self.model.rope_deltas = rope_deltas
        inputs["position_ids"] = torch.cat((text_position_ids, mrope_position_ids), dim=0)

        num_chunks = -(input_ids_length // -prefill_seq_len)
        padded_len = num_chunks * prefill_seq_len
        inputs["position_ids"] = F.pad(
            inputs["position_ids"], pad=(0, padded_len - input_ids_length), mode="constant", value=-1
        )
        return inputs

    def get_inputs_info(self) -> List[IOInfo]:
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=torch.float32,
                shape=("grid_height", 3, "patch_size", "patch_size"),
            ),
        ]
