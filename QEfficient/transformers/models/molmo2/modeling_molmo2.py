# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from QEfficient.transformers.cache_utils import QEffDynamicCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qeff_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(module, query, key, value, attention_mask, scaling, **kwargs):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if attention_mask is not None:
        attn_weights = torch.where(attention_mask, torch.tensor(-10000.0, dtype=torch.float32), attn_weights)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffMolmo2RotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = getattr(config, "rope_theta", 1000000.0)
        self.original_max_seq_len = config.max_position_embeddings

        dim = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=self.original_max_seq_len,
            device=device or torch.device("cpu"),
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class QEffMolmo2Attention(nn.Module):
    def __qeff_init__(self):
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[QEffDynamicCache] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.att_proj(hidden_states)
        query_states, key_states, value_states = qkv.split(self.fused_dims, dim=-1)
        value_states = value_states.view(hidden_shape)

        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type != "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.view(hidden_shape)
        key_states = key_states.view(hidden_shape)

        if self.q_norm is not None and self.k_norm is not None and self.qk_norm_type == "qwen3":
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states, key_states = qeff_apply_rotary_pos_emb(query_states, key_states, cos_cached, sin_cached)

        if past_key_values is not None:
            cache_kwargs = {"batch_index": batch_index, "position_ids": position_ids}
            if comp_ctx_lengths is not None:
                cache_kwargs["CCL"] = comp_ctx_lengths
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output, _ = eager_attention_forward(
            self, query_states, key_states, value_states, attention_mask, self.scaling
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.attn_out(attn_output)
        return attn_output, None


class QEffMolmo2DecoderLayer(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[QEffDynamicCache] = None,
        cos_cached: Optional[torch.Tensor] = None,
        sin_cached: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cos_cached=cos_cached,
            sin_cached=sin_cached,
            batch_index=batch_index,
            comp_ctx_lengths=comp_ctx_lengths,
        )

        hidden_states = residual + self.dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.ff_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.dropout(hidden_states)

        return hidden_states, None


class QEffMolmo2TextModel(nn.Module):
    def __qeff_init__(self):
        self.rotary_emb = QEffMolmo2RotaryEmbedding(config=self.config)
        self.cos_cached = nn.Parameter(self.rotary_emb.cos_cached, requires_grad=False)
        self.sin_cached = nn.Parameter(self.rotary_emb.sin_cached, requires_grad=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        **kwargs,
    ):
        if inputs_embeds is None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
            inputs_embeds = self.wte(input_ids)

        if past_key_values is None:
            past_key_values = QEffDynamicCache()

        hidden_states = inputs_embeds

        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        # For ONNX export / CPU validation the cache is fresh (seq_length=0). Use the current input
        # seq_len so _create_causal_mask produces a [B,1,S,S] causal mask. In the AI 100 runtime
        # flow the cache is pre-allocated with ctx_len and that value is used instead.
        cache_seq_length = past_key_values.get_seq_length(0, position_ids)
        target_length = cache_seq_length if cache_seq_length > 0 else hidden_states.shape[1]

        attention_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=target_length,
        )

        for decoder_block in self.blocks[: self.config.num_hidden_layers]:
            hidden_states, _ = decoder_block(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cos_cached=cos,
                sin_cached=sin,
                batch_index=batch_index,
                comp_ctx_lengths=comp_ctx_lengths,
            )

        hidden_states = self.ln_f(hidden_states)

        return hidden_states, past_key_values


class QEffMolmo2EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.model.transformer.blocks[0].__class__}

    def forward(self, pixel_values, image_token_pooling):
        # pixel_values: [B, num_crops, num_patches, pixels_per_patch]
        # image_token_pooling: [B, num_pooled_tokens, pool_dim]
        # Molmo2VisionBackbone returns flat 2D [B * num_pooled_tokens, hidden].
        # Reshape to 3D [B, num_pooled_tokens, hidden] so the decoder can do batched gather.
        vision_embeds = self.model.model.vision_backbone(pixel_values, image_token_pooling)
        batch_size = pixel_values.shape[0]
        vision_embeds = vision_embeds.view(batch_size, -1, vision_embeds.shape[-1])
        return vision_embeds


class QEffMolmo2DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.config

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.model.vision_backbone.image_vit.blocks[0].__class__}

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        inputs_embeds = self.model.model.transformer.wte(input_ids)

        image_patch_id = self.config.image_patch_id
        selected = input_ids == image_patch_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.shape[0]).view(-1, 1)
        # vision_embeds is 3D [B, num_pooled_tokens, hidden]; gather produces [B, S, hidden].
        image_features_expanded = vision_embeds[indices0, indices1]
        image_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded + inputs_embeds, inputs_embeds)

        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_embeds)

        hidden_states, past_key_values = self.model.model.transformer.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )

        logits = self.model.lm_head(hidden_states)
        batch_arange = torch.arange(logits.shape[0])
        logit_index = position_ids.to(torch.int32).argmax(1)
        logits = logits[batch_arange, logit_index]

        next_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_idx, next_idx, image_idx)
        return logits, vision_embeds, image_idx, past_key_values


class QEffMolmo2Model(nn.Module):
    def get_qeff_vision_encoder(self):
        return QEffMolmo2EncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffMolmo2DecoderWrapper(self)

    def forward(
        self,
        pixel_values,
        image_token_pooling,
        input_ids,
        position_ids,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
    ):
        # Vision: run ViT + adapter. Reshape 2D [N, hidden] return to 3D [B, N/B, hidden].
        vision_embeds = self.model.vision_backbone(pixel_values, image_token_pooling)
        batch_size = pixel_values.shape[0]
        vision_embeds = vision_embeds.view(batch_size, -1, vision_embeds.shape[-1])

        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)

        inputs_embeds = self.model.transformer.wte(input_ids)

        image_patch_id = self.config.image_patch_id
        selected = input_ids == image_patch_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds[indices0, indices1]
        image_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded + inputs_embeds, inputs_embeds)

        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_embeds)

        hidden_states, past_key_values = self.model.transformer.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            use_cache=True,
        )

        logits = self.lm_head(hidden_states)
        batch_arange = torch.arange(logits.shape[0])
        logit_index = position_ids.to(torch.int32).argmax(1)
        logits = logits[batch_arange, logit_index]

        next_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_idx, next_idx, image_idx)

        return logits, pixel_values, image_idx, past_key_values

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        num_images: int = None,
        img_size: int = None,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 1024
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN

        vit_config = self.config.vit_config

        patch_size = vit_config.image_patch_size
        default_size = vit_config.image_default_input_size
        num_patches_per_crop = (default_size[0] // patch_size) * (default_size[1] // patch_size)
        pixels_per_patch = patch_size * patch_size * 3

        if num_images is None:
            num_images = 1
        # Number of crops per image (max): 1 global resize + 4 quadrant crops at default resolution.
        # Molmo2's build_batched_images batches crops per image; use 3 for 378x378 (global + 2 crops).
        num_crops = 3
        # Number of pooled tokens after 2x2 adapter pooling; varies by image grid (processor output).
        # For a 378x378 image with default grid this is 518 (empirically from processor).
        num_pooled_tokens = 518
        pool_dim = 4  # 2x2 pool

        vision = [
            {
                "batch_size": batch_size,
                "num_crops": num_crops,
                "num_patches": num_patches_per_crop,
                "pixels_per_patch": pixels_per_patch,
                "num_pooled_tokens": num_pooled_tokens,
                "pool_dim": pool_dim,
            }
        ]

        valid_size = num_pooled_tokens

        if comp_ctx_lengths_prefill is not None and comp_ctx_lengths_decode is not None:
            lang = []
            for i in range(len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                    "valid_size": valid_size,
                    "vision_batch_size": batch_size,
                }
                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size
                lang.append(lang_prefill)

            for i in range(len(comp_ctx_lengths_decode)):
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                    "valid_size": valid_size,
                    "vision_batch_size": batch_size,
                }
                if continuous_batching:
                    lang_decode["full_batch_size"] = kv_cache_batch_size
                lang.append(lang_decode)
        else:
            lang_prefill = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "valid_size": valid_size,
                "vision_batch_size": batch_size,
            }
            if continuous_batching:
                lang_prefill["full_batch_size"] = kv_cache_batch_size
            if full_batch_size:
                lang_prefill["full_batch_exec_size"] = full_batch_size

            lang_decode = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "valid_size": valid_size,
                "vision_batch_size": batch_size,
            }
            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size

            lang = [lang_prefill, lang_decode]

        specializations = {}
        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}

        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["vision_embeds"] = {0: "vision_batch_size", 1: "valid_size"}

        vision_dynamic_axes["pixel_values"] = {0: "batch_size", 1: "num_crops", 2: "num_patches", 3: "pixels_per_patch"}
        vision_dynamic_axes["image_token_pooling"] = {0: "batch_size", 1: "num_pooled_tokens", 2: "pool_dim"}

        num_layers = self.config.text_config.num_hidden_layers

        for i in range(num_layers):
            lang_dynamic_axes[f"past_key.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }
            lang_dynamic_axes[f"past_value.{i}"] = {
                0: "full_batch_size" if continuous_batching else "batch_size",
                2: "ctx_len",
            }

        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]

        num_layers = self.config.text_config.num_hidden_layers
        for i in range(num_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return lang_output_names
        return output_names

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        **kwargs,
    ):
        vit_config = self.config.vit_config
        text_config = self.config.text_config

        patch_size = vit_config.image_patch_size
        default_size = vit_config.image_default_input_size
        num_patches_per_crop = (default_size[0] // patch_size) * (default_size[1] // patch_size)
        pixels_per_patch = patch_size * patch_size * 3

        num_crops = 3
        num_pooled_tokens = 518
        pool_dim = 4
        valid_size = num_pooled_tokens

        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS

        vision_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros(
            (bs, num_crops, num_patches_per_crop, pixels_per_patch), dtype=torch.float32
        )
        vision_inputs["image_token_pooling"] = torch.zeros((bs, num_pooled_tokens, pool_dim), dtype=torch.int64)

        lang_inputs = {}
        lang_inputs["input_ids"] = torch.zeros((bs, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((bs, valid_size, text_config.hidden_size), dtype=torch.float32)
        lang_inputs["position_ids"] = (
            torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
            .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
            .repeat(bs, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((1, 1), dtype=torch.int64)

        kv_cache_shape = get_padding_shape_from_config(
            config=text_config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )

        lang_inputs["past_key_values"] = [[] for _ in range(text_config.num_hidden_layers)]
        for i in range(text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int8)
        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=torch.float32,
                shape=("batch_size", "num_crops", "num_patches", "pixels_per_patch"),
            ),
            IOInfo(
                name="image_token_pooling",
                datatype=torch.int64,
                shape=("batch_size", "num_pooled_patches"),
            ),
        ]
