# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from pathlib import Path

import onnx
import torch
import yaml
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerCausalLMOutputWithPast,
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerModel,
    MuseGlimmerModelOutputWithPast,
    MuseGlimmerRMSNorm,
    MuseGlimmerTextAttention,
    MuseGlimmerTextCenteredRMSNorm,
    MuseGlimmerTextDecoderLayer,
    MuseGlimmerTextModel,
    repeat_kv,
)

from QEfficient.customop.rms_norm import CustomRMSNormFunc
from QEfficient.transformers.cache_utils import QEffSlidingWindowCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffMuseGlimmerRMSNormAIC(MuseGlimmerRMSNorm):
    """Export Muse RMSNorm through the compiler's numerically stable custom op."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            return super().forward(hidden_states)
        if self.with_scale:
            weight = self.weight
        else:
            weight = hidden_states.new_ones(hidden_states.shape[-1])
        return CustomRMSNormFunc.apply(hidden_states, weight, self.eps).to(hidden_states.dtype)


class QEffMuseGlimmerTextCenteredRMSNormAIC(MuseGlimmerTextCenteredRMSNorm):
    """Preserve Muse's centered RMSNorm scale while using the custom-op export."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not torch.onnx.is_in_onnx_export():
            return super().forward(hidden_states)
        return CustomRMSNormFunc.apply(hidden_states, self.weight + 1.0, self.eps).to(hidden_states.dtype)


class QEffMuseGlimmerTextAttention(MuseGlimmerTextAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        **kwargs,
    ):
        if past_key_values is None or isinstance(past_key_values, Cache):
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states = self.qk_norm(query_states) * self.qk_scale_factor
        key_states = self.qk_norm(key_states)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (self._rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (self._rotate_half(key_states) * sin)

        if past_key_values is not None:
            cache_kwargs = {
                "batch_index": batch_index,
                "position_ids": position_ids,
                "is_sliding": self.is_local_attention,
            }
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = torch.where(
                attention_mask.bool(),
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=hidden_states.dtype),
                attn_weights,
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = attn_output * torch.sigmoid(self.gate_proj(hidden_states))
        return self.o_proj(attn_output), attn_weights

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class QEffMuseGlimmerTextDecoderLayer(MuseGlimmerTextDecoderLayer):
    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        comp_ctx_lengths=None,
        batch_index=None,
        **kwargs,
    ):
        if isinstance(past_key_values, QEffSlidingWindowCache):
            target_length = (
                past_key_values.sliding_window_len
                if self.self_attn.is_local_attention
                else past_key_values.max_cache_len
            )
            attention_mask = _create_causal_mask(
                position_ids=position_ids,
                target_length=target_length,
                sliding_window=target_length if self.self_attn.is_local_attention else None,
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            **kwargs,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.post_feedforward_layernorm(self.mlp(hidden_states))
        return residual + hidden_states


class QEffMuseGlimmerTextModel(MuseGlimmerTextModel):
    def __qeff_init__(self):
        full_layer = next(
            (index + 1 for index, layer_type in enumerate(self.config.layer_types) if layer_type == "full_attention"),
            self.config.num_hidden_layers,
        )
        self.config._sliding_window_pattern = full_layer

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        comp_ctx_lengths=None,
        batch_index=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if past_key_values is None or isinstance(past_key_values, Cache):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                **kwargs,
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        past_key_values = QEffSlidingWindowCache.from_legacy_cache(self.config, past_key_values)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for index, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings if self.config.layer_rope_theta[index] else None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
            )
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values.to_legacy_cache(),
        )


class QEffMuseGlimmerModel(MuseGlimmerModel):
    def get_image_features(self, pixel_values: torch.FloatTensor, image_grid_thw: torch.LongTensor, **kwargs):
        outputs = super().get_image_features(pixel_values, image_grid_thw, **kwargs)
        outputs.pooler_output = tuple(features.clamp(-60000.0, 60000.0) for features in outputs.pooler_output)
        return outputs

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        comp_ctx_lengths=None,
        batch_index=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            multimodal_mask = (input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)
            llm_input_ids = torch.where(multimodal_mask, torch.zeros_like(input_ids), input_ids)
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        for values, grid, token_id in (
            (pixel_values, image_grid_thw, self.config.image_token_id),
            (pixel_values_videos, video_grid_thw, self.config.video_token_id),
        ):
            if values is None:
                continue
            features = self.get_image_features(values, grid).pooler_output
            features = torch.cat(features, dim=0).to(inputs_embeds.dtype)
            selected = input_ids == token_id
            indices = selected.to(torch.int64).cumsum(1) - 1
            indices = torch.where(selected, indices, torch.zeros_like(indices))
            expanded = features[indices]
            inputs_embeds = torch.where(selected.unsqueeze(-1), expanded, inputs_embeds)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            **kwargs,
        )
        return MuseGlimmerModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class QEffMuseGlimmerForConditionalGeneration(MuseGlimmerForConditionalGeneration):
    def __qeff_init__(self):
        self.language_model = self.model.language_model

    def generate_npi_file(self, onnx_path: str | Path, model_name: str | None = None) -> str:
        """Keep numerically sensitive Muse operators in FP32 on AI 100."""
        del model_name
        onnx_path = Path(onnx_path)
        model = onnx.load(str(onnx_path), load_external_data=False)
        fp32_ops = {"CustomRMSNorm", "Sigmoid", "Softmax"}
        fp32_names = [
            output_name
            for node in [*model.graph.node, *(node for function in model.functions for node in function.node)]
            if node.op_type in fp32_ops
            for output_name in node.output
            if output_name
        ]
        npi_path = onnx_path.with_name(f"{onnx_path.stem}_muse_glimmer_npi.yaml")
        with open(npi_path, "w") as fp:
            yaml.safe_dump({"FP32NodeInstanceNames": list(dict.fromkeys(fp32_names))}, fp, sort_keys=False)
        return str(npi_path)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        past_key_values=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        attention_mask=None,
        comp_ctx_lengths=None,
        batch_index=None,
        use_cache=True,
        **kwargs,
    ):
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if position_ids is not None and past_key_values is not None:
            logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
            hidden_states = hidden_states[torch.arange(hidden_states.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states) * self.config.text_config.output_multiplier
        softcap = self.config.text_config.final_logit_softcapping
        logits = torch.tanh(logits / softcap) * softcap
        return MuseGlimmerCausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)

    def get_output_names(self, kv_offload: bool = False):
        output_names = ["logits"]
        for layer in range(self.config.text_config.num_hidden_layers):
            output_names.extend([f"past_key.{layer}_RetainedState", f"past_value.{layer}_RetainedState"])
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="pixel_values", datatype=torch.float32, shape=("num_patches", 1176)),
        ]

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: list[int] | None = None,
        continuous_batching: bool = False,
        **kwargs,
    ):
        batch_size = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        prefill_seq_len = int(kwargs.get("prefill_seq_len") or constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        ctx_len = int(kwargs.get("ctx_len") or max(prefill_seq_len, constants.INTERN_CTX_LEN))
        grid_h = int(kwargs.get("grid_h") or 4)
        grid_w = int(kwargs.get("grid_w") or 4)
        grid_t = int(kwargs.get("time") or 1)
        vision_tokens = grid_t * grid_h * grid_w // self.config.vision_config.merge_size**2
        input_ids = torch.zeros((batch_size, prefill_seq_len), dtype=torch.int64)
        input_ids[:, :vision_tokens] = self.config.image_token_id
        position_ids = torch.arange(prefill_seq_len, dtype=torch.int64).view(1, -1).repeat(batch_size, 1)
        patch_width = self.config.vision_config.patch_temporal * 3 * self.config.vision_config.patch_size**2
        pixel_values = torch.zeros((grid_t * grid_h * grid_w, patch_width), dtype=self.config.torch_dtype)
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int64)

        pattern = next(
            (index + 1 for index, kind in enumerate(self.config.text_config.layer_types) if kind == "full_attention"),
            self.config.text_config.num_hidden_layers,
        )
        past_key_values = []
        for layer in range(self.config.text_config.num_hidden_layers):
            cache_len = self.config.text_config.sliding_window if (layer + 1) % pattern else ctx_len
            shape = (
                batch_size,
                self.config.text_config.num_key_value_heads,
                cache_len,
                self.config.text_config.head_dim,
            )
            past_key_values.append(
                (torch.zeros(shape, dtype=self.config.torch_dtype), torch.zeros(shape, dtype=self.config.torch_dtype))
            )
        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "past_key_values": past_key_values,
        }
        if comp_ctx_lengths is not None:
            inputs["comp_ctx_lengths"] = torch.full((1,), ctx_len, dtype=torch.int64)
        if continuous_batching:
            inputs["batch_index"] = torch.arange(batch_size, dtype=torch.int64)
        return inputs

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int | None = None,
        height: int = 56,
        width: int = 56,
        time: int = 1,
        **compiler_options,
    ):
        compiler_options.pop("comp_ctx_lengths_prefill", None)
        compiler_options.pop("comp_ctx_lengths_decode", None)
        compiler_options.pop("kv_cache_batch_size", None)
        if img_size is not None:
            height = width = img_size
        patch_size = self.config.vision_config.patch_size
        grid_h = height // patch_size
        grid_w = width // patch_size
        common = {
            "batch_size": batch_size,
            "ctx_len": ctx_len,
            "sliding_window": min(self.config.text_config.sliding_window, ctx_len),
            "grid_h": grid_h,
            "grid_w": grid_w,
            "time": time,
            "num_patches": time * grid_h * grid_w,
        }
        return [
            {**common, "seq_len": prefill_seq_len},
            {**common, "seq_len": 1},
        ], compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: list[int] | None = None,
        continuous_batching: bool = False,
        **kwargs,
    ):
        axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "pixel_values": {0: "num_patches"},
            "image_grid_thw": {0: "num_images"},
        }
        pattern = next(
            (index + 1 for index, kind in enumerate(self.config.text_config.layer_types) if kind == "full_attention"),
            self.config.text_config.num_hidden_layers,
        )
        for layer in range(self.config.text_config.num_hidden_layers):
            cache_axis = "ctx_len" if not (layer + 1) % pattern else "sliding_window"
            axes[f"past_key.{layer}"] = {0: "batch_size", 2: cache_axis}
            axes[f"past_value.{layer}"] = {0: "batch_size", 2: cache_axis}
        if comp_ctx_lengths is not None:
            axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}
        if continuous_batching:
            axes["batch_index"] = {0: "batch_size"}
        return axes
