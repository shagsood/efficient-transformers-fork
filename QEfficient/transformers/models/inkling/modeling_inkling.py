# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""QEfficient wrappers for Inkling multimodal conditional-generation models."""

import torch
import torch.nn.functional as F
import transformers.models.inkling.modeling_inkling as _hf_inkling
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.inkling.modeling_inkling import (
    InklingAttention,
    InklingCausalLMOutputWithPast,
    InklingForConditionalGeneration,
    InklingModel,
    InklingModelOutputWithPast,
    InklingMoE,
    InklingShortConvolution,
    InklingTextModel,
    eager_attention_forward,
)
from transformers.models.inkling.modeling_inkling import (
    plan_out_scales as _hf_plan_out_scales,
)

from QEfficient.customop.ctx_scatter_gather import (
    CtxGatherFunc3DGeneralized,
    CtxScatterFunc3DGeneralized,
    CtxScatterFunc3DInt,
)
from QEfficient.transformers.cache_utils import (
    CtxGatherFuncCB3D,
    CtxScatterFuncCB3D,
    QEffDynamicLayer,
)
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


def _plan_out_scales_with_python_dims(*args, **kwargs):
    """Keep HF's scale plan while supplying nn.Linear with Python integer dimensions."""

    scales = _hf_plan_out_scales(*args, **kwargs)
    return [tuple(int(value) for value in scale) for scale in scales]


# transformers 5.14.1 forwards scalar tensors from plan_out_scales into
# nn.Linear, which torch 2.7 rejects. QEff imports before from_pretrained, so
# normalize those dimensions without changing the scale plan or checkpoint.
_hf_inkling.plan_out_scales = _plan_out_scales_with_python_dims


class QEffInklingDynamicCache(Cache):
    """Static export cache containing KV and four convolution states per layer."""

    def __init__(self, config):
        super().__init__(layers=[])
        self.config = config
        self.kv_layers = [QEffDynamicLayer() for _ in range(config.num_hidden_layers)]
        self.conv_states = [[None] * config.number_of_conv_states for _ in range(config.num_hidden_layers)]

    @classmethod
    def from_legacy_cache(
        cls,
        config,
        past_key_values: tuple[tuple[torch.FloatTensor, ...], ...] | None = None,
    ) -> "QEffInklingDynamicCache":
        cache = cls(config)
        if past_key_values is None:
            return cache
        for layer_idx, layer_state in enumerate(past_key_values):
            if not layer_state:
                continue
            key_states, value_states, *conv_states = layer_state
            cache.kv_layers[layer_idx] = QEffDynamicLayer.from_tensors(key_states, value_states)
            cache.conv_states[layer_idx] = list(conv_states)
        return cache

    def __len__(self):
        return len(self.kv_layers)

    def __getitem__(self, layer_idx):
        if isinstance(layer_idx, slice):
            return tuple(self[idx] for idx in range(*layer_idx.indices(len(self))))
        layer = self.kv_layers[layer_idx]
        return (layer.keys, layer.values, *self.conv_states[layer_idx])

    @property
    def key_cache(self):
        return [layer.keys for layer in self.kv_layers]

    @property
    def value_cache(self):
        return [layer.values for layer in self.kv_layers]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_layers[layer_idx].update(key_states, value_states, cache_kwargs)

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        del cache_position
        layer = self.kv_layers[layer_idx]
        return 0 if layer.keys is None else layer.keys.shape[-2]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx, layer in enumerate(self.kv_layers):
            if layer.keys is not None:
                beam_idx_device = beam_idx.to(layer.keys.device)
                layer.keys = layer.keys.index_select(0, beam_idx_device)
                layer.values = layer.values.index_select(0, beam_idx_device)
            for state_idx, state in enumerate(self.conv_states[layer_idx]):
                if state is not None:
                    self.conv_states[layer_idx][state_idx] = state.index_select(0, beam_idx.to(state.device))

    def to_legacy_cache(self) -> tuple[tuple[torch.Tensor, ...], ...]:
        legacy_cache = ()
        for layer_idx, layer in enumerate(self.kv_layers):
            states = [layer.keys, layer.values, *self.conv_states[layer_idx]]
            legacy_cache += (tuple(torch.empty(0) if state is None else state for state in states),)
        return legacy_cache


def _qeff_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    position_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a causal depthwise convolution and return its fixed-size next state."""

    state_len = conv_state.shape[-1]
    positions = position_ids[0].flatten()
    state_positions = torch.zeros(state_len, dtype=positions.dtype, device=positions.device)
    last_state_indices = torch.argsort(torch.cat([state_positions, positions], dim=0))[-state_len:]

    hidden_states_with_cache = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    updated_conv_state = hidden_states_with_cache.index_select(2, last_state_indices.long())
    output = F.conv1d(
        hidden_states_with_cache,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=hidden_states.shape[1],
    )
    return output[:, :, -hidden_states.shape[-1] :].to(hidden_states.dtype), updated_conv_state


class QEffInklingShortConvolution(InklingShortConvolution):
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: QEffInklingDynamicCache | None = None,
        position_ids: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        input_dtype = hidden_states.dtype
        residual = hidden_states
        if position_ids is not None:
            hidden_states = torch.where(
                (position_ids >= 0).unsqueeze(-1), hidden_states, torch.zeros_like(hidden_states)
            )
        hidden_states = hidden_states.float().transpose(1, 2)

        if past_key_values is None:
            output = F.conv1d(
                hidden_states,
                self.conv1d.weight.float(),
                self.conv1d.bias,
                padding=self.conv_kernel_size - 1,
                groups=hidden_states.shape[1],
            )[:, :, : hidden_states.shape[-1]]
        else:
            state_all = past_key_values.conv_states[self.layer_idx][self.conv_idx]
            if state_all is None:
                state_all = hidden_states.new_zeros(
                    hidden_states.shape[0], hidden_states.shape[1], self.conv_kernel_size
                )
            if batch_index is not None:
                batch_index = batch_index if batch_index.ndim == 2 else batch_index.reshape(batch_index.shape[0], 1)
                channel_indices = torch.arange(state_all.shape[1], dtype=torch.int64, device=state_all.device)[None, :]
                state = CtxGatherFuncCB3D.apply(state_all, batch_index.to(state_all.device), channel_indices)
            else:
                state = state_all

            output, next_state = _qeff_causal_conv1d_update(
                hidden_states,
                state,
                self.conv1d.weight.squeeze(1).float(),
                position_ids,
                self.conv1d.bias,
            )
            if batch_index is not None:
                batch_index = batch_index.to(state_all.device)
                channel_indices = torch.arange(state_all.shape[1], dtype=torch.int64, device=state_all.device)[None, :]
                next_state = CtxScatterFuncCB3D.apply(state_all, batch_index, channel_indices, next_state)
            past_key_values.conv_states[self.layer_idx][self.conv_idx] = next_state

        output = output.transpose(1, 2)
        return (output + residual.float()).to(input_dtype)


class QEffInklingAttention(InklingAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: QEffInklingDynamicCache | None = None,
        position_ids: torch.LongTensor | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]

        query_states = self.q_proj(hidden_states)
        key_states = self.k_sconv(
            self.k_proj(hidden_states),
            past_key_values=past_key_values,
            position_ids=position_ids,
            batch_index=batch_index,
        )
        value_states = self.v_sconv(
            self.v_proj(hidden_states),
            past_key_values=past_key_values,
            position_ids=position_ids,
            batch_index=batch_index,
        )
        relative_states = self.r_proj(hidden_states)

        query_states = self.q_norm(query_states.reshape(*input_shape, self.num_heads, self.head_dim)).transpose(1, 2)
        key_states = self.k_norm(key_states.reshape(*input_shape, self.num_key_value_heads, self.head_dim)).transpose(
            1, 2
        )
        value_states = value_states.reshape(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"position_ids": position_ids, "batch_index": batch_index}
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[..., : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        kv_length = key_states.shape[-2]
        query_positions = position_ids[0]
        key_positions = torch.arange(kv_length, device=hidden_states.device)
        relative_states = relative_states.reshape(*input_shape, self.num_heads, self.config.d_rel)
        position_bias = self.rel_logits_proj(relative_states, query_positions, key_positions)

        if not self.is_sliding and self.config.log_scaling_n_floor is not None:
            effective_n = (query_positions + 1).float().clamp(min=1.0)
            tau = 1.0 + self.config.log_scaling_alpha * torch.log(
                (effective_n / self.config.log_scaling_n_floor).clamp(min=1.0)
            )
            tau = tau.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            query_states = (query_states.float() * tau).to(query_states.dtype)
            position_bias = (position_bias.float() * tau).to(position_bias.dtype)

        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            position_bias=position_bias,
            **kwargs,
        )
        attn_output = self.o_proj(attn_output.reshape(*input_shape, self.num_heads * self.head_dim).contiguous())
        return attn_output, attn_weights


class QEffInklingMoE(InklingMoE):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len
        flat_states = hidden_states.reshape(num_tokens, hidden_size)
        _, topk_weights, topk_indices, shared_gammas = self.gate(hidden_states)

        selected = topk_indices.flatten()
        gate_up_proj = self.experts.gate_up_proj[selected].transpose(1, 2)
        down_proj = self.experts.down_proj[selected].transpose(1, 2)
        repeated_states = (
            flat_states.unsqueeze(1)
            .expand(num_tokens, self.gate.top_k, hidden_size)
            .contiguous()
            .reshape(num_tokens * self.gate.top_k, 1, hidden_size)
        )
        gate_states, up_states = torch.bmm(repeated_states, gate_up_proj).chunk(2, dim=-1)
        expert_states = self.experts.act_fn(gate_states) * up_states
        expert_states = torch.bmm(expert_states, down_proj).reshape(num_tokens, self.gate.top_k, hidden_size)
        routed_states = (expert_states * topk_weights.unsqueeze(-1)).sum(dim=1)

        num_shared_experts = self.shared_experts.n_shared_experts
        shared_inputs = flat_states.unsqueeze(0).expand(num_shared_experts, num_tokens, hidden_size)
        shared_gammas = shared_gammas.reshape(num_tokens, num_shared_experts, 1).transpose(0, 1)
        shared_gate = torch.bmm(shared_inputs, self.shared_experts.gate_proj.transpose(1, 2))
        shared_up = torch.bmm(shared_inputs, self.shared_experts.up_proj.transpose(1, 2))
        shared_activated = self.shared_experts.act_fn(shared_gate) * shared_up * shared_gammas
        shared_down = torch.bmm(shared_activated, self.shared_experts.down_proj.transpose(1, 2))
        shared_states = shared_down.float().sum(dim=0).to(hidden_states.dtype)
        return (routed_states + shared_states).reshape(batch_size, seq_len, hidden_size)


def _build_matched_idx_from_cumsum(selected: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = selected.shape
    int32_max = torch.iinfo(torch.int32).max
    int32_max_scalar = torch.tensor(int32_max, dtype=torch.int32, device=selected.device)
    token_idx = torch.arange(seq_len, dtype=torch.int32, device=selected.device).unsqueeze(0).expand(batch_size, -1)
    valid_prefix = torch.cumsum(selected.to(torch.int32), dim=1)
    valid_dest = valid_prefix - 1
    scatter_pos = torch.where(selected, valid_dest, int32_max_scalar)
    matched_idx = torch.full_like(token_idx, int32_max)
    matched_idx = CtxScatterFunc3DInt.apply(
        matched_idx.unsqueeze(-1),
        scatter_pos,
        token_idx.unsqueeze(-1),
    ).squeeze(-1)
    return matched_idx


def _cumsum_scatter_gather_update_inkling_expert_blocked(
    hidden_states: torch.Tensor,
    selected: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    routing_weight: torch.Tensor,
    expert_out: torch.Tensor,
    act_fn,
    packed_chunk_size: int,
) -> torch.Tensor:
    batch_size, seq_len = selected.shape
    packed_chunk_size = max(1, min(packed_chunk_size, seq_len))

    matched_idx = _build_matched_idx_from_cumsum(selected)
    valid_rows = selected.to(torch.int32).sum(dim=1).unsqueeze(1)
    hidden_states_expanded = hidden_states.unsqueeze(0).expand(batch_size, -1, -1)

    for packed_start in range(0, seq_len, packed_chunk_size):
        packed_stop = packed_start + packed_chunk_size
        chunk_matched_idx = matched_idx[:, packed_start:packed_stop]
        chunk_size = chunk_matched_idx.shape[1]
        row_range = torch.arange(chunk_size, dtype=torch.int32, device=hidden_states.device).unsqueeze(0)

        hidden_chunk = CtxGatherFunc3DGeneralized.apply(hidden_states_expanded, chunk_matched_idx)
        gate_states, up_states = torch.bmm(hidden_chunk, gate_up_proj).chunk(2, dim=-1)
        down_chunk = torch.bmm(act_fn(gate_states) * up_states, down_proj)

        routing_chunk = CtxGatherFunc3DGeneralized.apply(routing_weight, chunk_matched_idx)
        down_chunk = down_chunk * routing_chunk

        expert_out_chunk = CtxGatherFunc3DGeneralized.apply(expert_out, chunk_matched_idx)
        updated_chunk = expert_out_chunk + down_chunk

        chunk_valid_rows = torch.clamp(
            valid_rows - packed_start,
            min=torch.zeros_like(valid_rows),
            max=torch.full_like(valid_rows, chunk_size),
        )
        updated_chunk = torch.where(
            (row_range < chunk_valid_rows).unsqueeze(-1), updated_chunk, torch.zeros_like(updated_chunk)
        )
        expert_out = CtxScatterFunc3DGeneralized.apply(expert_out, chunk_matched_idx, updated_chunk)

    return expert_out


class QEffPrefillChunkedInklingMoE(QEffInklingMoE):
    supports_moe_prefill_blocking = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len
        flat_states = hidden_states.reshape(num_tokens, hidden_size)
        _, topk_weights, topk_indices, shared_gammas = self.gate(hidden_states)

        num_experts = self.experts.num_experts
        num_nsp = getattr(self, "expert_blocking_num_nsp", num_experts)
        packed_chunk_size = getattr(self, "expert_blocking_packed_chunk_size", num_tokens)
        if num_experts % num_nsp != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by expert_blocking_num_nsp ({num_nsp})")

        routing_weights = flat_states.new_zeros((num_tokens, num_experts))
        routing_weights.scatter_(1, topk_indices, topk_weights)

        local_experts = num_experts // num_nsp
        routing_weights_by_expert = (
            routing_weights.transpose(0, 1).contiguous().view(local_experts, num_nsp, num_tokens).transpose(0, 1)
        ).contiguous()
        gate_up_proj = (
            self.experts.gate_up_proj.view(local_experts, num_nsp, 2 * self.experts.intermediate_dim, hidden_size)
            .transpose(0, 1)
            .transpose(2, 3)
            .contiguous()
        )
        down_proj = (
            self.experts.down_proj.view(local_experts, num_nsp, hidden_size, self.experts.intermediate_dim)
            .transpose(0, 1)
            .transpose(2, 3)
            .contiguous()
        )

        expert_out = flat_states.new_zeros((num_nsp, num_tokens, hidden_size))
        routing_weights_unsqueezed = routing_weights_by_expert.unsqueeze(-1)
        for slot in range(local_experts):
            expert_out = _cumsum_scatter_gather_update_inkling_expert_blocked(
                hidden_states=flat_states,
                selected=routing_weights_by_expert[:, slot, :] > 0,
                gate_up_proj=gate_up_proj[:, slot],
                down_proj=down_proj[:, slot],
                routing_weight=routing_weights_unsqueezed[:, slot],
                expert_out=expert_out,
                act_fn=self.experts.act_fn,
                packed_chunk_size=packed_chunk_size,
            )

        routed_states = expert_out.float().sum(dim=0).to(hidden_states.dtype)

        num_shared_experts = self.shared_experts.n_shared_experts
        shared_inputs = flat_states.unsqueeze(0).expand(num_shared_experts, num_tokens, hidden_size)
        shared_gammas = shared_gammas.reshape(num_tokens, num_shared_experts, 1).transpose(0, 1)
        shared_gate = torch.bmm(shared_inputs, self.shared_experts.gate_proj.transpose(1, 2))
        shared_up = torch.bmm(shared_inputs, self.shared_experts.up_proj.transpose(1, 2))
        shared_activated = self.shared_experts.act_fn(shared_gate) * shared_up * shared_gammas
        shared_down = torch.bmm(shared_activated, self.shared_experts.down_proj.transpose(1, 2))
        shared_states = shared_down.float().sum(dim=0).to(hidden_states.dtype)
        return (routed_states + shared_states).reshape(batch_size, seq_len, hidden_size)


class QEffInklingTextModel(InklingTextModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: QEffInklingDynamicCache | tuple[tuple[torch.Tensor, ...], ...] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_legacy_cache = False
        if past_key_values is not None and not isinstance(past_key_values, QEffInklingDynamicCache):
            return_legacy_cache = True
            past_key_values = QEffInklingDynamicCache.from_legacy_cache(self.config, past_key_values)
        elif use_cache and past_key_values is None:
            past_key_values = QEffInklingDynamicCache(self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        target_length = past_key_values.get_seq_length(0) if past_key_values is not None else inputs_embeds.shape[1]
        if comp_ctx_lengths is not None:
            target_length = comp_ctx_lengths.shape[-1]
        full_attention_mask = _create_causal_mask(
            position_ids=position_ids, target_length=target_length, sliding_window=None
        )
        sliding_attention_mask = _create_causal_mask(
            position_ids=position_ids,
            target_length=target_length,
            sliding_window=self.config.sliding_window_size,
        )
        full_attention_mask = torch.where(
            full_attention_mask,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=inputs_embeds.dtype),
            torch.tensor(0.0, dtype=inputs_embeds.dtype),
        )
        sliding_attention_mask = torch.where(
            sliding_attention_mask,
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=inputs_embeds.dtype),
            torch.tensor(0.0, dtype=inputs_embeds.dtype),
        )

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            layer_mask = sliding_attention_mask if decoder_layer.self_attn.is_sliding else full_attention_mask
            residual = hidden_states
            hidden_states = decoder_layer.input_layernorm(hidden_states)
            hidden_states, _ = decoder_layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=layer_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                **kwargs,
            )
            hidden_states = decoder_layer.attn_sconv(
                hidden_states,
                past_key_values=past_key_values,
                position_ids=position_ids,
                batch_index=batch_index,
            )
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = decoder_layer.mlp_sconv(
                hidden_states,
                past_key_values=past_key_values,
                position_ids=position_ids,
                batch_index=batch_index,
            )
            hidden_states = residual + hidden_states

        hidden_states = self.norm(hidden_states)
        if return_legacy_cache:
            past_key_values = past_key_values.to_legacy_cache()
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


def _merge_placeholder_features(input_ids, inputs_embeds, features, token_id, feature_mask=None):
    selected = input_ids == token_id
    feature_indices = (selected.to(torch.int64).cumsum(1) - 1).clamp(min=0)
    feature_table = features.flatten(0, features.ndim - 2)
    if feature_mask is None:
        feature_table = feature_table.unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        expanded = torch.gather(
            feature_table,
            1,
            feature_indices.unsqueeze(-1).expand(-1, -1, feature_table.shape[-1]),
        )
    else:
        valid_features = feature_mask.flatten().to(torch.int64)
        source_indices = valid_features.cumsum(0) - 1
        gather_weights = torch.logical_and(
            feature_indices.unsqueeze(-1) == source_indices.unsqueeze(0).unsqueeze(0),
            valid_features.unsqueeze(0).unsqueeze(0) != 0,
        ).to(feature_table.dtype)
        expanded = torch.matmul(gather_weights, feature_table)
    return torch.where(selected.unsqueeze(-1), expanded, inputs_embeds)


class QEffInklingModel(InklingModel):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        **kwargs,
    ) -> InklingModelOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.language_model.embed_norm(self.get_input_embeddings()(input_ids))

        image_features = None
        if pixel_values is not None and input_ids is not None and input_ids.shape[1] != 1:
            image_features = self.get_image_features(pixel_values).pooler_output.to(inputs_embeds.dtype)
            image_features = image_features.clamp(-60000, 60000)
            inputs_embeds = _merge_placeholder_features(
                input_ids, inputs_embeds, image_features, self.config.image_token_id
            )

        audio_features = None
        if audio_input_ids is not None and input_ids is not None and input_ids.shape[1] != 1:
            audio_features = self.audio_tower(audio_input_ids).last_hidden_state.to(inputs_embeds.dtype)
            inputs_embeds = _merge_placeholder_features(
                input_ids,
                inputs_embeds,
                audio_features,
                self.config.audio_token_id,
                feature_mask=audio_input_ids_mask,
            )

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            **kwargs,
        )
        return InklingModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class QEffInklingForConditionalGeneration(InklingForConditionalGeneration):
    def get_submodules_for_export(self) -> type[nn.Module]:
        return set()

    def get_onnx_past_key_value_names(self, layer_idx: int, layer_state=None) -> list[str]:
        del layer_state
        num_conv_states = self.config.text_config.number_of_conv_states
        return [
            f"past_key.{layer_idx}",
            f"past_value.{layer_idx}",
            *(f"conv_state.{layer_idx}.{state_idx}" for state_idx in range(num_conv_states)),
        ]

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        comp_ctx_lengths: torch.LongTensor | None = None,
        batch_index: torch.LongTensor | None = None,
        **kwargs,
    ) -> InklingCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            **kwargs,
        )
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).unsqueeze(1), logit_index]
        hidden_states = hidden_states / self.config.text_config.logits_mup_width_multiplier
        logits = self.lm_head(hidden_states).float()
        unpadded_vocab_size = self.config.text_config.unpadded_vocab_size
        if unpadded_vocab_size is not None and unpadded_vocab_size < logits.shape[-1]:
            logits = logits[..., :unpadded_vocab_size]
        return InklingCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: list[int] | None = None,
        continuous_batching: bool = False,
        **kwargs,
    ):
        batch_size = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        full_batch_size = constants.ONNX_EXPORT_EXAMPLE_FBS if continuous_batching else batch_size
        seq_len = int(kwargs.get("prefill_seq_len", constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN))
        ctx_len = int(kwargs.get("ctx_len", seq_len))
        audio_feature_len = int(kwargs.get("audio_feature_len", 4))
        num_audios = int(kwargs.get("num_audios", batch_size))
        num_patches = int(kwargs.get("num_patches", 1))
        dtype = getattr(self.config, "torch_dtype", None) or torch.float32

        inputs = {
            "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(batch_size, 1),
            "pixel_values": torch.zeros(
                (
                    num_patches,
                    self.config.vision_config.temporal_patch_size,
                    self.config.vision_config.patch_size,
                    self.config.vision_config.patch_size,
                    self.config.vision_config.num_channels,
                ),
                dtype=dtype,
            ),
            "audio_input_ids": torch.zeros(
                (num_audios, audio_feature_len, self.config.audio_config.n_mel_bins), dtype=torch.int64
            ),
            "audio_input_ids_mask": torch.ones((num_audios, audio_feature_len), dtype=torch.int64),
            "past_key_values": [[] for _ in range(self.config.text_config.num_hidden_layers)],
        }

        kv_cache_shape = get_padding_shape_from_config(
            config=self.config.text_config,
            batch_size=full_batch_size,
            seq_len=ctx_len,
        )
        for layer_idx, layer in enumerate(self.model.language_model.layers):
            inputs["past_key_values"][layer_idx].extend(
                [torch.zeros(kv_cache_shape, dtype=dtype), torch.zeros(kv_cache_shape, dtype=dtype)]
            )
            conv_modules = [
                layer.self_attn.k_sconv,
                layer.self_attn.v_sconv,
                layer.attn_sconv,
                layer.mlp_sconv,
            ]
            for conv_module in conv_modules:
                inputs["past_key_values"][layer_idx].append(
                    torch.zeros(
                        (full_batch_size, conv_module.conv1d.in_channels, conv_module.conv_kernel_size),
                        dtype=torch.float32,
                    )
                )

        if continuous_batching:
            inputs["batch_index"] = torch.arange(batch_size).view(batch_size, 1)
        if comp_ctx_lengths is not None:
            inputs["comp_ctx_lengths"] = torch.zeros((max(comp_ctx_lengths),), dtype=torch.int64)
        return inputs

    def get_specializations(self, batch_size: int, ctx_len: int, **compiler_options):
        prefill_seq_len = compiler_options.pop("prefill_seq_len", constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        compiler_options.pop("comp_ctx_lengths_prefill", None)
        compiler_options.pop("comp_ctx_lengths_decode", None)
        compiler_options.pop("img_size", None)
        audio_feature_len = compiler_options.pop("audio_feature_len", 4)
        num_audios = compiler_options.pop("num_audios", batch_size)
        num_patches = compiler_options.pop("num_patches", 1)
        common = {
            "batch_size": batch_size,
            "ctx_len": ctx_len,
            "audio_feature_len": audio_feature_len,
            "num_audios": num_audios,
            "num_patches": num_patches,
        }
        return [
            {"_graph_name": "Prefill", **common, "seq_len": prefill_seq_len},
            {"_graph_name": "Decode", **common, "seq_len": 1},
        ], compiler_options

    def get_onnx_dynamic_axes(self, comp_ctx_lengths=None, continuous_batching: bool = False):
        batch_axis_name = "full_batch_size" if continuous_batching else "batch_size"
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "pixel_values": {0: "num_patches"},
            "audio_input_ids": {0: "num_audios", 1: "audio_feature_len"},
            "audio_input_ids_mask": {0: "num_audios", 1: "audio_feature_len"},
        }
        num_conv_states = self.config.text_config.number_of_conv_states
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            dynamic_axes[f"past_key.{layer_idx}"] = {0: batch_axis_name, 2: "ctx_len"}
            dynamic_axes[f"past_value.{layer_idx}"] = {0: batch_axis_name, 2: "ctx_len"}
            for state_idx in range(num_conv_states):
                dynamic_axes[f"conv_state.{layer_idx}.{state_idx}"] = {0: batch_axis_name}
        if continuous_batching:
            dynamic_axes["batch_index"] = {0: "batch_size"}
        if comp_ctx_lengths is not None:
            dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}
        return dynamic_axes

    def get_output_names(self):
        output_names = ["logits"]
        num_conv_states = self.config.text_config.number_of_conv_states
        for layer_idx in range(self.config.text_config.num_hidden_layers):
            output_names.extend(
                [
                    f"past_key.{layer_idx}_RetainedState",
                    f"past_value.{layer_idx}_RetainedState",
                    *(f"conv_state.{layer_idx}.{state_idx}_RetainedState" for state_idx in range(num_conv_states)),
                ]
            )
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=torch.float32,
                shape=("num_patches", "temporal_patch", "patch_h", "patch_w", "channels"),
            ),
            IOInfo(
                name="audio_input_ids",
                datatype=torch.int64,
                shape=("num_audios", "audio_feature_len", "num_mel_bins"),
            ),
            IOInfo(
                name="audio_input_ids_mask",
                datatype=torch.int64,
                shape=("num_audios", "audio_feature_len"),
            ),
        ]
