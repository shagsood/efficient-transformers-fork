# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""QEfficient wrappers for Qwen3-ASR."""

from typing import Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_asr.modeling_qwen3_asr import (
    Qwen3ASRAudioAttention,
    Qwen3ASRCausalLMOutputWithPast,
    Qwen3ASREncoder,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRModel,
    Qwen3ASRModelOutputWithPast,
    get_audio_cu_seqlens,
)

from QEfficient.transformers.models.qwen3.modeling_qwen3 import QEffQwen3DecoderLayer
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils import constants


class QEffQwen3ASRAudioAttention(Qwen3ASRAudioAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, self.head_dim)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(seq_length, self.embed_dim)
        return self.out_proj(attn_output)


class QEffQwen3ASREncoder(Qwen3ASREncoder):
    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        batch_size, num_mel_bins, padded_feature_length = input_features.shape
        chunk_len = self.n_window * 2

        if padded_feature_length % chunk_len != 0:
            raise ValueError(
                "Qwen3ASREncoder expects `padded_feature_length` to be a multiple of "
                f"`n_window * 2` ({chunk_len}), but got {padded_feature_length}."
            )

        num_chunks = padded_feature_length // chunk_len
        feature_lens = input_features_mask.sum(-1).to(torch.long)
        chunk_lengths = (
            input_features_mask.view(batch_size, num_chunks, chunk_len).sum(dim=-1).reshape(-1).to(torch.long)
        )
        cu_seqlens = get_audio_cu_seqlens(
            chunk_lengths, feature_lens, self.n_window_infer, self.n_window, kwargs=kwargs
        )

        chunked = (
            input_features.view(batch_size, num_mel_bins, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
        )

        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))
        total_chunks, conv_channels, freq_bins, time_steps = conv_out.size()
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous().view(total_chunks, time_steps, conv_channels * freq_bins)
        )
        conv_out += self.positional_embedding.positional_embedding[:time_steps].to(conv_out.dtype)

        # The QEff compile path pads/truncates to whole, all-valid audio chunks.
        # Avoid upstream boolean packing (`nonzero`) because QAIC requires static NonZero shapes.
        hidden_states = conv_out.reshape(-1, conv_out.shape[-1])

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(hidden_states, cu_seqlens, **kwargs)
            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class QEffQwen3ASRModel(Qwen3ASRModel):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        input_features_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: torch.FloatTensor = None,
        use_cache: Optional[bool] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3ASRModelOutputWithPast]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_embeds = None
        if input_features is not None and input_ids is not None and input_ids.shape[1] != 1:
            audio_embeds = self.get_audio_features(
                input_features,
                input_features_mask,
                return_dict=True,
            ).pooler_output

            selected = input_ids == self.config.audio_token_id
            indices1 = selected.to(torch.int64).cumsum(1) - 1
            indices1 = indices1.clamp(min=0)
            indices0 = torch.zeros_like(indices1)
            audio_features_expanded = audio_embeds.unsqueeze(0)[indices0, indices1]
            audio_input_embeds = torch.where(selected.unsqueeze(-1), audio_features_expanded, inputs_embeds)
            inputs_embeds = audio_input_embeds

        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            **kwargs,
        )

        return Qwen3ASRModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=audio_embeds,
        )


class QEffQwen3ASRForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffQwen3DecoderLayer}

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_features: torch.FloatTensor = None,
        input_features_mask: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: torch.FloatTensor = None,
        labels: torch.LongTensor = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen3ASRCausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
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
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()

        return Qwen3ASRCausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            audio_hidden_states=outputs.audio_hidden_states,
        )

    def get_dummy_inputs(
        self,
        comp_ctx_lengths: Optional[list[int]] = None,
        continuous_batching: bool = False,
        **kwargs,
    ):
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        seq_len = int(kwargs.get("prefill_seq_len", constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN))
        feature_len = int(kwargs.get("feature_len", self.config.audio_config.n_window * 2))
        num_mel_bins = self.config.audio_config.num_mel_bins

        inputs = {
            "input_features": torch.zeros((bs, num_mel_bins, feature_len), dtype=self.config.torch_dtype),
            "input_features_mask": torch.ones((bs, feature_len), dtype=torch.int64),
            "input_ids": torch.zeros((bs, seq_len), dtype=torch.int64),
            "position_ids": torch.arange(seq_len, dtype=torch.int64).view(1, seq_len).repeat(bs, 1),
            "past_key_values": [[] for _ in range(self.config.text_config.num_hidden_layers)],
        }

        kv_cache_shape = get_padding_shape_from_config(
            config=self.config.text_config,
            batch_size=constants.ONNX_EXPORT_EXAMPLE_FBS if continuous_batching else bs,
            seq_len=seq_len,
        )
        for i in range(self.config.text_config.num_hidden_layers):
            for _ in ("key", "value"):
                inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=self.config.torch_dtype))

        if continuous_batching:
            inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if comp_ctx_lengths is not None:
            inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int64)

        return inputs

    def get_specializations(self, batch_size: int, encoder_ctx_len, ctx_len, **compiler_options):
        if encoder_ctx_len is None:
            encoder_ctx_len = self.config.audio_config.n_window * 2
        else:
            chunk_len = self.config.audio_config.n_window * 2
            encoder_ctx_len = ((encoder_ctx_len + chunk_len - 1) // chunk_len) * chunk_len

        prefill = {
            "_graph_name": "Prefill",
            "batch_size": batch_size,
            "seq_len": compiler_options.pop("prefill_seq_len", constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN),
            "ctx_len": ctx_len,
            "feature_len": encoder_ctx_len,
        }
        decode = {
            "_graph_name": "Decode",
            "batch_size": batch_size,
            "seq_len": 1,
            "ctx_len": ctx_len,
            "feature_len": self.config.audio_config.n_window * 2,
        }
        return [prefill, decode], compiler_options

    def get_onnx_dynamic_axes(self):
        dynamic_axes = {
            "input_features": {0: "batch_size", 2: "feature_len"},
            "input_features_mask": {0: "batch_size", 1: "feature_len"},
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
        }
        pkv_dynamic_axes = {0: "batch_size", 2: "ctx_len"}
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ("key", "value"):
                dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes
        return dynamic_axes

    def get_output_names(self):
        output_names = ["logits"]
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ("key", "value"):
                output_names.append(f"past_{kv}.{i}_RetainedState")
        return output_names

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="input_features", datatype=torch.float32, shape=("batch_size", "num_mel_bins", "feature_len")),
            IOInfo(name="input_features_mask", datatype=torch.int64, shape=("batch_size", "feature_len")),
        ]
