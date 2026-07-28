# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Vision-language wiring for ``deepseek-ai/DeepSeek-OCR-2`` (``model_type: deepseek_vl_v2``).

Composes the vision tower (:mod:`vision_deepseek_vl_v2`) with the MoE text decoder
(:mod:`modeling_deepseek_vl_v2`) and implements the QEfficient VLM export contract, so
the model can be compiled either as a single graph or as a vision/language QPC pair
(``kv_offload=True``).

Image injection
---------------
The reference splices vision embeddings into the token embeddings with an in-place
``masked_scatter_`` over a boolean ``images_seq_mask``. That is replaced here by the
gather/``torch.where`` form used by the other QEfficient VLMs: image positions are
identified by ``image_token_id`` (128815), a running ``image_idx`` tracks how many
embeddings have been consumed, and the selection is expressed as index arithmetic. This
is numerically identical for the reference's inputs while remaining export-safe (no
in-place mutation of a traced tensor, no data-dependent shapes).
"""

from typing import List, Optional, Type

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel

from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo, get_padding_shape_from_config
from QEfficient.utils.logging_utils import logger

from .configuration_deepseek_vl_v2 import DeepseekVLV2Config
from .modeling_deepseek_vl_v2 import QEffDeepseekVLV2ForCausalLM
from .vision_deepseek_vl_v2 import (
    QEffMlpProjector,
    QEffQwen2Decoder2Encoder,
    QEffSamBlock,
    QEffSamImageEncoderViT,
)


class QEffDeepseekVLV2VisionWrapper(nn.Module):
    """Vision half: pixels -> projected embeddings (with the view-separator row)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffSamBlock}

    def forward(self, pixel_values):
        return self.model.get_image_features(pixel_values)


class QEffDeepseekVLV2DecoderWrapper(nn.Module):
    """Language half: token ids + precomputed vision embeddings -> logits."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = self.model.language_model.config
        self.language_model = self.model.language_model

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {self.model.language_model.model.layers[0].__class__}

    def forward(self, input_ids, vision_embeds, position_ids, image_idx, past_key_values):
        inputs_embeds, image_idx = self.model.inject_image_features(input_ids, vision_embeds, image_idx)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits, vision_embeds, image_idx, outputs.past_key_values


class QEffDeepseekVLV2ForConditionalGeneration(PreTrainedModel):
    """Full OCR VLM: DeepEncoder-v2 vision tower + non-MLA MoE text decoder."""

    config_class = DeepseekVLV2Config
    _no_split_modules = None
    _supports_sdpa = False

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sam_model = QEffSamImageEncoderViT()
        self.qwen2_model = QEffQwen2Decoder2Encoder()
        self.projector = QEffMlpProjector(input_dim=896, n_embed=config.hidden_size)
        self.view_seperator = nn.Parameter(torch.zeros(config.hidden_size))
        self.language_model = QEffDeepseekVLV2ForCausalLM(config)
        self.image_token_id = getattr(config, "image_token_id", constants.DEEPSEEK_VL_V2_IMAGE_TOKEN_ID)
        self.post_init()

    def __qeff_init__(self):
        self.language_model.__qeff_init__()
        self.qwen2_model.__qeff_init__()

    # -- vision ------------------------------------------------------------

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``(B,3,1024,1024) -> (1, B*257, hidden)``.

        Matches the reference's global-view (``crop_mode=False``) composition: the
        projected 256 embeddings followed by the learned view separator.
        """
        features = self.projector(self.qwen2_model(self.sam_model(pixel_values)))
        bs, _, n_dim = features.shape
        separator = self.view_seperator.to(features.dtype).view(1, 1, n_dim).expand(bs, 1, n_dim)
        features = torch.cat([features, separator], dim=1)
        return features.reshape(1, -1, n_dim)

    # -- image injection ---------------------------------------------------

    def inject_image_features(self, input_ids, vision_embeds, image_idx):
        """Replace image-placeholder embeddings with vision embeddings."""
        input_embeds = self.language_model.model.embed_tokens(input_ids)
        b, n, c = input_embeds.shape
        input_embeds = input_embeds.reshape(b * n, c)
        image_input_ids = input_ids.reshape(b * n)

        selected = image_input_ids == self.image_token_id
        indices1 = selected.unsqueeze(0).to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.reshape(-1, c).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(0).unsqueeze(-1), image_features_expanded, input_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), input_embeds, image_input_embeds)
        next_image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        image_idx = torch.where(image_idx < next_image_idx, next_image_idx, image_idx)
        return inputs_embeds.reshape(b, n, c), image_idx

    # -- QEfficient export contract ---------------------------------------

    def get_qeff_vision_encoder(self):
        return QEffDeepseekVLV2VisionWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffDeepseekVLV2DecoderWrapper(self)

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        **compiler_options,
    ):
        if comp_ctx_lengths_prefill or comp_ctx_lengths_decode:
            raise NotImplementedError("comp_ctx_lengths (CCL) is not supported for deepseek_vl_v2.")
        prefill_seq_len = prefill_seq_len or constants.DEEPSEEK_VL_V2_PREFILL_SEQ_LEN
        ctx_len = ctx_len or constants.DEEPSEEK_VL_V2_CTX_LEN
        if img_size is None:
            img_size = getattr(self.config, "image_size", constants.DEEPSEEK_VL_V2_IMG_SIZE)
        if img_size != constants.DEEPSEEK_VL_V2_IMG_SIZE:
            logger.warning(
                f"Only img_size={constants.DEEPSEEK_VL_V2_IMG_SIZE} is validated for deepseek_vl_v2; got {img_size}."
            )
        vision_size = batch_size * constants.DEEPSEEK_VL_V2_FEATURE_SIZE

        vision = [{"batch_size": batch_size, "img_size": img_size}]
        lang_prefill = {
            "batch_size": 1 if continuous_batching else batch_size,
            "seq_len": prefill_seq_len,
            "ctx_len": ctx_len,
            "img_size": img_size,
            "vision_size": vision_size,
        }
        if continuous_batching:
            lang_prefill["full_batch_size"] = kv_cache_batch_size
        else:
            lang_prefill["batch_size"] = kv_cache_batch_size or batch_size
        if full_batch_size:
            lang_prefill["full_batch_exec_size"] = full_batch_size

        lang_decode = {
            "batch_size": full_batch_size if continuous_batching else batch_size,
            "seq_len": "1",
            "ctx_len": ctx_len,
            "img_size": img_size,
            "vision_size": vision_size,
        }
        if continuous_batching:
            lang_decode["full_batch_size"] = kv_cache_batch_size
        else:
            lang_decode["batch_size"] = kv_cache_batch_size or batch_size

        lang = [lang_prefill, lang_decode]
        if kv_offload:
            return {"vision": vision, "lang": lang}, compiler_options
        lang[0].pop("vision_size")
        lang[1].pop("vision_size")
        return lang, compiler_options

    def get_onnx_dynamic_axes(
        self,
        comp_ctx_lengths: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
    ):
        if comp_ctx_lengths is not None:
            raise NotImplementedError("comp_ctx_lengths (CCL) is not supported for deepseek_vl_v2.")
        vision_dynamic_axes = {"pixel_values": {0: "batch_size", 2: "img_size", 3: "img_size"}}
        lang_dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "position_ids": {0: "batch_size", 1: "seq_len"},
            "vision_embeds": {1: "vision_size"},
        }
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        pkv_dynamic_axes = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
        for i in range(self.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_dynamic_axes[f"past_{kv}.{i}"] = pkv_dynamic_axes

        if kv_offload:
            return {"vision": vision_dynamic_axes, "lang": lang_dynamic_axes}
        return {**vision_dynamic_axes, **lang_dynamic_axes}

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return {"vision": vision_output_names, "lang": lang_output_names}
        lang_output_names.insert(1, "pixel_values_RetainedState")
        lang_output_names.insert(2, "image_idx_output")
        return lang_output_names

    def get_dummy_inputs(
        self,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        comp_ctx_lengths: Optional[List[int]] = None,
        **kwargs,
    ):
        if comp_ctx_lengths is not None:
            raise NotImplementedError("comp_ctx_lengths (CCL) is not supported for deepseek_vl_v2.")
        prefill_seq_len = int(kwargs.get("prefill_seq_len") or constants.DEEPSEEK_VL_V2_PREFILL_SEQ_LEN)
        img_size = getattr(self.config, "image_size", constants.DEEPSEEK_VL_V2_IMG_SIZE)
        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS

        vision_inputs = {
            "pixel_values": torch.zeros(
                (bs, constants.DEEPSEEK_VL_V2_NUM_CHANNELS, img_size, img_size), dtype=torch.float32
            )
        }
        lang_inputs = {
            "input_ids": torch.zeros((bs, prefill_seq_len), dtype=torch.int64),
            "vision_embeds": torch.zeros(
                (1, constants.DEEPSEEK_VL_V2_FEATURE_SIZE * bs, self.config.hidden_size), dtype=torch.float32
            ),
            "position_ids": torch.arange(prefill_seq_len, dtype=torch.int64).view(1, prefill_seq_len).repeat(bs, 1),
            "image_idx": torch.zeros((1, 1), dtype=torch.int64),
        }

        kv_cache_shape = get_padding_shape_from_config(
            config=self.config, batch_size=fbs if continuous_batching else bs, seq_len=prefill_seq_len
        )
        lang_inputs["past_key_values"] = [[] for _ in range(self.config.num_hidden_layers)]
        for i in range(self.config.num_hidden_layers):
            for _ in ["key", "value"]:
                lang_inputs["past_key_values"][i].append(torch.zeros(kv_cache_shape, dtype=torch.float32))
        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        if kv_offload:
            return {"vision": vision_inputs, "lang": lang_inputs}
        lang_inputs.pop("vision_embeds")
        return {**vision_inputs, **lang_inputs}

    def forward(self, input_ids, pixel_values, position_ids, image_idx, past_key_values):
        vision_embeds = self.get_image_features(pixel_values)
        inputs_embeds, image_idx = self.inject_image_features(input_ids, vision_embeds, image_idx)
        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs.last_hidden_state[torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.language_model.lm_head(hidden_states).float()
        return logits, pixel_values, image_idx, outputs.past_key_values

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=torch.float32,
                shape=("batch_size", 3, "img_size", "img_size"),
            ),
        ]
