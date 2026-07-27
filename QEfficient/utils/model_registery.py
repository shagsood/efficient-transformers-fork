# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

# Placeholder for all non-transformer models
from QEfficient.transformers.models.deepseek_vl_v2.configuration_deepseek_vl_v2 import DeepseekVLV2Config
from QEfficient.transformers.models.deepseek_vl_v2.vlm_deepseek_vl_v2 import (
    QEffDeepseekVLV2ForConditionalGeneration,
)
from QEfficient.transformers.models.llama_swiftkv.modeling_llama_swiftkv import (
    QEffLlamaSwiftKVConfig,
    QEffLlamaSwiftKVForCausalLM,
)

# Map of model type to config class, Modelling class and transformer model architecture class
MODEL_TYPE_TO_CONFIG_CLS_AND_ARCH_CLS = {
    "llama_swiftkv": [QEffLlamaSwiftKVConfig, QEffLlamaSwiftKVForCausalLM, AutoModelForCausalLM],
    # DeepSeek-OCR-2: not in transformers (its checkpoint ships trust_remote_code modeling
    # pinned to an older release), so the config/model pair is registered here.
    "deepseek_vl_v2": [
        DeepseekVLV2Config,
        QEffDeepseekVLV2ForConditionalGeneration,
        AutoModelForImageTextToText,
    ],
}

# loop over all the model types which are not present in transformers and register them
for model_type, model_cls in MODEL_TYPE_TO_CONFIG_CLS_AND_ARCH_CLS.items():
    # Register the model config class based on the model type. This will be first element in the tuple
    AutoConfig.register(model_type, model_cls[0])

    # Register the non transformer library Class and config class using AutoModelClass
    model_cls[2].register(model_cls[0], model_cls[1])
