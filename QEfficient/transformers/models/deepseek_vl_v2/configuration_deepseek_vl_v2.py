# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DeepseekVLV2Config(PretrainedConfig):
    r"""
    Configuration for the text decoder of ``deepseek-ai/DeepSeek-OCR-2`` (HF class
    ``DeepseekOCR2ForCausalLM``, ``model_type: deepseek_vl_v2``).

    This is a DeepSeek-V2-family MoE decoder with **multi-latent attention disabled**
    (``use_mla=False``): the attention is plain Llama-style multi-head attention and the
    FFN is a DeepSeek MoE block (routed + shared experts). It is defined as a
    self-contained config because neither the ``trust_remote_code`` reference (pinned to
    transformers 4.46.3) nor the transformers-native ``deepseek_v2`` (MLA-only, strict
    config) can represent/run this checkpoint on transformers 5.x.

    Defaults below match ``language_config`` of ``deepseek-ai/DeepSeek-OCR-2``.
    """

    model_type = "deepseek_vl_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=1280,
        intermediate_size=6848,
        moe_intermediate_size=896,
        num_hidden_layers=12,
        num_attention_heads=10,
        num_key_value_heads=10,
        n_shared_experts=2,
        n_routed_experts=64,
        routed_scaling_factor=1.0,
        num_experts_per_tok=6,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        norm_topk_prob=False,
        scoring_func="softmax",
        topk_method="greedy",
        n_group=1,
        topk_group=1,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        use_mla=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.use_mla = use_mla

        # plain MHA: single per-head dim derived from hidden_size / heads
        self.head_dim = self.hidden_size // self.num_attention_heads

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
