# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Stage 1→2 test for Qwen3.5-35B-A3B VLM path (`Qwen3_5MoeForConditionalGeneration`).

Validates that QEff transforms produce matching outputs to HF baseline on a tiny
random-weight VLM config (text backbone + vision encoder).
"""

import copy

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForConditionalGeneration

from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform


def get_tiny_vlm_config():
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)

    # Shrink text config
    tc = config.text_config
    tc.num_hidden_layers = 4
    tc.hidden_size = 64
    tc.intermediate_size = 128
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 16
    tc.linear_key_heads = 2
    tc.linear_value_heads = 4
    tc.linear_key_head_dim = 8
    tc.linear_value_head_dim = 8
    tc.linear_conv_kernel_dim = 4
    tc.moe_intermediate_size = 32
    tc.num_experts = 4
    tc.num_experts_per_tok = 2
    tc.vocab_size = 1000
    tc.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]

    # Shrink vision config
    vc = config.vision_config
    vc.depth = 2
    vc.hidden_size = 64
    vc.intermediate_size = 128
    vc.num_heads = 4
    vc.out_hidden_size = tc.hidden_size  # must match text hidden_size for injection
    vc.patch_size = 16
    vc.spatial_merge_size = 2

    return config


def test_vlm_stage12_classes_swap():
    """Verify QEff transforms swap the VLM classes correctly."""
    torch.manual_seed(42)
    config = get_tiny_vlm_config()
    model = Qwen3_5MoeForConditionalGeneration(config).float().eval()

    CustomOpsTransform.apply(model)
    KVCacheTransform.apply(model)

    from QEfficient.transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        QEffQwen3_5MoeForConditionalGeneration,
        QEffQwen3_5MoeModel,
        QEffQwen3_5MoeTextModel,
    )

    assert isinstance(model, QEffQwen3_5MoeForConditionalGeneration), "Top-level VLM class not swapped"
    assert isinstance(model.model, QEffQwen3_5MoeModel), "Multimodal Model class not swapped"
    assert isinstance(model.model.language_model, QEffQwen3_5MoeTextModel), "Language model not swapped"
    print("PASS: VLM class swap — all classes replaced")


def test_vlm_stage12_forward_no_image():
    """Text-only forward path through the VLM wrapper should match HF baseline
    (when no pixel_values, the wrapper just runs the text model)."""
    torch.manual_seed(42)
    config = get_tiny_vlm_config()

    model_hf = Qwen3_5MoeForConditionalGeneration(config).float().eval()
    state = copy.deepcopy(model_hf.state_dict())

    input_ids = torch.randint(0, config.text_config.vocab_size, (1, 5))

    # HF baseline
    with torch.no_grad():
        out_hf = model_hf(input_ids=input_ids, use_cache=False)
    # HF may return tuple (logits, ...) or ModelOutput with .logits
    logits_hf = out_hf[0] if isinstance(out_hf, tuple) else out_hf.logits

    # QEff transformed — also returns tuple (logits, past_kv) or just logits
    model_q = Qwen3_5MoeForConditionalGeneration(config).float().eval()
    model_q.load_state_dict(state)
    CustomOpsTransform.apply(model_q)
    KVCacheTransform.apply(model_q)

    position_ids = torch.arange(5).unsqueeze(0).unsqueeze(0).expand(4, 1, -1)
    with torch.no_grad():
        out_q = model_q(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    logits_q = out_q[0] if isinstance(out_q, tuple) else out_q

    top10_hf = set(logits_hf[0, -1].topk(10).indices.tolist())
    top10_q = set(logits_q[0, -1].topk(10).indices.tolist())
    overlap = len(top10_hf & top10_q)
    max_diff = (logits_hf[0, -1] - logits_q[0, -1]).abs().max().item()
    print(f"VLM text-only path: top-10 overlap={overlap}/10, max_diff={max_diff:.6e}")
    assert overlap >= 7, f"Top-10 overlap {overlap}/10 < 7"
    print("PASS: VLM stage1→2 text-only forward")


if __name__ == "__main__":
    test_vlm_stage12_classes_swap()
    test_vlm_stage12_forward_no_image()
