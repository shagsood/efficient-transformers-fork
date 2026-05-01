# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Stage 1→2 validation for Qwen3.5-35B-A3B (qwen3_5_moe).

Tests that QEff transforms produce matching outputs to HF baseline on a tiny
random-weight model. This is a custom test because QEFFAutoModelForCausalLM
cannot handle the hybrid 3-state-type architecture (KV + conv + recurrent).

Usage:
    python tests/transformers/models/test_qwen3_5_moe_stage12.py
    # or via pytest:
    pytest tests/transformers/models/test_qwen3_5_moe_stage12.py -v -s
"""

import copy

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

from QEfficient.transformers.models.pytorch_transforms import CustomOpsTransform, KVCacheTransform


def get_tiny_config():
    config = AutoConfig.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)
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
    tc.layer_types = [
        "linear_attention", "linear_attention", "linear_attention", "full_attention"
    ]
    return tc


def test_stage12_hf_vs_qeff():
    torch.manual_seed(42)
    tc = get_tiny_config()

    model_hf = Qwen3_5MoeForCausalLM(tc).float().eval()
    state = copy.deepcopy(model_hf.state_dict())

    input_ids = torch.randint(0, tc.vocab_size, (1, 5))
    position_ids = torch.arange(5).unsqueeze(0).unsqueeze(0).expand(4, 1, -1)

    with torch.no_grad():
        out_hf = model_hf(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    logits_hf = out_hf.logits

    model_qeff = Qwen3_5MoeForCausalLM(tc).float().eval()
    model_qeff.load_state_dict(state)
    CustomOpsTransform.apply(model_qeff)
    KVCacheTransform.apply(model_qeff)

    with torch.no_grad():
        out_qeff = model_qeff(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    logits_qeff = out_qeff.logits

    max_diff = (logits_hf - logits_qeff).abs().max().item()
    top10_hf = set(logits_hf[0, -1].topk(10).indices.tolist())
    top10_qeff = set(logits_qeff[0, -1].topk(10).indices.tolist())
    overlap = len(top10_hf & top10_qeff)

    print(f"S1 (HF)   top-10: {logits_hf[0, -1].topk(10).indices.tolist()}")
    print(f"S2 (QEff) top-10: {logits_qeff[0, -1].topk(10).indices.tolist()}")
    print(f"Max diff: {max_diff:.6e}")
    print(f"Top-10 overlap: {overlap}/10")

    assert overlap >= 7, f"S1→S2 FAIL: top-10 overlap {overlap}/10 < 7"
    assert max_diff < 1.0, f"S1→S2 FAIL: max diff {max_diff:.6e} >= 1.0"
    print("PASS: S1→S2 (HF vs QEff transforms)")


if __name__ == "__main__":
    test_stage12_hf_vs_qeff()
