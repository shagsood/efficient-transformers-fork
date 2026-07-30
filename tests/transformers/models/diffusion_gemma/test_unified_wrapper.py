# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Focused sanity tests for the DiffusionGemma single-QPC unified wrapper's
export contract (see Task 3 of docs/superpowers/plans/2026-07-10-diffusion-
gemma-single-qpc.md).

Exercises get_dummy_inputs / get_specializations / get_onnx_dynamic_axes /
get_output_names against the existing tiny-random config in
tests/configs/image_text_model_configs.json (per Rule 13 — no per-arch
tiny script, reuse the canonical config). CPU-only; no hardware or export.

Rule 13 note: the canonical parametrized harness
(test_image_text_to_text_models.py) currently skips diffusion under
kv_offload=True and routes kv_offload=False through
_QEFFAutoModelForImageTextToTextSingleQPC which never touches the
UnifiedWrapper. This file gives the wrapper contract a direct sanity
check until the harness is parametrized on `use_unified_wrapper`.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig


def _load_tiny_config():
    root = Path(__file__).resolve().parents[4]
    cfg_path = root / "tests" / "configs" / "image_text_model_configs.json"
    with open(cfg_path) as f:
        payload = json.load(f)
    # Search every top-level bucket — diffusion_gemma's tiny entry lives under
    # image_text_embedding_models today, but callers of this helper shouldn't
    # care which bucket it's filed under.
    for bucket in payload.values():
        if not isinstance(bucket, list):
            continue
        for m in bucket:
            if isinstance(m, dict) and m.get("model_name") == "google/diffusiongemma-26B-A4B-it":
                return m
    raise RuntimeError("tiny diffusion_gemma config missing from image_text_model_configs.json")


@pytest.fixture(scope="module")
def unified_wrapper():
    from QEfficient.transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        QEffDiffusionGemmaForBlockDiffusion,
        QEffDiffusionGemmaUnifiedWrapper,
    )

    tiny = _load_tiny_config()
    params = tiny["additional_params"]
    cfg = AutoConfig.for_model(tiny["model_type"], trust_remote_code=True, **params)
    cfg.name_or_path = tiny["model_name"]

    with torch.device("meta"):
        hf = QEffDiffusionGemmaForBlockDiffusion._from_config(cfg)
    return QEffDiffusionGemmaUnifiedWrapper(hf), cfg


def test_get_specializations_returns_one_fixed_shape_spec(unified_wrapper):
    wrapper, cfg = unified_wrapper
    specs, opts = wrapper.get_specializations(batch_size=1, prefill_seq_len=32, ctx_len=256)
    assert isinstance(specs, list) and len(specs) == 1
    spec = specs[0]
    assert spec["_graph_name"] == "Unified"
    assert spec["seq_len"] == 32
    assert spec["canvas_len"] > 1
    assert "ctx_len" in spec
    assert "sliding_window" in spec
    assert opts == {}


def test_get_onnx_dynamic_axes_covers_all_layers(unified_wrapper):
    wrapper, cfg = unified_wrapper
    axes = wrapper.get_onnx_dynamic_axes()
    n_layers = cfg.text_config.num_hidden_layers
    for i in range(n_layers):
        assert f"past_key.{i}" in axes
        assert f"past_value.{i}" in axes
    for key in ("input_ids", "position_ids", "vision_embeds", "mm_token_type_ids",
                "decoder_input_ids", "decoder_position_ids", "self_conditioning_logits",
                "encoder_attention_mask"):
        assert key in axes, f"missing dynamic axis for {key}"


def test_get_output_names_retained_state(unified_wrapper):
    wrapper, cfg = unified_wrapper
    names = wrapper.get_output_names()
    # Positional outputs match forward's return: canvas logits, image index,
    # then KV retained-state pairs.
    assert names[0] == "canvas_logits"
    assert "image_idx_output" in names
    # KV RetainedState pairs: 2 per layer.
    n_layers = cfg.text_config.num_hidden_layers
    kv_retained = [n for n in names if n.startswith(("past_key.", "past_value.")) and n.endswith("_RetainedState")]
    assert len(kv_retained) == 2 * n_layers


def test_get_dummy_inputs_covers_forward_signature(unified_wrapper):
    wrapper, cfg = unified_wrapper
    di = wrapper.get_dummy_inputs()
    expected = {
        "input_ids", "position_ids", "vision_embeds", "image_idx", "mm_token_type_ids",
        "decoder_input_ids", "decoder_position_ids", "self_conditioning_logits",
        "encoder_attention_mask", "is_encode", "past_key_values",
    }
    assert set(di.keys()) == expected
    n_layers = cfg.text_config.num_hidden_layers
    assert isinstance(di["past_key_values"], list) and len(di["past_key_values"]) == n_layers
