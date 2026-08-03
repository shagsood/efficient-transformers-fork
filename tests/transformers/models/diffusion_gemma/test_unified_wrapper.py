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

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from transformers import AutoConfig


def _load_debug_helpers():
    root = Path(__file__).resolve().parents[4]
    path = (
        root
        / "examples"
        / "image_text_to_text"
        / "models"
        / "diffusion_gemma"
        / "diffusion_gemma_debug.py"
    )
    spec = importlib.util.spec_from_file_location("diffusion_gemma_debug", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        "encoder_attention_mask", "execution_mode", "past_key_values",
    }
    assert set(di.keys()) == expected
    assert di["execution_mode"].tolist() == [0]
    n_layers = cfg.text_config.num_hidden_layers
    assert isinstance(di["past_key_values"], list) and len(di["past_key_values"]) == n_layers


def test_trace_comparison_ignores_transport_but_detects_token_drift(tmp_path):
    debug = _load_debug_helpers()
    left = tmp_path / "single.jsonl"
    right = tmp_path / "dual.jsonl"
    base = {
        "event": "denoise_step",
        "step": 0,
        "accepted_count": 3,
        "denoiser_tokens": {"sha256": "same"},
    }
    left.write_text(json.dumps({**base, "backend": "single-qpc", "kv_behavior": "retained"}) + "\n")
    right.write_text(json.dumps({**base, "backend": "dual-qpc", "kv_behavior": "host-copy"}) + "\n")
    assert debug.compare_traces(left, right) == []

    right.write_text(json.dumps({**base, "backend": "dual-qpc", "accepted_count": 2}) + "\n")
    mismatches = debug.compare_traces(left, right)
    assert len(mismatches) == 1
    assert "accepted_count" in mismatches[0]


def test_array_summary_fingerprints_values():
    debug = _load_debug_helpers()
    first = debug.array_summary(torch.tensor([[1, 2]], dtype=torch.int64).numpy(), include_values=True)
    second = debug.array_summary(torch.tensor([[1, 3]], dtype=torch.int64).numpy(), include_values=True)
    assert first["values"] == [[1, 2]]
    assert first["sha256"] != second["sha256"]


def test_hf_sampler_is_deterministic_for_identical_logits():
    debug = _load_debug_helpers()
    logits = torch.linspace(-2, 2, 2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8).numpy()
    left = debug.HostSampler("hf-torch", seed=1234, vocab_size=8, canvas_length=4)
    right = debug.HostSampler("hf-torch", seed=1234, vocab_size=8, canvas_length=4)

    left_canvas = left.initialize_canvas(batch_size=2)
    right_canvas = right.initialize_canvas(batch_size=2)
    assert (left_canvas == right_canvas).all()

    left_step = left.step(logits, left_canvas, step=0, max_steps=48)
    right_step = right.step(logits, right_canvas, step=0, max_steps=48)
    for field in ("next_canvas", "denoiser_canvas", "argmax_canvas", "final_canvas", "accepted_mask"):
        assert (getattr(left_step, field) == getattr(right_step, field)).all()


def test_hf_sampler_stops_only_when_argmax_is_stable_and_confident():
    debug = _load_debug_helpers()
    sampler = debug.HostSampler(
        "hf-torch",
        seed=7,
        vocab_size=3,
        canvas_length=2,
        stability_threshold=1,
        confidence_threshold=0.005,
    )
    canvas = sampler.initialize_canvas()
    confident_logits = torch.tensor([[[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]]]).numpy()

    first = sampler.step(confident_logits, canvas, step=0, max_steps=48)
    second = sampler.step(confident_logits, first.next_canvas, step=1, max_steps=48)

    assert not first.should_stop
    assert second.should_stop
    assert second.mean_entropy < 0.005


def test_hf_sampler_returns_argmax_canvas_when_adaptive_stop_triggers():
    debug = _load_debug_helpers()
    sampler = debug.HostSampler("hf-torch", seed=11, vocab_size=3, canvas_length=2, stability_threshold=0)
    canvas = sampler.initialize_canvas()
    logits = torch.tensor([[[20.0, -20.0, -20.0], [-20.0, 20.0, -20.0]]]).numpy()

    result = sampler.step(logits, canvas, step=0, max_steps=48)

    assert result.should_stop
    assert (result.final_canvas == logits.argmax(-1)).all()


def test_historical_numpy_sampler_remains_available():
    debug = _load_debug_helpers()
    sampler = debug.HostSampler("numpy-gumbel", seed=1234, vocab_size=8, canvas_length=4)
    canvas = sampler.initialize_canvas()
    logits = torch.zeros((1, 4, 8), dtype=torch.float32).numpy()

    result = sampler.step(logits, canvas, step=0, max_steps=48)

    assert result.next_canvas.shape == canvas.shape
    assert result.final_canvas.shape == canvas.shape
    assert result.stop_reason in (None, "full-acceptance")


def test_hf_sampler_uses_hf_temperature_schedule_for_feedback():
    debug = _load_debug_helpers()
    sampler = debug.HostSampler("hf-torch", seed=1, vocab_size=3, canvas_length=1)
    canvas = sampler.initialize_canvas()
    logits = torch.tensor([[[0.3, 0.6, 0.9]]], dtype=torch.float32).numpy()

    first = sampler.step(logits, canvas, step=0, max_steps=48)
    last = sampler.step(logits, first.next_canvas, step=47, max_steps=48)

    assert first.temperature == pytest.approx(0.8)
    assert last.temperature == pytest.approx(0.4 + 0.4 / 48)
    torch.testing.assert_close(
        torch.from_numpy(first.self_conditioning_logits), torch.from_numpy(logits) / first.temperature
    )


def test_logits_sample_comparator_reports_numeric_drift(tmp_path):
    debug = _load_debug_helpers()
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    logits = torch.arange(2 * 8, dtype=torch.float32).reshape(1, 2, 8).numpy()
    left_event = {"event": "denoise_step", "block": 0, "step": 0, "logits_sample": debug.logits_sample(logits)}
    changed = logits.copy()
    changed[0, 0, 0] += 0.5
    right_event = {
        "event": "denoise_step",
        "block": 0,
        "step": 0,
        "logits_sample": debug.logits_sample(changed),
    }
    left.write_text(json.dumps(left_event) + "\n")
    right.write_text(json.dumps(right_event) + "\n")

    rows, missing = debug.compare_logits_samples(left, right)

    assert missing == []
    assert rows[0][2] == pytest.approx(0.5)


def test_first_decode_mode_disables_uniform_self_conditioning(unified_wrapper):
    from QEfficient.transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        QEffDiffusionGemmaDecoderModel,
    )

    _, cfg = unified_wrapper
    tiny_cfg = deepcopy(cfg)
    tiny_cfg.text_config.vocab_size = 16
    tiny_cfg.text_config.num_hidden_layers = 0
    tiny_cfg.text_config.layer_types = []
    decoder = QEffDiffusionGemmaDecoderModel(tiny_cfg)
    decoder.layers = nn.ModuleList()

    canvas = torch.tensor([[1, 2]], dtype=torch.int64)
    positions = torch.tensor([[3, 4]], dtype=torch.int64)
    zero_logits = torch.zeros((1, 2, 16), dtype=torch.float32)
    without_feedback = decoder(
        decoder_input_ids=canvas,
        decoder_position_ids=positions,
        self_conditioning_logits=None,
    ).last_hidden_state
    masked_feedback = decoder(
        decoder_input_ids=canvas,
        decoder_position_ids=positions,
        self_conditioning_logits=zero_logits,
        self_conditioning_mask=torch.tensor([False]),
    ).last_hidden_state
    enabled_feedback = decoder(
        decoder_input_ids=canvas,
        decoder_position_ids=positions,
        self_conditioning_logits=zero_logits,
        self_conditioning_mask=torch.tensor([True]),
    ).last_hidden_state

    torch.testing.assert_close(masked_feedback, without_feedback)
    assert not torch.allclose(enabled_feedback, without_feedback)


def test_unified_decoder_mode_matches_split_decoder_on_tiny_model(unified_wrapper):
    from QEfficient.transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        QEffDiffusionGemmaForBlockDiffusion,
        QEffDiffusionGemmaUnifiedWrapper,
    )

    class InactiveEncoder(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.vocab_size = vocab_size

        def forward(self, input_ids, image_idx, past_key_values, **kwargs):
            del kwargs
            logits = torch.zeros((input_ids.shape[0], 1, self.vocab_size), dtype=torch.float32)
            cache = [(layer.keys, layer.values) for layer in past_key_values.layers]
            return logits, image_idx, cache

    _, cfg = unified_wrapper
    tiny_cfg = deepcopy(cfg)
    tiny_cfg.canvas_length = 4
    tiny_cfg.image_token_id = 31
    tiny_cfg.text_config.vocab_size = 32
    torch.manual_seed(0)
    model = QEffDiffusionGemmaForBlockDiffusion._from_config(tiny_cfg).eval()
    unified = QEffDiffusionGemmaUnifiedWrapper(model).eval()
    inputs = unified.get_dummy_inputs()
    unified.encoder_prefill = InactiveEncoder(tiny_cfg.text_config.vocab_size)
    inputs["execution_mode"] = torch.tensor([2], dtype=torch.int64)
    split_cache = model.get_dummy_pkv_cache(
        config=tiny_cfg.text_config,
        batch_size=inputs["input_ids"].shape[0],
        seq_len=inputs["input_ids"].shape[1],
    )

    with torch.no_grad():
        unified_logits = unified(**inputs)[0]
        split_logits = unified.canvas_decode(
            decoder_input_ids=inputs["decoder_input_ids"],
            decoder_position_ids=inputs["decoder_position_ids"],
            self_conditioning_logits=inputs["self_conditioning_logits"],
            past_key_values=split_cache,
            encoder_attention_mask=inputs["encoder_attention_mask"],
            execution_mode=inputs["execution_mode"],
        )[0]

    torch.testing.assert_close(unified_logits, split_logits, rtol=0, atol=0)


def test_decoder_only_unified_probe_matches_split_decoder(unified_wrapper):
    from QEfficient.transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        QEffDiffusionGemmaForBlockDiffusion,
    )

    _, cfg = unified_wrapper
    tiny_cfg = deepcopy(cfg)
    tiny_cfg.canvas_length = 4
    tiny_cfg.image_token_id = 31
    tiny_cfg.text_config.vocab_size = 32
    torch.manual_seed(0)
    model = QEffDiffusionGemmaForBlockDiffusion._from_config(tiny_cfg).eval()
    probe = model.get_qeff_decoder_only_unified_probe().eval()
    split = model.get_qeff_canvas_decode().eval()
    inputs = probe.get_dummy_inputs()
    inputs["execution_mode"] = torch.tensor([2], dtype=torch.int64)
    with torch.no_grad():
        probe_logits = probe(**inputs)[0]
        split_logits = split(**inputs)[0]
    torch.testing.assert_close(probe_logits, split_logits, rtol=0, atol=0)


def test_full_logit_comparison_reports_mad_and_argmax():
    from examples.image_text_to_text.models.diffusion_gemma import diffusion_gemma_debug as debug

    reference = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    candidate = np.array([[[1.25, 0.0], [0.0, 0.5]]], dtype=np.float32)
    metrics = debug.compare_full_logits(reference, candidate)
    assert metrics["mean_abs"] == pytest.approx(0.1875)
    assert metrics["max_abs"] == pytest.approx(0.5)
    assert metrics["argmax_agreement"] == 1.0
