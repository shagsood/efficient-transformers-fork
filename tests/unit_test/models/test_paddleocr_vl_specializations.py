# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

from types import SimpleNamespace

from QEfficient.transformers.models.paddleocr_vl.modeling_paddleocr_vl import (
    QEffPaddleOCRVLForConditionalGeneration,
)
from QEfficient.utils._utils import to_named_specializations


def _model_with_minimal_config():
    model = object.__new__(QEffPaddleOCRVLForConditionalGeneration)
    object.__setattr__(
        model,
        "config",
        SimpleNamespace(
            vision_config=SimpleNamespace(patch_size=14, spatial_merge_size=2),
            text_config=SimpleNamespace(
                hidden_size=8,
                num_attention_heads=2,
                num_hidden_layers=1,
                num_key_value_heads=1,
            ),
        ),
    )
    return model


def test_rectangular_single_qpc_specializations_bind_language_grid_dimensions():
    model = _model_with_minimal_config()

    specializations, _ = model.get_specializations(
        batch_size=1,
        prefill_seq_len=64,
        ctx_len=1024,
        height=308,
        width=476,
        kv_cache_batch_size=1,
    )
    named = to_named_specializations(specializations)

    assert named == [
        {
            "name": "Prefill",
            "symbols": {
                "batch_size": "1",
                "ctx_len": "1024",
                "grid_h": "22",
                "grid_height": "748",
                "grid_w": "34",
                "grid_width": "588",
                "seq_len": "64",
                "vision_batch_size": "1",
                "vision_size": "187",
            },
        },
        {
            "name": "Decode",
            "symbols": {
                "batch_size": "1",
                "ctx_len": "1024",
                "grid_h": "22",
                "grid_height": "748",
                "grid_w": "34",
                "grid_width": "588",
                "seq_len": "1",
                "vision_batch_size": "1",
                "vision_size": "187",
            },
        },
    ]

    dummy_inputs = model.get_dummy_inputs(height=308, width=476)
    assert dummy_inputs["image_grid_thw"].tolist() == [[1, 22, 34]]
    assert tuple(dummy_inputs["pixel_values"].shape) == (748, 3, 14, 14)
