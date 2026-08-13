# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
PaddleOCR-VL on Cloud AI 100 — single-QPC image-text-to-text inference.

PaddleOCR-VL is a compact (~1B) document-OCR vision-language model: a NaViT-style
dynamic-resolution vision encoder + a 2x2 spatial-merge projector + an ERNIE-4.5-0.3B
causal decoder that uses multimodal RoPE (M-RoPE, mrope_section=[16,24,24]).

M-RoPE means the decoder consumes 4D position_ids of shape (4, batch, seq_len):
row 0 is the 1D text position, rows 1-3 are the temporal/height/width positions for
image tokens. QEfficient's stock `QEFFAutoModelForImageTextToText.generate()` builds
2D position_ids, which is incompatible with this 4D input, so this example drives the
compiled QPC directly via QAICInferenceSession and builds the 4D positions with the
model's own `prepare_inputs_for_generation()` (which calls `get_rope_index()`), then
advances them by 1 per decoded token — the same convention Qwen2.5-VL / GLM-OCR use.

Single-QPC (`kv_offload=False`) runs the vision encoder, projector, and decoder in one
graph, feeding `pixel_values` directly. Dynamic-resolution images are resized to a fixed
`img_size` so the vision-token count matches the compiled specialization.

Usage:
    python paddleocr_vl.py
    python paddleocr_vl.py --image-url <url> --prompt "OCR:" --precision mxfp6+mxint8_kv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.modeling_auto import get_compilation_dims

MODEL_ID = "PaddlePaddle/PaddleOCR-VL"
DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
)


def build_4d_position_ids(qeff_model, inputs, prefill_seq_len, batch_size=1):
    """4D M-RoPE position_ids (4, batch, padded_len) via the model's get_rope_index().

    Padded positions are set to -1 by prepare_inputs_for_generation; the compiled graph
    gathers the final-token logits at argmax(position_ids[0]), so padding is ignored.
    """
    prepped = qeff_model.model.prepare_inputs_for_generation(
        inputs={
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_grid_thw": inputs["image_grid_thw"],
        },
        prefill_seq_len=prefill_seq_len,
        batch_size=batch_size,
    )
    return prepped["position_ids"].numpy().astype(np.int64)


def make_empty_kv(text_config, ctx_len, dtype, batch_size=1):
    """Zero-initialised past_key/past_value host buffers for every decoder layer."""
    shape = (batch_size, text_config.num_key_value_heads, ctx_len, text_config.head_dim)
    return {
        name: np.zeros(shape, dtype=dtype)
        for i in range(text_config.num_hidden_layers)
        for name in (f"past_key.{i}", f"past_value.{i}")
    }


def resize_and_pad(image, height, width):
    """Fit an image inside the compiled frame without changing its aspect ratio."""
    scale = min(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def qpc_input_dtype(session, input_name):
    binding = session.bindings[session.binding_index_map[input_name]]
    return session.aic_to_np_dtype_mapping[binding.type]


def run(
    model_id,
    image,
    prompt,
    precision,
    device_ids,
    img_size,
    height,
    width,
    prefill_seq_len,
    ctx_len,
    generation_len,
):
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, padding=True)

    # STEP 1: single-QPC model (vision + projector + decoder fused in one graph)
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        model_id, attn_implementation="eager", kv_offload=False, torch_dtype=torch.float32
    )

    # STEP 2: compile. NaViT is dynamic-resolution, so the image is resized to img_size
    #         below; that fixes the vision-token count to match this specialization.
    qpc_path = str(
        qeff_model.compile(
            img_size=img_size,
            height=height,
            width=width,
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            batch_size=1,
            num_cores=16,
            num_devices=len(device_ids),
            mxfp6_matmul=precision in ("mxfp6", "mxfp6+mxint8_kv"),
            mxint8_kv_cache=precision == "mxfp6+mxint8_kv",
        )
    )
    compile_seconds = time.perf_counter() - started

    # STEP 3: build inputs. The padded frame matches the fixed QPC while keeping
    # document geometry intact for PaddleOCR's smart_resize path.
    target_height = height if height is not None else img_size
    target_width = width if width is not None else img_size
    image = resize_and_pad(image, target_height, target_width)
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    target_pixels = target_height * target_width
    inputs = processor(
        images=image,
        text=chat,
        return_tensors="pt",
        min_pixels=target_pixels,
        max_pixels=target_pixels,
    )
    patch_size = qeff_model.model.config.vision_config.patch_size
    expected_patches = (target_height // patch_size) * (target_width // patch_size)
    actual_patches = inputs["pixel_values"].shape[0]
    if actual_patches != expected_patches:
        raise ValueError(
            f"Processor produced {actual_patches} patches, but the {target_height}x{target_width} QPC expects "
            f"{expected_patches}."
        )
    real_len = inputs["input_ids"].shape[1]

    pos_4d = build_4d_position_ids(qeff_model, inputs, prefill_seq_len)
    padded_len = pos_4d.shape[-1]
    pad_id = processor.tokenizer.pad_token_id or 0
    input_ids = F.pad(inputs["input_ids"], (0, padded_len - real_len), value=pad_id).numpy()
    pixel_values = inputs["pixel_values"].numpy().astype(np.float32)
    image_idx = np.zeros((1, 1), dtype=np.int64)

    # STEP 4: run the QPC — chunked prefill, then greedy decode with per-step 4D positions
    _, compiled_ctx, _ = get_compilation_dims(qpc_path)
    ctx_len = compiled_ctx or ctx_len
    session = QAICInferenceSession(qpc_path, device_ids=device_ids, activate=False)
    session.skip_buffers(
        [x for x in session.input_names + session.output_names if x.startswith("past_") or x.endswith("_RetainedState")]
    )
    session.activate()

    in_names = set(session.input_names)
    kv_inputs = [name for name in in_names if name.startswith("past_key.") or name.startswith("past_value.")]
    if not kv_inputs:
        raise RuntimeError("QPC has no host KV-cache input bindings.")
    kv = make_empty_kv(qeff_model.model.config.text_config, ctx_len, qpc_input_dtype(session, kv_inputs[0]))
    num_chunks = -(-real_len // prefill_seq_len)  # ceil

    inference_started = time.perf_counter()
    first_token_at = None
    last = None
    for c in range(num_chunks):
        s, e = c * prefill_seq_len, (c + 1) * prefill_seq_len
        feed = {
            "input_ids": input_ids[:, s:e],
            "position_ids": pos_4d[:, :, s:e],
            "pixel_values": pixel_values,
            "image_idx": image_idx,
        }
        feed.update(kv)
        last = session.run({k: v for k, v in feed.items() if k in in_names})
        if first_token_at is None:
            first_token_at = time.perf_counter()
        if "image_idx_output" in last:
            image_idx = last["image_idx_output"].astype(np.int64)
        # pixel_values becomes device-retained after the first chunk (if compiled so)
        if "pixel_values_RetainedState" in session.output_names:
            session.skip_buffers(["pixel_values"])
            pixel_values = None

    logits = last["logits"]
    logits = logits[0, 0] if logits.ndim == 3 else logits
    tokens = [int(logits.argmax())]
    cur_pos = int(pos_4d[0, 0, :real_len].max()) + 1

    eos = qeff_model.model.config.text_config.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or [])

    for _ in range(generation_len - 1):
        if tokens[-1] in eos:
            break
        feed = {
            "input_ids": np.array([[tokens[-1]]], dtype=np.int64),
            "position_ids": np.full((4, 1, 1), cur_pos, dtype=np.int64),
            "image_idx": image_idx,
        }
        if pixel_values is not None:
            feed["pixel_values"] = pixel_values
        feed.update(kv)
        out = session.run({k: v for k, v in feed.items() if k in in_names})
        if "image_idx_output" in out:
            image_idx = out["image_idx_output"].astype(np.int64)
        lg = out["logits"]
        lg = lg[0, 0] if lg.ndim == 3 else lg
        tokens.append(int(lg.argmax()))
        cur_pos += 1
    session.deactivate()

    print("Generated:", repr(processor.tokenizer.decode(tokens, skip_special_tokens=True)))
    inference_seconds = time.perf_counter() - inference_started
    ttft_seconds = first_token_at - inference_started
    decode_tokens = max(len(tokens) - 1, 0)
    decode_seconds = max(inference_seconds - ttft_seconds, 0.0)
    print(f"Compile: {compile_seconds:.3f}s")
    print(f"TTFT: {ttft_seconds:.3f}s")
    print(f"Decode: {decode_tokens / decode_seconds:.2f} tok/s" if decode_seconds else "Decode: n/a")
    print(f"End-to-end: {time.perf_counter() - started:.3f}s")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL single-QPC inference on Cloud AI 100")
    parser.add_argument("--model-id", default=MODEL_ID)
    image_source = parser.add_mutually_exclusive_group()
    image_source.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    image_source.add_argument("--image-path", type=Path)
    parser.add_argument("--prompt", default="OCR:")
    parser.add_argument("--precision", choices=["fp16", "mxfp6", "mxfp6+mxint8_kv"], default="mxfp6+mxint8_kv")
    parser.add_argument("--device-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--prefill-seq-len", type=int, default=512)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--generation-len", type=int, default=128)
    args = parser.parse_args()

    if args.img_size is not None and (args.height is not None or args.width is not None):
        parser.error("Pass either --img-size or both --height and --width.")
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be passed together.")
    if args.img_size is None and args.height is None:
        args.img_size = 392

    if args.image_path is not None:
        image = Image.open(args.image_path).convert("RGB")
    else:
        response = requests.get(args.image_url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
    run(
        args.model_id,
        image,
        args.prompt,
        args.precision,
        args.device_ids,
        args.img_size,
        args.height,
        args.width,
        args.prefill_seq_len,
        args.ctx_len,
        args.generation_len,
    )


if __name__ == "__main__":
    main()
