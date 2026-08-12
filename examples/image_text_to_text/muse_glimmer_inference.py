# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Compile and run Muse Glimmer image-to-text inference on Cloud AI 100.

Example:
    python examples/image_text_to_text/muse_glimmer_inference.py \
        --device-ids 4,5,6,7 --precision mxfp6

Use ``--precision fp16`` as the control path. ``--precision mxfp6`` enables
MXFP6 matmuls with FP16 KV cache. ``--precision mxfp6+mxint8-kv`` is exposed for
debugging the MXINT8 KV-cache path and should not be treated as validated unless
the model's parity evidence covers it. The printed timing values are
single-request measurements, intended as deployment telemetry rather than a
benchmark claim.
"""

import argparse
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL = "meta-models/Muse-Glimmer-30B"
DEFAULT_IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"

PRECISION_FLAGS = {
    "fp16": {"mxfp6_matmul": False, "mxint8_kv_cache": False},
    "mxfp6": {"mxfp6_matmul": True, "mxint8_kv_cache": False},
    "mxfp6+mxint8-kv": {"mxfp6_matmul": True, "mxint8_kv_cache": True},
    "mxfp6+mxint8_kv": {"mxfp6_matmul": True, "mxint8_kv_cache": True},
}


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def parse_device_ids(value):
    device_ids = [int(item) for item in value.split(",") if item.strip()]
    if not device_ids:
        raise argparse.ArgumentTypeError("--device-ids must contain at least one QID")
    return device_ids


def validate_image_size(image_size, patch_size):
    if image_size <= 0:
        raise ValueError("--image-size must be positive")
    if image_size % patch_size:
        raise ValueError(f"--image-size {image_size} must be divisible by Muse patch size {patch_size}")


def load_image(path, url, image_size):
    if path:
        image = Image.open(path).convert("RGB")
    else:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
    return image.resize((image_size, image_size))


def prepare_prefill(processor, tokenizer, args):
    image = load_image(args.image, args.image_url, args.image_size)
    prompt = (
        "<|begin_of_text|><|start|>user<|message|><|patch|>"
        f"{args.prompt}<|eot|><|start|>assistant"
    )
    raw_inputs = processor(text=prompt, images=image, return_tensors="pt")
    valid_len = int(raw_inputs["input_ids"].shape[1])
    if valid_len > args.prefill_seq_len:
        raise ValueError(f"Prompt length {valid_len} exceeds --prefill-seq-len {args.prefill_seq_len}")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = torch.full((1, args.prefill_seq_len), int(pad_token_id), dtype=torch.int64)
    input_ids[:, :valid_len] = raw_inputs["input_ids"].to(torch.int64)
    position_ids = torch.full((1, args.prefill_seq_len), -1, dtype=torch.int64)
    position_ids[:, :valid_len] = torch.arange(valid_len, dtype=torch.int64)

    return {
        "input_ids": input_ids.numpy(),
        "position_ids": position_ids.numpy(),
        "pixel_values": as_numpy(raw_inputs["pixel_values"]).astype(np.float32),
        "image_grid_thw": as_numpy(raw_inputs["image_grid_thw"]).astype(np.int64),
    }, valid_len


def zero_past_inputs(input_names, config, ctx_len):
    full_attention_interval = next(
        (index + 1 for index, kind in enumerate(config.text_config.layer_types) if kind == "full_attention"),
        config.text_config.num_hidden_layers,
    )
    past = {}
    for layer in range(config.text_config.num_hidden_layers):
        cache_len = config.text_config.sliding_window if (layer + 1) % full_attention_interval else ctx_len
        shape = (1, config.text_config.num_key_value_heads, cache_len, config.text_config.head_dim)
        for kind in ("past_key", "past_value"):
            name = f"{kind}.{layer}"
            if name in input_names:
                past[name] = np.zeros(shape, dtype=np.float16)
    return past


def session_run(session, values):
    inputs = {name: value for name, value in values.items() if name in session.input_names}
    missing = set(session.input_names) - set(inputs)
    if missing:
        raise RuntimeError(f"Missing required QPC inputs: {sorted(missing)}")
    return session.run(inputs)


def next_token(logits):
    logits = np.asarray(logits)
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    return int(logits[0].argmax())


def decode_generated_tokens(tokenizer, generated):
    text = tokenizer.decode(generated, skip_special_tokens=True)
    # Muse's chat template can leave routing fragments in raw greedy output when
    # decoding only the newly generated tokens. Keep the raw token ids above and
    # present a cleaner user-facing string here.
    return text.replace("to=self", "").strip()


def compile_or_load(model, args):
    if args.qpc_path:
        qpc_path = Path(args.qpc_path).expanduser()
        if not (qpc_path / "programqpc.bin").is_file():
            raise FileNotFoundError(f"No programqpc.bin under --qpc-path: {qpc_path}")
        return qpc_path, 0.0

    started = time.perf_counter()
    qpc_path = model.compile(
        num_cores=args.num_cores,
        num_devices=len(args.device_ids),
        batch_size=1,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        height=args.image_size,
        width=args.image_size,
        **PRECISION_FLAGS[args.precision],
    )
    return Path(qpc_path), time.perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description="Muse Glimmer Cloud AI 100 inference")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--image", help="Local image path; overrides --image-url")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--image-size", type=int, default=56)
    parser.add_argument("--prefill-seq-len", type=int, default=640)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=0, help="Extra prefill runs before measured inference")
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--device-ids", type=parse_device_ids, required=True, help="Comma-separated Cloud AI 100 QIDs")
    parser.add_argument("--precision", choices=PRECISION_FLAGS, default="mxfp6")
    parser.add_argument("--qpc-path", help="Reuse an existing QPC instead of compiling")
    parser.add_argument("--compile-only", action="store_true", help="Compile or validate --qpc-path and exit before runtime")
    args = parser.parse_args()

    if args.generation_len < 1:
        parser.error("--generation-len must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")

    print(f"Loading {args.model_name}")
    model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_name, kv_offload=False)
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    validate_image_size(args.image_size, model.model.config.vision_config.patch_size)

    qpc_path, compile_seconds = compile_or_load(model, args)
    print(f"QPC: {qpc_path}")
    if compile_seconds:
        print(f"Compile time: {compile_seconds:.2f} s")
    if args.compile_only:
        return

    prefill, valid_len = prepare_prefill(processor, tokenizer, args)
    session_started = time.perf_counter()
    session = QAICInferenceSession(str(qpc_path), device_ids=args.device_ids)
    session_load_seconds = time.perf_counter() - session_started

    try:
        prefill.update(zero_past_inputs(session.input_names, model.model.config, args.ctx_len))
        for _ in range(args.warmup_runs):
            session_run(session, prefill)

        prefill_started = time.perf_counter()
        outputs = session_run(session, prefill)
        prefill_seconds = time.perf_counter() - prefill_started
        generated = [next_token(outputs["logits"])]

        decode_seconds = 0.0
        for step in range(1, args.generation_len):
            decode_inputs = {
                "input_ids": np.array([[generated[-1]]], dtype=np.int64),
                "position_ids": np.array([[valid_len + step - 1]], dtype=np.int64),
                "pixel_values": prefill["pixel_values"],
                "image_grid_thw": prefill["image_grid_thw"],
            }
            decode_inputs.update(
                {
                    name.removesuffix("_RetainedState"): value
                    for name, value in outputs.items()
                    if name.endswith("_RetainedState")
                }
            )
            started = time.perf_counter()
            outputs = session_run(session, decode_inputs)
            decode_seconds += time.perf_counter() - started
            generated.append(next_token(outputs["logits"]))
            if generated[-1] == tokenizer.eos_token_id:
                break
    finally:
        session.deactivate()

    decoded = decode_generated_tokens(tokenizer, generated)
    decode_tokens = max(len(generated) - 1, 0)
    total_runtime_seconds = prefill_seconds + decode_seconds
    print(f"Generated token ids: {generated}")
    print(f"Generated text: {decoded!r}")
    print("Performance (single request):")
    if args.warmup_runs:
        print(f"  Warmup prefill runs: {args.warmup_runs}")
    print(f"  QPC load: {session_load_seconds * 1000:.1f} ms")
    print(f"  Prefill: {prefill_seconds * 1000:.1f} ms ({valid_len} input tokens)")
    if decode_tokens:
        print(f"  Decode: {decode_seconds * 1000:.1f} ms ({decode_tokens} tokens)")
        print(f"  Decode throughput: {decode_tokens / decode_seconds:.2f} tokens/s")
    print(f"  End-to-end measured runtime: {total_runtime_seconds * 1000:.1f} ms")


if __name__ == "__main__":
    main()
