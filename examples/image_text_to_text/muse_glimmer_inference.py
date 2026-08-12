# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Muse Glimmer image-text and text-only inference on Cloud AI 100."""

import argparse
import re
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText

MODEL_ID = "meta-models/Muse-Glimmer-30B"
IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"


def parse_device_ids(value):
    device_ids = [int(device_id) for device_id in value.split(",")]
    if not device_ids:
        raise argparse.ArgumentTypeError("--device-ids must contain at least one device")
    return device_ids


def load_image(image_url, image_size):
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB").resize((image_size, image_size))


def prepare_inputs(processor, model, mode, prompt, image_url, image_size, prefill_seq_len, ctx_len):
    content = [{"type": "text", "text": prompt}]
    if mode == "image":
        content.insert(0, {"type": "image"})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_strength="low",
    )
    images = load_image(image_url, image_size) if mode == "image" else None
    inputs = processor(text=text, images=images, return_tensors="pt")

    if mode == "text":
        patch_size = model.model.config.vision_config.patch_size
        grid_size = image_size // patch_size
        dummy = model.model.get_dummy_inputs(
            prefill_seq_len=prefill_seq_len,
            ctx_len=ctx_len,
            grid_h=grid_size,
            grid_w=grid_size,
        )
        inputs["pixel_values"] = dummy["pixel_values"]
        inputs["image_grid_thw"] = dummy["image_grid_thw"]

    inputs["input_ids"] = inputs["input_ids"].to(torch.int64)
    inputs["attention_mask"] = inputs["attention_mask"].to(torch.int64)
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float32)
    inputs["image_grid_thw"] = inputs["image_grid_thw"].to(torch.int64)
    return inputs


def decode_response(tokenizer, generated_ids):
    raw_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    match = re.search(r"to=user<\|message\|>(.*?)(?:<\|eot\|>|$)", raw_text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No final user response was generated; increase --generation-len."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_ID)
    parser.add_argument("--mode", choices=("image", "text"), default="image")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--image-url", default=IMAGE_URL)
    parser.add_argument("--image-size", type=int, default=56)
    parser.add_argument("--prefill-seq-len", type=int, default=640)
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--generation-len", type=int, default=32)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--device-ids", type=parse_device_ids, default=parse_device_ids("0,1,2,3"))
    parser.add_argument("--precision", choices=("fp16", "mxfp6"), default="mxfp6")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = QEFFAutoModelForImageTextToText.from_pretrained(args.model_name, kv_offload=False)
    inputs = prepare_inputs(
        processor,
        model,
        args.mode,
        args.prompt,
        args.image_url,
        args.image_size,
        args.prefill_seq_len,
        args.ctx_len,
    )

    model.compile(
        batch_size=1,
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        height=args.image_size,
        width=args.image_size,
        num_cores=args.num_cores,
        num_devices=len(args.device_ids),
        mxfp6_matmul=args.precision == "mxfp6",
        mxint8_kv_cache=False,
        node_precision_info=True,
    )

    output = model.generate(
        inputs=inputs,
        device_ids=args.device_ids,
        generation_len=args.generation_len,
    )
    print(decode_response(processor.tokenizer, output.generated_ids))
    print(
        "Performance: "
        f"prefill={output.perf_metrics.prefill_time:.3f}s, "
        f"decode={output.perf_metrics.decode_perf:.2f} tok/s, "
        f"end-to-end={output.perf_metrics.total_perf:.2f} tok/s"
    )


if __name__ == "__main__":
    main()
