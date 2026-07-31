# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoProcessor

from QEfficient import QEFFAutoModelForSpeechSeq2Seq
from QEfficient.transformers.models.modeling_auto import _load_qwen3_asr_native_checkpoint


def parse_device_ids(value: str):
    return [int(device_id) for device_id in value.split(",") if device_id.strip()]


def qwen3_asr_prompt(processor):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "dummy"},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)


def trim_audio_tokens(input_ids: torch.Tensor, audio_token_id: int, target_audio_tokens: int) -> torch.Tensor:
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    audio_positions = (input_ids[0] == audio_token_id).nonzero(as_tuple=False).flatten()
    if target_audio_tokens < audio_positions.numel():
        keep[0, audio_positions[target_audio_tokens:]] = False
    return input_ids[keep].view(1, -1)


def prepare_inputs(model, processor, model_name: str):
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    audio = ds[0]["audio"]["array"].reshape(-1)
    sampling_rate = ds[0]["audio"]["sampling_rate"]
    inputs = processor(
        text=qwen3_asr_prompt(processor),
        audio=audio,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    )

    chunk_len = model.config.audio_config.n_window * 2
    original_feature_len = inputs["input_features"].shape[-1]
    aligned_feature_len = (original_feature_len // chunk_len) * chunk_len
    inputs["input_features"] = inputs["input_features"][..., :aligned_feature_len]
    inputs["input_features_mask"] = inputs["input_features_mask"][..., :aligned_feature_len].to(torch.int64)

    with torch.no_grad():
        audio_features = model.get_audio_features(
            inputs["input_features"],
            inputs["input_features_mask"],
            return_dict=True,
        ).pooler_output
    original_audio_tokens = int((inputs["input_ids"] == model.config.audio_token_id).sum().item())
    target_audio_tokens = int(audio_features.shape[0])
    inputs["input_ids"] = trim_audio_tokens(inputs["input_ids"], model.config.audio_token_id, target_audio_tokens)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1], dtype=torch.int64).view(1, -1)

    metadata = {
        "model": model_name,
        "sample": "hf-internal-testing/librispeech_asr_dummy validation[0]",
        "feature_len": f"{original_feature_len} -> {aligned_feature_len}",
        "audio_tokens": f"{original_audio_tokens} -> {target_audio_tokens}",
        "prefill_seq_len": int(inputs["input_ids"].shape[1]),
    }
    return inputs, metadata


def stage_logits(model, inputs: Dict[str, torch.Tensor]) -> np.ndarray:
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs["input_features_mask"],
            attention_mask=inputs["attention_mask"],
            position_ids=inputs["position_ids"],
            use_cache=True,
        )
    return outputs.logits.detach().float().cpu().numpy()


def onnx_logits(onnx_path: Optional[str], inputs: Dict[str, torch.Tensor], ctx_len: int) -> Optional[np.ndarray]:
    if onnx_path is None:
        return None

    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    feeds = {
        "input_ids": inputs["input_ids"].cpu().numpy().astype(np.int64),
        "input_features": inputs["input_features"].cpu().numpy().astype(np.float32),
        "position_ids": inputs["position_ids"].cpu().numpy().astype(np.int64),
    }
    for session_input in session.get_inputs():
        name = session_input.name
        if name in feeds:
            continue
        if name.startswith("past_key.") or name.startswith("past_value."):
            feeds[name] = np.zeros((1, 8, ctx_len, 128), dtype=np.float32)
    try:
        outputs = session.run(["logits"], feeds)
    except Exception as exc:
        print(f"Skipping ORT stage: {exc}", file=sys.stderr)
        return None
    return outputs[0].astype(np.float32)


def qpc_logits_from_io_dir(io_dir: Optional[str]) -> Optional[np.ndarray]:
    if io_dir is None:
        return None
    io_root = Path(io_dir)
    metadata_path = io_root / "aic_batch_io.json"
    with metadata_path.open() as fp:
        metadata = json.load(fp)
    for entry in metadata["IO-files"][0]:
        if entry["io-direction"] == "out" and entry["map-to"] == "logits":
            return np.fromfile(io_root / entry["path"], dtype=np.float32).reshape(entry["dims"])
    raise ValueError(f"No prefill logits entry found in {metadata_path}")


def top_logits(logits: np.ndarray, processor, top_k: int):
    row = logits.reshape(-1, logits.shape[-1])[-1]
    token_ids = np.argsort(row)[-top_k:][::-1]
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": processor.tokenizer.decode([int(token_id)]),
            "logit": float(row[token_id]),
        }
        for rank, token_id in enumerate(token_ids, start=1)
    ]


def compare(a: np.ndarray, b: np.ndarray):
    a_flat = a.reshape(-1, a.shape[-1])[-1].astype(np.float64)
    b_flat = b.reshape(-1, b.shape[-1])[-1].astype(np.float64)
    denom = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    return {
        "argmax_a": int(a_flat.argmax()),
        "argmax_b": int(b_flat.argmax()),
        "max_abs": float(np.max(np.abs(a_flat - b_flat))),
        "mean_abs": float(np.mean(np.abs(a_flat - b_flat))),
        "cosine": float(np.dot(a_flat, b_flat) / denom) if denom else float("nan"),
    }


def largest_diffs(a: np.ndarray, b: np.ndarray, processor, top_k: int):
    a_row = a.reshape(-1, a.shape[-1])[-1]
    b_row = b.reshape(-1, b.shape[-1])[-1]
    diffs = np.abs(a_row - b_row)
    token_ids = np.argsort(diffs)[-top_k:][::-1]
    return [
        {
            "rank": rank,
            "token_id": int(token_id),
            "token": processor.tokenizer.decode([int(token_id)]),
            "a": float(a_row[token_id]),
            "b": float(b_row[token_id]),
            "abs_diff": float(diffs[token_id]),
        }
        for rank, token_id in enumerate(token_ids, start=1)
    ]


def write_top_table(lines: list[str], title: str, rows: Iterable[dict]):
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| rank | token_id | token | logit |")
    lines.append("|---:|---:|---|---:|")
    for row in rows:
        token = row["token"].replace("|", "\\|").replace("\n", "\\n")
        lines.append(f"| {row['rank']} | {row['token_id']} | `{token}` | {row['logit']:.6f} |")
    lines.append("")


def render_report(stages: Dict[str, np.ndarray], processor, metadata: dict, top_k: int) -> str:
    lines = ["# Qwen3-ASR Stage Logit Samples", ""]
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")

    for name, logits in stages.items():
        write_top_table(lines, f"{name} Top Logits", top_logits(logits, processor, top_k))

    stage_names = list(stages)
    if len(stage_names) > 1:
        lines.append("## Edge Summary")
        lines.append("")
        lines.append("| Edge | argmax_a | argmax_b | max_abs | mean_abs | cosine |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for left, right in zip(stage_names, stage_names[1:]):
            metric = compare(stages[left], stages[right])
            lines.append(
                f"| {left}=={right} | {metric['argmax_a']} | {metric['argmax_b']} | "
                f"{metric['max_abs']:.6f} | {metric['mean_abs']:.6f} | {metric['cosine']:.9f} |"
            )
        lines.append("")

    if "ORT" in stages and "QPC" in stages:
        lines.append("## Largest ORT vs QPC Logit Differences")
        lines.append("")
        lines.append("| rank | token_id | token | ORT | QPC | abs_diff |")
        lines.append("|---:|---:|---|---:|---:|---:|")
        for row in largest_diffs(stages["ORT"], stages["QPC"], processor, top_k):
            token = row["token"].replace("|", "\\|").replace("\n", "\\n")
            lines.append(
                f"| {row['rank']} | {row['token_id']} | `{token}` | "
                f"{row['a']:.6f} | {row['b']:.6f} | {row['abs_diff']:.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Print Qwen3-ASR HF/QEff/ORT/QPC prefill logit samples.")
    parser.add_argument("--model-name", default="Qwen/Qwen3-ASR-0.6B", help="HuggingFace Qwen3-ASR model ID")
    parser.add_argument("--onnx-path", help="Optional exported ONNX path for the ORT stage")
    parser.add_argument("--qpc-io-dir", help="Optional write_io directory containing QPC prefill logits")
    parser.add_argument("--ctx-len", type=int, default=180, help="KV cache context length used by ONNX/QPC")
    parser.add_argument("--top-k", type=int, default=8, help="Number of top logits and largest diffs to print")
    parser.add_argument("--output", help="Optional markdown output path")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model_name)
    hf_model = _load_qwen3_asr_native_checkpoint(args.model_name, {"torch_dtype": torch.float32})
    hf_model.eval()
    inputs, metadata = prepare_inputs(hf_model, processor, args.model_name)

    stages = {"HF": stage_logits(hf_model, inputs)}
    del hf_model

    qeff_model = QEFFAutoModelForSpeechSeq2Seq.from_pretrained(args.model_name, torch_dtype=torch.float32)
    qeff_model.model.eval()
    stages["QEff"] = stage_logits(qeff_model.model, inputs)

    ort = onnx_logits(args.onnx_path, inputs, args.ctx_len)
    if ort is not None:
        stages["ORT"] = ort

    qpc = qpc_logits_from_io_dir(args.qpc_io_dir)
    if qpc is not None:
        stages["QPC"] = qpc

    report = render_report(stages, processor, metadata, args.top_k)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")


if __name__ == "__main__":
    main()
