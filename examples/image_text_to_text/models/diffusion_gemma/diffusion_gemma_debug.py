# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Deterministic trace and graph-difference helpers for DiffusionGemma examples."""

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch

MATCHED_FP32_OPS = {"CustomRMSNorm", "Clip", "Softmax", "Add", "Sub", "Mul", "Div", "Tanh", "Pow", "ReduceMean"}


@dataclass
class SamplerStep:
    next_canvas: np.ndarray
    denoiser_canvas: np.ndarray
    argmax_canvas: np.ndarray
    final_canvas: np.ndarray
    accepted_mask: np.ndarray
    self_conditioning_logits: np.ndarray
    temperature: float
    mean_entropy: float
    should_stop: bool
    stop_reason: str | None

    @property
    def accepted_count(self):
        return int(self.accepted_mask.sum())


class HostSampler:
    """Shared deterministic host sampler for the single- and dual-QPC runners."""

    POLICIES = ("numpy-gumbel", "hf-torch")

    def __init__(
        self,
        policy,
        seed,
        vocab_size,
        canvas_length,
        entropy_bound=0.1,
        t_min=0.4,
        t_max=0.8,
        stability_threshold=1,
        confidence_threshold=0.005,
    ):
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown sampler policy {policy!r}; expected one of {self.POLICIES}")
        self.policy = policy
        self.vocab_size = vocab_size
        self.canvas_length = canvas_length
        self.entropy_bound = entropy_bound
        self.t_min = t_min
        self.t_max = t_max
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self._numpy_rng = np.random.RandomState(seed if seed >= 0 else None)
        self._torch_rng = torch.Generator(device="cpu")
        if seed >= 0:
            self._torch_rng.manual_seed(seed)
        else:
            self._torch_rng.seed()
        self._accepted_mask = None
        self._argmax_history = None

    def initialize_canvas(self, batch_size=1):
        self._accepted_mask = None
        self._argmax_history = None
        if self.policy == "hf-torch":
            return torch.randint(
                0,
                self.vocab_size,
                (batch_size, self.canvas_length),
                generator=self._torch_rng,
                dtype=torch.int64,
            ).numpy()
        return self._numpy_rng.randint(
            0, self.vocab_size, size=(batch_size, self.canvas_length), dtype=np.int64
        )

    def step(self, logits, current_canvas, step, max_steps):
        if self.policy == "hf-torch":
            return self._hf_step(logits, current_canvas, step, max_steps)
        return self._numpy_step(logits, current_canvas, step, max_steps)

    def _temperature(self, step, max_steps, match_hf):
        if match_hf:
            cur_step = max_steps - step
            return self.t_min + (self.t_max - self.t_min) * (cur_step / max_steps)
        return self.t_min + (self.t_max - self.t_min) * (max_steps - 1 - step) / max(1, max_steps - 1)

    def _hf_step(self, logits, current_canvas, step, max_steps):
        temperature = self._temperature(step, max_steps, match_hf=True)
        processed_logits = torch.from_numpy(np.asarray(logits, dtype=np.float32)) / temperature
        probabilities = torch.softmax(processed_logits, dim=-1, dtype=torch.float32)
        batch_size, canvas_length, vocab_size = probabilities.shape
        denoiser = torch.multinomial(
            probabilities.reshape(-1, vocab_size), num_samples=1, generator=self._torch_rng
        ).squeeze(-1).reshape(batch_size, canvas_length)
        argmax_canvas = torch.argmax(processed_logits, dim=-1)

        entropy = torch.distributions.Categorical(logits=processed_logits).entropy()
        sorted_entropy, sorted_indices = torch.sort(entropy, dim=-1)
        selected = torch.cumsum(sorted_entropy, dim=-1) - sorted_entropy <= self.entropy_bound
        accepted_mask = torch.scatter(torch.zeros_like(selected), -1, sorted_indices, selected)
        current = torch.from_numpy(np.asarray(current_canvas, dtype=np.int64))
        accepted_canvas = torch.where(accepted_mask, denoiser, current)

        random_canvas = torch.randint(
            0,
            self.vocab_size,
            accepted_canvas.shape,
            generator=self._torch_rng,
            dtype=torch.int64,
        )
        next_canvas = torch.where(~accepted_mask, random_canvas, accepted_canvas)

        if self.stability_threshold == 0:
            stable = torch.ones(batch_size, dtype=torch.bool)
        else:
            if self._argmax_history is None:
                self._argmax_history = torch.full(
                    (self.stability_threshold, batch_size, canvas_length), -1, dtype=torch.int64
                )
            stable = (self._argmax_history == argmax_canvas[None, :, :]).all(dim=-1).all(dim=0)
            self._argmax_history = torch.roll(self._argmax_history, shifts=-1, dims=0)
            self._argmax_history[-1] = argmax_canvas
        mean_entropy = entropy.mean(dim=-1)
        should_stop = bool(torch.all(stable & (mean_entropy < self.confidence_threshold)))
        return SamplerStep(
            next_canvas=next_canvas.numpy(),
            denoiser_canvas=denoiser.numpy(),
            argmax_canvas=argmax_canvas.numpy(),
            final_canvas=argmax_canvas.numpy(),
            accepted_mask=accepted_mask.numpy(),
            self_conditioning_logits=processed_logits.numpy(),
            temperature=float(temperature),
            mean_entropy=float(mean_entropy.mean()),
            should_stop=should_stop,
            stop_reason="stable-and-confident" if should_stop else None,
        )

    def _numpy_step(self, logits, current_canvas, step, max_steps):
        temperature = self._temperature(step, max_steps, match_hf=False)
        processed_logits = np.asarray(logits, dtype=np.float32) / temperature
        gumbel = -np.log(
            -np.log(self._numpy_rng.uniform(size=processed_logits.shape).astype(np.float32) + 1e-20) + 1e-20
        )
        denoiser = (processed_logits + gumbel).argmax(-1).astype(np.int64)
        shifted = processed_logits - processed_logits.max(-1, keepdims=True)
        log_softmax = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
        entropy = -(np.exp(log_softmax) * log_softmax).sum(-1)
        order = np.argsort(entropy, axis=-1)
        sorted_entropy = np.take_along_axis(entropy, order, axis=-1)
        selected = (np.cumsum(sorted_entropy, axis=-1) - sorted_entropy) <= self.entropy_bound
        accepted_this_step = np.zeros_like(selected)
        np.put_along_axis(accepted_this_step, order, selected, axis=-1)
        new_canvas = np.where(accepted_this_step, denoiser, current_canvas)
        if self._accepted_mask is None:
            self._accepted_mask = np.zeros_like(accepted_this_step)
        self._accepted_mask |= accepted_this_step
        next_canvas = np.where(
            ~self._accepted_mask,
            self._numpy_rng.randint(0, self.vocab_size, size=current_canvas.shape, dtype=np.int64),
            new_canvas,
        )
        should_stop = bool(self._accepted_mask.all())
        return SamplerStep(
            next_canvas=next_canvas,
            denoiser_canvas=denoiser,
            argmax_canvas=processed_logits.argmax(-1).astype(np.int64),
            final_canvas=new_canvas,
            accepted_mask=self._accepted_mask.copy(),
            self_conditioning_logits=np.asarray(logits, dtype=np.float32),
            temperature=float(temperature),
            mean_entropy=float(entropy.mean()),
            should_stop=should_stop,
            stop_reason="full-acceptance" if should_stop else None,
        )


def array_summary(value, include_values=False):
    """Return a compact, exact fingerprint; optionally include small integer tensors."""
    array = np.ascontiguousarray(value)
    summary = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
        "nonzero": int(np.count_nonzero(array)),
    }
    if array.size:
        summary["min"] = array.min().item()
        summary["max"] = array.max().item()
    if include_values:
        summary["values"] = array.tolist()
    return summary


def logits_sample(value):
    """Record a small deterministic cross-section of a large canvas-logits tensor."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected [batch, canvas, vocab] logits, got shape {array.shape}")
    positions = sorted({0, array.shape[1] // 2, array.shape[1] - 1})
    vocab = sorted({0, 1, array.shape[2] // 4, array.shape[2] // 2, 3 * array.shape[2] // 4,
                    array.shape[2] - 2, array.shape[2] - 1})
    sample = array[0][np.ix_(positions, vocab)]
    return {"position_indices": positions, "vocab_indices": vocab, "values": sample.tolist()}


def npi_tensor_count(path):
    """Count the quoted tensor names emitted by the example NPI writers."""
    return len(re.findall(r"'[^']+'", Path(path).read_text(encoding="utf-8")))


def write_matched_accum_npi(onnx_path):
    """Select the same semantic FP32 accumulation classes in every graph."""
    graph = onnx.load(onnx_path, load_external_data=False).graph
    tensors = set()
    for node in graph.node:
        is_attention_score_matmul = node.op_type == "MatMul" and (
            node.name.endswith("/self_attn/MatMul") or node.name.endswith("/self_attn/MatMul_1")
        )
        is_self_conditioning = "/self_conditioning/" in node.name
        if node.op_type in MATCHED_FP32_OPS or is_attention_score_matmul or is_self_conditioning:
            tensors.update(name for name in node.output if name)

    path = os.path.join(os.path.dirname(onnx_path), "npi_fp32_matched_accum.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("FP32NodeInstanceNames: [")
        handle.write(", ".join(f"'{name}'" for name in sorted(tensors)))
        handle.write("]\n")
    print(f"  matched fp32 accumulation island: {len(tensors)} tensors -> {path}")
    return path


def kv_sample_manifest(buffers, valid_length, sample_width=16):
    """Sample the first and last valid KV positions without scanning huge cache tensors."""
    manifest = {}
    for name, value in sorted(buffers.items()):
        if not name.startswith(("past_key.", "past_value.")):
            continue
        canonical_name = name.removesuffix("_RetainedState").removesuffix("_out")
        array = np.asarray(value)
        if array.ndim != 4 or not array.shape[2]:
            continue
        last_position = min(valid_length - 1, array.shape[2] - 1)
        sample = np.concatenate(
            (array[0, 0, 0, :sample_width], array[0, 0, last_position, :sample_width])
        ).astype(np.float32)
        manifest[canonical_name] = {
            "shape": list(array.shape),
            "sample": sample.tolist(),
            "sample_sha256": hashlib.sha256(np.ascontiguousarray(sample).view(np.uint8)).hexdigest(),
        }
    return manifest


class TraceWriter:
    def __init__(self, path, backend):
        self.path = Path(path) if path else None
        self.backend = backend
        self._handle = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8")

    @property
    def enabled(self):
        return self._handle is not None

    def event(self, event, **fields):
        if not self._handle:
            return
        payload = {"event": event, "backend": self.backend, **fields}
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self):
        if self._handle:
            self._handle.close()
            self._handle = None


def _read_trace(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare_traces(left_path, right_path):
    """Compare correctness-relevant trace fields, ignoring backend and wall time."""
    left = _read_trace(left_path)
    right = _read_trace(right_path)
    ignored = {"backend", "wall_seconds", "qpc_path", "device_ids", "kv_behavior", "kv_buffers"}
    mismatches = []
    if len(left) != len(right):
        mismatches.append(f"event count: {len(left)} != {len(right)}")
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        lhs = {key: value for key, value in lhs.items() if key not in ignored}
        rhs = {key: value for key, value in rhs.items() if key not in ignored}
        if lhs != rhs:
            keys = sorted(key for key in lhs.keys() | rhs.keys() if lhs.get(key) != rhs.get(key))
            mismatches.append(
                f"event {index} ({lhs.get('event')} vs {rhs.get('event')}): differing fields {', '.join(keys)}"
            )
    return mismatches


def compare_kv_samples(left_path, right_path):
    left_prefill = next(event for event in _read_trace(left_path) if event["event"] == "prefill")
    right_prefill = next(event for event in _read_trace(right_path) if event["event"] == "prefill")
    left = left_prefill.get("kv_samples", {})
    right = right_prefill.get("kv_samples", {})
    rows = []
    for name in sorted(left.keys() & right.keys()):
        lhs = np.asarray(left[name]["sample"], dtype=np.float32)
        rhs = np.asarray(right[name]["sample"], dtype=np.float32)
        denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        cosine = float(np.dot(lhs, rhs) / denominator) if denominator else float(lhs.shape == rhs.shape and np.array_equal(lhs, rhs))
        rows.append((name, float(np.max(np.abs(lhs - rhs))), cosine))
    return rows, sorted(left.keys() ^ right.keys())


def compare_logits_samples(left_path, right_path):
    """Return max-absolute and cosine differences for matching sampled denoising logits."""
    left = {
        (event["block"], event["step"]): event["logits_sample"]
        for event in _read_trace(left_path)
        if event["event"] == "denoise_step" and event.get("logits_sample")
    }
    right = {
        (event["block"], event["step"]): event["logits_sample"]
        for event in _read_trace(right_path)
        if event["event"] == "denoise_step" and event.get("logits_sample")
    }
    rows = []
    for key in sorted(left.keys() & right.keys()):
        lhs_meta, rhs_meta = left[key], right[key]
        if (lhs_meta["position_indices"], lhs_meta["vocab_indices"]) != (
            rhs_meta["position_indices"], rhs_meta["vocab_indices"]
        ):
            rows.append((*key, float("inf"), float("nan")))
            continue
        lhs = np.asarray(lhs_meta["values"], dtype=np.float32).reshape(-1)
        rhs = np.asarray(rhs_meta["values"], dtype=np.float32).reshape(-1)
        denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
        cosine = float(np.dot(lhs, rhs) / denominator) if denominator else float(np.array_equal(lhs, rhs))
        rows.append((*key, float(np.max(np.abs(lhs - rhs))), cosine))
    return rows, sorted(left.keys() ^ right.keys())


def compare_full_logits(reference, candidate):
    """Return raw-logit diagnostics for one fixed pre-sampling decoder call."""
    ref = np.asarray(reference, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    if ref.shape != cand.shape:
        raise ValueError(f"logit shape mismatch: {ref.shape} != {cand.shape}")
    diff = np.abs(ref - cand)
    ref_flat, cand_flat = ref.reshape(-1), cand.reshape(-1)
    denom = float(np.linalg.norm(ref_flat) * np.linalg.norm(cand_flat))
    ref_ids = ref.argmax(axis=-1)
    cand_ids = cand.argmax(axis=-1)
    return {
        "shape": list(ref.shape),
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "cosine": float(np.dot(ref_flat, cand_flat) / denom) if denom else float(np.array_equal(ref, cand)),
        "argmax_agreement": float(np.mean(ref_ids == cand_ids)),
    }


def compare_full_kv(reference, candidate):
    """Return per-buffer state diagnostics for identically named KV tensors."""
    ref_keys, cand_keys = set(reference), set(candidate)
    if ref_keys != cand_keys:
        return {"missing_reference": sorted(cand_keys - ref_keys), "missing_candidate": sorted(ref_keys - cand_keys)}
    return {name: compare_full_logits(reference[name], candidate[name]) for name in sorted(ref_keys)}


def _graph_stats(path):
    graph = onnx.load(path, load_external_data=False).graph
    ops = Counter(node.op_type for node in graph.node)
    prefixes = Counter()
    for node in graph.node:
        if "/decoder/" in node.name or node.name.startswith("/decoder"):
            prefixes["decoder"] += 1
        elif (
            "/encoder/" in node.name
            or node.name.startswith("/encoder")
            or "/encoder_prefill/" in node.name
            or "/language_model/" in node.name
            or node.name.startswith("/language_model")
        ):
            prefixes["encoder_or_language_model"] += 1
        else:
            prefixes["shared_or_other"] += 1
    return {
        "path": str(path),
        "nodes": len(graph.node),
        "inputs": [value.name for value in graph.input],
        "outputs": [value.name for value in graph.output],
        "ops": dict(sorted(ops.items())),
        "regions": dict(sorted(prefixes.items())),
    }


def write_graph_report(output_path, single_path, encoder_path, decoder_path):
    graphs = {
        "single_qpc": _graph_stats(single_path),
        "dual_encoder": _graph_stats(encoder_path),
        "dual_decoder": _graph_stats(decoder_path),
    }
    single = graphs["single_qpc"]
    rows = [
        ("single encoder prefill", single, "execution_mode=0; encoder output/KV selected"),
        ("single encoder commit", single, "execution_mode=1; encoder output/KV selected"),
        ("single decoder step 0", single, "execution_mode=2; self-conditioning disabled"),
        ("single decoder later", single, "execution_mode=3; previous logits used for self-conditioning"),
        ("dual encoder", graphs["dual_encoder"], "encoder-only graph; emits host-visible KV"),
        ("dual decoder", graphs["dual_decoder"], "decoder-only graph; reads host-provided KV"),
    ]
    lines = [
        "# DiffusionGemma graph-difference report",
        "",
        "The four single-QPC modes use the same compiled specialization. `execution_mode` feeds",
        "`Where` selections, so this ONNX structure does not prove that inactive encoder/decoder computation is skipped.",
        "",
        "| Mode | Nodes | Encoder/language-region nodes | Decoder-region nodes | Runtime distinction |",
        "|---|---:|---:|---:|---|",
    ]
    for name, stats, distinction in rows:
        regions = stats["regions"]
        lines.append(
            f"| {name} | {stats['nodes']} | {regions.get('encoder_or_language_model', 0)} | "
            f"{regions.get('decoder', 0)} | {distinction} |"
        )
    lines.extend(["", "## Graph signatures", ""])
    for name, stats in graphs.items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Path: `{stats['path']}`",
                f"- Inputs: `{', '.join(stats['inputs'])}`",
                f"- Outputs: `{', '.join(stats['outputs'])}`",
                f"- Operator counts: `{json.dumps(stats['ops'], sort_keys=True)}`",
                "",
            ]
        )
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare = subparsers.add_parser("compare-traces")
    compare.add_argument("left")
    compare.add_argument("right")
    compare_kv = subparsers.add_parser("compare-kv")
    compare_kv.add_argument("left")
    compare_kv.add_argument("right")
    compare_logits = subparsers.add_parser("compare-logits")
    compare_logits.add_argument("left")
    compare_logits.add_argument("right")
    graph = subparsers.add_parser("graph-report")
    graph.add_argument("--single", required=True)
    graph.add_argument("--encoder", required=True)
    graph.add_argument("--decoder", required=True)
    graph.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "compare-traces":
        mismatches = compare_traces(args.left, args.right)
        if mismatches:
            print("\n".join(mismatches))
            raise SystemExit(1)
        print("Traces match on all correctness-relevant fields.")
    elif args.command == "compare-kv":
        rows, missing = compare_kv_samples(args.left, args.right)
        for name, max_abs, cosine in rows:
            print(f"{name}: max_abs={max_abs:.8g} cosine={cosine:.8f}")
        if missing:
            print(f"Only present on one side: {', '.join(missing)}")
            raise SystemExit(1)
    elif args.command == "compare-logits":
        rows, missing = compare_logits_samples(args.left, args.right)
        for block, step, max_abs, cosine in rows:
            print(f"block={block} step={step}: max_abs={max_abs:.8g} cosine={cosine:.8f}")
        if missing:
            print(f"Only present on one side: {missing}")
            raise SystemExit(1)
    else:
        write_graph_report(args.output, args.single, args.encoder, args.decoder)
        print(args.output)


if __name__ == "__main__":
    main()
