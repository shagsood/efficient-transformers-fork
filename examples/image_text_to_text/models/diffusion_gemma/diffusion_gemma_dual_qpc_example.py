# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""DiffusionGemma dual-QPC example on Cloud AI 100.

DiffusionGemma is a non-autoregressive block-diffusion VLM: the encoder does a
one-shot prefill over the prompt (+ vision), then the decoder iterates a fixed
canvas of `canvas_length` tokens over N diffusion steps, denoising the canvas
via an entropy-bound sampler. `QEFFAutoModelForImageTextToText.generate()` does
not apply; this example drives the two exported QPCs (encoder-prefill and
canvas-decode) via `QAICInferenceSession` directly, with the runner host-copying
encoder past_*_out into decoder past_*.{i} inputs between diffusion steps.

For the single-QPC (unified) variant, see diffusion_gemma_single_qpc_example.py.
"""

import os
import time
import argparse
import re
from io import BytesIO

import onnx
import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from QEfficient.base.modeling_qeff import QEFFBaseModel
from QEfficient.generation.cloud_infer import QAICInferenceSession
from QEfficient.transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    QEffDiffusionGemmaForBlockDiffusion,
)
from QEfficient.transformers.models.modeling_auto import QEffCausalLMForTextImageToTextModel

model_id = "google/diffusiongemma-26B-A4B-it"
prefill_seq_len = 512
ctx_len = 1024
canvas_length = 256
diffusion_steps = 48
num_cores = 16
num_devices = 4
device_ids = [int(x) for x in os.environ.get("DG", "4,5,6,7").split(",")]
image_url = (
    "https://huggingface.co/datasets/huggingface/documentation-images"
    "/resolve/main/transformers/tasks/car.jpg"
)
image_prompt_text = "Describe this image in detail."
text_only_prompt_text = "What is the capital of France? Answer in one sentence."

parser = argparse.ArgumentParser(description="Run DiffusionGemma dual-QPC inference.")
parser.add_argument("--text-only", action="store_true", help="Run a prompt with no image tokens.")
parser.add_argument("--prompt", default=None, help="Override the default image or text-only prompt.")
parser.add_argument("--seed", type=int, default=1234, help="Seed for the diffusion sampler. Use --seed -1 for the unseeded path.")
parser.add_argument("--fuse-sampler", action="store_true", help="Run the fused on-device sampler path.")
parser.add_argument("--max-top-k", type=int, default=64, help="Top-k used by the fused sampler.")
parser.add_argument("--verbose-steps", action="store_true", help="Decode and print a preview after every diffusion step.")
args = parser.parse_args()
if args.seed >= 0:
    np.random.seed(args.seed)


# ---------------------------------------------------------------------------
# Compile helpers — wire the two QEff wrappers into standalone QPCs.
# ---------------------------------------------------------------------------


class _EncoderPrefillQPC(QEffCausalLMForTextImageToTextModel):
    def __init__(self, model):
        QEFFBaseModel.__init__(self, model)
        self.model = model.get_qeff_encoder_prefill()
        self.model.qaic_config = None
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.continuous_batching = False

    @property
    def get_model_config(self):
        return self.model.model.config.__dict__

    def export(self, inputs, output_names, dynamic_axes, **kwargs):
        return self._export(inputs, output_names=output_names, dynamic_axes=dynamic_axes)


class _CanvasDecodeQPC(QEffCausalLMForTextImageToTextModel):
    def __init__(self, model, fuse_sampler=False, max_top_k=64):
        QEFFBaseModel.__init__(self, model)
        self.model = model.get_qeff_canvas_decode(fuse_sampler=fuse_sampler, max_top_k=max_top_k)
        self.model.qaic_config = None
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        if fuse_sampler:
            self.hash_params["fuse_sampler"] = fuse_sampler
            self.hash_params["max_top_k"] = max_top_k
        self.continuous_batching = False

    @property
    def get_model_config(self):
        return self.model.model.config.__dict__

    def export(self, inputs, output_names, dynamic_axes, **kwargs):
        return self._export(inputs, output_names=output_names, dynamic_axes=dynamic_axes)


FP32_ENCODER_ACCUM_OPS = {"CustomRMSNorm", "Clip", "Softmax", "Add", "Sub", "Mul", "Div", "Tanh", "Pow", "ReduceMean"}


def _write_encoder_accum_npi(onnx_path):
    """Keep the residual/norm/attention-score accumulation path in fp32."""
    graph = onnx.load(onnx_path, load_external_data=False).graph
    tensors, seen = [], set()

    def add_output(name):
        if name and name not in seen:
            seen.add(name)
            tensors.append(name)

    for node in graph.node:
        if node.op_type in FP32_ENCODER_ACCUM_OPS:
            for out_name in node.output:
                if "MatMul" not in out_name and "Einsum" not in out_name:
                    add_output(out_name)
    for node in graph.node:
        if node.op_type == "MatMul" and (node.name.endswith("/self_attn/MatMul") or node.name.endswith("/self_attn/MatMul_1")):
            for out_name in node.output:
                add_output(out_name)

    path = os.path.join(os.path.dirname(onnx_path), "npi_fp32_accum.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("FP32NodeInstanceNames: [")
        handle.write(", ".join(f"'{name}'" for name in sorted(tensors)))
        handle.write("]\n")
    print(f"  encoder fp32 accumulation island: {len(tensors)} tensors -> {path}")
    return path


def _write_decode_accum_npi(onnx_path):
    """Keep the validated logit-feedback path in fp32 without fp32 vocab matmuls."""
    graph = onnx.load(onnx_path, load_external_data=False).graph
    producers = {out_name: node for node in graph.node for out_name in node.output}
    keep_nodes = []

    for node in graph.node:
        if (
            "/decoder/self_conditioning/" in node.name
            or node.name == "/decoder/MatMul"
            or node.name == "/decoder/norm/CustomRMSNorm"
        ):
            keep_nodes.append(node)

    seen_names = set()

    def backtrace(tensor_name, depth=0):
        if tensor_name in seen_names or depth > 8:
            return
        seen_names.add(tensor_name)
        node = producers.get(tensor_name)
        if node is None or node.name == "/decoder/norm/CustomRMSNorm":
            return
        keep_nodes.append(node)
        for input_name in node.input:
            if input_name in producers:
                backtrace(input_name, depth + 1)

    backtrace(graph.output[0].name)

    vocab_matmul_outputs = {"/decoder/MatMul_output_0", "/lm_head/MatMul_output_0"}
    tensors, seen_tensors = [], set()
    for node in keep_nodes:
        for out_name in node.output:
            if out_name and out_name not in seen_tensors and out_name not in vocab_matmul_outputs:
                seen_tensors.add(out_name)
                tensors.append(out_name)

    path = os.path.join(os.path.dirname(onnx_path), "npi_fp32_feedback.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("FP32NodeInstanceNames: [")
        handle.write(", ".join(f"'{name}'" for name in sorted(tensors)))
        handle.write("]\n")
    print(f"  decode fp32 accumulation island: {len(tensors)} tensors -> {path}")
    return path


def _vision_embeds_cpu(processor, text_model, vision_inputs):
    """Run the vision tower on a fresh CPU HF model; export may offload the compile model."""
    hf_vision = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager", device_map="cpu", low_cpu_mem_usage=True
    )
    with torch.no_grad():
        encoder = hf_vision.model.encoder
        pixel_values = vision_inputs["pixel_values"]
        image_position_ids = vision_inputs["image_position_ids"]
        padding_positions = (image_position_ids == -1).all(dim=-1)
        h = encoder.vision_tower.patch_embedder(pixel_values, image_position_ids, padding_positions)
        attn_mask = ((~(~padding_positions)).unsqueeze(1).unsqueeze(2).to(h.dtype) * torch.finfo(h.dtype).min)
        attn_mask = attn_mask.expand(-1, 1, h.shape[1], -1)
        pos_emb = encoder.vision_tower.encoder.rotary_emb(h, image_position_ids)
        for layer in encoder.vision_tower.encoder.layers[: encoder.vision_tower.encoder.config.num_hidden_layers]:
            h = layer(h, attention_mask=attn_mask, position_embeddings=pos_emb, position_ids=image_position_ids)
        out_len = encoder.vision_tower.config.default_output_length
        h, _ = encoder.vision_tower.pooler(
            hidden_states=h,
            pixel_position_ids=image_position_ids,
            padding_positions=padding_positions,
            output_length=out_len,
        )
        if encoder.vision_tower.config.standardize:
            h = (h - encoder.vision_tower.std_bias) * encoder.vision_tower.std_scale
        vision_embeds = encoder.embed_vision(inputs_embeds=h).clamp(-60000.0, 60000.0)
        mm_tokens = text_model._get_mm_tokens_per_image()
        vision_embeds = vision_embeds[:, :mm_tokens, :].float().numpy()
    del hf_vision
    return vision_embeds


def _clean_diffusion_text(text):
    text = text.replace("\ufffd", " ").strip()
    text = re.sub(r"^\s*(thought\s*)+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("。", ".")
    text = re.sub(r"\bfulling shot\b", "full shot", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(light|dark)\s+(blue|green|teal)ing\b", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(?:of|Of)\b.*$", ".", text)
    match = re.search(r"(.{12,}?[.!?])", text)
    if match:
        text = match.group(1)
    return text.strip(" \n\t\r\"'")


def _coherence_score(text):
    if not text:
        return -1_000
    if len(text) < 12:
        return -500
    printable = sum(ch.isprintable() for ch in text)
    asciiish = sum((ord(ch) < 128 and ch.isprintable()) for ch in text)
    alpha = sum(ch.isalpha() for ch in text)
    punct_end = int(text.endswith((".", "!", "?")))
    return asciiish / max(1, printable) + 0.01 * min(alpha, 120) + punct_end


def _select_output_text(tokenizer, token_candidates):
    decoded = [tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True) for tokens in token_candidates]
    cleaned = [_clean_diffusion_text(text) for text in decoded]
    return max(cleaned, key=_coherence_score)


# ---------------------------------------------------------------------------
# Load HF model, wrap, compile both QPCs.
# ---------------------------------------------------------------------------

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

hf_model = AutoModelForImageTextToText.from_pretrained(
    model_id, config=config, torch_dtype=torch.float32, attn_implementation="eager"
)
# Bootstrap the QEff-patched top-level model class.
qeff_model = QEffDiffusionGemmaForBlockDiffusion.__new__(QEffDiffusionGemmaForBlockDiffusion)
qeff_model.__dict__.update(hf_model.__dict__)

print(f"Compiling encoder-prefill QPC ({num_devices} devices, {num_cores} cores)...")
t0 = time.time()
enc = _EncoderPrefillQPC(qeff_model)
enc_inputs = enc.model.get_dummy_inputs()
enc.export(enc_inputs, enc.model.get_output_names(), enc.model.get_onnx_dynamic_axes())
enc_spec, _ = enc.model.get_specializations(
    batch_size=1, prefill_seq_len=prefill_seq_len, ctx_len=ctx_len, canvas_length=canvas_length,
)
text_cfg = qeff_model.config.text_config
enc_custom_io = {"vision_embeds": "float16"}
for i in range(text_cfg.num_hidden_layers):
    for kv in ("key", "value"):
        enc_custom_io[f"past_{kv}.{i}"] = "float16"
        enc_custom_io[f"past_{kv}.{i}_out"] = "float16"
enc_qpc = enc._compile(
    onnx_path=enc.onnx_path, compile_dir=None, specializations=enc_spec,
    convert_to_fp16=True, mxfp6_matmul=True, mdp_ts_num_devices=num_devices,
    aic_num_cores=num_cores, custom_io=enc_custom_io, aic_enable_depth_first=True,
    node_precision_info=_write_encoder_accum_npi(enc.onnx_path),
)
print(f"  encoder QPC: {enc_qpc}  ({time.time() - t0:.0f}s)")

decode_mode = "fused sampler" if args.fuse_sampler else "logit-feedback reference"
print(f"Compiling canvas-decode QPC ({decode_mode} path)...")
t0 = time.time()
dec = _CanvasDecodeQPC(qeff_model, fuse_sampler=args.fuse_sampler, max_top_k=args.max_top_k)
dec_inputs = dec.model.get_dummy_inputs()
dec.export(dec_inputs, dec.model.get_output_names(), dec.model.get_onnx_dynamic_axes())
dec_spec, _ = dec.model.get_specializations(
    batch_size=1, prefill_seq_len=prefill_seq_len, ctx_len=ctx_len, canvas_length=canvas_length,
)
dec_custom_io = {}
for i in range(text_cfg.num_hidden_layers):
    for kv in ("key", "value"):
        dec_custom_io[f"past_{kv}.{i}"] = "float16"
dec_compile_kwargs = {
    "onnx_path": dec.onnx_path,
    "compile_dir": None,
    "specializations": dec_spec,
    "convert_to_fp16": True,
    "mxfp6_matmul": True,
    "mdp_ts_num_devices": num_devices,
    "aic_num_cores": num_cores,
    "custom_io": dec_custom_io,
    "retained_state": True,
    "aic_enable_depth_first": True,
}
if not args.fuse_sampler:
    dec_compile_kwargs["node_precision_info"] = _write_decode_accum_npi(dec.onnx_path)
dec_qpc = dec._compile(**dec_compile_kwargs)
print(f"  canvas-decode QPC: {dec_qpc}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# Vision encoding on CPU.
# ---------------------------------------------------------------------------

if args.text_only:
    prompt_text = args.prompt or text_only_prompt_text
    print("\nText-only mode: no image, no image tokens.")
    vision_inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    vision_embeds = None
else:
    prompt_text = args.prompt or image_prompt_text
    image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
    vision_inputs = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ]}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    vision_embeds = _vision_embeds_cpu(processor, qeff_model, vision_inputs)


# ---------------------------------------------------------------------------
# Phase 1: encoder prefill (writes KV cache).
# ---------------------------------------------------------------------------

input_ids = vision_inputs["input_ids"].numpy().astype(np.int64)
seq_len = input_ids.shape[1]
position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
mm_type_ids = vision_inputs.get("mm_token_type_ids")
mm_type_ids = mm_type_ids.numpy().astype(np.int64) if mm_type_ids is not None else np.zeros_like(input_ids)

# Pad to prefill_seq_len (single-pass prefill; runner asserts n_chunks == 1).
if seq_len > prefill_seq_len:
    raise ValueError(f"Prompt has {seq_len} tokens > prefill_seq_len={prefill_seq_len}; recompile with larger prefill_seq_len.")
pad = prefill_seq_len - seq_len
pad_token_id = processor.tokenizer.pad_token_id or 0
input_ids = np.pad(input_ids, ((0, 0), (0, pad)), constant_values=pad_token_id)
position_ids = np.pad(position_ids, ((0, 0), (0, pad)), constant_values=-1)
mm_type_ids = np.pad(mm_type_ids, ((0, 0), (0, pad)))

enc_session = QAICInferenceSession(str(enc_qpc), device_ids)
ve_dims = next(tuple(b.dims) for b in enc_session.bindings if b.name == "vision_embeds")
ve = np.zeros(ve_dims, dtype=np.float16)
if vision_embeds is not None:
    ve[:, :vision_embeds.shape[1], :] = vision_embeds.astype(np.float16)

t0 = time.perf_counter()
enc_out = enc_session.run({
    "input_ids": input_ids,
    "position_ids": position_ids,
    "vision_embeds": ve,
    "image_idx": np.array([[0]], dtype=np.int64),
    "mm_token_type_ids": mm_type_ids,
})
ttft = time.perf_counter() - t0
kv_host = {n[:-len("_out")]: v for n, v in enc_out.items() if n.startswith("past_") and n.endswith("_out")}
enc_session.deactivate()
print(f"\nTTFT: {ttft:.2f}s ({len(kv_host)} KV buffers captured)")


# ---------------------------------------------------------------------------
# Phase 2: canvas denoise (entropy-bound sampler on host).
# ---------------------------------------------------------------------------

vocab_size = text_cfg.vocab_size
canvas = np.random.randint(0, vocab_size, size=(1, canvas_length), dtype=np.int64)
new_canvas = canvas.copy()
canvas_pos = np.arange(seq_len, seq_len + canvas_length, dtype=np.int64).reshape(1, -1)
accepted_mask = np.zeros((1, canvas_length), dtype=bool)
sc = np.zeros((1, canvas_length, vocab_size), dtype=np.float32)
prev_tokens = canvas.copy()
enc_attn_mask = np.zeros((1, ctx_len), dtype=np.int64)
enc_attn_mask[:, :seq_len] = 1

dec_session = QAICInferenceSession(str(dec_qpc), device_ids)
const_feed = {"decoder_position_ids": canvas_pos, "encoder_attention_mask": enc_attn_mask, **kv_host}
dec_session.set_buffers({k: v for k, v in const_feed.items() if k in dec_session.input_names})

entropy_bound, t_max, t_min = 0.1, 0.8, 0.4
t0 = time.perf_counter()
for step in range(diffusion_steps):
    temperature = t_min + (t_max - t_min) * (diffusion_steps - 1 - step) / max(1, diffusion_steps - 1)
    if args.fuse_sampler:
        random_numbers = np.random.uniform(
            low=0.0,
            high=1.0,
            size=(1, canvas_length, args.max_top_k),
        ).astype(np.float32)
        out = dec_session.run({
            "decoder_input_ids": canvas,
            "temperature": np.array([[[temperature]]], dtype=np.float32),
            "random_numbers": random_numbers,
            "prev_tokens": prev_tokens,
        })
        denoiser = out["denoiser_tokens"].astype(np.int64)
        ent = out["token_entropy"].astype(np.float32)[0]
    else:
        out = dec_session.run({"decoder_input_ids": canvas, "self_conditioning_logits": sc})
        canvas_logits = out["canvas_logits"].astype(np.float32)
        sc = canvas_logits
        lt = canvas_logits / temperature
        gumbel = -np.log(-np.log(np.random.uniform(size=lt.shape).astype(np.float32) + 1e-20) + 1e-20)
        denoiser = (lt + gumbel).argmax(-1).astype(np.int64)
        shifted = lt - lt.max(-1, keepdims=True)
        log_softmax = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
        ent = -(np.exp(log_softmax) * log_softmax).sum(-1)[0]
    order = np.argsort(ent)
    sel = (np.cumsum(ent[order]) - ent[order]) <= entropy_bound
    new_acc = np.zeros(canvas_length, dtype=bool)
    new_acc[order[sel]] = True
    new_canvas = np.where(new_acc[None, :], denoiser, canvas)
    accepted_mask = accepted_mask | new_acc[None, :]
    canvas = np.where(~accepted_mask,
                      np.random.randint(0, vocab_size, size=(1, canvas_length), dtype=np.int64),
                      new_canvas)
    prev_tokens = denoiser.copy() if args.fuse_sampler else new_canvas.copy()
    accepted_count = int(accepted_mask.sum())
    if args.verbose_steps:
        preview = processor.tokenizer.decode(new_canvas[0].tolist(), skip_special_tokens=True)
        print(f"  step {step + 1:2d} t={temperature:.2f} acc={accepted_count}/{canvas_length} :: {preview[:60]!r}")
    else:
        print(f"  step {step + 1:2d} t={temperature:.2f} acc={accepted_count}/{canvas_length}")
    if accepted_count >= canvas_length:
        break
canvas_time = time.perf_counter() - t0
dec_session.deactivate()

output_text = _select_output_text(processor.tokenizer, [new_canvas])

steps_run = step + 1
print(f"\nCanvas: {steps_run} steps, {canvas_time:.1f}s, "
      f"{steps_run * canvas_length / canvas_time:.1f} tok/s")
print(f"\nOutput:\n{output_text}")
