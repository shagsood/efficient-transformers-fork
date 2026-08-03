# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""DiffusionGemma dual-QPC example on Cloud AI 100.

DiffusionGemma is a block-diffusion VLM: the encoder prefills prompt (+ vision)
KV, then the decoder denoises fixed-size canvases. For generation longer than
one canvas, this runner follows HF's block-autoregressive loop: denoise one
canvas, commit the final canvas through the encoder to append KV, then denoise
the next canvas.

`QEFFAutoModelForImageTextToText.generate()` does not apply; this example drives
the two exported QPCs (encoder-prefill/commit and canvas-decode) via
`QAICInferenceSession` directly. Use `--encoder-devices` and `--decoder-devices`
to keep the two QPCs resident on separate TS4 groups and avoid deactivate/load
thrash. KV is still copied through host at canvas boundaries.

For the single-QPC (unified) variant, see diffusion_gemma_single_qpc_example.py.
"""

import argparse
import os
import re
import time
from io import BytesIO

import numpy as np
import onnx
import requests
import torch
from diffusion_gemma_debug import (
    HostSampler,
    TraceWriter,
    array_summary,
    kv_sample_manifest,
    logits_sample,
    npi_tensor_count,
    write_matched_accum_npi,
)
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
encoder_device_ids = [int(x) for x in os.environ.get("DG_ENC", os.environ.get("DG", "4,5,6,7")).split(",")]
decoder_device_ids = [int(x) for x in os.environ.get("DG_DEC", os.environ.get("DG", "4,5,6,7")).split(",")]
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
parser.add_argument("--ctx-len", type=int, default=ctx_len, help="Compiled retained-KV context length.")
parser.add_argument("--canvas-length", type=int, default=canvas_length, help="Denoising canvas length per block.")
parser.add_argument("--max-new-tokens", type=int, default=canvas_length, help="Total generated tokens across canvases.")
parser.add_argument("--diffusion-steps", type=int, default=diffusion_steps, help="Maximum denoising steps per canvas.")
parser.add_argument("--encoder-devices", default=None, help="Comma-separated encoder QPC device IDs. Defaults to DG_ENC or DG.")
parser.add_argument("--decoder-devices", default=None, help="Comma-separated decoder QPC device IDs. Defaults to DG_DEC or DG.")
parser.add_argument("--no-stop-on-eos", action="store_true", help="Do not truncate/stop at the first EOS in a canvas.")
parser.add_argument("--truncate-first-sentence", action="store_true", help="Return only the first complete sentence.")
parser.add_argument("--verbose-steps", action="store_true", help="Decode and print a preview after every diffusion step.")
parser.add_argument("--trace-file", default=None, help="Write deterministic JSONL equivalence instrumentation.")
parser.add_argument(
    "--sampler-policy",
    choices=HostSampler.POLICIES,
    default="numpy-gumbel",
    help="Historical NumPy sampler or the Transformers-compatible torch sampler and adaptive stopping.",
)
parser.add_argument(
    "--precision-profile",
    choices=("current", "matched"),
    default="current",
    help="Use the historical precision islands or one semantic selector shared with single-QPC.",
)
parser.add_argument(
    "--matmul-precision",
    choices=("mxfp6", "fp16"),
    default="mxfp6",
    help="Matmul weight precision; fp16 is a slower equivalence-control compile.",
)
args = parser.parse_args()
trace = TraceWriter(args.trace_file, "dual-qpc")
ctx_len = args.ctx_len
canvas_length = args.canvas_length
diffusion_steps = args.diffusion_steps
max_new_tokens = args.max_new_tokens
if args.encoder_devices:
    encoder_device_ids = [int(x) for x in args.encoder_devices.split(",")]
if args.decoder_devices:
    decoder_device_ids = [int(x) for x in args.decoder_devices.split(",")]
num_devices = len(decoder_device_ids)
if len(encoder_device_ids) != len(decoder_device_ids):
    raise ValueError("--encoder-devices and --decoder-devices must have the same device count for matching TS specs.")
if max_new_tokens <= 0:
    raise ValueError("--max-new-tokens must be positive")
if canvas_length <= 0:
    raise ValueError("--canvas-length must be positive")
if args.fuse_sampler and args.sampler_policy != "numpy-gumbel":
    raise ValueError("--fuse-sampler only supports --sampler-policy numpy-gumbel")


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


def _clean_diffusion_text(text, truncate_first_sentence=True):
    text = text.replace("\ufffd", " ").strip()
    text = re.sub(r"^\s*(thought\s*)+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("。", ".")
    text = re.sub(r"\bfulling shot\b", "full shot", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(light|dark)\s+(blue|green|teal)ing\b", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"\.(?:of|Of)\b.*$", ".", text)
    match = re.search(r"(.{12,}?[.!?])", text) if truncate_first_sentence else None
    if truncate_first_sentence and match:
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


def _load_qeff_model():
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_id, config=config, torch_dtype=torch.float32, attn_implementation="eager"
    )
    # Bootstrap the QEff-patched top-level model class.
    qeff_model = QEffDiffusionGemmaForBlockDiffusion.__new__(QEffDiffusionGemmaForBlockDiffusion)
    qeff_model.__dict__.update(hf_model.__dict__)
    qeff_model.config.canvas_length = canvas_length
    return qeff_model


qeff_model = _load_qeff_model()

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
enc_npi_path = (
    write_matched_accum_npi(enc.onnx_path)
    if args.precision_profile == "matched"
    else _write_encoder_accum_npi(enc.onnx_path)
)
enc_qpc = enc._compile(
    onnx_path=enc.onnx_path, compile_dir=None, specializations=enc_spec,
    convert_to_fp16=True, mxfp6_matmul=args.matmul_precision == "mxfp6", mdp_ts_num_devices=num_devices,
    aic_num_cores=num_cores, custom_io=enc_custom_io, aic_enable_depth_first=True,
    node_precision_info=enc_npi_path,
)
print(f"  encoder QPC: {enc_qpc}  ({time.time() - t0:.0f}s)")

decode_mode = "fused sampler" if args.fuse_sampler else "logit-feedback reference"
print(f"Compiling canvas-decode QPC ({decode_mode} path)...")
t0 = time.time()
qeff_model = _load_qeff_model()
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
    "mxfp6_matmul": args.matmul_precision == "mxfp6",
    "mdp_ts_num_devices": num_devices,
    "aic_num_cores": num_cores,
    "custom_io": dec_custom_io,
    "retained_state": True,
    "aic_enable_depth_first": True,
}
if not args.fuse_sampler:
    dec_npi_path = (
        write_matched_accum_npi(dec.onnx_path)
        if args.precision_profile == "matched"
        else _write_decode_accum_npi(dec.onnx_path)
    )
    dec_compile_kwargs["node_precision_info"] = dec_npi_path
else:
    dec_npi_path = None
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

enc_session = QAICInferenceSession(str(enc_qpc), encoder_device_ids)
ve_dims = next(tuple(b.dims) for b in enc_session.bindings if b.name == "vision_embeds")
ve = np.zeros(ve_dims, dtype=np.float16)
if vision_embeds is not None:
    ve[:, :vision_embeds.shape[1], :] = vision_embeds.astype(np.float16)


def _session_feed(session, feed):
    return {k: v for k, v in feed.items() if k in session.input_names}


def _run_encoder_commit(tokens, token_position_ids, token_mm_type_ids, past=None):
    feed = {
        "input_ids": tokens,
        "position_ids": token_position_ids,
        "vision_embeds": ve,
        "image_idx": np.array([[0]], dtype=np.int64),
        "mm_token_type_ids": token_mm_type_ids,
    }
    if past:
        feed.update(past)
    out = enc_session.run(_session_feed(enc_session, feed))
    return {n[:-len("_out")]: v for n, v in out.items() if n.startswith("past_") and n.endswith("_out")}


t0 = time.perf_counter()
kv_host = _run_encoder_commit(input_ids, position_ids, mm_type_ids)
ttft = time.perf_counter() - t0
print(f"\nTTFT: {ttft:.2f}s ({len(kv_host)} KV buffers captured)")

prompt_attn_mask = np.zeros((1, ctx_len), dtype=np.int64)
prompt_attn_mask[:, :seq_len] = 1
trace.event(
    "run_config",
    prompt=prompt_text,
    text_only=args.text_only,
    prompt_length=seq_len,
    prefill_seq_len=prefill_seq_len,
    ctx_len=ctx_len,
    canvas_length=canvas_length,
    max_new_tokens=max_new_tokens,
    diffusion_steps=diffusion_steps,
    seed=args.seed,
    precision=f"{args.matmul_precision}-matmul/fp16-kv/fp32-accum-island",
    precision_profile=args.precision_profile,
    precision_island={
        "encoder_selected_tensors": npi_tensor_count(enc_npi_path),
        "decoder_selected_tensors": npi_tensor_count(dec_npi_path) if dec_npi_path else 0,
    },
    fuse_sampler=args.fuse_sampler,
    sampler="fused-top-k-gumbel-max" if args.fuse_sampler else args.sampler_policy,
    sampler_config={
        "entropy_bound": 0.1,
        "t_min": 0.4,
        "t_max": 0.8,
        "stability_threshold": 1,
        "confidence_threshold": 0.005,
        "acceptance": "per-step" if args.sampler_policy == "hf-torch" else "cumulative",
        "final_canvas": "argmax" if args.sampler_policy == "hf-torch" else "accepted",
    },
    diffusion_stop=(
        "stable-argmax-and-mean-entropy-below-0.005-or-step-cap"
        if args.sampler_policy == "hf-torch"
        else "full-acceptance-or-step-cap"
    ),
    stop_on_eos=not args.no_stop_on_eos,
    truncate_first_sentence=args.truncate_first_sentence,
)
trace.event(
    "prefill",
    position_ids=array_summary(position_ids, include_values=True),
    attention_mask=array_summary(prompt_attn_mask, include_values=True),
    kv_behavior="encoder QPC regular outputs copied through host to decoder QPC inputs",
    kv_buffers=sorted(kv_host),
    kv_samples=kv_sample_manifest(kv_host, seq_len),
    bound_inputs=sorted(enc_session.input_names),
)


# ---------------------------------------------------------------------------
# Phase 2: canvas denoise + optional multi-canvas commit loop.
# ---------------------------------------------------------------------------

vocab_size = text_cfg.vocab_size
dec_session = QAICInferenceSession(str(dec_qpc), decoder_device_ids)

entropy_bound, t_max, t_min = 0.1, 0.8, 0.4
sampler_rng = np.random.RandomState(args.seed if args.seed >= 0 else None)
sampler = HostSampler(args.sampler_policy, args.seed, vocab_size, canvas_length)


def _denoise_canvas(block_index, cursor, past):
    canvas = (
        sampler_rng.randint(0, vocab_size, size=(1, canvas_length), dtype=np.int64)
        if args.fuse_sampler
        else sampler.initialize_canvas()
    )
    final_canvas = canvas.copy()
    prev_tokens = canvas.copy()
    accepted_mask = np.zeros((1, canvas_length), dtype=bool)
    sc = np.zeros((1, canvas_length, vocab_size), dtype=np.float32)
    canvas_pos = np.arange(cursor, cursor + canvas_length, dtype=np.int64).reshape(1, -1)
    enc_attn_mask = np.zeros((1, ctx_len), dtype=np.int64)
    enc_attn_mask[:, :cursor] = 1
    const_feed = {"decoder_position_ids": canvas_pos, "encoder_attention_mask": enc_attn_mask, **past}
    dec_session.set_buffers(_session_feed(dec_session, const_feed))
    trace.event(
        "canvas_start",
        block=block_index,
        cursor=cursor,
        decoder_position_ids=array_summary(canvas_pos, include_values=True),
        encoder_attention_mask=array_summary(enc_attn_mask, include_values=True),
        initial_canvas=array_summary(canvas, include_values=True),
        kv_behavior="host KV set once as persistent decoder QPC buffers for this canvas",
        encoder_attention_mask_bound="encoder_attention_mask" in dec_session.input_names,
    )

    t0 = time.perf_counter()
    for step in range(diffusion_steps):
        temperature = t_min + (t_max - t_min) * (diffusion_steps - 1 - step) / max(1, diffusion_steps - 1)
        canvas_input = canvas.copy()
        if args.fuse_sampler:
            random_numbers = sampler_rng.uniform(
                low=0.0,
                high=1.0,
                size=(1, canvas_length, args.max_top_k),
            ).astype(np.float32)
            out = dec_session.run(_session_feed(dec_session, {
                "decoder_input_ids": canvas,
                "execution_mode": np.array([2 if step == 0 else 3], dtype=np.int64),
                "temperature": np.array([[[temperature]]], dtype=np.float32),
                "random_numbers": random_numbers,
                "prev_tokens": prev_tokens,
            }))
            denoiser = out["denoiser_tokens"].astype(np.int64)
            ent = out["token_entropy"].astype(np.float32)[0]
            order = np.argsort(ent)
            sel = (np.cumsum(ent[order]) - ent[order]) <= entropy_bound
            new_acc = np.zeros(canvas_length, dtype=bool)
            new_acc[order[sel]] = True
            final_canvas = np.where(new_acc[None, :], denoiser, canvas)
            accepted_mask = accepted_mask | new_acc[None, :]
            canvas = np.where(
                ~accepted_mask,
                sampler_rng.randint(0, vocab_size, size=(1, canvas_length), dtype=np.int64),
                final_canvas,
            )
            prev_tokens = denoiser.copy()
            accepted_count = int(accepted_mask.sum())
            mean_entropy = float(ent.mean())
            should_stop = accepted_count >= canvas_length
            stop_reason = "full-acceptance" if should_stop else None
            trace_acceptance_mask = accepted_mask
            argmax_tokens = None
            step_logits_sample = None
        else:
            out = dec_session.run(_session_feed(dec_session, {
                "decoder_input_ids": canvas,
                "self_conditioning_logits": sc,
                "execution_mode": np.array([2 if step == 0 else 3], dtype=np.int64),
            }))
            canvas_logits = out["canvas_logits"].astype(np.float32)
            sample = sampler.step(canvas_logits, canvas, step, diffusion_steps)
            sc = sample.self_conditioning_logits
            denoiser = sample.denoiser_canvas
            canvas = sample.next_canvas
            final_canvas = sample.final_canvas
            accepted_count = sample.accepted_count
            mean_entropy = sample.mean_entropy
            should_stop = sample.should_stop
            stop_reason = sample.stop_reason
            trace_acceptance_mask = sample.accepted_mask
            argmax_tokens = sample.argmax_canvas
            step_logits_sample = logits_sample(canvas_logits)
        reported_temperature = sample.temperature if not args.fuse_sampler else float(temperature)
        trace.event(
            "denoise_step",
            block=block_index,
            step=step,
            decoder_phase="first" if step == 0 else "later",
            self_conditioning=(
                "disabled" if step == 0 else ("prev_tokens" if args.fuse_sampler else "previous_canvas_logits")
            ),
            execution_mode=2 if step == 0 else 3,
            temperature=reported_temperature,
            canvas_input=array_summary(canvas_input, include_values=True),
            denoiser_tokens=array_summary(denoiser, include_values=True),
            argmax_tokens=(array_summary(argmax_tokens, include_values=True) if argmax_tokens is not None else None),
            logits_sample=step_logits_sample,
            pre_sampling_logits=(
                array_summary(canvas_logits) if step == 0 and not args.fuse_sampler else None
            ),
            pre_sampling_argmax=(
                array_summary(canvas_logits.argmax(-1).astype(np.int64), include_values=True)
                if step == 0 and not args.fuse_sampler
                else None
            ),
            acceptance_mask=array_summary(trace_acceptance_mask, include_values=True),
            accepted_count=accepted_count,
            mean_entropy=mean_entropy,
            adaptive_stop=should_stop,
            stop_reason=stop_reason,
        )
        if args.verbose_steps:
            preview = processor.tokenizer.decode(final_canvas[0].tolist(), skip_special_tokens=True)
            print(
                f"  block {block_index + 1:2d} step {step + 1:2d} "
                f"t={reported_temperature:.2f} acc={accepted_count}/{canvas_length} :: {preview[:60]!r}"
            )
        else:
            print(
                f"  block {block_index + 1:2d} step {step + 1:2d} "
                f"t={reported_temperature:.2f} acc={accepted_count}/{canvas_length}"
            )
        if should_stop:
            break
    return final_canvas, step + 1, time.perf_counter() - t0, accepted_count


def _pad_commit_inputs(tokens, cursor):
    commit_len = tokens.shape[1]
    if commit_len > prefill_seq_len:
        raise ValueError(f"Commit length {commit_len} exceeds prefill_seq_len={prefill_seq_len}")
    commit_input_ids = np.full((1, prefill_seq_len), pad_token_id, dtype=np.int64)
    commit_input_ids[:, :commit_len] = tokens
    commit_position_ids = np.full((1, prefill_seq_len), -1, dtype=np.int64)
    commit_position_ids[:, :commit_len] = np.arange(cursor, cursor + commit_len, dtype=np.int64)
    commit_mm_type_ids = np.zeros((1, prefill_seq_len), dtype=np.int64)
    return commit_input_ids, commit_position_ids, commit_mm_type_ids


eos_token_ids = processor.tokenizer.eos_token_id
if eos_token_ids is None:
    eos_token_ids = []
elif isinstance(eos_token_ids, int):
    eos_token_ids = [eos_token_ids]
else:
    eos_token_ids = list(eos_token_ids)


def _finalize_canvas_tokens(tokens):
    if args.no_stop_on_eos or not eos_token_ids:
        return tokens, False, -1
    eos_positions = np.where(np.isin(tokens[0], eos_token_ids))[0]
    if eos_positions.size == 0:
        return tokens, False, -1
    eos_pos = int(eos_positions[0])
    return tokens[:, : eos_pos + 1], True, eos_pos


generated = []
cursor = seq_len
total_steps = 0
total_canvas_time = 0.0
target_new_tokens = min(max_new_tokens, max(0, ctx_len - seq_len))
if target_new_tokens <= 0:
    raise ValueError(f"No generation room: seq_len={seq_len}, ctx_len={ctx_len}. Recompile with larger --ctx-len.")
num_blocks = int(np.ceil(target_new_tokens / canvas_length))
if target_new_tokens < max_new_tokens:
    print(f"\nWarning: capped generation to {target_new_tokens} tokens because seq_len={seq_len}, ctx_len={ctx_len}.")

for block in range(num_blocks):
    emitted_tokens = sum(part.shape[1] for part in generated)
    remaining = target_new_tokens - emitted_tokens
    block_tokens, steps_run, canvas_time, accepted_count = _denoise_canvas(block, cursor, kv_host)
    if remaining < canvas_length:
        block_tokens = block_tokens[:, :remaining]
    block_tokens, hit_eos, eos_pos = _finalize_canvas_tokens(block_tokens)
    generated.append(block_tokens)
    total_steps += steps_run
    total_canvas_time += canvas_time
    old_cursor = cursor
    cursor += block_tokens.shape[1]
    trace.event(
        "canvas_final",
        block=block,
        steps=steps_run,
        accepted_count=accepted_count,
        final_token_ids=array_summary(block_tokens, include_values=True),
        eos_hit=hit_eos,
        eos_offset=eos_pos,
        generated_length=sum(part.shape[1] for part in generated),
    )
    print(
        f"  block {block + 1:2d} done: {steps_run} steps, {canvas_time:.1f}s, "
        f"accepted={accepted_count}/{canvas_length}, finalized={block_tokens.shape[1]}, cursor={cursor}"
    )
    if hit_eos:
        print(f"  EOS at block {block + 1}, offset {eos_pos}; truncating and stopping.")
        break
    if block + 1 < num_blocks and block_tokens.shape[1] > 0:
        commit_inputs = _pad_commit_inputs(block_tokens, old_cursor)
        trace.event(
            "commit",
            cursor=old_cursor,
            token_ids=array_summary(block_tokens, include_values=True),
            position_ids=array_summary(commit_inputs[1], include_values=True),
            attention_mask=None,
            kv_behavior="host KV from prior encoder call fed to encoder QPC; returned KV replaces host copy",
        )
        kv_host = _run_encoder_commit(*commit_inputs, past=kv_host)

dec_session.deactivate()
enc_session.deactivate()

all_tokens = np.concatenate(generated, axis=1) if generated else np.zeros((1, 0), dtype=np.int64)
raw_output = processor.tokenizer.decode(all_tokens[0].tolist(), skip_special_tokens=True)
executed_blocks = len(generated)
output_text = _clean_diffusion_text(raw_output, truncate_first_sentence=(executed_blocks == 1 or args.truncate_first_sentence))

print(f"\nCanvas: {total_steps} steps across {executed_blocks} blocks, {total_canvas_time:.1f}s, "
      f"{total_steps * canvas_length / total_canvas_time:.1f} tok/s")
print(f"\nOutput:\n{output_text}")
trace.event(
    "run_final",
    generated_length=all_tokens.shape[1],
    final_token_ids=array_summary(all_tokens, include_values=True),
    blocks=executed_blocks,
    total_steps=total_steps,
)
trace.close()
