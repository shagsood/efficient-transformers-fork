# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""DiffusionGemma single-QPC / single-specialization example on Cloud AI 100.

The unified wrapper packs encoder-prefill and canvas-decode into one QPC with
one fixed-shape specialization:

  input_ids=[1,prefill_seq_len], decoder_input_ids=[1,canvas_length]

The runtime tensor ``is_encode`` selects behavior:

  * is_encode=1: encoder-prefill writes retained KV.
  * is_encode=0: canvas-decode reads retained KV and denoises the canvas.

The encoder runs once. Decoder outputs are not fed back to the encoder; only the
canvas/self-conditioning state changes across denoising iterations.
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

parser = argparse.ArgumentParser(description="Run DiffusionGemma single-QPC single-specialization inference.")
parser.add_argument("--text-only", action="store_true", help="Run a prompt with no image tokens.")
parser.add_argument("--prompt", default=None, help="Override the default image or text-only prompt.")
parser.add_argument("--seed", type=int, default=1234, help="Seed for the diffusion sampler. Use --seed -1 for unseeded.")
parser.add_argument("--verbose-steps", action="store_true", help="Decode and print a preview after every diffusion step.")
args = parser.parse_args()
if args.seed >= 0:
    np.random.seed(args.seed)


# ---------------------------------------------------------------------------
# Compile helper for the unified single-QPC wrapper.
# ---------------------------------------------------------------------------


class _UnifiedQPC(QEffCausalLMForTextImageToTextModel):
    def __init__(self, model):
        QEFFBaseModel.__init__(self, model)
        self.model = model.get_qeff_unified_wrapper()
        self.model.qaic_config = None
        self.hash_params["qeff_auto_class"] = self.__class__.__name__
        self.continuous_batching = False

    @property
    def get_model_config(self):
        return self.model.model.config.__dict__

    def export(self, inputs, output_names, dynamic_axes, **kwargs):
        return self._export(inputs, output_names=output_names, dynamic_axes=dynamic_axes)


FP32_ACCUM_OPS = {"CustomRMSNorm", "Clip", "Softmax", "Add", "Sub", "Mul", "Div", "Tanh", "Pow", "ReduceMean"}


def _write_unified_accum_npi(onnx_path):
    graph = onnx.load(onnx_path, load_external_data=False).graph
    producers = {out_name: node for node in graph.node for out_name in node.output}
    keep_nodes = []

    for node in graph.node:
        if node.op_type in FP32_ACCUM_OPS:
            keep_nodes.append(node)
        if "/decoder/self_conditioning/" in node.name or node.name.endswith("/decoder/norm/CustomRMSNorm"):
            keep_nodes.append(node)

    seen_names = set()

    def backtrace(tensor_name, depth=0):
        if tensor_name in seen_names or depth > 8:
            return
        seen_names.add(tensor_name)
        node = producers.get(tensor_name)
        if node is None:
            return
        keep_nodes.append(node)
        for input_name in node.input:
            if input_name in producers:
                backtrace(input_name, depth + 1)

    if graph.output:
        backtrace(graph.output[0].name)

    initializer_names = {init.name for init in graph.initializer}

    def depends_on_initializer(tensor_name, depth=0):
        if tensor_name in initializer_names:
            return True
        if depth > 4:
            return False
        producer = producers.get(tensor_name)
        if producer is None:
            return False
        return any(depends_on_initializer(input_name, depth + 1) for input_name in producer.input)

    vocab_matmul_outputs = {"/decoder/MatMul_output_0", "/lm_head/MatMul_output_0"}
    tensors, seen_tensors = [], set()
    for node in keep_nodes:
        for out_name in node.output:
            if not out_name or out_name in seen_tensors or out_name in vocab_matmul_outputs:
                continue
            if node.op_type == "MatMul" and any(depends_on_initializer(inp) for inp in node.input):
                continue
            if node.op_type in {"Cast", "Transpose", "Reshape", "DequantizeLinear", "QuantizeLinear"} and depends_on_initializer(out_name):
                continue
            seen_tensors.add(out_name)
            tensors.append(out_name)

    path = os.path.join(os.path.dirname(onnx_path), "npi_fp32_unified_accum.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("FP32NodeInstanceNames: [")
        handle.write(", ".join(f"'{name}'" for name in sorted(tensors)))
        handle.write("]\n")
    print(f"  unified fp32 accumulation island: {len(tensors)} tensors -> {path}")
    return path


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


# ---------------------------------------------------------------------------
# Load HF model, wrap, compile one unified QPC (one specialization).
# ---------------------------------------------------------------------------

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

hf_model = AutoModelForImageTextToText.from_pretrained(
    model_id, config=config, torch_dtype=torch.float32, attn_implementation="eager"
)
qeff_model = QEffDiffusionGemmaForBlockDiffusion.__new__(QEffDiffusionGemmaForBlockDiffusion)
qeff_model.__dict__.update(hf_model.__dict__)

print(f"Compiling unified single-QPC ({num_devices} devices, {num_cores} cores)...")
t0 = time.time()
uni = _UnifiedQPC(qeff_model)
uni.export(uni.model.get_dummy_inputs(), uni.model.get_output_names(), uni.model.get_onnx_dynamic_axes())
uni_spec, _ = uni.model.get_specializations(
    batch_size=1, prefill_seq_len=prefill_seq_len, ctx_len=ctx_len, canvas_length=canvas_length,
)
text_cfg = qeff_model.config.text_config
uni_custom_io = {"vision_embeds": "float16"}
for i in range(text_cfg.num_hidden_layers):
    for kv in ("key", "value"):
        uni_custom_io[f"past_{kv}.{i}"] = "float16"
        uni_custom_io[f"past_{kv}.{i}_RetainedState"] = "float16"
uni_qpc = uni._compile(
    onnx_path=uni.onnx_path, compile_dir=None, specializations=uni_spec,
    convert_to_fp16=True, mxfp6_matmul=True, mdp_ts_num_devices=num_devices,
    aic_num_cores=num_cores, custom_io=uni_custom_io, retained_state=True,
    aic_enable_depth_first=True, node_precision_info=_write_unified_accum_npi(uni.onnx_path),
)
print(f"  unified QPC: {uni_qpc}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# CPU vision encoding.
# ---------------------------------------------------------------------------

if args.text_only:
    prompt_text = args.prompt or text_only_prompt_text
    print("\nText-only mode: no image, no image tokens.")
    vision_inputs = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True,
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

    with torch.no_grad():
        encoder = hf_model.model.encoder
        pixel_values = vision_inputs["pixel_values"]
        image_position_ids = vision_inputs["image_position_ids"]
        padding_positions = (image_position_ids == -1).all(dim=-1)
        h = encoder.vision_tower.patch_embedder(pixel_values, image_position_ids, padding_positions)
        attn_mask = ((~(~padding_positions)).unsqueeze(1).unsqueeze(2).to(h.dtype) * torch.finfo(h.dtype).min)
        attn_mask = attn_mask.expand(-1, 1, h.shape[1], -1)
        pos_emb = encoder.vision_tower.encoder.rotary_emb(h, image_position_ids)
        for layer in encoder.vision_tower.encoder.layers[:encoder.vision_tower.encoder.config.num_hidden_layers]:
            h = layer(h, attention_mask=attn_mask, position_embeddings=pos_emb, position_ids=image_position_ids)
        out_len = encoder.vision_tower.config.default_output_length
        h, _ = encoder.vision_tower.pooler(
            hidden_states=h, pixel_position_ids=image_position_ids,
            padding_positions=padding_positions, output_length=out_len,
        )
        if encoder.vision_tower.config.standardize:
            h = (h - encoder.vision_tower.std_bias) * encoder.vision_tower.std_scale
        vision_embeds = encoder.embed_vision(inputs_embeds=h).clamp(-60000.0, 60000.0)
        mm_tokens = hf_model.config.mm_tokens_per_image
        vision_embeds = vision_embeds[:, :mm_tokens, :].float().numpy()


# ---------------------------------------------------------------------------
# Prepare prompt inputs (padded to prefill_seq_len for single-pass prefill).
# ---------------------------------------------------------------------------

input_ids = vision_inputs["input_ids"].numpy().astype(np.int64)
seq_len = input_ids.shape[1]
position_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
mm_type_ids = vision_inputs.get("mm_token_type_ids")
mm_type_ids = mm_type_ids.numpy().astype(np.int64) if mm_type_ids is not None else np.zeros_like(input_ids)

if seq_len > prefill_seq_len:
    raise ValueError(
        f"Prompt has {seq_len} tokens > prefill_seq_len={prefill_seq_len}; "
        f"recompile with larger prefill_seq_len."
    )
pad = prefill_seq_len - seq_len
pad_token_id = processor.tokenizer.pad_token_id or 0
input_ids = np.pad(input_ids, ((0, 0), (0, pad)), constant_values=pad_token_id)
position_ids = np.pad(position_ids, ((0, 0), (0, pad)), constant_values=-1)
mm_type_ids = np.pad(mm_type_ids, ((0, 0), (0, pad)))


# ---------------------------------------------------------------------------
# Open one QAICInferenceSession. Prefill call (canvas_len=1) writes KV into
# retained-state buffers; Decode calls (canvas_len=canvas_length) read them.
# ---------------------------------------------------------------------------

session = QAICInferenceSession(str(uni_qpc), device_ids)
ve_dims = next(tuple(b.dims) for b in session.bindings if b.name == "vision_embeds")
ve = np.zeros(ve_dims, dtype=np.float16)
if vision_embeds is not None:
    ve[:, :vision_embeds.shape[1], :] = vision_embeds.astype(np.float16)
vocab_size = text_cfg.vocab_size
canvas_pos = np.arange(seq_len, seq_len + canvas_length, dtype=np.int64).reshape(1, -1)
enc_attn_mask = np.zeros((1, ctx_len), dtype=np.int64)
enc_attn_mask[:, :seq_len] = 1

t0 = time.perf_counter()
prefill_out = session.run({
    "input_ids": input_ids,
    "position_ids": position_ids,
    "vision_embeds": ve,
    "image_idx": np.array([[0]], dtype=np.int64),
    "mm_token_type_ids": mm_type_ids,
    "decoder_input_ids": np.zeros((1, canvas_length), dtype=np.int64),
    "decoder_position_ids": canvas_pos,
    "self_conditioning_logits": np.zeros((1, canvas_length, vocab_size), dtype=np.float32),
    "encoder_attention_mask": enc_attn_mask,
    "is_encode": np.array([1], dtype=np.int64),
})
ttft = time.perf_counter() - t0
retained_buffers = [n for n in session.input_names + session.output_names if n.startswith("past_")]
session.skip_buffers(retained_buffers)
print(f"\nTTFT: {ttft:.2f}s ({len([n for n in prefill_out if n.startswith('past_')])} KV buffers retained)")


# ---------------------------------------------------------------------------
# Canvas denoise loop.
# ---------------------------------------------------------------------------

canvas = np.random.randint(0, vocab_size, size=(1, canvas_length), dtype=np.int64)
new_canvas = canvas.copy()
canvas_pos = np.arange(seq_len, seq_len + canvas_length, dtype=np.int64).reshape(1, -1)
accepted_mask = np.zeros((1, canvas_length), dtype=bool)
sc = np.zeros((1, canvas_length, vocab_size), dtype=np.float32)

entropy_bound, t_max, t_min = 0.1, 0.8, 0.4
t0 = time.perf_counter()
for step in range(diffusion_steps):
    temperature = t_min + (t_max - t_min) * (diffusion_steps - 1 - step) / max(1, diffusion_steps - 1)
    feed = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "vision_embeds": ve,
        "image_idx": np.array([[0]], dtype=np.int64),
        "mm_token_type_ids": mm_type_ids,
        "decoder_input_ids": canvas,
        "decoder_position_ids": canvas_pos,
        "self_conditioning_logits": sc,
        "encoder_attention_mask": enc_attn_mask,
        "is_encode": np.array([0], dtype=np.int64),
    }
    feed = {k: v for k, v in feed.items() if k in session.input_names}
    out = session.run(feed)
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
    accepted_count = int(accepted_mask.sum())
    if args.verbose_steps:
        preview = processor.tokenizer.decode(new_canvas[0].tolist(), skip_special_tokens=True)
        print(f"  step {step + 1:2d} t={temperature:.2f} acc={accepted_count}/{canvas_length} :: {preview[:60]!r}")
    else:
        print(f"  step {step + 1:2d} t={temperature:.2f} acc={accepted_count}/{canvas_length}")
    if accepted_count >= canvas_length:
        break
canvas_time = time.perf_counter() - t0
session.deactivate()

output_text = _clean_diffusion_text(processor.tokenizer.decode(new_canvas[0].tolist(), skip_special_tokens=True))

steps_run = step + 1
print(f"\nCanvas: {steps_run} steps, {canvas_time:.1f}s, "
      f"{steps_run * canvas_length / canvas_time:.1f} tok/s")
print(f"\nOutput:\n{output_text}")
