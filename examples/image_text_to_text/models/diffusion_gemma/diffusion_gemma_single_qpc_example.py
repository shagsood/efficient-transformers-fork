# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""DiffusionGemma single-QPC (unified) example on Cloud AI 100.

The unified wrapper packs encoder-prefill and canvas-decode into one QPC with
two specializations (Prefill: seq_len=<prefill>, canvas_len=1; Decode: seq_len=1,
canvas_len=<canvas>). Both branches trace and the compiler is expected to fold
the unused one per specialization. The runner drives one QAICInferenceSession
and host-copies past_*_RetainedState (Prefill output) into past_*.{i} (Decode
input) between diffusion steps, same contract as the dual-QPC path.

Status: this graph currently exceeds the AI 100 per-core VA budget at
TS<=16 (SDK 1.22.x) and TS>=24 exceeds the multi-device connection cap.
See the dual-QPC example (diffusion_gemma_example.py) for a working end-to-end
run. Kept here as a reference for the single-QPC pipeline shape.
"""

import time
from io import BytesIO

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
diffusion_steps = 24
num_cores = 16
num_devices = 16
device_ids = list(range(num_devices))
image_url = (
    "https://huggingface.co/datasets/huggingface/documentation-images"
    "/resolve/main/transformers/tasks/car.jpg"
)
prompt_text = "Describe this image in detail."


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


# ---------------------------------------------------------------------------
# Load HF model, wrap, compile one unified QPC (two specializations).
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
    aic_enable_depth_first=True,
)
print(f"  unified QPC: {uni_qpc}  ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# CPU vision encoding.
# ---------------------------------------------------------------------------

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
    attn_mask = ((~(~padding_positions)).unsqueeze(1).unsqueeze(2).to(h.dtype)
                 * torch.finfo(h.dtype).min)
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
input_ids = np.pad(input_ids, ((0, 0), (0, pad)))
position_ids = np.pad(position_ids, ((0, 0), (0, pad)))
mm_type_ids = np.pad(mm_type_ids, ((0, 0), (0, pad)))


# ---------------------------------------------------------------------------
# Open one QAICInferenceSession. Prefill call (canvas_len=1) writes KV into
# retained-state buffers; Decode calls (canvas_len=canvas_length) read them.
# ---------------------------------------------------------------------------

session = QAICInferenceSession(str(uni_qpc), device_ids)
ve_dims = next(tuple(b.dims) for b in session.bindings if b.name == "vision_embeds")
ve = np.zeros(ve_dims, dtype=np.float16)
ve[:, :vision_embeds.shape[1], :] = vision_embeds.astype(np.float16)
vocab_size = text_cfg.vocab_size

t0 = time.perf_counter()
prefill_out = session.run({
    "input_ids": input_ids,
    "position_ids": position_ids,
    "vision_embeds": ve,
    "image_idx": np.array([[0]], dtype=np.int64),
    "mm_token_type_ids": mm_type_ids,
    "decoder_input_ids": np.zeros((1, 1), dtype=np.int64),
    "decoder_position_ids": np.zeros((1, 1), dtype=np.int64),
    "self_conditioning_logits": np.zeros((1, 1, vocab_size), dtype=np.float32),
})
ttft = time.perf_counter() - t0
kv_host = {n[:-len("_RetainedState")]: v
           for n, v in prefill_out.items()
           if n.startswith("past_") and n.endswith("_RetainedState")}
print(f"\nTTFT: {ttft:.2f}s ({len(kv_host)} KV buffers)")


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
        **kv_host,
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
    if accepted_mask.sum() >= canvas_length:
        break
canvas_time = time.perf_counter() - t0
session.deactivate()

output_text = processor.tokenizer.decode(new_canvas[0].tolist(), skip_special_tokens=True)

print(f"\nCanvas: {diffusion_steps} steps, {canvas_time:.1f}s, "
      f"{diffusion_steps * canvas_length / canvas_time:.1f} tok/s")
print(f"\nOutput:\n{output_text}")
