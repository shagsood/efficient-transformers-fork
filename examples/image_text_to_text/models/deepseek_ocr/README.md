# DeepSeek-OCR-2 Inference

This directory contains an example script for running document OCR on `deepseek-ai/DeepSeek-OCR-2`
(`model_type: deepseek_vl_v2`) via `QEFFAutoModelForImageTextToText`.

The model is a two-stage "DeepEncoder v2" vision tower (SAM ViT-B backbone -> Qwen2-style
global-attention stage -> linear projector) feeding a 12-layer DeepSeek-V2-family
Mixture-of-Experts text decoder (64 routed + 2 shared experts, top-6 routing, MLA disabled —
plain multi-head attention). `kv_offload=True` compiles the vision tower and the language
decoder as two separate QPCs, so the image is encoded once per request while decode iterates
on the language QPC alone.

The checkpoint's own modeling code targets an older Transformers release, so QEfficient
supplies a self-contained implementation for this architecture — `trust_remote_code=True` on
`from_pretrained` only loads its config/tokenizer, not its modeling code.

## Known limitation

Compile with `num_devices=1`. With tensor slicing enabled (`num_devices > 1`), the compiler
drops the `vision_embeds` input from the language QPC, so the decoder never sees the image and
silently transcribes nothing from it (no error — the runtime only warns
`Buffer: "vision_embeds" not found`). This is a compiler defect, not a modeling gap; `num_devices=1`
is token-exact against the fp32 CPU reference at both `fp16` and `mxfp6+mxint8_kv`.

To run the example script:
```sh
python deepseek_ocr_example.py
```

Expected output for the sample document image in the script:
```sh
κε

Table 3: A summary of the performance of cGFR equations in critically ill patients with AKI, whose  \( {}^{1} \) CrCl was less than 60 mL min
```

This output is token-exact between the AI 100 device (both `fp16` and `mxfp6+mxint8_kv`) and the
fp32 CPU reference.
