# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
DeepSeek-OCR-2 document OCR on Cloud AI 100.

Runs the full vision-language pipeline: a document image is encoded by the DeepEncoder-v2
vision tower (SAM ViT-B backbone -> Qwen2-style global-attention stage -> linear
projector), the resulting embeddings replace the image placeholder tokens in the prompt,
and the Mixture-of-Experts text decoder generates the transcription.

``kv_offload=True`` compiles the vision tower and the language decoder as two QPCs, so the
image is encoded once per request while decode iterates on the language QPC alone.
"""

from io import BytesIO

import requests
from PIL import Image
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.utils import constants

model_id = "deepseek-ai/DeepSeek-OCR-2"

# Load the model. The checkpoint's own modeling code targets an older Transformers
# release, so QEfficient supplies a self-contained implementation for this architecture.
qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
    model_id,
    attn_implementation="eager",
    kv_offload=True,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

qeff_model.compile(
    img_size=constants.DEEPSEEK_VL_V2_IMG_SIZE,
    prefill_seq_len=512,
    ctx_len=1024,
    num_cores=16,
    num_devices=4,
    batch_size=1,
    mxfp6_matmul=True,
    mxint8_kv_cache=True,
    aic_enable_depth_first=True,
)

image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_docvqa/resolve/main/document_2.png"
image = Image.open(BytesIO(requests.get(image_url, stream=True).content)).convert("RGB")

# The prompt reserves one placeholder token per vision embedding; the vision tower's
# output is spliced into exactly those positions.
prompt = "<image>\nFree OCR. "

output = qeff_model.generate(
    tokenizer=tokenizer,
    images=image,
    prompt=prompt,
    generation_len=256,
)
print(output)
