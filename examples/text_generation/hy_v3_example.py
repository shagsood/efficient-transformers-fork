# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""tencent/Hy3 MoE disaggregated inference on Cloud AI 100.

Compiles separate prefill and decode QPCs and passes the KV cache between
them — at this parameter count a single QPC exceeds on-chip memory.
"""

import argparse

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

MODEL_ID = "tencent/Hy3"


def main():
    parser = argparse.ArgumentParser(description="hy_v3 MoE disaggregated on Cloud AI 100")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-devices", type=int, default=8)
    parser.add_argument("--generation-len", type=int, default=1000)
    args = parser.parse_args()

    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_id)

    # Step 1: prefill QPC — subfunctions on, full prefill_seq_len
    model.compile(
        num_cores=16, num_devices=args.num_devices,
        mxfp6_matmul=True, mxint8_kv_cache=True,
        batch_size=1, prefill_seq_len=128, ctx_len=args.ctx_len,
        mos=1, aic_enable_depth_first=True,
        prefill_only=True, use_onnx_subfunctions=True,
    )

    # Step 2: decode QPC — subfunctions off, prefill_seq_len=1
    model.compile(
        num_cores=16, num_devices=args.num_devices,
        mxfp6_matmul=True, mxint8_kv_cache=True,
        batch_size=1, prefill_seq_len=1, ctx_len=args.ctx_len,
        mos=1, aic_enable_depth_first=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    output = model.generate(prompts=[args.prompt], tokenizer=tokenizer, generation_len=args.generation_len)
    print(output)


if __name__ == "__main__":
    main()
