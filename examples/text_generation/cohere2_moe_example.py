# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CohereLabs/North-Mini-Code-1.0 MoE text generation on Cloud AI 100."""

import argparse

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM

MODEL_ID = "CohereLabs/North-Mini-Code-1.0"


def main():
    parser = argparse.ArgumentParser(description="cohere2_moe MoE on Cloud AI 100")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--generation-len", type=int, default=1000)
    args = parser.parse_args()

    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_id)
    model.compile(
        num_cores=16, aic_hw_version="ai100", num_devices=args.num_devices,
        mxfp6_matmul=True, mxint8_kv_cache=True,
        batch_size=1, prefill_seq_len=128, ctx_len=args.ctx_len,
        mos=1, aic_enable_depth_first=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    output = model.generate(prompts=[args.prompt], tokenizer=tokenizer, generation_len=args.generation_len)
    print(output)


if __name__ == "__main__":
    main()
