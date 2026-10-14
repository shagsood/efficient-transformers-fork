# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""North Mini Code disaggregated MoE inference on Cloud AI 100."""

import argparse
import os

import numpy as np
from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "CohereLabs/North-Mini-Code-1.0"
PREFILL_SEQ_LEN = 128


def _devices(value):
    return [int(item) for item in value.strip("[]").split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Cohere2 MoE disaggregated inference")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--device-group", type=_devices, default=_devices(os.environ.get("DEVICE_GROUP", "0,1,2,3")))
    parser.add_argument("--generation-len", type=int, default=100)
    args = parser.parse_args()

    if len(args.device_group) < 4 or len(args.device_group) % 2 or len(set(args.device_group)) != len(args.device_group) or any(qid < 0 for qid in args.device_group):
        parser.error("--device-group must contain unique, non-negative QIDs: an even number of at least four")
    half = len(args.device_group) // 2
    prefill_devices = args.device_group[:half]
    decode_devices = args.device_group[half:]

    model = QEFFAutoModelForCausalLM.from_pretrained(args.model_id, continuous_batching=True)
    decode_qpc = model.compile(
        num_cores=16,
        aic_hw_version="ai100",
        num_devices=half,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        batch_size=1,
        full_batch_size=1,
        prefill_seq_len=1,
        ctx_len=args.ctx_len,
        mos=1,
        aic_enable_depth_first=True,
        offload_pt_weights=False,
        split_retained_state_io=True,
        retain_full_kv=True,
        use_onnx_subfunctions=True,
    )
    prefill_qpc = model.compile(
        num_cores=16,
        aic_hw_version="ai100",
        num_devices=half,
        mxfp6_matmul=True,
        mxint8_kv_cache=True,
        batch_size=1,
        full_batch_size=1,
        prefill_seq_len=PREFILL_SEQ_LEN,
        ctx_len=args.ctx_len,
        qaic_config={"moe_config": {"expert_parallel_chunk_size": 128}},
        mdp_num_partitions=1,
        split_retained_state_io=True,
        retain_full_kv=True,
        mos=1,
        aic_enable_depth_first=False,
        prefill_only=True,
        enable_chunking=True,
        use_onnx_subfunctions=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    prompt_len = int(tokenizer(args.prompt, return_tensors="np")["input_ids"].shape[1])
    padded_len = ((prompt_len + PREFILL_SEQ_LEN - 1) // PREFILL_SEQ_LEN) * PREFILL_SEQ_LEN
    encoded = tokenizer(args.prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs = {
        "input_ids": encoded["input_ids"],
        "position_ids": np.where(encoded["attention_mask"], np.arange(padded_len), -1),
        "batch_index": np.array([[0]], dtype=np.int64),
    }

    prefill_session = QAICInferenceSession(
        prefill_qpc,
        device_ids=prefill_devices,
        kv_dma_share=True,
        stages=1,
        full_batch_size=1,
        cluster_id="prefill",
    )
    decode_session = QAICInferenceSession(
        decode_qpc,
        device_ids=decode_devices,
        kv_dma_share=True,
        full_batch_size=1,
        cluster_id="decode",
    )
    kv_caches = [np.zeros(shape, dtype=dtype) for shape, dtype in decode_session.kv_cache_info]
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    prefill_out = None
    for offset in range(0, padded_len, PREFILL_SEQ_LEN):
        last_chunk = offset + PREFILL_SEQ_LEN == padded_len
        chunk = {
            "input_ids": inputs["input_ids"][:, offset : offset + PREFILL_SEQ_LEN],
            "position_ids": inputs["position_ids"][:, offset : offset + PREFILL_SEQ_LEN],
            "batch_index": inputs["batch_index"],
        }
        exec_idx = prefill_session.np_run_pipeline(
            chunk, last_chunk=last_chunk, kv_cache_buffers=kv_caches if last_chunk else None
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        prefill_out = prefill_session.get_outputs(index=exec_idx)

    tokens = [int(np.argmax(prefill_out["logits"]))]
    decode_inputs = {
        "input_ids": np.array([[tokens[0]]], dtype=np.int64),
        "position_ids": np.array([[int(np.max(inputs["position_ids"])) + 1]], dtype=np.int64),
        "batch_index": np.array([[0]], dtype=np.int64),
    }
    for _ in range(max(0, args.generation_len - 1)):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches,
            [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx,
            buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        output = decode_session.get_outputs(index=exec_idx)
        logits = output["logits"].reshape(1, -1, output["logits"].shape[-1])[:, -1, :]
        tokens.append(int(np.argmax(logits, axis=-1)[0]))
        decode_inputs["input_ids"] = np.array([[tokens[-1]]], dtype=np.int64)
        decode_inputs["position_ids"] += 1

    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
