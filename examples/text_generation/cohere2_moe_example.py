# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""CohereLabs/North-Mini-Code-1.0 MoE disaggregated inference on Cloud AI 100.

Compiles separate prefill and decode QPCs and passes KV through QEff's
retained-state DMA handoff. The two QPCs use disjoint device groups.
"""

import argparse
import os
from time import perf_counter

import numpy as np

from transformers import AutoTokenizer

from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.generation.cloud_infer import QAICInferenceSession

MODEL_ID = "CohereLabs/North-Mini-Code-1.0"


def _parse_qids(value, name):
    qids = [int(item) for item in value.strip("[]").split(",") if item.strip()]
    if not qids or len(set(qids)) != len(qids) or any(qid < 0 for qid in qids):
        raise ValueError(f"{name} must contain unique, non-negative QIDs")
    return qids


def _validate_qpc(path, name):
    path = os.path.abspath(os.fspath(path))
    if not os.path.isfile(os.path.join(path, "programqpc.bin")):
        raise FileNotFoundError(f"{name} QPC does not contain programqpc.bin: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="cohere2_moe MoE disaggregated on Cloud AI 100")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default="The future of AI is")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--device-group", default=os.environ.get("DEVICE_GROUP", "0,1,2,3,4,5,6,7"))
    parser.add_argument("--prefill-device-group", default=os.environ.get("PREFILL_DEVICE_GROUP"))
    parser.add_argument("--decode-device-group", default=os.environ.get("DECODE_DEVICE_GROUP"))
    parser.add_argument("--prefill-qpc-path", default=os.environ.get("QEFF_PREFILL_QPC_PATH"))
    parser.add_argument("--decode-qpc-path", default=os.environ.get("QEFF_DECODE_QPC_PATH"))
    parser.add_argument("--generation-len", type=int, default=1000)
    args = parser.parse_args()
    try:
        if bool(args.prefill_device_group) != bool(args.decode_device_group):
            raise ValueError("--prefill-device-group and --decode-device-group must be provided together")
        if args.prefill_device_group:
            prefill_devices = _parse_qids(args.prefill_device_group, "--prefill-device-group")
            decode_devices = _parse_qids(args.decode_device_group, "--decode-device-group")
        else:
            device_group = _parse_qids(args.device_group, "--device-group")
            if len(device_group) < 4 or len(device_group) % 2:
                raise ValueError("--device-group must contain an even number of at least four QIDs")
            half = len(device_group) // 2
            prefill_devices, decode_devices = device_group[:half], device_group[half:]
        if set(prefill_devices) & set(decode_devices):
            raise ValueError("prefill and decode QID groups must be disjoint")
    except ValueError as error:
        parser.error(str(error))

    model = None
    if not (args.prefill_qpc_path and args.decode_qpc_path):
        model = QEFFAutoModelForCausalLM.from_pretrained(args.model_id, continuous_batching=True)

    if args.decode_qpc_path:
        decode_qpc = _validate_qpc(args.decode_qpc_path, "decode")
    else:
        assert model is not None
        decode_qpc = model.compile(
            num_cores=16, aic_hw_version="ai100", num_devices=len(decode_devices),
            mxfp6_matmul=True, mxint8_kv_cache=True,
            batch_size=1, full_batch_size=1, prefill_seq_len=1, ctx_len=args.ctx_len,
            # Decode-BMM replicates all 128 experts per SoC (~58 GB/SoC). Pack
            # experts across the decode SoCs so the QPC fits 32 GB devices.
            qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 1}},
            mos=1, aic_enable_depth_first=True,
            offload_pt_weights=False, split_retained_state_io=True,
            retain_full_kv=True, use_onnx_subfunctions=False,
        )

    if args.prefill_qpc_path:
        prefill_qpc = _validate_qpc(args.prefill_qpc_path, "prefill")
    else:
        assert model is not None
        prefill_qpc = model.compile(
            num_cores=16, aic_hw_version="ai100", num_devices=len(prefill_devices),
            mxfp6_matmul=True, mxint8_kv_cache=True,
            batch_size=1, full_batch_size=1, prefill_seq_len=128, ctx_len=args.ctx_len,
            qaic_config={"moe_config": {"flavour": "expert_parallel", "expert_parallel_chunk_size": 128}},
            mdp_num_partitions=1, split_retained_state_io=True, retain_full_kv=True,
            mos=1, aic_enable_depth_first=False, prefill_only=True,
            enable_chunking=True, use_onnx_subfunctions=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    encoded = tokenizer(args.prompt, return_tensors="np")
    prompt_len = int(encoded["input_ids"].shape[1])
    padded_len = ((prompt_len + 128 - 1) // 128) * 128
    encoded = tokenizer(args.prompt, return_tensors="np", padding="max_length", max_length=padded_len)
    inputs = {
        "input_ids": encoded["input_ids"],
        "position_ids": np.where(encoded["attention_mask"], np.arange(padded_len), -1),
        "batch_index": np.array([[0]], dtype=np.int64),
    }

    prefill_session = QAICInferenceSession(
        prefill_qpc, device_ids=prefill_devices, kv_dma_share=True,
        stages=1, full_batch_size=1, cluster_id="prefill",
    )
    decode_session = QAICInferenceSession(
        decode_qpc, device_ids=decode_devices, kv_dma_share=True,
        full_batch_size=1, cluster_id="decode",
    )
    kv_caches = [np.zeros(shape, dtype=dtype) for shape, dtype in decode_session.kv_cache_info]
    decode_kv_map = decode_session.decode_buff_map + decode_session.decode_rs_kv_only_buff_map

    prefill_start = perf_counter()
    prefill_out = None
    for offset in range(0, padded_len, 128):
        chunk = {key: value for key, value in inputs.items()}
        chunk["input_ids"] = inputs["input_ids"][:, offset : offset + 128]
        chunk["position_ids"] = inputs["position_ids"][:, offset : offset + 128]
        exec_idx = prefill_session.np_run_pipeline(
            chunk, last_chunk=offset + 128 == padded_len,
            kv_cache_buffers=kv_caches if offset + 128 == padded_len else None,
        )
        prefill_session.complete_inf(exec_idx, is_prefill=True)
        prefill_out = prefill_session.get_outputs(index=exec_idx)
    prefill_time = perf_counter() - prefill_start

    tokens = [int(np.argmax(prefill_out["logits"]))]
    decode_inputs = {
        "input_ids": np.array([[tokens[0]]], dtype=np.int64),
        "position_ids": np.array([[int(np.max(inputs["position_ids"])) + 1]], dtype=np.int64),
        "batch_index": np.array([[0]], dtype=np.int64),
    }
    decode_steps = max(0, args.generation_len - 1)
    decode_start = perf_counter()
    for _ in range(decode_steps):
        decode_session.set_data_for_kv_handoff(
            kv_caches + kv_caches, [("batch_index", 0), ("ctx_start", 0)],
            index=decode_session.decode_execObj_idx, buff_map=decode_kv_map,
        )
        exec_idx = decode_session.np_run(decode_inputs, is_prefill=False)
        decode_session.complete_inf(exec_idx, is_prefill=False)
        output = decode_session.get_outputs(index=exec_idx)
        logits = output["logits"].reshape(1, -1, output["logits"].shape[-1])[:, -1, :]
        tokens.append(int(np.argmax(logits, axis=-1)[0]))
        decode_inputs["input_ids"] = np.array([[tokens[-1]]], dtype=np.int64)
        decode_inputs["position_ids"] += 1
    decode_time = perf_counter() - decode_start
    print(f"Prefill devices: {prefill_devices}; decode devices: {decode_devices}")
    print(f"Prefill time: {prefill_time:.3f} s ({padded_len / prefill_time:.2f} tokens/s)")
    if decode_steps:
        print(f"Decode time: {decode_time:.3f} s ({decode_steps / decode_time:.2f} tokens/s)")
    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
