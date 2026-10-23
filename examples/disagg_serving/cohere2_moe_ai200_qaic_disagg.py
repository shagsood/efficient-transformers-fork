# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Serve Cohere2-MoE AI200 QEff QPCs through qaic_disagg.

The QPCs must be compiled by cohere2_moe_ai200_disagg.py for the same 8-prefill
and 4-decode device topology.
"""

import argparse
import json
import os
import sys
from pathlib import Path

MODEL_ID = "CohereLabs/North-Mini-Code-1.0"


def parse_qids(value: str, name: str, expected: int) -> list[int]:
    qids = [int(item) for item in value.split(",") if item]
    if len(qids) != expected or len(set(qids)) != expected or any(qid < 0 for qid in qids):
        raise ValueError(f"{name} requires {expected} unique non-negative QIDs")
    return qids


def qpc_path(value: str, name: str) -> str:
    path = Path(value).resolve()
    if not (path / "programqpc.bin").is_file():
        raise ValueError(f"{name} does not contain programqpc.bin: {path}")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cohere2-MoE AI200 QPCs with qaic_disagg")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prefill-qpc-path", required=True)
    parser.add_argument("--decode-qpc-path", required=True)
    parser.add_argument("--prefill-device-group", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--decode-device-group", default="8,9,10,11")
    parser.add_argument("--ctx-len", type=int, default=1024)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--prefill-port", type=int, default=18001)
    parser.add_argument("--decode-port", type=int, default=18002)
    parser.add_argument("--kv-handoff-port", type=int, default=18565)
    args = parser.parse_args()

    try:
        prefill_qids = parse_qids(args.prefill_device_group, "--prefill-device-group", 8)
        decode_qids = parse_qids(args.decode_device_group, "--decode-device-group", 4)
        if set(prefill_qids) & set(decode_qids):
            raise ValueError("prefill and decode device groups must be disjoint")
        prefill_qpc = qpc_path(args.prefill_qpc_path, "--prefill-qpc-path")
        decode_qpc = qpc_path(args.decode_qpc_path, "--decode-qpc-path")
    except ValueError as error:
        parser.error(str(error))

    try:
        from vllm.platforms import current_platform
        from vllm.model_executor.models.cohere2_moe import Cohere2MoeForCausalLM  # noqa: F401

        current_platform.package_root = "vllm_qaic"
        import qaic_disagg.__main__ as disagg
    except ImportError as error:
        parser.error(
            "qaic_disagg requires the Qualcomm vLLM-QAIC environment with Cohere2-MoE vLLM support: "
            f"{error}"
        )

    site_dir = Path(__file__).with_name("cohere2_moe_qaic_site")
    real_popen = disagg.subprocess.Popen

    def patched_popen(command, *popen_args, **popen_kwargs):
        if not (isinstance(command, (list, tuple)) and list(command[:2]) == ["vllm", "serve"]):
            return real_popen(command, *popen_args, **popen_kwargs)

        command = list(command)
        child_port = int(command[command.index("--port") + 1])
        if "--model-tag" in command:
            index = command.index("--model-tag")
            del command[index : index + 2]
        config_index = command.index("--additional-config") + 1
        config = json.loads(command[config_index])
        if child_port == args.decode_port:
            config["device_group"] = list(range(len(decode_qids)))
        command[config_index] = json.dumps(config)

        visible_qids = {
            args.prefill_port: args.prefill_device_group,
            args.decode_port: args.decode_device_group,
        }.get(child_port)
        if visible_qids is not None:
            environment = dict(popen_kwargs.get("env") or os.environ)
            environment["QAIC_VISIBLE_DEVICES"] = visible_qids
            environment["VLLM_QAIC_USE_LEGACY_SLICING_SPEC"] = "1"
            environment["PYTHONPATH"] = f"{site_dir}:{environment.get('PYTHONPATH', '')}"
            popen_kwargs["env"] = environment
        return real_popen(command, *popen_args, **popen_kwargs)

    disagg.subprocess.Popen = patched_popen
    prefill_config = {
        "qpc_path": prefill_qpc,
        "prefill_only": True,
        "split-model-io": True,
        "allow-mxint8-mdp-io": True,
    }
    decode_config = {
        "qpc_path": decode_qpc,
        "prefill_only": False,
        "split-model-io": True,
        "allow-mxint8-mdp-io": True,
    }
    sys.argv = [
        "qaic_disagg",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--prefill-port",
        str(args.prefill_port),
        "--decode-port",
        str(args.decode_port),
        "--kv-handOff-port",
        str(args.kv_handoff_port),
        "--prefill-device-group",
        args.prefill_device_group,
        "--decode-device-group",
        args.decode_device_group,
        "--prefill-pipeline-parallel-size",
        "1",
        "--prefill-max-num-seqs",
        "1",
        "--decode-max-num-seqs",
        "1",
        "--max-model-len",
        str(args.ctx_len),
        "--dtype",
        "float16",
        "--quantization",
        "mxfp6",
        "--kv-cache-dtype",
        "mxint8",
        "--prefill-override-qaic-config",
        json.dumps(prefill_config),
        "--decode-override-qaic-config",
        json.dumps(decode_config),
        "-vvv",
    ]
    disagg.main()


if __name__ == "__main__":
    main()
