#!/usr/bin/env python3
"""Compare fixed DiffusionGemma raw tensors across HF, QEff, ORT, and QPC.

Each input is an ``.npz`` capture with ``canvas_logits`` and optional retained
KV arrays (`past_key.*`, `past_value.*`).  Capture decoder mode 2 and mode 3
separately with identical canvas, positions, mask, and self-conditioning
inputs.  This tool intentionally compares tensors before sampling.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from diffusion_gemma_debug import compare_full_kv, compare_full_logits


def load_capture(path):
    with np.load(path) as capture:
        if "canvas_logits" not in capture:
            raise ValueError(f"{path}: missing canvas_logits")
        logits = capture["canvas_logits"]
        kv = {name: capture[name] for name in capture.files if name.startswith(("past_key.", "past_value."))}
    return logits, kv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True)
    parser.add_argument("--qeff", required=True)
    parser.add_argument("--ort", required=True)
    parser.add_argument("--qpc", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("first", "later"), required=True)
    parser.add_argument("--precision", required=True)
    args = parser.parse_args()

    captures = {name: load_capture(getattr(args, name)) for name in ("hf", "qeff", "ort", "qpc")}
    rows = {}
    for left, right in (("hf", "qeff"), ("qeff", "ort"), ("ort", "qpc")):
        left_logits, left_kv = captures[left]
        right_logits, right_kv = captures[right]
        rows[f"{left.upper()}=={right.upper()}"] = {
            "logits": compare_full_logits(left_logits, right_logits),
            "kv": compare_full_kv(left_kv, right_kv),
        }
    output = {"mode": args.mode, "precision": args.precision, "edges": rows}
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
