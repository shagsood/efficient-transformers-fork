# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""Full-weight Muse Glimmer HF == QEff == ORT == QPC prefill parity.

This correctness script reuses an existing QPC. If --onnx-path is omitted, it
exports the QEff model once so ORT can run the same fixed prefill feed.
"""

import argparse
import gc
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from QEfficient import QEFFAutoModelForImageTextToText
from QEfficient.generation.cloud_infer import QAICInferenceSession

DEFAULT_MODEL = 'meta-models/Muse-Glimmer-30B'
DEFAULT_IMAGE_URL = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg'


def parse_device_ids(value):
    device_ids = [int(item) for item in value.split(',') if item.strip()]
    if not device_ids:
        raise argparse.ArgumentTypeError('--device-ids must contain at least one QID')
    return device_ids


def resolve_qpc_path(path):
    qpc_path = Path(path).expanduser()
    if (qpc_path / 'programqpc.bin').is_file():
        return qpc_path
    if qpc_path.exists():
        matches = sorted(qpc_path.rglob('programqpc.bin'))
        if matches:
            return matches[0].parent
        raise FileNotFoundError(
            f'No programqpc.bin found under --qpc-path: {qpc_path}. '
            'The directory exists, but it is not a complete compiled QPC.'
        )
    raise FileNotFoundError(
        f'--qpc-path does not exist: {qpc_path}. Re-run compile/inference or pass the current QPC directory.'
    )


def load_image(path, url, image_size):
    if path:
        image = Image.open(path).convert('RGB')
    else:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
    return image.resize((image_size, image_size))


def build_prompt(prompt):
    return '<|begin_of_text|><|start|>user<|message|><|patch|>' + prompt + '<|eot|><|start|>assistant'


def prepare_prefill(processor, tokenizer, args):
    image = load_image(args.image, args.image_url, args.image_size)
    raw = processor(text=build_prompt(args.prompt), images=image, return_tensors='pt')
    valid_len = int(raw['input_ids'].shape[1])
    if valid_len > args.prefill_seq_len:
        raise ValueError(f'Prompt length {valid_len} exceeds --prefill-seq-len {args.prefill_seq_len}')

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    input_ids = torch.full((1, args.prefill_seq_len), int(pad_token_id), dtype=torch.int64)
    input_ids[:, :valid_len] = raw['input_ids'].to(torch.int64)
    position_ids = torch.full((1, args.prefill_seq_len), -1, dtype=torch.int64)
    position_ids[:, :valid_len] = torch.arange(valid_len, dtype=torch.int64)

    padded = {
        'input_ids': input_ids,
        'position_ids': position_ids,
        'pixel_values': raw['pixel_values'].to(torch.float32),
        'image_grid_thw': raw['image_grid_thw'].to(torch.int64),
    }
    raw_inputs = {
        'input_ids': raw['input_ids'].to(torch.int64),
        'position_ids': torch.arange(valid_len, dtype=torch.int64).view(1, -1),
        'pixel_values': raw['pixel_values'].to(torch.float32),
        'image_grid_thw': raw['image_grid_thw'].to(torch.int64),
    }
    return raw_inputs, padded, valid_len


def select_prefill_logits(logits, valid_len):
    logits = np.asarray(logits)
    if logits.ndim == 3 and logits.shape[1] > 1:
        return logits[:, valid_len - 1, :]
    if logits.ndim == 3:
        return logits[:, -1, :]
    return logits


def numpy_dict(values):
    result = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            result[name] = value.detach().cpu().numpy()
        else:
            result[name] = np.asarray(value)
    return result


def full_attention_interval(config):
    return next(
        (index + 1 for index, kind in enumerate(config.text_config.layer_types) if kind == 'full_attention'),
        config.text_config.num_hidden_layers,
    )


def zero_past_inputs(input_names, config, ctx_len):
    interval = full_attention_interval(config)
    past = {}
    for layer in range(config.text_config.num_hidden_layers):
        cache_len = config.text_config.sliding_window if (layer + 1) % interval else ctx_len
        shape = (1, config.text_config.num_key_value_heads, cache_len, config.text_config.head_dim)
        for kind in ('past_key', 'past_value'):
            name = f'{kind}.{layer}'
            if name in input_names:
                past[name] = np.zeros(shape, dtype=np.float16)
    return past


def exported_onnx_path(export_result):
    candidate = export_result[-1] if isinstance(export_result, (list, tuple)) else export_result
    candidate = Path(candidate)
    if candidate.is_dir():
        matches = sorted(candidate.rglob('*.onnx'))
        if not matches:
            raise FileNotFoundError(f'No ONNX file under export directory {candidate}')
        return matches[0]
    return candidate


def cosine_similarity(reference, candidate):
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(ref) * np.linalg.norm(cand)
    return float(np.dot(ref, cand) / denom) if denom else float('nan')


def compare_logits(edge, reference, candidate, tol_cos):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape != candidate.shape:
        return {'edge': edge, 'passed': False, 'reason': f'shape mismatch {reference.shape} vs {candidate.shape}'}
    ref_2d = reference.reshape(-1, reference.shape[-1])
    cand_2d = candidate.reshape(-1, candidate.shape[-1])
    argmax_ref = int(ref_2d[-1].argmax())
    argmax_candidate = int(cand_2d[-1].argmax())
    cosine = cosine_similarity(reference, candidate)
    diff = reference.astype(np.float64) - candidate.astype(np.float64)
    return {
        'edge': edge,
        'passed': argmax_ref == argmax_candidate and cosine >= tol_cos,
        'argmax_ref': argmax_ref,
        'argmax_candidate': argmax_candidate,
        'argmax_ok': argmax_ref == argmax_candidate,
        'cosine': cosine,
        'max_abs': float(np.max(np.abs(diff))),
        'mean_abs': float(np.mean(np.abs(diff))),
    }


def print_result(result):
    if 'reason' in result:
        print(f"{result['edge']}: FAIL {result['reason']}")
        return
    status = 'PASS' if result['passed'] else 'FAIL'
    print(
        f"{result['edge']}: {status} "
        f"argmax_ref={result['argmax_ref']} argmax_candidate={result['argmax_candidate']} "
        f"cosine={result['cosine']:.6f} max_abs={result['max_abs']:.6g} mean_abs={result['mean_abs']:.6g}"
    )


def write_report(path, args, valid_len, comparisons):
    verdict = 'PASS' if all(item['passed'] for item in comparisons) else 'FAIL'
    lines = [
        '# Muse Glimmer Full-Weight Parity',
        '',
        f'- Model: {args.model_name}',
        f'- Precision: {args.precision}',
        f'- QPC: {args.qpc_path}',
        f'- Prompt tokens: {valid_len}',
        f'- Verdict: {verdict}',
        '',
        '| Edge | Verdict | Argmax | Cosine | Max abs | Mean abs |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for item in comparisons:
        if 'reason' in item:
            lines.append(f"| {item['edge']} | FAIL | {item['reason']} | | | |")
            continue
        status = 'PASS' if item['passed'] else 'FAIL'
        lines.append(
            f"| {item['edge']} | {status} | {item['argmax_ref']} / {item['argmax_candidate']} | "
            f"{item['cosine']:.6f} | {item['max_abs']:.6g} | {item['mean_abs']:.6g} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Muse Glimmer full-weight HF == QEff == ORT == QPC parity')
    parser.add_argument('--model-name', default=DEFAULT_MODEL)
    parser.add_argument('--prompt', default='Describe this image.')
    parser.add_argument('--image', help='Local image path; overrides --image-url')
    parser.add_argument('--image-url', default=DEFAULT_IMAGE_URL)
    parser.add_argument('--image-size', type=int, default=56)
    parser.add_argument('--prefill-seq-len', type=int, default=640)
    parser.add_argument('--ctx-len', type=int, default=1024)
    parser.add_argument('--device-ids', type=parse_device_ids, required=True)
    parser.add_argument('--qpc-path', required=True)
    parser.add_argument('--onnx-path', help='Reuse an exported ONNX instead of exporting')
    parser.add_argument('--export-dir', default='muse_glimmer_full_parity_export')
    parser.add_argument('--precision', choices=['fp16', 'mxfp6'], default='mxfp6')
    parser.add_argument('--tol-cos', type=float, default=0.999)
    parser.add_argument('--report-path', help='Optional markdown report path')
    args = parser.parse_args()

    qpc_path = resolve_qpc_path(args.qpc_path)
    args.qpc_path = str(qpc_path)

    print(f'Loading processor: {args.model_name}')
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = getattr(processor, 'tokenizer', processor)
    raw_inputs, padded_inputs, valid_len = prepare_prefill(processor, tokenizer, args)
    print(f'Prompt tokens: {valid_len}')

    dtype = torch.float16
    print('Running HF PyTorch prefill...')
    started = time.perf_counter()
    hf_model = AutoModelForImageTextToText.from_pretrained(
        args.model_name, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True
    ).eval()
    with torch.no_grad():
        hf_outputs = hf_model(**raw_inputs, use_cache=False)
    hf_logits = select_prefill_logits(hf_outputs.logits.detach().float().cpu().numpy(), valid_len)
    del hf_outputs, hf_model
    gc.collect()
    print(f'HF stage: {time.perf_counter() - started:.2f} s')

    print('Loading QEff model...')
    qeff_model = QEFFAutoModelForImageTextToText.from_pretrained(
        args.model_name, kv_offload=False, trust_remote_code=True, dtype=dtype
    )
    print('Running QEff PyTorch prefill...')
    dummy = qeff_model.model.get_dummy_inputs(
        prefill_seq_len=args.prefill_seq_len,
        ctx_len=args.ctx_len,
        height=args.image_size,
        width=args.image_size,
    )
    dummy.update(padded_inputs)
    with torch.no_grad():
        qeff_outputs = qeff_model.model(**dummy)
    qeff_logits = select_prefill_logits(qeff_outputs.logits.detach().float().cpu().numpy(), valid_len)
    del qeff_outputs

    print('Running ORT prefill...')
    if args.onnx_path:
        onnx_path = Path(args.onnx_path).expanduser()
    else:
        export_result = qeff_model.export(
            args.export_dir,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            height=args.image_size,
            width=args.image_size,
            offload_pt_weights=False,
        )
        onnx_path = exported_onnx_path(export_result)
    print(f'ONNX: {onnx_path}')
    ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    ort_inputs = numpy_dict(padded_inputs)
    for item in ort_session.get_inputs():
        name = item.name
        if name.startswith('past_key.'):
            ort_inputs[name] = dummy['past_key_values'][int(name.rsplit('.', 1)[1])][0].detach().cpu().numpy()
        elif name.startswith('past_value.'):
            ort_inputs[name] = dummy['past_key_values'][int(name.rsplit('.', 1)[1])][1].detach().cpu().numpy()
    required = {item.name for item in ort_session.get_inputs()}
    ort_logits = select_prefill_logits(ort_session.run(['logits'], {name: ort_inputs[name] for name in required})[0], valid_len)
    del ort_session
    gc.collect()

    print('Running QPC prefill...')
    qaic_session = QAICInferenceSession(str(qpc_path), device_ids=args.device_ids)
    try:
        qpc_inputs = numpy_dict(padded_inputs)
        qpc_inputs.update(zero_past_inputs(qaic_session.input_names, qeff_model.model.config, args.ctx_len))
        qpc_feed = {name: value for name, value in qpc_inputs.items() if name in qaic_session.input_names}
        missing = set(qaic_session.input_names) - set(qpc_feed)
        if missing:
            raise RuntimeError(f'Missing required QPC inputs: {sorted(missing)}')
        qpc_outputs = qaic_session.run(qpc_feed)
        qpc_logits = select_prefill_logits(qpc_outputs['logits'], valid_len)
    finally:
        qaic_session.deactivate()

    comparisons = [
        compare_logits('HF==QEff', hf_logits, qeff_logits, args.tol_cos),
        compare_logits('QEff==ORT', qeff_logits, ort_logits, args.tol_cos),
        compare_logits('ORT==QPC', ort_logits, qpc_logits, args.tol_cos),
    ]
    for item in comparisons:
        print_result(item)
    if args.report_path:
        write_report(Path(args.report_path), args, valid_len, comparisons)
        print(f'Report: {args.report_path}')
    if not all(item['passed'] for item in comparisons):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
