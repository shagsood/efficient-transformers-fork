# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import argparse
import io
import wave
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, TextStreamer

from QEfficient import QEFFAutoModelForMultimodalLM


MODEL_ID = "thinkingmachines/Inkling-Small"
SAMPLE_RATE = 16000
DEFAULT_PREFILL_SEQ_LEN = 128
DEFAULT_CTX_LEN = 1024
DEFAULT_GENERATION_LEN = 128
DEFAULT_AUDIO_SECONDS = 0.2
INKLING_PAD_TOKEN = "<|endoftext|>"
INKLING_STOP_TOKEN = "<|content_model_end_sampling|>"


def parse_device_ids(value: str | None):
    if value is None or value.strip() == "":
        return None
    return [int(device_id) for device_id in value.split(",") if device_id.strip()]


def load_image(path_or_url: str | None) -> Image.Image:
    if path_or_url is None:
        return Image.new("RGB", (40, 40), color=(90, 110, 180))
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, stream=True, timeout=30)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def synthesize_audio(seconds: float = DEFAULT_AUDIO_SECONDS, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    samples = max(1, int(seconds * sampling_rate))
    t = np.arange(samples, dtype=np.float32) / sampling_rate
    return (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def _decode_pcm_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wav:
        sampling_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32), sampling_rate


def load_audio(path_or_url: str | None) -> tuple[np.ndarray | str, int]:
    if path_or_url is None:
        return synthesize_audio(), SAMPLE_RATE
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=30)
        response.raise_for_status()
        try:
            return _decode_pcm_wav(response.content)
        except wave.Error:
            return path_or_url, SAMPLE_RATE

    path = Path(path_or_url)
    try:
        return _decode_pcm_wav(path.read_bytes())
    except wave.Error:
        return str(path), SAMPLE_RATE


def configure_tokenizer(processor):
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = INKLING_PAD_TOKEN
    if getattr(tokenizer, "eos_token_id", None) is None:
        tokenizer.eos_token = INKLING_STOP_TOKEN


def inkling_messages(prompt: str, image: Image.Image, audio):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "audio", "audio": audio},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def prepare_inputs(processor, prompt: str, image: Image.Image, audio, sampling_rate: int, prefill_seq_len: int):
    inputs = processor.apply_chat_template(
        inkling_messages(prompt, image, audio),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        reasoning_effort="none",
        processor_kwargs={
            "sampling_rate": sampling_rate,
            "padding": "max_length",
            "max_length": prefill_seq_len,
        },
    )

    if "audio_input_ids_mask" in inputs:
        inputs["audio_input_ids_mask"] = inputs["audio_input_ids_mask"].to(torch.int64)
    inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1], dtype=torch.int64).view(1, -1)
    return inputs


def processor_dimensions(inputs) -> dict[str, int]:
    pixel_values = inputs["pixel_values"]
    audio_input_ids = inputs["audio_input_ids"]
    return {
        "num_patches": int(pixel_values.shape[0]),
        "num_audios": int(audio_input_ids.shape[0]),
        "audio_feature_len": int(audio_input_ids.shape[1]),
    }


def main():
    parser = argparse.ArgumentParser(description="Inkling-Small image+audio+text generation on Cloud AI 100")
    parser.add_argument("--model-id", default=MODEL_ID, help="Hugging Face model ID or local model directory")
    parser.add_argument("--image", help="Image path or URL. Defaults to a generated 40x40 RGB image.")
    parser.add_argument(
        "--audio",
        help="PCM WAV path/URL or processor-readable audio path/URL. Defaults to a 0.2s tone.",
    )
    parser.add_argument(
        "--prompt",
        default="Describe the image and the audio together.",
        help="User text prompt paired with the image and audio.",
    )
    parser.add_argument("--prefill-seq-len", type=int, default=DEFAULT_PREFILL_SEQ_LEN)
    parser.add_argument("--ctx-len", type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--generation-len", type=int, default=DEFAULT_GENERATION_LEN)
    parser.add_argument("--num-cores", type=int, default=16)
    parser.add_argument("--num-devices", type=int, default=8)
    parser.add_argument("--device-ids", type=parse_device_ids, default=None)
    parser.add_argument("--compile-dir", default="qpc_inkling_mm_model")
    parser.add_argument("--qpc-path", help="Existing QPC path. If set, compile is skipped.")
    parser.add_argument("--no-mxfp6", action="store_true", help="Disable MXFP6 matmul compression.")
    parser.add_argument("--no-mxint8-kv-cache", action="store_true", help="Disable MXINT8 KV-cache compression.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders.")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face token.")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
    )
    configure_tokenizer(processor)

    image = load_image(args.image)
    audio, sampling_rate = load_audio(args.audio)
    inputs = prepare_inputs(processor, args.prompt, image, audio, sampling_rate, args.prefill_seq_len)
    dims = processor_dimensions(inputs)

    qeff_model = QEFFAutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        token=args.hf_token,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
        kv_offload=False,
    )

    if args.qpc_path:
        qeff_model.qpc_path = Path(args.qpc_path)
    else:
        qeff_model.compile(
            compile_dir=args.compile_dir,
            batch_size=1,
            prefill_seq_len=args.prefill_seq_len,
            ctx_len=args.ctx_len,
            num_cores=args.num_cores,
            num_devices=args.num_devices,
            mxfp6_matmul=not args.no_mxfp6,
            mxint8_kv_cache=not args.no_mxint8_kv_cache,
            num_patches=dims["num_patches"],
            num_audios=dims["num_audios"],
            audio_feature_len=dims["audio_feature_len"],
            use_onnx_subfunctions=True,
        )

    streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    output = qeff_model.generate(
        inputs=inputs,
        streamer=streamer,
        device_ids=args.device_ids,
        generation_len=args.generation_len,
    )
    print()
    print(processor.tokenizer.batch_decode(output.generated_ids, skip_special_tokens=True))
    print(output)


if __name__ == "__main__":
    main()
