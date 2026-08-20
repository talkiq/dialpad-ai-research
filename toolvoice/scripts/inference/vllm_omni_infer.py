"""Confetti / When2Call audio inference for Qwen3-Omni and Qwen2.5-Omni using in-process vLLM.

Reads audio from data/<benchmark>/audio/<tts_model>/<voice>/ (produced by
../audio_generation/generate_tts_audio.py) and writes model-response CSVs to
model_responses/<model>_responses/ at the repo root, using the same file-naming
and column conventions (`model_response` + `gold_answer`) the evaluation
scripts consume.

Usage:
    python vllm_omni_infer.py --benchmark confetti
    python vllm_omni_infer.py --benchmark when2call --tts-model gpt-4o-mini-tts --voice coral
"""
import argparse
import ast
import os
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"

MODELS = [
    ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "qwen3"),
    ("Qwen/Qwen2.5-Omni-7B", "qwen2_5"),
]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL_RESPONSES_DIR = DATA_DIR.parent / "model_responses"

BENCHMARK_CONFIG = {
    "confetti": {
        "dataset_csv": DATA_DIR / "confetti" / "confetti_benchmark.csv",
        "sampling_params": dict(temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384),
        "limit_mm_per_prompt": {"audio": 3},
        "system_prompt": "You are an intelligent voice assistant. Given conversation history, audio, and tools, decide whether to call a tool.",
    },
    "when2call": {
        "dataset_csv": DATA_DIR / "when2call" / "when2call_benchmark.csv",
        "sampling_params": dict(temperature=0.3, top_p=0.9, max_tokens=2048),
        "limit_mm_per_prompt": {"audio": 1},
        "system_prompt": 'Return ONLY valid JSON:\n{\n  "Type": "TOOL_CALL | FOLLOW_UP_QUESTION | CANNOT_ANSWER",\n  "Response": "..."\n}',
    },
}


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    return audio, sr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["confetti", "when2call"], required=True)
    ap.add_argument("--tts-model", default="gpt-4o-mini-tts", help="TTS model subfolder under data/<benchmark>/audio/.")
    ap.add_argument("--voice", default="ash", help="Voice subfolder under the TTS model folder.")
    ap.add_argument("--tensor-parallel-size", type=int, default=None,
                    help="Defaults to the number of visible CUDA devices.")
    args = ap.parse_args()

    tp_size = args.tensor_parallel_size or max(1, torch.cuda.device_count())
    if torch.cuda.device_count() < tp_size:
        raise SystemExit(f"--tensor-parallel-size {tp_size} but only {torch.cuda.device_count()} CUDA device(s) visible.")

    cfg = BENCHMARK_CONFIG[args.benchmark]
    audio_root = DATA_DIR / args.benchmark / "audio" / args.tts_model / args.voice
    df = pd.read_csv(cfg["dataset_csv"])
    sampling_params = SamplingParams(**cfg["sampling_params"])

    for model_path, model_type in MODELS:
        if model_type == "qwen3":
            from transformers import Qwen3OmniMoeProcessor

            processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        else:
            from transformers import Qwen2_5OmniProcessor

            processor = Qwen2_5OmniProcessor.from_pretrained(model_path)

        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.95,
            max_model_len=32768,
            limit_mm_per_prompt=cfg["limit_mm_per_prompt"],
        )

        rows = []
        for idx, row in df.iterrows():
            tools = ast.literal_eval(row["tools"])
            audio_path = str(audio_root / f"output_audio{idx}.wav")
            audio, sr = load_audio(audio_path)

            messages = [{"role": "system", "content": [{"type": "text", "text": cfg["system_prompt"]}]}]
            if args.benchmark == "confetti":
                for turn in ast.literal_eval(row["context"]):
                    messages.append({"role": turn["role"], "content": [{"type": "text", "text": turn["content"]}]})
            messages.append({"role": "user", "content": [{"type": "audio", "audio": audio_path}]})

            prompt = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
            out = llm.generate([{"prompt": prompt, "multi_modal_data": {"audio": (audio, sr)}}], sampling_params)[0]

            rows.append({"id": row["id"], "model_response": out.outputs[0].text, "gold_answer": row["gold_answer"]})

        model_name = model_path.split("/")[-1]
        out_dir = MODEL_RESPONSES_DIR / f"{model_name}_responses"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = "BFCL_v2_conversations_clean_with_context" if args.benchmark == "confetti" else "When2call"
        out_csv = out_dir / f"{prefix}_{model_name}_{args.tts_model}_{args.voice}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
