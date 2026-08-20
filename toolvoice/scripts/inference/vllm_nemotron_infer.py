"""Confetti / When2Call audio inference for Nemotron-3-Nano-Omni via a vLLM OpenAI-compatible server.

Note: Nemotron is an additional model, not one of the seven evaluated in the paper.

Reads audio from data/<benchmark>/audio/<tts_model>/<voice>/ (produced by
../audio_generation/generate_tts_audio.py) and writes model-response CSVs to
model_responses/<model>_responses/ at the repo root. The Confetti output keeps
the model's text in `model_response` (the column the evaluation scripts read)
and native tool calls, when the server parses them, in `tool_calls` using the
same `[{"id", "name", "args_json"}]` format the realtime inference scripts
write, so `evaluate_confetti.py --response-column tool_calls` parses it directly.

Start the server first (BF16 needs TP>1 on 80GB GPUs):
    pip install "vllm[audio]"
    vllm serve nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 \\
      --host 0.0.0.0 --port 8000 \\
      --tensor-parallel-size 8 \\
      --max-model-len 32768 \\
      --trust-remote-code \\
      --allowed-local-media-path / \\
      --reasoning-parser nemotron_v3 \\
      --enable-auto-tool-choice \\
      --tool-call-parser qwen3_coder

Usage:
    python vllm_nemotron_infer.py --benchmark confetti
    python vllm_nemotron_infer.py --benchmark when2call --tts-model gpt-4o-mini-tts --voice coral
"""
import argparse
import ast
import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

MODEL = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MODEL_RESPONSES_DIR = DATA_DIR.parent / "model_responses"

BENCHMARK_CONFIG = {
    "confetti": {
        "dataset_csv": DATA_DIR / "confetti" / "confetti_benchmark.csv",
        "sampling": dict(max_tokens=16384, temperature=0.6, top_p=0.95),
        "extra_body": {
            "top_k": 20,
            "thinking_token_budget": 16384 + 1024,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_budget": 16384},
        },
        "system_prompt": "You are an intelligent voice assistant. Given conversation history, audio, and tools, decide whether to call a tool.",
    },
    "when2call": {
        "dataset_csv": DATA_DIR / "when2call" / "when2call_benchmark.csv",
        "sampling": dict(max_tokens=2048, temperature=0.2),
        "extra_body": {"top_k": 1, "chat_template_kwargs": {"enable_thinking": False}},
        "system_prompt": 'Return ONLY valid JSON:\n{\n  "Type": "TOOL_CALL | FOLLOW_UP_QUESTION | CANNOT_ANSWER",\n  "Response": "..."\n}',
    },
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["confetti", "when2call"], required=True)
    ap.add_argument("--tts-model", default="gpt-4o-mini-tts", help="TTS model subfolder under data/<benchmark>/audio/.")
    ap.add_argument("--voice", default="ash", help="Voice subfolder under the TTS model folder.")
    args = ap.parse_args()

    cfg = BENCHMARK_CONFIG[args.benchmark]
    client = OpenAI(base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"), api_key="EMPTY")
    audio_root = DATA_DIR / args.benchmark / "audio" / args.tts_model / args.voice
    df = pd.read_csv(cfg["dataset_csv"])

    rows = []
    for idx, row in df.iterrows():
        tools = ast.literal_eval(row["tools"])
        audio_url = (audio_root / f"output_audio{idx}.wav").resolve().as_uri()

        messages = [{"role": "system", "content": cfg["system_prompt"]}]
        if args.benchmark == "confetti":
            for turn in ast.literal_eval(row["context"]):
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": audio_url}}]})

        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=tools, extra_body=cfg["extra_body"], **cfg["sampling"]
        )
        choice = response.choices[0].message

        if args.benchmark == "confetti":
            tool_calls = [
                {"id": tc.id, "name": tc.function.name, "args_json": tc.function.arguments}
                for tc in (choice.tool_calls or [])
            ]
            rows.append({
                "id": row["id"],
                "model_response": choice.content,
                "tool_calls": json.dumps(tool_calls, ensure_ascii=False),
                "reasoning": getattr(choice, "reasoning_content", None),
                "gold_answer": row["gold_answer"],
            })
        else:
            rows.append({"id": row["id"], "model_response": choice.content, "gold_answer": row["gold_answer"]})

    model_name = MODEL.split("/")[-1]
    out_dir = MODEL_RESPONSES_DIR / f"{model_name}_responses"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "BFCL_v2_conversations_clean_with_context" if args.benchmark == "confetti" else "When2call"
    out_csv = out_dir / f"{prefix}_{model_name}_{args.tts_model}_{args.voice}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()
