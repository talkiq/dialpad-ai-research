"""Qwen-Omni (2.5 or 3) audio inference for Confetti / When2Call, via in-process vLLM.

Usage:
    python qwen_omni_infer.py --model-version 3 --benchmark confetti
    python qwen_omni_infer.py --model-version 2.5 --benchmark when2call
"""
import argparse
import ast
import json
import os
from pathlib import Path

import pandas as pd
import torch
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"

DEFAULT_MODEL_PATHS = {
    "2.5": ["Qwen/Qwen2.5-Omni-7B"],
    "3": ["Qwen/Qwen3-Omni-30B-A3B-Instruct", "Qwen/Qwen3-Omni-30B-A3B-Thinking"],
}

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

BENCHMARK_CSV = {
    "confetti": DATA_DIR / "confetti" / "confetti_benchmark.csv",
    "when2call": DATA_DIR / "when2call" / "when2call_benchmark.csv",
}

MODEL_RESPONSES_DIR = DATA_DIR.parent / "model_responses"

CONFETTI_PROMPT = (
    "You will be given a conversation context in text format and an audio input from "
    "the user. You are also provided with a list of tools that you can leverage to "
    "answer the user query. If a tool is appropriate, return a tool call with clear "
    "JSON arguments. If not, answer in plain text."
)

WHEN2CALL_PROMPT = """You are an AI assistant that helps users decide when to call tools. You will receive an audio input from the user and a list of available tools. Your task is to determine the appropriate action:

1. TOOL_CALL: If the user's request can be fulfilled by calling one of the available tools, call the tool with appropriate arguments.

2. FOLLOW_UP_QUESTION: If you need more information to fulfill the request or determine which tool to call, ask a clarifying follow-up question.

3. CANNOT_ANSWER: If the user's request cannot be answered based on the information available, just inform the user that you do not know the answer.

**General Rule**: If no tool is provided as context, do not make any tool calls.

Please provide your response in JSON format with following keys: (i) Type, (ii) Response.
The value of Type can be one of the following: (i) TOOL_CALL, (ii) FOLLOW_UP_QUESTION, (iii) CANNOT_ANSWER."""


def build_processor(model_version, model_path):
    if model_version == "2.5":
        from transformers import Qwen2_5OmniProcessor

        return Qwen2_5OmniProcessor.from_pretrained(model_path)
    from transformers import Qwen3OmniMoeProcessor

    return Qwen3OmniMoeProcessor.from_pretrained(model_path)


def build_messages(benchmark, system_prompt, task_prompt, conversation, audio_path):
    user_content = []
    if benchmark == "confetti":
        user_content.append({"type": "text", "text": f"Instructions: {task_prompt}\n\nConversation Context: {conversation}"})
    user_content.append({"type": "audio", "audio": audio_path})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def parse_tools(model_version, raw_tools):
    """The benchmark CSVs store `tools` as a Python-literal (single-quoted) list of dicts,
    which apply_chat_template needs parsed into actual dicts (for either model version)."""
    if not isinstance(raw_tools, str):
        return raw_tools
    try:
        return json.loads(raw_tools)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(raw_tools)
    except (ValueError, SyntaxError):
        return None


def run_row(llm, processor, sampling_params, model_version, benchmark, messages, tools, audio_path):
    text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)

    from qwen_omni_utils import process_mm_info

    try:
        audios, _, _ = process_mm_info(messages, use_audio_in_video=False)
    except Exception:
        return None
    inputs = {"prompt": text, "multi_modal_data": {"audio": audios if audios else []}}
    if model_version != "2.5":
        inputs["mm_processor_kwargs"] = {"use_audio_in_video": False}

    outputs = llm.generate([inputs], sampling_params=sampling_params)
    return outputs[0].outputs[0].text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-version", choices=["2.5", "3"], required=True)
    ap.add_argument("--benchmark", choices=["confetti", "when2call"], required=True)
    ap.add_argument("--model-paths", nargs="+", default=None)
    ap.add_argument("--dataset-csv", default=None)
    args = ap.parse_args()

    model_paths = args.model_paths or DEFAULT_MODEL_PATHS[args.model_version]
    dataset_csv = args.dataset_csv or BENCHMARK_CSV[args.benchmark]
    tts_types = ["gemini-2.5-flash-tts", "gemini-2.5-pro-tts", "gpt-4o-mini-tts"]
    voice_types = ["kore", "orus", "ash", "coral"]

    system_prompt_base = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
        "perceiving auditory and visual inputs, as well as generating text and speech."
        if args.model_version == "2.5"
        else "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen.\n\n"
    )
    task_prompt = CONFETTI_PROMPT if args.benchmark == "confetti" else WHEN2CALL_PROMPT
    system_prompt = system_prompt_base if args.model_version == "2.5" else system_prompt_base + task_prompt

    for model_path in model_paths:
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={"image": 3, "video": 3, "audio": 3},
            max_num_seqs=8,
            max_model_len=32768,
            seed=1234,
        )
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384)
        processor = build_processor(args.model_version, model_path)

        for tts_type in tts_types:
            for voice_type in voice_types:
                if voice_type in ("kore", "orus") and tts_type == "gpt-4o-mini-tts":
                    continue
                if voice_type in ("ash", "coral") and tts_type != "gpt-4o-mini-tts":
                    continue

                df = pd.read_csv(dataset_csv)
                rows_out = {"model_response": [], "gold_answer": [], "available_tools": []}
                if args.benchmark == "confetti":
                    rows_out["conversation_context"] = []
                else:
                    rows_out["uuids"] = []

                for idx, row in df.iterrows():
                    audio_path = str(DATA_DIR / args.benchmark / "audio" / tts_type / voice_type / f"output_audio{idx}.wav")
                    if args.benchmark == "confetti":
                        raw_tools, conversation, gold = row["tools"], row["context"], row["gold_answer"]
                    else:
                        raw_tools, conversation, gold = row["tools"], "", row["gold_answer"]
                    tools = parse_tools(args.model_version, raw_tools)

                    messages = build_messages(args.benchmark, system_prompt, task_prompt, conversation, audio_path)
                    answer = run_row(llm, processor, sampling_params, args.model_version, args.benchmark, messages, tools, audio_path)
                    if answer is None:
                        continue
                    print(answer)

                    rows_out["model_response"].append(answer)
                    rows_out["gold_answer"].append(gold)
                    rows_out["available_tools"].append(raw_tools)
                    if args.benchmark == "confetti":
                        rows_out["conversation_context"].append(conversation)
                    else:
                        rows_out["uuids"].append(row["id"])

                model_name = model_path.split("/")[-1]
                out_dir = MODEL_RESPONSES_DIR / f"{model_name}_responses"
                out_dir.mkdir(parents=True, exist_ok=True)
                if args.benchmark == "confetti":
                    out_name = f"BFCL_v2_conversations_clean_with_context_{model_name}_{tts_type}_{voice_type}.csv"
                else:
                    out_name = f"When2call_{model_name}_{tts_type}_{voice_type}.csv"
                pd.DataFrame(rows_out).to_csv(out_dir / out_name, index=False)


if __name__ == "__main__":
    main()
