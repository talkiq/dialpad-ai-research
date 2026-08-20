#!/usr/bin/env python3
"""ASR-cascade text-only runner for the local Qwen3-Omni model: sends each transcribed
question + context + tools to Qwen3-Omni as a text-only chat turn (no audio) via
in-process vLLM, so the transcript can be scored by evaluate_confetti.py /
evaluate_when2call.py the same way as the direct-voice inference scripts.

Reads the ASR CSVs produced by transcribe_audio.py + enrich_asr_csvs.py (see
cascade_asr/README.md) and writes `<input>_<model>_text_only.csv`.

Usage:
    python asr_cascade_qwen_confetti.py --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
"""
import argparse
import ast
import json
import os

import pandas as pd
import torch
from vllm import LLM, SamplingParams

CONFETTI_PROMPT = (
    "You will be given a conversation context in text format and a user question "
    "transcribed from speech. You are also provided with a list of tools that you can "
    "leverage to answer the user query. If a tool is appropriate, return a tool call "
    "with clear JSON arguments. If not, answer in plain text."
)

WHEN2CALL_PROMPT = """You are an AI assistant that helps users decide when to call tools. You will receive a user question transcribed from speech and a list of available tools. Your task is to determine the appropriate action:

1. TOOL_CALL: If the user's request can be fulfilled by calling one of the available tools, call the tool with appropriate arguments.

2. FOLLOW_UP_QUESTION: If you need more information to fulfill the request or determine which tool to call, ask a clarifying follow-up question.

3. CANNOT_ANSWER: If the user's request cannot be answered based on the information available, just inform the user that you do not know the answer.

**General Rule**: If no tool is provided as context, do not make any tool calls.

Please provide your response in JSON format with following keys: (i) Type, (ii) Response.
The value of Type can be one of the following: (i) TOOL_CALL, (ii) FOLLOW_UP_QUESTION, (iii) CANNOT_ANSWER."""

IN_CSVS = {
    "confetti": "GPT-4o-Mini-STT-Confetti-GPT.csv",
    "when2call": "GPT-4o-Mini-STT-When2Call-GPT.csv",
}


_TYPE_MAP = {"dict": "object", "list": "array", "float": "number", "int": "integer", "str": "string", "bool": "boolean"}


def _fix_schema_types(schema):
    if isinstance(schema, dict):
        current = schema.get("type")
        if current == "any":
            schema.pop("type", None)
        elif current in _TYPE_MAP:
            schema["type"] = _TYPE_MAP[current]
        for prop_schema in schema.get("properties", {}).values():
            _fix_schema_types(prop_schema)
        if "items" in schema:
            _fix_schema_types(schema["items"])
    return schema


def parse_tools(raw_tools):
    """Tool cells are either a Python-literal list of dicts (Confetti) or a
    Python-literal list of JSON-encoded schema strings (When2Call)."""
    if not isinstance(raw_tools, str):
        return raw_tools
    parsed = None
    try:
        parsed = json.loads(raw_tools)
    except json.JSONDecodeError:
        pass
    if parsed is None:
        try:
            parsed = ast.literal_eval(raw_tools)
        except (ValueError, SyntaxError):
            return None

    if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
        parsed = [json.loads(item) for item in parsed]

    if isinstance(parsed, list):
        for schema in parsed:
            _fix_schema_types(schema)
    return parsed


def build_messages(benchmark, task_prompt, context, question):
    system_prompt = "You are Qwen-Omni, a smart voice assistant created by Alibaba Qwen.\n\n" + task_prompt
    if benchmark == "confetti":
        user_text = f"Conversation Context: {context}\n\nUser question: {question}"
    else:
        user_text = f"User question: {question}"
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_text}]},
    ]


def process_csv(llm, processor, sampling_params, benchmark, in_csv):
    task_prompt = CONFETTI_PROMPT if benchmark == "confetti" else WHEN2CALL_PROMPT
    base = os.path.splitext(os.path.basename(in_csv))[0]
    model_name = llm.llm_engine.model_config.model.split("/")[-1]
    out_csv = f"{base}_{model_name}_text_only.csv"

    df = pd.read_csv(in_csv)
    df = df[df["audio_condition"] == "clean"].reset_index(drop=True)
    print(f"[{in_csv}] rows after audio_condition='clean' filter: {len(df)}", flush=True)

    results = []
    for idx, row in df.iterrows():
        question = str(row.get("transcription_text") or "")
        context = str(row.get("context") or "")
        tools = parse_tools(row.get("tool"))

        messages = build_messages(benchmark, task_prompt, context, question)
        text = processor.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([{"prompt": text}], sampling_params=sampling_params)
        answer = outputs[0].outputs[0].text
        print(f"[{idx}] {answer[:200]!r}", flush=True)

        row_dict = row.to_dict()
        row_dict["assistant_text"] = answer
        row_dict["model_response"] = answer
        row_dict["model_intent"] = "tool_call" if "<tool_call>" in answer else "text"
        results.append(row_dict)

    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"Done. Wrote {len(results)} rows to {out_csv}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--benchmarks", nargs="+", choices=list(IN_CSVS), default=list(IN_CSVS))
    args = ap.parse_args()

    from transformers import Qwen3OmniMoeProcessor

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_seqs=8,
        max_model_len=32768,
        seed=1234,
    )
    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=16384)
    processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_path)

    for benchmark in args.benchmarks:
        in_csv = IN_CSVS[benchmark]
        if not os.path.exists(in_csv):
            print(f"[WARN] {in_csv} not found, skipping {benchmark}", flush=True)
            continue
        process_csv(llm, processor, sampling_params, benchmark, in_csv)


if __name__ == "__main__":
    main()
