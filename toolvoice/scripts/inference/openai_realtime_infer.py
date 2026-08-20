#!/usr/bin/env python3
"""OpenAI Realtime (WebSocket) inference for Confetti / When2Call.

For Confetti, captures raw tool-call intent (model_intent / assistant_text / tool_calls).
For When2Call, classifies the response into TOOL_CALL / FOLLOW_UP_QUESTION / CANNOT_ANSWER /
DIRECT_ANSWER by asking the model to emit a JSON {"Type", "Response"} object.

Tool schema cells accept: a JSON array of tool schemas, a single JSON tool schema,
a JSON/comma-separated list of tool names, a Python-literal list of JSON strings
(When2Call's format), or a file reference "@path/to/tools.json".

Uses the GA Realtime protocol (`session.type`, `output_modalities`, 24kHz-minimum audio
input) for both gpt-realtime-1.5 and gpt-realtime-2; `--realtime-version` just picks the
default model name.

Usage:
    export OPENAI_API_KEY=sk-...
    python openai_realtime_infer.py --benchmark confetti --realtime-version 1.5
    python openai_realtime_infer.py --benchmark when2call --realtime-version 2
"""
import argparse
import ast
import asyncio
import base64
import csv
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import websockets

CHUNK_BYTES = 32000
MIN_AUDIO_BYTES = 9600  # 200 ms of 24kHz mono PCM16, conservative Realtime minimum

DEFAULT_MODEL = {"1.5": "gpt-realtime-1.5", "2": "gpt-realtime-2"}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CONFETTI_INSTRUCTIONS = (
    "You are a helpful assistant. You will be given a conversation context "
    "in text format and an audio input from the user. "
    "You are also provided with a list of tools that you can leverage to answer the user query. "
    "If a tool is appropriate, return a tool call with clear JSON arguments. If not, answer in plain text."
)

WHEN2CALL_INSTRUCTIONS = (
    "You are an AI assistant that helps users decide when to call tools. "
    "You will receive an audio input from the user and a list of available tools. "
    "Your task is to determine the appropriate action:\n\n"
    "1. TOOL_CALL: If the user's request can be fulfilled by calling one of the available tools, "
    "call the tool with appropriate arguments.\n\n"
    "2. FOLLOW_UP_QUESTION: If you need more information to fulfill the request or determine "
    "which tool to call, ask a clarifying follow-up question.\n\n"
    "Respond naturally based on the user's audio input."
    "3. CANNOT_ANSWER: If the user's request "
    "cannot be answered based on the information available, just inform the user that you do not know the answer.\n\n"
    "**General Rule**: If no tool is provided as context, do not make any tool calls."
    "Please provide your response in JSON format with following keys: (i) Type, (ii) Response."
    "The value of Type can be one of the following:"
    "(i) TOOL_CALL, (ii) FOLLOW_UP_QUESTION, (iii) CANNOT_ANSWER."
)


def read_json_from_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_tool_cell(cell: str) -> Union[List[Any], Dict[str, Any], List[str]]:
    if not cell:
        return []
    s = cell.strip()

    if s.startswith("@"):
        return read_json_from_file(s[1:].strip())

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
            try:
                return [json.loads(item) for item in parsed]
            except Exception:
                return parsed
        return parsed
    except Exception:
        pass

    return [x.strip() for x in s.split(",") if x.strip()]


def sanitize_tool_name(name: str) -> str:
    """OpenAI Realtime tool names must match ^[a-zA-Z0-9_-]+$."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def normalize_to_function_tools(raw_tools: Union[List[Any], Dict[str, Any], List[str]]) -> List[Dict[str, Any]]:
    def fix_types(schema: Any) -> Any:
        if isinstance(schema, dict):
            type_mapping = {"dict": "object", "list": "array", "float": "number", "int": "integer", "str": "string", "bool": "boolean"}
            current_type = schema.get("type")
            if current_type == "any":
                schema.pop("type", None)
            elif current_type in type_mapping:
                schema["type"] = type_mapping[current_type]
            if "properties" in schema:
                for prop_schema in schema["properties"].values():
                    fix_types(prop_schema)
            if "items" in schema:
                fix_types(schema["items"])
        return schema

    def coerce_schema(d: Dict[str, Any]) -> Dict[str, Any]:
        name = d.get("name")
        if not name or not isinstance(name, str):
            raise ValueError(f"Tool schema is missing a valid 'name': {d}")
        sanitized_name = sanitize_tool_name(name)

        params = d.get("parameters", {"type": "object"})
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        if "type" not in params:
            params["type"] = "object"
        if "properties" not in params:
            params["properties"] = {}
        fix_types(params)

        description = d.get("description", "") or ""
        if name != sanitized_name and not description:
            description = f"Original name: {name}"

        return {"type": "function", "name": sanitized_name, "description": description, "parameters": params}

    if isinstance(raw_tools, dict):
        return [coerce_schema(raw_tools)]

    if isinstance(raw_tools, list) and raw_tools:
        if all(isinstance(x, str) for x in raw_tools):
            return [
                {"type": "function", "name": sanitize_tool_name(name), "description": "", "parameters": {"type": "object", "properties": {}}}
                for name in raw_tools if name
            ]
        tools = []
        for item in raw_tools:
            if isinstance(item, dict):
                tools.append(coerce_schema(item))
            elif isinstance(item, str):
                tools.append({"type": "function", "name": sanitize_tool_name(item), "description": "", "parameters": {"type": "object", "properties": {}}})
            else:
                raise ValueError(f"Unsupported tool entry: {item!r}")
        return tools

    return []


def yield_pcm16_chunks(data: bytes, chunk_bytes: int = CHUNK_BYTES):
    if len(data) < MIN_AUDIO_BYTES:
        data += bytes(MIN_AUDIO_BYTES - len(data))
    for i in range(0, len(data), chunk_bytes):
        yield base64.b64encode(data[i:i + chunk_bytes]).decode("ascii")


def read_pcm16_chunks(path: str, chunk_bytes: int = CHUNK_BYTES):
    """The GA Realtime protocol requires input rate >= 24000; always decode via ffmpeg."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    if os.path.getsize(path) < 12:
        raise ValueError(f"File too small to be valid audio: {path}")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "24000", "-"],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise ValueError(f"Error decoding audio '{path}': {e}")
    yield from yield_pcm16_chunks(proc.stdout, chunk_bytes)


class ToolCallAssembler:
    """Collect streamed tool-call names and JSON-arg strings."""

    def __init__(self):
        self.names: Dict[str, str] = {}
        self.buffers: Dict[str, str] = {}

    def set_name(self, call_id: str, name: str):
        self.names[call_id] = name

    def add_delta(self, call_id: str, delta: str):
        self.buffers.setdefault(call_id, "")
        self.buffers[call_id] += delta

    def finalize(self, call_id: str) -> Tuple[str, str]:
        name = self.names.get(call_id, "")
        args_json = self.buffers.get(call_id, "") or "{}"
        self.names.pop(call_id, None)
        self.buffers.pop(call_id, None)
        return name, args_json


async def run_realtime_session(ws_url, headers, chunk_reader, audio_path, tool_cell, instructions, context_text=None):
    """Runs one Realtime turn; returns (assistant_text, tool_calls)."""
    tools = normalize_to_function_tools(parse_tool_cell(tool_cell))
    assistant_chunks: List[str] = []
    tool_calls: List[Dict[str, str]] = []
    assembler = ToolCallAssembler()

    # GA protocol: session needs a `type`, text output is `output_modalities`,
    # and audio settings (incl. turn_detection) live under `audio.input`.
    session = {
        "type": "realtime",
        "instructions": instructions,
        "output_modalities": ["text"],
        "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None}},
        "tools": tools,
    }

    async with websockets.connect(ws_url, additional_headers=headers, max_size=20_000_000) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": session}))

        if context_text and context_text.strip():
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": context_text.strip()}]},
            }))

        for b64 in chunk_reader(audio_path):
            await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await ws.send(json.dumps({"type": "response.create"}))

        async for raw in ws:
            evt = json.loads(raw)
            et = evt.get("type")

            if et == "error":
                raise RuntimeError(f"Realtime error: {evt}")
            if et == "response.output_text.delta":
                assistant_chunks.append(evt.get("delta", ""))
                continue
            if et == "response.output_item.added":
                item = evt.get("item", {})
                if item.get("type") == "function_call":
                    call_id, name = item.get("id"), item.get("name", "")
                    if call_id and name:
                        assembler.set_name(call_id, name)
                continue
            if et == "response.function_call_arguments.delta":
                call_id = evt.get("item_id")
                if call_id:
                    assembler.add_delta(call_id, evt.get("delta", ""))
                continue
            if et == "response.function_call_arguments.done":
                call_id = evt.get("item_id")
                if not call_id:
                    continue
                name, args_json = assembler.finalize(call_id)
                tool_calls.append({"id": call_id, "name": name, "args_json": args_json})
                continue
            if et == "response.done":
                break

        await ws.close()

    return "".join(assistant_chunks).strip(), tool_calls


async def run_row_confetti(ws_url, headers, chunk_reader, audio_path, context_text, tool_cell) -> Dict[str, Any]:
    assistant_text, tool_calls = await run_realtime_session(ws_url, headers, chunk_reader, audio_path, tool_cell, CONFETTI_INSTRUCTIONS, context_text)
    return {"model_intent": "tool_call" if tool_calls else "text", "assistant_text": assistant_text, "tool_calls": tool_calls}


async def run_row_when2call(ws_url, headers, chunk_reader, audio_path, tool_cell) -> Dict[str, Any]:
    assistant_text, tool_calls = await run_realtime_session(ws_url, headers, chunk_reader, audio_path, tool_cell, WHEN2CALL_INSTRUCTIONS)

    if tool_calls:
        return {"response_type": "tool_call", "response": json.dumps(tool_calls, ensure_ascii=False)}

    cleaned = assistant_text.replace("```", "").replace("json", "")
    try:
        parsed = json.loads(cleaned)
        return {"response_type": parsed.get("Type"), "response": parsed.get("Response")}
    except Exception:
        return {"response_type": "DIRECT_ANSWER", "response": assistant_text}


def write_results_csv(out_csv: str, out_fields: List[str], results: List[Dict[str, Any]]) -> None:
    tmp_csv = f"{out_csv}.tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for row in results:
            w.writerow(row)
    os.replace(tmp_csv, out_csv)


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", choices=["confetti", "when2call"], required=True)
    ap.add_argument("--realtime-version", choices=["1.5", "2"], default="1.5")
    ap.add_argument("--model", default=None, help=f"Overrides the default model for the chosen realtime version ({DEFAULT_MODEL}).")
    args = ap.parse_args()

    model_name = args.model or os.environ.get("OPENAI_REALTIME_MODEL", DEFAULT_MODEL[args.realtime_version])
    ws_url = f"wss://api.openai.com/v1/realtime?model={model_name}"
    headers = [("Authorization", f"Bearer {os.environ.get('OPENAI_API_KEY', '')}")]
    chunk_reader = read_pcm16_chunks

    tts_types = ["gemini-2.5-flash-tts", "gemini-2.5-pro-tts", "gpt-4o-mini-tts"]
    voice_types = ["kore", "orus", "ash", "coral"]
    output_dir = DATA_DIR.parent / "model_responses" / f"{model_name}_responses"
    os.makedirs(output_dir, exist_ok=True)

    for tts_type in tts_types:
        for voice_type in voice_types:
            if voice_type in ("kore", "orus") and tts_type == "gpt-4o-mini-tts":
                continue
            if voice_type in ("ash", "coral") and tts_type != "gpt-4o-mini-tts":
                continue

            if args.benchmark == "confetti":
                in_csv = DATA_DIR / "confetti" / "confetti_benchmark.csv"
                out_csv = f"{output_dir}/BFCL_v2_conversations_clean_with_context_{model_name}_{tts_type}_{voice_type}.csv"
                id_col, extra_fields = "id", ["assistant_text", "tool_calls", "model_intent"]
            else:
                in_csv = DATA_DIR / "when2call" / "when2call_benchmark.csv"
                out_csv = f"{output_dir}/when2call_{model_name}_final_{tts_type}_{voice_type}.csv"
                id_col, extra_fields = "id", ["final_response_type", "final_response", "response_type", "response"]

            df = pd.read_csv(in_csv)
            out_fields = list(df.columns) + extra_fields
            if os.path.exists(out_csv):
                existing_df = pd.read_csv(out_csv)
                results = existing_df.to_dict("records")
                completed_ids = set(existing_df.get(id_col, pd.Series(dtype=str)).astype(str))
                print(f"[RESUME] {out_csv}: loaded {len(results)} existing rows")
            else:
                results, completed_ids = [], set()

            for idx, r in df.iterrows():
                row_id = str(r.get(id_col, idx))
                if row_id in completed_ids:
                    continue
                audio_path = str(DATA_DIR / args.benchmark / "audio" / tts_type / voice_type / f"output_audio{idx}.wav")
                if not os.path.exists(audio_path):
                    continue
                tool_cell = r.get("tools", "") or ""

                try:
                    if args.benchmark == "confetti":
                        res = await run_row_confetti(ws_url, headers, chunk_reader, audio_path, r.get("context") or "", tool_cell)
                        update = {"assistant_text": res["assistant_text"], "tool_calls": json.dumps(res["tool_calls"], ensure_ascii=False), "model_intent": res["model_intent"]}
                    else:
                        res = await run_row_when2call(ws_url, headers, chunk_reader, audio_path, tool_cell)
                        update = {"final_response_type": res["response_type"], "final_response": res["response"], "response_type": res["response_type"], "response": res["response"]}
                    print(idx, res)
                    row_dict = r.to_dict()
                    row_dict.update(update)
                    results.append(row_dict)
                    completed_ids.add(row_id)
                    write_results_csv(out_csv, out_fields, results)
                except Exception as e:
                    print(f"[{idx}] ERROR: {e}")
                    error_text = str(e).lower()
                    if any(s in error_text for s in ("realtime error", "create_connection", "invalid_model", "not supported in realtime mode")):
                        raise
                    err = json.dumps({"error": str(e)}, ensure_ascii=False)
                    update = {"assistant_text": "", "tool_calls": err, "model_intent": "error"} if args.benchmark == "confetti" else {"final_response_type": "error", "final_response": err, "response_type": "error", "response": err}
                    row_dict = r.to_dict()
                    row_dict.update(update)
                    results.append(row_dict)
                    completed_ids.add(row_id)
                    write_results_csv(out_csv, out_fields, results)

            print(f"Done. Wrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    asyncio.run(main())
