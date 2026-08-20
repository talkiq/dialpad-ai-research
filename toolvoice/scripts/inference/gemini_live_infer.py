#!/usr/bin/env python3
"""Gemini Live (audio -> text + tool intent) inference for Confetti / When2Call.

For Confetti, captures raw tool-call intent (model_intent / assistant_text / tool_calls).
For When2Call, classifies the response into TOOL_CALL / FOLLOW_UP_QUESTION / CANNOT_ANSWER /
DIRECT_ANSWER by asking the model to emit a JSON {"Type", "Response"} object.

Tool schema cells accept: a JSON array of tool schemas, a single JSON tool schema,
a JSON/comma-separated list of tool names, a Python-literal list of JSON strings
(When2Call's format), or a file reference "@path/to/tools.json".

Requirements:
    pip install google-genai pandas
    ffmpeg must be available on PATH

Env:
    export GEMINI_API_KEY=...   (or GOOGLE_API_KEY)

Usage:
    python gemini_live_infer.py --benchmark confetti
    python gemini_live_infer.py --benchmark when2call
"""
import argparse
import ast
import asyncio
import csv
import json
import os
import re
import subprocess
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

warnings.filterwarnings("ignore")

import pandas as pd
from google import genai
from google.genai import types

GEMINI_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CONFETTI_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. You will be given a conversation context "
    "in text format and an audio input from the user. "
    "You are also provided with a list of tools that you can leverage to "
    "answer the user query. If a tool is appropriate, respond by issuing "
    "one or more function calls with clear JSON arguments. "
    "If no tools are appropriate, answer in plain text only."
)

WHEN2CALL_SYSTEM_INSTRUCTION = (
    "You are an AI assistant that helps users decide when to call tools. "
    "You will receive an audio input from the user and a list of available tools. "
    "Your task is to determine the appropriate action:\n\n"
    "1. TOOL_CALL: If the user's request can be fulfilled by calling one of the available tools, "
    "call the tool with appropriate arguments.\n\n"
    "2. FOLLOW_UP_QUESTION: If you need more information to fulfill the request or determine "
    "which tool to call, ask a clarifying follow-up question.\n\n"
    "3. CANNOT_ANSWER: If the user's request cannot be answered based on the information available, "
    "inform the user that you do not know the answer.\n\n"
    "General rule: If no tools are provided, do not make any tool calls.\n\n"
    "IMPORTANT OUTPUT FORMAT:\n"
    "Always respond with a single JSON object with the keys:\n"
    "  - \"Type\": one of [\"TOOL_CALL\", \"FOLLOW_UP_QUESTION\", \"CANNOT_ANSWER\", \"DIRECT_ANSWER\"]\n"
    "  - \"Response\": your natural language response or a brief description.\n"
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
        # A list of JSON strings (When2Call's tools-column format)
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
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def normalize_to_gemini_tools(raw_tools: Union[List[Any], Dict[str, Any], List[str]]) -> List[Dict[str, Any]]:
    """Coerce arbitrary tool-schema input into Gemini function_declarations."""

    def fix_types(schema: Any) -> Any:
        if isinstance(schema, dict):
            type_mapping = {"dict": "object", "list": "array", "float": "number", "int": "integer", "str": "string", "bool": "boolean"}
            current_type = schema.get("type")
            if current_type == "any":
                schema.pop("type", None)
            elif current_type in type_mapping:
                schema["type"] = type_mapping[current_type]
            if "enum" in schema:
                del schema["enum"]
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

        return {"name": sanitized_name, "description": description, "parameters": params}

    if isinstance(raw_tools, dict):
        return [coerce_schema(raw_tools)]

    if isinstance(raw_tools, list) and raw_tools:
        if all(isinstance(x, str) for x in raw_tools):
            return [
                {"name": sanitize_tool_name(name), "description": "", "parameters": {"type": "object", "properties": {}}}
                for name in raw_tools if name
            ]
        tools = []
        for item in raw_tools:
            if isinstance(item, dict):
                tools.append(coerce_schema(item))
            elif isinstance(item, str):
                tools.append({"name": sanitize_tool_name(item), "description": "", "parameters": {"type": "object", "properties": {}}})
            else:
                raise ValueError(f"Unsupported tool entry: {item!r}")
        return tools

    return []


def read_audio_as_pcm16(path: str) -> bytes:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        proc = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", "-"],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise ValueError(f"Error processing audio file '{path}': {e}") from e
    return proc.stdout


async def run_live_session(client, audio_path, tool_cell, system_instruction, context_text=None):
    """Runs one Gemini Live turn; returns (assistant_text, tool_calls)."""
    function_decls = normalize_to_gemini_tools(parse_tool_cell(tool_cell))
    tools_config = [types.Tool(function_declarations=function_decls)] if function_decls else None

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=system_instruction,
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=tools_config,
    )

    assistant_chunks, tool_calls = [], []

    async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
        if context_text and context_text.strip():
            await session.send_client_content(
                turns=types.Content(role="user", parts=[types.Part(text=f"Context: {context_text.strip()}")]),
                turn_complete=False,
            )

        audio_bytes = read_audio_as_pcm16(audio_path)
        await session.send_realtime_input(audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=16000"))
        # 1.5s of silence to trigger VAD end-of-speech detection (16kHz * 2 bytes * 1.5s)
        await session.send_realtime_input(audio=types.Blob(data=bytes(48000), mime_type="audio/pcm;rate=16000"))
        await asyncio.sleep(1)

        try:
            async with asyncio.timeout(30):
                async for response in session.receive():
                    if response.text:
                        assistant_chunks.append(response.text)
                    sc = getattr(response, "server_content", None)
                    if sc:
                        out_trans = getattr(sc, "output_transcription", None)
                        if out_trans and getattr(out_trans, "text", None):
                            assistant_chunks.append(out_trans.text)
                    if getattr(response, "tool_call", None):
                        for fc in response.tool_call.function_calls or []:
                            tool_calls.append({"id": fc.id or "", "name": fc.name or "", "args_json": json.dumps(fc.args or {}, ensure_ascii=False)})
                        break
                    if sc and getattr(sc, "turn_complete", False):
                        break
        except asyncio.TimeoutError:
            pass

    return "".join(assistant_chunks).strip(), tool_calls


async def run_row_confetti(client, audio_path, context_text, tool_cell) -> Dict[str, Any]:
    assistant_text, tool_calls = await run_live_session(client, audio_path, tool_cell, CONFETTI_SYSTEM_INSTRUCTION, context_text)
    return {"model_intent": "tool_call" if tool_calls else "text", "assistant_text": assistant_text, "tool_calls": tool_calls}


async def run_row_when2call(client, audio_path, tool_cell) -> Dict[str, Any]:
    assistant_text, tool_calls = await run_live_session(client, audio_path, tool_cell, WHEN2CALL_SYSTEM_INSTRUCTION)

    if tool_calls:
        return {"response_type": "tool_call", "response": json.dumps(tool_calls, ensure_ascii=False)}
    if not assistant_text:
        return {"response_type": "error", "response": json.dumps({"error": "Empty response from model"}, ensure_ascii=False)}

    cleaned = assistant_text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
        return {"response_type": parsed.get("Type", "DIRECT_ANSWER"), "response": parsed.get("Response", "")}
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
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set. Get an API key from Google AI Studio and export GEMINI_API_KEY.")
    if os.environ.get("GEMINI_API_KEY"):
        os.environ.pop("GOOGLE_API_KEY", None)
    client = genai.Client(api_key=api_key)

    tts_types = ["gemini-2.5-flash-tts", "gemini-2.5-pro-tts", "gpt-4o-mini-tts"]
    voice_types = ["kore", "orus", "ash", "coral"]
    output_dir = DATA_DIR.parent / "model_responses" / f"{GEMINI_MODEL}_responses"
    os.makedirs(output_dir, exist_ok=True)

    for tts_type in tts_types:
        for voice_type in voice_types:
            if voice_type in ("kore", "orus") and tts_type == "gpt-4o-mini-tts":
                continue
            if voice_type in ("ash", "coral") and tts_type != "gpt-4o-mini-tts":
                continue

            if args.benchmark == "confetti":
                in_csv = DATA_DIR / "confetti" / "confetti_benchmark.csv"
                out_csv = f"{output_dir}/BFCL_v2_conversations_clean_with_context_{GEMINI_MODEL}_{tts_type}_{voice_type}.csv"
                id_col = "id"
                extra_fields = ["assistant_text", "tool_calls", "model_intent"]
            else:
                in_csv = DATA_DIR / "when2call" / "when2call_benchmark.csv"
                out_csv = f"{output_dir}/when2call_{GEMINI_MODEL}_final_{tts_type}_{voice_type}.csv"
                id_col = "id"
                extra_fields = ["response_type", "response", "final_response_type", "final_response"]

            if not os.path.exists(in_csv):
                print(f"[WARN] Input CSV not found, skipping: {in_csv}")
                continue

            df = pd.read_csv(in_csv)
            out_fields = list(df.columns) + extra_fields
            if os.path.exists(out_csv):
                existing_df = pd.read_csv(out_csv)
                results: List[Dict[str, Any]] = existing_df.to_dict("records")
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

                try:
                    if args.benchmark == "confetti":
                        res = await run_row_confetti(client, audio_path, r.get("context") or "", r.get("tools") or "")
                        update = {"assistant_text": res["assistant_text"], "tool_calls": json.dumps(res["tool_calls"], ensure_ascii=False), "model_intent": res["model_intent"]}
                    else:
                        res = await run_row_when2call(client, audio_path, r.get("tools", "") or "")
                        update = {"response_type": res["response_type"], "response": res["response"], "final_response_type": res["response_type"], "final_response": res["response"]}
                    print(idx, res)
                    row_dict = r.to_dict()
                    row_dict.update(update)
                    results.append(row_dict)
                    completed_ids.add(row_id)
                    write_results_csv(out_csv, out_fields, results)
                except Exception as e:
                    print(f"[{idx}] ERROR: {e}")
                    if "model" in str(e).lower() or "not found" in str(e).lower():
                        raise
                    err = json.dumps({"error": str(e)}, ensure_ascii=False)
                    update = {"assistant_text": "", "tool_calls": err, "model_intent": "error"} if args.benchmark == "confetti" else {"response_type": "error", "response": err, "final_response_type": "error", "final_response": err}
                    row_dict = r.to_dict()
                    row_dict.update(update)
                    results.append(row_dict)
                    completed_ids.add(row_id)
                    write_results_csv(out_csv, out_fields, results)

            print(f"Done. Wrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    asyncio.run(main())
