#!/usr/bin/env python3
"""ASR-cascade text-only runner: sends each transcribed question + context + tools to
OpenAI Realtime in text mode (no audio) and captures tool-call intent, so the transcript
can be scored by evaluate_confetti.py the same way as the direct-voice inference scripts.

Reads the ASR CSVs produced by transcribe_audio.py + enrich_asr_csvs.py (see
cascade_asr/README.md) and writes `<input>_<model>_text_only.csv`.

Tool schema cells accept: a JSON array of tool schemas, a single JSON tool schema,
a JSON/comma-separated list of tool names, or a file reference "@path/to/tools.json".

Usage:
  export OPENAI_API_KEY=sk-...
  python asr_cascade_openai_confetti.py
"""

import os
import csv
import json
import ast
import asyncio
from typing import Any, Dict, List, Tuple, Union

import websockets
import pandas as pd

# ---------------- Config ----------------
OPENAI_REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
OPENAI_WS_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_REALTIME_MODEL}"
HEADERS = (("Authorization", f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"),)


# -------------- I/O helpers --------------

def read_json_from_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_tool_cell(cell: str) -> Union[List[Any], Dict[str, Any], List[str]]:
    """
    Returns either:
    - list of tool schemas (dicts)
    - single tool schema (dict)
    - list of tool names (strings)

    Accepts:
    - JSON array (schemas or names)
    - JSON object (single schema)
    - Python literal (ast.literal_eval)
    - comma-separated names
    - file reference "@path/to/tools.json"
    - list whose elements are themselves JSON-encoded schema strings
    """
    if not cell:
        return []
    s = cell.strip()

    if s.startswith("@"):
        return read_json_from_file(s[1:].strip())

    parsed = None
    try:
        parsed = json.loads(s)
    except Exception:
        pass

    if parsed is None:
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            pass

    if parsed is None:
        return [x.strip() for x in s.split(",") if x.strip()]

    if isinstance(parsed, list) and parsed and all(isinstance(x, str) for x in parsed):
        decoded = []
        all_dicts = True
        for x in parsed:
            x_stripped = x.strip()
            try:
                inner = json.loads(x_stripped)
            except Exception:
                try:
                    inner = ast.literal_eval(x_stripped)
                except Exception:
                    inner = x
            decoded.append(inner)
            if not isinstance(inner, dict):
                all_dicts = False
        if all_dicts:
            return decoded

    return parsed


def normalize_schema_types(schema: Any) -> Any:
    """Translate Python-style type strings (dict/list/str/...) to JSON schema types."""
    type_mapping = {
        "dict": "object",
        "list": "array",
        "float": "number",
        "int": "integer",
        "str": "string",
        "bool": "boolean",
    }
    if isinstance(schema, dict):
        t = schema.get("type")
        if t == "any":
            schema.pop("type", None)
        elif isinstance(t, str) and t in type_mapping:
            schema["type"] = type_mapping[t]
        if "properties" in schema and isinstance(schema["properties"], dict):
            for v in schema["properties"].values():
                normalize_schema_types(v)
        if "items" in schema:
            normalize_schema_types(schema["items"])
    elif isinstance(schema, list):
        for v in schema:
            normalize_schema_types(v)
    return schema


def sanitize_function_name(name: str) -> str:
    """OpenAI Realtime requires names matching ^[A-Za-z][A-Za-z0-9_-]*$."""
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out) or "tool"
    if not s[0].isalpha():
        s = "t_" + s
    return s


def normalize_to_function_tools(raw_tools: Union[List[Any], Dict[str, Any], List[str]]) -> List[Dict[str, Any]]:
    """
    Turn input into a list of OpenAI Realtime function tools:
    {
      "type": "function",
      "name": str,
      "description": str,
      "parameters": {...}
    }

    Rules:
    - If list of dicts: coerce each to a function tool
    - If single dict: wrap and coerce
    - If list of strings (names): produce permissive schemas for each
    - Add defaults if fields missing:
        description: ""
        parameters: {"type": "object", "additionalProperties": True}
    """

    def coerce_schema(d: Dict[str, Any]) -> Dict[str, Any]:
        raw_name = d.get("name")
        if not raw_name or not isinstance(raw_name, str):
            raise ValueError(f"Tool schema is missing a valid 'name': {d}")

        params = d.get("parameters", {"type": "object"})
        if not isinstance(params, dict):
            params = {"type": "object"}
        normalize_schema_types(params)
        if "type" not in params:
            params["type"] = "object"

        # Let arbitrary fields through when the author didn't specify properties
        if "properties" not in params:
            params.setdefault("additionalProperties", True)

        tool = {
            "type": "function",
            "name": sanitize_function_name(raw_name),
            "description": d.get("description", "") or "",
            "parameters": params,
        }
        return tool

    # Single dict -> list
    if isinstance(raw_tools, dict):
        return [coerce_schema(raw_tools)]

    # List input
    if isinstance(raw_tools, list) and raw_tools:
        # Is this a list of names?
        if all(isinstance(x, str) for x in raw_tools):
            tools = []
            for nm in raw_tools:
                if not nm:
                    continue
                tools.append({
                    "type": "function",
                    "name": sanitize_function_name(nm),
                    "description": "",
                    "parameters": {"type": "object", "additionalProperties": True},
                })
            return tools

        # Otherwise assume list of dict-like schemas
        tools = []
        for item in raw_tools:
            if isinstance(item, dict):
                tools.append(coerce_schema(item))
            elif isinstance(item, str):
                tools.append({
                    "type": "function",
                    "name": sanitize_function_name(item),
                    "description": "",
                    "parameters": {"type": "object", "additionalProperties": True},
                })
            else:
                raise ValueError(f"Unsupported tool entry: {item!r}")
        return tools

    # Empty or unrecognized -> no tools
    return []


# -------------- Tool-call capture --------------

class ToolCallAssembler:
    """Collect streamed tool-call names and JSON-arg strings."""

    def __init__(self):
        self.names: Dict[str, str] = {}   # id -> name
        self.buffers: Dict[str, str] = {} # id -> arg json

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


async def run_row(question_text: str, context_text: str, tool_cell: str) -> Dict[str, Any]:
    """
    Returns:
    {
      "model_intent": "tool_call" | "text",
      "assistant_text": str,
      "tool_calls": [
        {
          "id": str,
          "name": str,
          "args_json": str
        }
      ]
    }

    Sends context + question as a user text message to Realtime.
    """
    # Parse and normalize tools
    raw_tools = parse_tool_cell(tool_cell)
    tools = normalize_to_function_tools(raw_tools)

    assistant_chunks: List[str] = []
    tool_calls: List[Dict[str, str]] = []
    assembler = ToolCallAssembler()

    # Build user message from context + question
    parts = []
    if context_text and context_text.strip():
        parts.append("Conversation context:\n" + context_text.strip())
    if question_text and question_text.strip():
        parts.append("User question:\n" + question_text.strip())
    user_text = "\n\n".join(parts) if parts else ""

    instructions = (
        "You are a helpful assistant. You will be given a conversation context "
        "and a user question in text format. "
        "You are also provided with a list of tools that you can leverage to answer the user query. "
        "If a tool is appropriate, return a tool call with "
        "clear JSON arguments. If not, answer in plain text."
    )
    session = {"type": "realtime", "instructions": instructions, "output_modalities": ["text"], "tools": tools}

    async with websockets.connect(OPENAI_WS_URL, additional_headers=HEADERS, max_size=20_000_000) as ws:
        # 1) Configure session
        await ws.send(json.dumps({"type": "session.update", "session": session}))

        # 2) Add user message (context + question)
        if user_text:
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            }))

        # 3) Ask for a response
        await ws.send(json.dumps({"type": "response.create"}))

        # 4) Capture either tool-calls or text
        async for raw in ws:
            evt = json.loads(raw)
            et = evt.get("type")

            if et == "error":
                raise RuntimeError(f"Realtime error: {evt}")

            # Text response delta
            if et == "response.output_text.delta":
                assistant_chunks.append(evt.get("delta", ""))
                continue

            # Function call detected via output_item.added
            if et == "response.output_item.added":
                item = evt.get("item", {})
                if item.get("type") == "function_call":
                    call_id = item.get("id")
                    name = item.get("name", "")
                    if call_id and name:
                        assembler.set_name(call_id, name)
                continue

            # Function call arguments delta
            if et == "response.function_call_arguments.delta":
                call_id = evt.get("item_id")
                delta = evt.get("delta", "")
                if call_id:
                    assembler.add_delta(call_id, delta)
                continue

            # Function call arguments done
            if et == "response.function_call_arguments.done":
                call_id = evt.get("item_id")
                if not call_id:
                    continue
                name, args_json = assembler.finalize(call_id)
                tool_calls.append({"id": call_id, "name": name, "args_json": args_json})
                # DO NOT send tool.output. We're only capturing intent.
                continue

            # Response complete
            if et == "response.done":
                break

        await ws.close()

    return {
        "model_intent": "tool_call" if tool_calls else "text",
        "assistant_text": "".join(assistant_chunks).strip(),
        "tool_calls": tool_calls
    }


# -------------- CSV orchestration --------------

def _write_csv(out_csv: str, out_fields: List[str], results: List[Dict[str, Any]]) -> None:
    tmp = f"{out_csv}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for row in results:
            w.writerow(row)
    os.replace(tmp, out_csv)


async def process_csv(in_csv: str) -> None:
    base = os.path.splitext(os.path.basename(in_csv))[0]
    out_csv = f"{base}_{OPENAI_REALTIME_MODEL}_text_only.csv"

    df = pd.read_csv(in_csv)
    df = df[df["audio_condition"] == "clean"].reset_index(drop=True)
    print(f"[{in_csv}] rows after audio_condition='clean' filter: {len(df)}", flush=True)

    def _coerce(v):
        if v is None:
            return ""
        if isinstance(v, float) and pd.isna(v):
            return ""
        return str(v)

    out_fields = list(df.columns) + ["assistant_text", "tool_calls", "model_intent"]
    id_col = "relative_path" if "relative_path" in df.columns else "filename"

    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        results: List[Dict[str, Any]] = existing.to_dict("records")
        completed_ids = set(existing.get(id_col, pd.Series(dtype=str)).astype(str))
        print(f"[RESUME] {out_csv}: loaded {len(results)} existing rows", flush=True)
    else:
        results = []
        completed_ids = set()

    for idx, r in df.iterrows():
        row_id = str(r.get(id_col, idx))
        if row_id in completed_ids:
            continue

        question = _coerce(r.get("transcription_text"))
        context = _coerce(r.get("context"))
        tool_cell = _coerce(r.get("tool"))

        print(f"[{idx}] question={str(question)[:80].replace(chr(10), ' ')}{'...' if len(str(question)) > 80 else ''}", flush=True)

        res = None
        last_err = None
        for attempt in range(3):
            try:
                res = await run_row(question, context, tool_cell)
                break
            except Exception as e:
                last_err = e
                print(f"[{idx}] attempt {attempt + 1} failed: {e}", flush=True)
                await asyncio.sleep(2 * (attempt + 1))

        row_dict = r.to_dict()
        if res is not None:
            print(res, flush=True)
            row_dict.update({
                "assistant_text": res["assistant_text"],
                "tool_calls": json.dumps(res["tool_calls"], ensure_ascii=False),
                "model_intent": res["model_intent"],
            })
        else:
            print(f"[{idx}] ERROR after retries: {last_err}", flush=True)
            row_dict.update({
                "assistant_text": "",
                "tool_calls": json.dumps({"error": str(last_err)}, ensure_ascii=False),
                "model_intent": "error",
            })
        results.append(row_dict)
        completed_ids.add(row_id)
        _write_csv(out_csv, out_fields, results)

    print(f"Done. Wrote {len(results)} rows to {out_csv}", flush=True)


async def main():
    in_csvs = [
        "GPT-4o-Mini-STT-Confetti-GPT.csv",
        "GPT-4o-Mini-STT-When2Call-GPT.csv",
    ]
    for in_csv in in_csvs:
        await process_csv(in_csv)


if __name__ == "__main__":
    asyncio.run(main())
