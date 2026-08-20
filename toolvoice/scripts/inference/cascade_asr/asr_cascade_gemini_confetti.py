#!/usr/bin/env python3
"""
multimodal_gemini_confetti.py (text-only)

Per-row Gemini Live runner for the Confetti / BFCL_v2 dataset using TEXT input only.
Mirrors text-only/multimodal_openai_confetti.py:
- Reads CSV with columns: question (text), context, tool
- Normalizes tools to Gemini FunctionDeclarations
- Sends context + text question to Gemini Live (response_modalities=["TEXT"])
- DOES NOT execute tools; only captures tool-call intent
- Writes output CSV with assistant_text, tool_calls (JSON), model_intent

Env:
  export GEMINI_API_KEY=...

Usage:
  python multimodal_gemini_confetti.py
"""

import os
import csv
import json
import ast
import asyncio
import warnings
from typing import Any, Dict, List, Union

warnings.filterwarnings("ignore")

import pandas as pd

from google import genai
from google.genai import types

# ---------------- Config ----------------

GEMINI_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant. You will be given a conversation context "
    "and a user question in text format. "
    "You are also provided with a list of tools that you can leverage to "
    "answer the user query. If a tool is appropriate, respond by issuing "
    "one or more function calls with clear JSON arguments. "
    "If no tools are appropriate, answer in plain text only."
)


# -------------- I/O helpers --------------

def read_json_from_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_tool_cell(cell: str) -> Union[List[Any], Dict[str, Any], List[str]]:
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

    # If list elements are themselves JSON-encoded schema strings, parse them.
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


def sanitize_function_name(name: str) -> str:
    """Gemini/OpenAI require names matching ^[A-Za-z][A-Za-z0-9_-]*$. Replace dots etc."""
    out = []
    for i, ch in enumerate(name):
        if ch.isalnum() or ch in ("_", "-"):
            out.append(ch)
        else:
            out.append("_")
    s = "".join(out) or "tool"
    if not s[0].isalpha():
        s = "t_" + s
    return s


def normalize_to_gemini_tools(raw_tools: Union[List[Any], Dict[str, Any], List[str]]) -> List[Dict[str, Any]]:
    """Turn input into a list of Gemini function declarations."""

    def fix_types(schema: Any) -> Any:
        if isinstance(schema, dict):
            type_mapping = {
                "dict": "object",
                "list": "array",
                "float": "number",
                "int": "integer",
                "str": "string",
                "bool": "boolean",
            }
            current_type = schema.get("type")
            if current_type == "any":
                schema.pop("type", None)
            elif current_type in type_mapping:
                schema["type"] = type_mapping[current_type]

            if "enum" in schema:
                del schema["enum"]

            if "properties" in schema:
                for _, prop_schema in schema["properties"].items():
                    fix_types(prop_schema)
            if "items" in schema:
                fix_types(schema["items"])
        return schema

    def coerce_schema(d: Dict[str, Any]) -> Dict[str, Any]:
        raw_name = d.get("name")
        if not raw_name or not isinstance(raw_name, str):
            raise ValueError(f"Tool schema is missing a valid 'name': {d}")

        params = d.get("parameters", {"type": "object"})
        if not isinstance(params, dict):
            params = {"type": "object"}
        if "type" not in params:
            params["type"] = "object"
        if "properties" not in params:
            params["properties"] = {}

        fix_types(params)

        return {
            "name": sanitize_function_name(raw_name),
            "description": d.get("description", "") or "",
            "parameters": params,
        }

    if isinstance(raw_tools, dict):
        return [coerce_schema(raw_tools)]

    if isinstance(raw_tools, list) and raw_tools:
        if all(isinstance(x, str) for x in raw_tools):
            tools = []
            for nm in raw_tools:
                if not nm:
                    continue
                tools.append({
                    "name": sanitize_function_name(nm),
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                })
            return tools

        tools = []
        for item in raw_tools:
            if isinstance(item, dict):
                tools.append(coerce_schema(item))
            elif isinstance(item, str):
                tools.append({
                    "name": sanitize_function_name(item),
                    "description": "",
                    "parameters": {"type": "object", "properties": {}},
                })
            else:
                raise ValueError(f"Unsupported tool entry: {item!r}")
        return tools

    return []


# -------------- Per-row Live call --------------

async def run_row(
    client: genai.Client,
    question_text: str,
    context_text: str,
    tool_cell: str,
) -> Dict[str, Any]:
    raw_tools = parse_tool_cell(tool_cell)
    function_decls = normalize_to_gemini_tools(raw_tools)

    tools_config = None
    if function_decls:
        tools_config = [types.Tool(function_declarations=function_decls)]

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=SYSTEM_INSTRUCTION,
        output_audio_transcription=types.AudioTranscriptionConfig(),
        tools=tools_config,
    )

    parts = []
    if context_text and context_text.strip():
        parts.append("Conversation context:\n" + context_text.strip())
    if question_text and question_text.strip():
        parts.append("User question:\n" + question_text.strip())
    user_text = "\n\n".join(parts) if parts else ""

    assistant_chunks: List[str] = []
    tool_calls: List[Dict[str, str]] = []

    async with client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
        await session.send_realtime_input(text=user_text or "")

        try:
            async with asyncio.timeout(60):
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
                            tool_calls.append(
                                {
                                    "id": fc.id or "",
                                    "name": fc.name or "",
                                    "args_json": json.dumps(fc.args or {}, ensure_ascii=False),
                                }
                            )
                        break

                    if sc and getattr(sc, "turn_complete", False):
                        break
        except asyncio.TimeoutError:
            pass

    return {
        "model_intent": "tool_call" if tool_calls else "text",
        "assistant_text": "".join(assistant_chunks).strip(),
        "tool_calls": tool_calls,
    }


# -------------- CSV orchestration --------------

def write_results_csv(out_csv: str, out_fields: List[str], results: List[Dict[str, Any]]) -> None:
    tmp_csv = f"{out_csv}.tmp"
    with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for row in results:
            w.writerow(row)
    os.replace(tmp_csv, out_csv)


async def process_csv(client: genai.Client, in_csv: str) -> None:
    base = os.path.splitext(os.path.basename(in_csv))[0]
    out_csv = f"{base}_{GEMINI_MODEL}_text_only.csv"

    df = pd.read_csv(in_csv)
    df = df[df["audio_condition"] == "clean"].reset_index(drop=True)
    print(f"[{in_csv}] rows after audio_condition='clean' filter: {len(df)}")

    out_fields = list(df.columns) + ["assistant_text", "tool_calls", "model_intent"]

    id_col = "relative_path" if "relative_path" in df.columns else "filename"

    if os.path.exists(out_csv):
        existing_df = pd.read_csv(out_csv)
        results: List[Dict[str, Any]] = existing_df.to_dict("records")
        completed_ids = set(existing_df.get(id_col, pd.Series(dtype=str)).astype(str))
        print(f"[RESUME] {out_csv}: loaded {len(results)} existing rows")
    else:
        results = []
        completed_ids = set()

    def _coerce(v):
        if v is None:
            return ""
        if isinstance(v, float) and pd.isna(v):
            return ""
        return str(v)

    for idx, r in df.iterrows():
        row_id = str(r.get(id_col, idx))
        if row_id in completed_ids:
            continue

        question = _coerce(r.get("transcription_text"))
        context = _coerce(r.get("context"))
        tool_cell = _coerce(r.get("tool"))

        res = None
        last_err = None
        for attempt in range(3):
            try:
                res = await run_row(client, question, context, tool_cell)
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "not found" in msg or "model" in msg and "live" not in msg:
                    raise
                print(f"[{idx}] attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2 * (attempt + 1))

        row_dict = r.to_dict()
        if res is not None:
            print(idx, res)
            row_dict.update(
                {
                    "assistant_text": res["assistant_text"],
                    "tool_calls": json.dumps(res["tool_calls"], ensure_ascii=False),
                    "model_intent": res["model_intent"],
                }
            )
        else:
            print(f"[{idx}] ERROR after retries: {last_err}")
            row_dict.update(
                {
                    "assistant_text": "",
                    "tool_calls": json.dumps({"error": str(last_err)}, ensure_ascii=False),
                    "model_intent": "error",
                }
            )
        results.append(row_dict)
        completed_ids.add(row_id)
        write_results_csv(out_csv, out_fields, results)

    print(f"Done. Wrote {len(results)} rows to {out_csv}")


async def main():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get an API key from Google AI Studio and export GEMINI_API_KEY."
        )
    if os.environ.get("GEMINI_API_KEY"):
        os.environ.pop("GOOGLE_API_KEY", None)

    client = genai.Client(api_key=api_key)

    in_csvs = [
        "GPT-4o-Mini-STT-Confetti-GPT.csv",
        "GPT-4o-Mini-STT-When2Call-GPT.csv",
    ]
    for in_csv in in_csvs:
        await process_csv(client, in_csv)


if __name__ == "__main__":
    asyncio.run(main())